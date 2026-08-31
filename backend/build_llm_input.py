"""Build one completed user's privacy-minimized LLM analysis input.

This module is intentionally read-only with respect to experiment.db. It does
not import or modify survey, routing, episode, or feature-calculation logic.

Usage:
    python build_llm_input.py --user-id USER_ID
    python build_llm_input.py --user-id USER_ID --output path/to/input.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = BACKEND_DIR / "data" / "experiment.db"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "data" / "user_analysis_input.json"
EPISODES = ("E1", "E2", "E3", "E4", "E5", "E6")
FEATURE_TABLES = {episode: f"{episode.lower()}_features" for episode in EPISODES}

STATED_FEATURE_FIELDS = (
    "risk_capacity_age",
    "investment_horizon",
    "risky_asset_experience",
    "experience_breadth",
    "derivative_experience",
    "stated_loss_tolerance",
    "investment_exposure",
    "financial_capacity",
    "return_seeking",
    "financial_literacy",
)

FEATURE_METADATA_FIELDS = {
    "session_id",
    "survey_id",
    "user_id",
    "feature_version",
    "computed_at",
    "calculated_at",
    "episode_status",
}

BOOLEAN_FIELDS = {
    "never_entered",
    "floor_reached",
    "e4_routing_fallback",
    "e4_upper_level_capped",
    "vulnerability_flag",
}

COMMON_DECISION_FIELDS = (
    "decision_point",
    "day",
    "risk_share_before",
    "risk_share_after",
    "delta_risk_share",
    "decision_time_ms",
    "normalized_price",
    "return_from_initial",
    "drawdown_from_peak",
    "trailing_return_5d",
    "return_since_previous_dp",
    "semantic_role",
    "market_phase",
)

EPISODE_DECISION_FIELDS = {
    "E3": (
        "allocation_floor",
        "floor_reached",
        "initial_preallocated_risk_share",
    ),
    "E4": (
        "abs_return_since_previous_dp",
        "max_abs_daily_return_since_previous_dp",
        "rolling_volatility_20d",
        "previous_dp_volatility_20d",
        "delta_volatility_20d",
        "volatility_percentile",
        "volatility_direction",
    ),
}

MARKET_SNAPSHOT_FIELDS = (
    "normalized_price",
    "return_from_initial",
    "drawdown_from_peak",
    "trailing_return_5d",
    "rolling_volatility_20d",
)


class LlmInputBuildError(RuntimeError):
    """Raised when a user is incomplete or stored data is inconsistent."""


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    resolved = database_path.resolve()
    if not resolved.is_file():
        raise LlmInputBuildError(f"Database does not exist: {resolved}")
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _clean_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field in BOOLEAN_FIELDS:
        return bool(value)
    if isinstance(value, float):
        return round(value, 8)
    return value


def _select_fields(row: sqlite3.Row, fields: Iterable[str]) -> dict[str, Any]:
    available = set(row.keys())
    return {
        field: _clean_value(field, row[field])
        for field in fields
        if field in available
    }


def _completed_sessions(
    connection: sqlite3.Connection, user_id: str
) -> dict[str, sqlite3.Row]:
    rows = connection.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY episode",
        (user_id,),
    ).fetchall()
    sessions = {str(row["episode"]): row for row in rows}
    missing = [episode for episode in EPISODES if episode not in sessions]
    incomplete = [
        episode
        for episode in EPISODES
        if episode in sessions and sessions[episode]["episode_status"] != "completed"
    ]
    if missing or incomplete:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if incomplete:
            details.append("incomplete=" + ",".join(incomplete))
        raise LlmInputBuildError(
            "Only users who completed Survey and Episodes 1-6 can be exported ("
            + "; ".join(details)
            + ")"
        )
    return sessions


def _stated_preference(
    connection: sqlite3.Connection, user_id: str
) -> dict[str, Any]:
    survey = connection.execute(
        "SELECT 1 FROM survey_results WHERE user_id = ?", (user_id,)
    ).fetchone()
    stated = connection.execute(
        "SELECT * FROM stated_features WHERE user_id = ?", (user_id,)
    ).fetchone()
    if survey is None or stated is None:
        raise LlmInputBuildError("Completed survey result and stated features are required")
    features = _select_fields(stated, STATED_FEATURE_FIELDS)
    missing = set(STATED_FEATURE_FIELDS) - set(features)
    if missing:
        raise LlmInputBuildError(
            "Missing stated feature columns: " + ", ".join(sorted(missing))
        )
    return {
        "features": features,
        "investor_protection_metadata": {
            "vulnerability_flag": bool(stated["vulnerability_flag"])
        },
    }


def _summary_features(
    connection: sqlite3.Connection,
    episode: str,
    session_id: str,
) -> dict[str, Any]:
    table = FEATURE_TABLES[episode]
    row = connection.execute(
        f"SELECT * FROM {table} WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None or row["episode_status"] != "completed":
        raise LlmInputBuildError(f"Completed {episode} feature row is required")
    return {
        field: _clean_value(field, value)
        for field, value in dict(row).items()
        if field not in FEATURE_METADATA_FIELDS
    }


def _decision_logs(
    connection: sqlite3.Connection,
    episode: str,
    session_id: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM behavior_events WHERE session_id = ? "
        "AND event_phase = 'allocation' ORDER BY decision_index",
        (session_id,),
    ).fetchall()
    if len(rows) != 7 or [int(row["decision_index"]) for row in rows] != list(
        range(1, 8)
    ):
        raise LlmInputBuildError(f"{episode} requires exactly seven ordered decisions")
    fields = COMMON_DECISION_FIELDS + EPISODE_DECISION_FIELDS.get(episode, ())
    return [_select_fields(row, fields) for row in rows]


def _adaptive_context(episode: str, session: sqlite3.Row) -> dict[str, Any]:
    if episode == "E3":
        fields = (
            "assigned_level",
            "routing_score",
            "scenario_max_drawdown",
            "entry_risk_share",
            "allocation_floor",
        )
    elif episode == "E4":
        fields = (
            "assigned_volatility_level",
            "e4_routing_score",
            "e4_routing_fallback",
            "e4_context_gap",
            "e4_upper_level_capped",
            "scenario_volatility_60d",
            "scenario_volatility_20d_min",
            "scenario_volatility_20d_max",
            "scenario_volatility_20d_q25",
            "scenario_volatility_20d_q75",
            "entry_risk_share",
        )
    else:
        raise ValueError(f"Adaptive context is not defined for {episode}")
    return _select_fields(session, fields)


def _parse_json_list(raw_value: Any, field_name: str) -> list[Any]:
    try:
        parsed = json.loads(str(raw_value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise LlmInputBuildError(f"Invalid {field_name} JSON") from exc
    if not isinstance(parsed, list):
        raise LlmInputBuildError(f"{field_name} must contain a JSON array")
    return parsed


def _public_sources(post: sqlite3.Row) -> list[dict[str, Any]]:
    sources = _parse_json_list(post["stimulus_pair_json"], "stimulus_pair_json")
    cleaned: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise LlmInputBuildError("Stimulus source must be an object")
        cleaned.append(
            {
                field: source.get(field)
                for field in ("source", "sentiment", "strength")
            }
        )
    return cleaned


def _information_events(
    connection: sqlite3.Connection, session_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM behavior_events WHERE session_id = ? "
        "ORDER BY decision_index, CASE event_phase "
        "WHEN 'pre_information' THEN 0 ELSE 1 END",
        (session_id,),
    ).fetchall()
    grouped: dict[int, dict[str, sqlite3.Row]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["decision_index"])][str(row["event_phase"])] = row
    if set(grouped) != {1, 2, 3}:
        raise LlmInputBuildError("E5 requires exactly three information decisions")

    events: list[dict[str, Any]] = []
    for index in range(1, 4):
        phases = grouped[index]
        if set(phases) != {"pre_information", "post_information"}:
            raise LlmInputBuildError(
                f"E5 DP{index} requires one PRE and one POST event"
            )
        pre = phases["pre_information"]
        post = phases["post_information"]
        if any(pre[field] != post[field] for field in MARKET_SNAPSHOT_FIELDS):
            raise LlmInputBuildError(
                f"E5 DP{index} PRE/POST market snapshots are inconsistent"
            )
        if pre["market_snapshot_id"] != post["market_snapshot_id"]:
            raise LlmInputBuildError(f"E5 DP{index} snapshot IDs do not match")

        display_order = _parse_json_list(post["display_order_json"], "display_order_json")
        events.append(
            {
                "decision_point": post["decision_point"],
                "day": int(post["day"]),
                "semantic_role": post["semantic_role"],
                "market_snapshot": _select_fields(post, MARKET_SNAPSHOT_FIELDS),
                "risk_share_before_pre": _clean_value(
                    "risk_share_before_pre", pre["risk_share_before_pre"]
                ),
                "pre_information_allocation": _clean_value(
                    "risk_share_pre_info", pre["risk_share_pre_info"]
                ),
                "pre_information_delta": _clean_value(
                    "pre_information_delta", pre["pre_information_delta"]
                ),
                "pre_information_decision_time_ms": int(pre["decision_time_ms"]),
                "sources": _public_sources(post),
                "display_order": display_order,
                "post_information_allocation": _clean_value(
                    "risk_share_post_info", post["risk_share_post_info"]
                ),
                "information_delta": _clean_value(
                    "information_delta", post["information_delta"]
                ),
                "aligned_source": post["aligned_source"],
                "post_information_decision_time_ms": int(post["decision_time_ms"]),
            }
        )
    return events


def _analysis_request() -> dict[str, Any]:
    dimension = {"score": None, "confidence": None, "reason": None}
    return {
        "instructions": [
            "Compare stated preference with revealed behavior without treating either as ground truth.",
            "Ground every reason in the supplied episode features or decisions.",
            "Return scores and confidence values between 0 and 1.",
            "Return JSON only and do not infer a real ticker, date range, or identity.",
        ],
        "required_output_format": {
            "risk_engagement": dict(dimension),
            "loss_resilience": dict(dimension),
            "volatility_tolerance": dict(dimension),
            "information_sensitivity": dict(dimension),
            "cross_context_consistency": dict(dimension),
            "revealed_investor_profile": None,
        },
        "revealed_investor_profile_allowed_values": [
            "안정형",
            "안정추구형",
            "위험중립형",
            "적극투자형",
            "공격투자형",
        ],
    }


def build_llm_input(database_path: Path, user_id: str) -> dict[str, Any]:
    """Read and validate exactly one completed user's analysis data."""
    if not user_id or len(user_id) > 128:
        raise LlmInputBuildError("A valid user_id is required")
    with _connect_read_only(database_path) as connection:
        sessions = _completed_sessions(connection, user_id)
        stated = _stated_preference(connection, user_id)
        behavioral: dict[str, Any] = {}
        for episode in EPISODES:
            session = sessions[episode]
            episode_payload: dict[str, Any] = {}
            if episode in {"E3", "E4"}:
                episode_payload["adaptive_context"] = _adaptive_context(
                    episode, session
                )
            episode_payload["summary_features"] = _summary_features(
                connection, episode, str(session["session_id"])
            )
            if episode == "E5":
                episode_payload["information_events"] = _information_events(
                    connection, str(session["session_id"])
                )
            else:
                episode_payload["decisions"] = _decision_logs(
                    connection, episode, str(session["session_id"])
                )
            behavioral[f"episode{episode[1:]}"] = episode_payload

    return {
        "stated_preference": stated,
        "behavioral_analysis": behavioral,
        "analysis_request": _analysis_request(),
    }


def write_llm_input(output_path: Path, payload: dict[str, Any]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one completed user's structured LLM analysis input"
    )
    parser.add_argument("--user-id", required=True, help="Exact experiment user_id")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            os.getenv("EXPERIMENT_DB_PATH", str(DEFAULT_DATABASE_PATH))
        ),
        help="SQLite experiment database path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path (default: backend/data/user_analysis_input.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = build_llm_input(args.database, args.user_id)
        write_llm_input(args.output, payload)
    except (LlmInputBuildError, sqlite3.DatabaseError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Created: {args.output.resolve()}")
    print("Exported users: 1")
    print("Completed episodes: E1, E2, E3, E4, E5, E6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

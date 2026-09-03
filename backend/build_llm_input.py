"""Build privacy-minimized, two-stage LLM inputs for one completed user.

Stage 1 (behavioral) never loads stated-preference values into the payload.
After the LLM returns ordinal behavioral dimensions, Python deterministically
calculates the revealed risk score/profile. Stage 2 (comparison) receives only
the fixed revealed result and stated preference.

Usage:
    python build_llm_input.py --stage behavioral --user-id USER_ID
    python build_llm_input.py --stage comparison --user-id USER_ID \
        --revealed-result revealed_dimensions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = BACKEND_DIR / "data" / "experiment.db"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "data" / "user_analysis_input.json"
DEFAULT_FEATURE_GUIDE_PATH = BACKEND_DIR / "data" / "llm_feature_guide.json"
DEFAULT_MANIFEST_PATH = BACKEND_DIR / "feature_schema_manifest.json"
EPISODES = ("E1", "E2", "E3", "E4", "E5", "E6")
FEATURE_TABLES = {episode: f"{episode.lower()}_features" for episode in EPISODES}

BOOLEAN_FIELDS = {
    "floor_reached",
    "e4_routing_fallback",
    "e4_upper_level_capped",
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
    """Raised when source data, manifest, or an LLM stage result is invalid."""


def load_feature_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LlmInputBuildError(f"Feature manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise LlmInputBuildError(f"Invalid feature manifest JSON: {manifest_path}") from exc

    required_top_level = {
        "schema_version",
        "feature_schema_version",
        "feature_selection_schema_version",
        "input_schema_versions",
        "policy",
        "revealed_profile_scoring",
        "behavioral_dimension_rubrics",
        "stated_preference",
        "episodes",
    }
    missing = required_top_level - set(manifest)
    if missing:
        raise LlmInputBuildError(
            "Feature manifest is missing: " + ", ".join(sorted(missing))
        )
    if set(manifest["episodes"]) != set(EPISODES):
        raise LlmInputBuildError("Feature manifest must define exactly E1 through E6")
    if set(manifest["input_schema_versions"]) != {"behavioral", "comparison"}:
        raise LlmInputBuildError(
            "Manifest input_schema_versions must define behavioral and comparison"
        )

    stated_specification = manifest["stated_preference"]
    if not isinstance(stated_specification.get("feature_version"), str):
        raise LlmInputBuildError("Stated-preference feature_version is required")
    stated_definitions = stated_specification.get("features")
    if not isinstance(stated_definitions, dict) or not stated_definitions:
        raise LlmInputBuildError("Stated-preference feature definitions are required")
    included_stated = _included_stated_feature_names(manifest)
    if not included_stated:
        raise LlmInputBuildError("No stated-preference features are included for LLM")
    for name, definition in stated_definitions.items():
        if type(definition.get("include_in_llm")) is not bool:
            raise LlmInputBuildError(
                f"stated_preference.{name} must explicitly define include_in_llm"
            )
        if not isinstance(definition.get("meaning"), str):
            raise LlmInputBuildError(
                f"stated_preference.{name} meaning is required"
            )
        if definition["include_in_llm"] is True and not isinstance(
            definition.get("direction"), str
        ):
            raise LlmInputBuildError(
                f"stated_preference.{name} direction is required when included"
            )
        if definition["include_in_llm"] is False and not isinstance(
            definition.get("exclusion_reason"), str
        ):
            raise LlmInputBuildError(
                f"stated_preference.{name} exclusion_reason is required when excluded"
            )

    for episode in EPISODES:
        specification = manifest["episodes"][episode]
        if not isinstance(specification.get("feature_version"), str):
            raise LlmInputBuildError(f"{episode} manifest feature_version is required")
        features = specification.get("features")
        if not isinstance(features, dict) or not features:
            raise LlmInputBuildError(f"{episode} manifest features are required")
        included = _included_feature_names(manifest, episode)
        if not included:
            raise LlmInputBuildError(f"{episode} has no LLM-included features")
        for name in included:
            definition = features[name]
            if not isinstance(definition.get("meaning"), str) or not isinstance(
                definition.get("direction"), str
            ):
                raise LlmInputBuildError(
                    f"{episode}.{name} requires reviewed meaning and direction"
                )

    scoring = manifest["revealed_profile_scoring"]
    ordinal_values = scoring.get("ordinal_values")
    core_dimensions = scoring.get("core_dimensions")
    modifier_dimensions = scoring.get("modifier_dimensions")
    bands = scoring.get("profile_bands")
    if not isinstance(ordinal_values, dict) or not ordinal_values:
        raise LlmInputBuildError("Manifest ordinal_values are required")
    if not isinstance(core_dimensions, list) or len(core_dimensions) != 3:
        raise LlmInputBuildError("Manifest must define three core dimensions")
    if not isinstance(modifier_dimensions, list):
        raise LlmInputBuildError("Manifest modifier_dimensions must be a list")
    if not isinstance(bands, list) or not bands:
        raise LlmInputBuildError("Manifest profile_bands are required")
    all_dimensions = set(core_dimensions) | set(modifier_dimensions)
    rubrics = manifest["behavioral_dimension_rubrics"]
    rubric_dimensions = rubrics.get("dimensions")
    if not isinstance(rubric_dimensions, dict) or set(rubric_dimensions) != all_dimensions:
        raise LlmInputBuildError(
            "Manifest rubrics must exactly match core and modifier dimensions"
        )
    for dimension, rubric in rubric_dimensions.items():
        if set(rubric.get("levels", {})) != set(ordinal_values):
            raise LlmInputBuildError(
                f"{dimension} rubric levels must match ordinal_values exactly"
            )
        if not rubric.get("primary_evidence") or not rubric.get("anchor_calculation"):
            raise LlmInputBuildError(
                f"{dimension} rubric requires primary evidence and anchor calculation"
            )
        anchor = rubric.get("anchor")
        if not isinstance(anchor, dict) or anchor.get("type") not in {
            "path_value",
            "weighted_mean",
        }:
            raise LlmInputBuildError(
                f"{dimension} rubric requires a supported machine-readable anchor"
            )
        if anchor["type"] == "path_value" and not isinstance(
            anchor.get("path"), str
        ):
            raise LlmInputBuildError(f"{dimension} path_value anchor requires path")
        if anchor["type"] == "weighted_mean":
            inputs = anchor.get("inputs")
            if not isinstance(inputs, list) or not inputs or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("weight"), (int, float))
                or float(item["weight"]) <= 0
                for item in inputs
            ):
                raise LlmInputBuildError(
                    f"{dimension} weighted_mean anchor inputs are invalid"
                )
        cutoffs = rubric.get("cutoffs")
        if not isinstance(cutoffs, list) or [
            cutoff.get("level") for cutoff in cutoffs
        ] != list(ordinal_values):
            raise LlmInputBuildError(
                f"{dimension} machine cutoffs must follow ordinal_values order"
            )
        previous: Mapping[str, Any] | None = None
        for cutoff in cutoffs:
            required_cutoff_fields = {
                "level", "min", "max", "min_inclusive", "max_inclusive"
            }
            if not required_cutoff_fields <= set(cutoff) or not isinstance(
                cutoff["min"], (int, float)
            ) or not isinstance(cutoff["max"], (int, float)):
                raise LlmInputBuildError(
                    f"{dimension} cutoff is not machine-readable"
                )
            if float(cutoff["min"]) > float(cutoff["max"]):
                raise LlmInputBuildError(
                    f"{dimension} cutoff min exceeds max"
                )
            if previous is not None:
                if float(previous["max"]) != float(cutoff["min"]):
                    raise LlmInputBuildError(
                        f"{dimension} cutoffs contain a gap or overlap"
                    )
                if bool(previous["max_inclusive"]) == bool(
                    cutoff["min_inclusive"]
                ):
                    raise LlmInputBuildError(
                        f"{dimension} cutoff boundary must belong to exactly one level"
                    )
            previous = cutoff
        if not isinstance(rubric.get("max_llm_adjustment_steps"), int) or int(
            rubric["max_llm_adjustment_steps"]
        ) < 0:
            raise LlmInputBuildError(
                f"{dimension} max_llm_adjustment_steps must be non-negative"
            )
    serialized_manifest = json.dumps(manifest, ensure_ascii=False)
    if "\ufffd" in serialized_manifest:
        raise LlmInputBuildError(
            "Feature manifest contains Unicode replacement characters; repair UTF-8 text"
        )
    return manifest


def _included_feature_names(
    manifest: Mapping[str, Any], episode: str
) -> tuple[str, ...]:
    features = manifest["episodes"][episode]["features"]
    return tuple(
        name
        for name, definition in features.items()
        if definition.get("include_in_llm") is True
    )


def _included_stated_feature_names(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    definitions = manifest["stated_preference"]["features"]
    return tuple(
        name
        for name, definition in definitions.items()
        if definition.get("include_in_llm") is True
    )


def build_feature_guide(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the reusable, versioned guide from the canonical manifest."""
    return {
        "manifest_schema_version": manifest["schema_version"],
        "feature_schema_version": manifest["feature_schema_version"],
        "feature_selection_schema_version": manifest[
            "feature_selection_schema_version"
        ],
        "policy": manifest["policy"],
        "shared_conventions": {
            "risk_share_and_allocation_changes": (
                "0~1 proportion; 0.10 means 10 percentage points"
            ),
            "returns_and_drawdowns": "decimal returns; -0.05 means -5%",
            "decision_time": "milliseconds",
            "null": "not observed or not validly calculable; never substitute zero",
            "evidence_rule": (
                "Composite values and their components are one evidence chain "
                "and must not be counted independently."
            ),
        },
        "common_decision_fields": {
            "risk_share_before": "risk allocation immediately before the decision",
            "risk_share_after": "risk allocation selected at the decision",
            "delta_risk_share": "risk_share_after - risk_share_before",
            "normalized_price": "current price with Day 1 normalized to 100",
            "return_from_initial": "return from Day 1 to the current day",
            "drawdown_from_peak": "decline from the running peak through the current day",
            "trailing_return_5d": "return over the preceding five trading days",
            "return_since_previous_dp": "return since the previous decision point",
            "semantic_role": "preconfigured experimental role of the decision point",
            "market_phase": "server-defined market-state tag",
        },
        "stated_preference": {
            "feature_version": manifest["stated_preference"]["feature_version"],
            "features": manifest["stated_preference"]["features"],
        },
        "episodes": {
            f"episode{episode[1:]}": {
                "feature_version": manifest["episodes"][episode]["feature_version"],
                "summary_features": {
                    name: {
                        key: value
                        for key, value in manifest["episodes"][episode]["features"][
                            name
                        ].items()
                        if key != "include_in_llm"
                    }
                    for name in _included_feature_names(manifest, episode)
                },
                **(
                    {
                        "adaptive_context": manifest["episodes"][episode][
                            "adaptive_context"
                        ]
                    }
                    if "adaptive_context" in manifest["episodes"][episode]
                    else {}
                ),
                **(
                    {
                        "information_event_fields": manifest["episodes"][episode][
                            "information_event_fields"
                        ]
                    }
                    if "information_event_fields" in manifest["episodes"][episode]
                    else {}
                ),
            }
            for episode in EPISODES
        },
        "revealed_profile_scoring": manifest["revealed_profile_scoring"],
        "behavioral_dimension_rubrics": manifest[
            "behavioral_dimension_rubrics"
        ],
    }


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
    survey = connection.execute(
        "SELECT 1 FROM survey_results WHERE user_id = ?", (user_id,)
    ).fetchone()
    if survey is None:
        raise LlmInputBuildError("Completed stated-preference survey is required")

    rows = connection.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY episode", (user_id,)
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
    connection: sqlite3.Connection,
    user_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    survey = connection.execute(
        "SELECT score,profile FROM survey_results WHERE user_id = ?", (user_id,)
    ).fetchone()
    stated = connection.execute(
        "SELECT * FROM stated_features WHERE user_id = ?", (user_id,)
    ).fetchone()
    if survey is None or stated is None:
        raise LlmInputBuildError("Survey result and stated features are required")
    actual_version = str(stated["feature_version"])
    expected_version = str(manifest["stated_preference"]["feature_version"])
    if actual_version != expected_version:
        raise LlmInputBuildError(
            "Stated-preference feature version mismatch: "
            f"DB={actual_version}, manifest={expected_version}. Review stated "
            "feature meanings and update the canonical manifest before export."
        )
    included_fields = _included_stated_feature_names(manifest)
    features = _select_fields(stated, included_fields)
    missing = set(included_fields) - set(features)
    if missing:
        raise LlmInputBuildError(
            "Missing stated feature columns: " + ", ".join(sorted(missing))
        )
    return {
        "feature_version": actual_version,
        "features": features,
        "survey_baseline": {
            "score": _clean_value("score", survey["score"]),
            "profile": survey["profile"],
        },
    }


def _summary_features(
    connection: sqlite3.Connection,
    episode: str,
    session_id: str,
    manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    row = connection.execute(
        f"SELECT * FROM {FEATURE_TABLES[episode]} WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None or row["episode_status"] != "completed":
        raise LlmInputBuildError(f"Completed {episode} feature row is required")

    actual_version = str(row["feature_version"])
    expected_version = str(manifest["episodes"][episode]["feature_version"])
    if actual_version != expected_version:
        raise LlmInputBuildError(
            f"{episode} feature version mismatch: DB={actual_version}, "
            f"manifest={expected_version}. Review feature meanings and update the "
            "canonical manifest before export."
        )

    allowlist = _included_feature_names(manifest, episode)
    missing = set(allowlist) - set(row.keys())
    if missing:
        raise LlmInputBuildError(
            f"Missing {episode} allowlisted feature columns: "
            + ", ".join(sorted(missing))
        )
    return actual_version, _select_fields(row, allowlist)


def _decision_logs(
    connection: sqlite3.Connection, episode: str, session_id: str
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


def _adaptive_context(
    episode: str,
    session: sqlite3.Row,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    definitions = manifest["episodes"][episode].get("adaptive_context")
    if not isinstance(definitions, dict) or not definitions:
        raise LlmInputBuildError(
            f"{episode} adaptive context is not defined in the manifest"
        )
    missing = set(definitions) - set(session.keys())
    if missing:
        raise LlmInputBuildError(
            f"Missing {episode} adaptive context columns: "
            + ", ".join(sorted(missing))
        )
    return _select_fields(session, definitions)


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
                "display_order": _parse_json_list(
                    post["display_order_json"], "display_order_json"
                ),
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


def _behavioral_analysis(
    connection: sqlite3.Connection,
    sessions: Mapping[str, sqlite3.Row],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    behavioral: dict[str, Any] = {}
    for episode in EPISODES:
        session = sessions[episode]
        episode_payload: dict[str, Any] = {}
        if "adaptive_context" in manifest["episodes"][episode]:
            episode_payload["adaptive_context"] = _adaptive_context(
                episode, session, manifest
            )
        version, summary = _summary_features(
            connection, episode, str(session["session_id"]), manifest
        )
        episode_payload["feature_version"] = version
        episode_payload["summary_features"] = summary
        if episode == "E5":
            episode_payload["information_events"] = _information_events(
                connection, str(session["session_id"])
            )
        else:
            episode_payload["decisions"] = _decision_logs(
                connection, episode, str(session["session_id"])
            )
        behavioral[f"episode{episode[1:]}"] = episode_payload
    return behavioral


def _resolve_json_path(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise LlmInputBuildError(f"Manifest JSON path does not exist: {path}")
        value = value[segment]
    return value


def _calculate_anchor_value(
    payload: Mapping[str, Any], anchor: Mapping[str, Any]
) -> float | None:
    anchor_type = anchor["type"]
    if anchor_type == "path_value":
        raw_value = _resolve_json_path(payload, str(anchor["path"]))
        return None if raw_value is None else float(raw_value)
    if anchor_type == "weighted_mean":
        weighted_values: list[tuple[float, float]] = []
        for item in anchor["inputs"]:
            raw_value = _resolve_json_path(payload, str(item["path"]))
            if raw_value is not None:
                weighted_values.append((float(raw_value), float(item["weight"])))
        if not weighted_values:
            return None
        total_weight = sum(weight for _, weight in weighted_values)
        return sum(value * weight for value, weight in weighted_values) / total_weight
    raise LlmInputBuildError(f"Unsupported rubric anchor type: {anchor_type}")


def _level_from_cutoffs(value: float, rubric: Mapping[str, Any]) -> str:
    for cutoff in rubric["cutoffs"]:
        minimum = float(cutoff["min"])
        maximum = float(cutoff["max"])
        above_minimum = value >= minimum if cutoff["min_inclusive"] else value > minimum
        below_maximum = value <= maximum if cutoff["max_inclusive"] else value < maximum
        if above_minimum and below_maximum:
            return str(cutoff["level"])
    raise LlmInputBuildError(
        f"Quantitative anchor {value} does not match any cutoff for "
        f"{rubric['rubric_id']}"
    )


def calculate_quantitative_baselines(
    behavioral_analysis: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Calculate manifest-driven anchors and immutable base levels in Python."""
    payload = {"behavioral_analysis": behavioral_analysis}
    baselines: dict[str, dict[str, Any]] = {}
    rubrics = manifest["behavioral_dimension_rubrics"]["dimensions"]
    for dimension, rubric in rubrics.items():
        anchor_value = _calculate_anchor_value(payload, rubric["anchor"])
        if anchor_value is None:
            baselines[dimension] = {
                "rubric_id": rubric["rubric_id"],
                "classification_status": "insufficient_evidence",
                "anchor_value": None,
                "base_level": None,
                "max_llm_adjustment_steps": rubric[
                    "max_llm_adjustment_steps"
                ],
            }
            continue
        baselines[dimension] = {
            "rubric_id": rubric["rubric_id"],
            "classification_status": "anchored",
            "anchor_value": round(anchor_value, 8),
            "base_level": _level_from_cutoffs(anchor_value, rubric),
            "max_llm_adjustment_steps": rubric["max_llm_adjustment_steps"],
        }
    return baselines


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _confidence_level_from_consistency(
    consistency: float | None, manifest: Mapping[str, Any]
) -> str:
    if consistency is None:
        return "low"
    thresholds = manifest["revealed_profile_scoring"][
        "cross_context_calibration"
    ]["confidence_thresholds"]
    if consistency >= float(thresholds["high_min"]):
        return "high"
    if consistency >= float(thresholds["medium_min"]):
        return "medium"
    return "low"


def _gap_direction(
    earlier: float, anchor: float, gap: float, manifest: Mapping[str, Any]
) -> str:
    rules = manifest["revealed_profile_scoring"]["cross_context_calibration"][
        "risk_adjustment"
    ]
    aligned_max = float(rules["aligned_gap_max_exclusive"])
    strong_min = float(rules["deterministic_gap_min_exclusive"])
    if gap < aligned_max:
        return "aligned"
    if anchor < earlier:
        return "strongly_lower" if gap > strong_min else "lower"
    return "strongly_higher" if gap > strong_min else "higher"


def calculate_cross_context_calibration(
    behavioral_analysis: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Calculate private E6 gaps and deterministic confidence/adjustment anchors."""
    episode3 = behavioral_analysis["episode3"]
    episode6 = behavioral_analysis["episode6"]
    e3_summary = episode3["summary_features"]
    e6_summary = episode6["summary_features"]

    pre_e6_risk = _optional_float(
        episode3["adaptive_context"].get("routing_score")
    )
    e6_risk = _optional_float(e6_summary.get("anchor_risk_exposure_auc"))
    risk_consistency = _optional_float(
        e6_summary.get("risk_engagement_consistency")
    )
    pre_e6_loss = _optional_float(
        e3_summary.get("behavior_resilience_score")
    )
    e6_loss = _optional_float(
        e6_summary.get("e6_behavior_resilience_score")
    )
    loss_consistency = _optional_float(
        e6_summary.get("loss_response_consistency")
    )
    cross_context = _optional_float(
        e6_summary.get("cross_context_consistency")
    )

    def gap_payload(
        earlier: float | None,
        anchor: float | None,
        consistency: float | None,
        *,
        with_adjustment: bool,
    ) -> dict[str, Any]:
        if earlier is None or anchor is None:
            return {
                "pre_e6_value": earlier,
                "e6_value": anchor,
                "gap": None,
                "direction": "insufficient_evidence",
                "consistency_value": consistency,
                "consistency_level": _confidence_level_from_consistency(
                    consistency, manifest
                ),
                "suggested_adjustment": None,
            }
        gap = abs(earlier - anchor)
        direction = _gap_direction(earlier, anchor, gap, manifest)
        suggestion: int | None = None
        if with_adjustment:
            rules = manifest["revealed_profile_scoring"][
                "cross_context_calibration"
            ]["risk_adjustment"]
            aligned_max = float(rules["aligned_gap_max_exclusive"])
            strong_min = float(rules["deterministic_gap_min_exclusive"])
            if gap < aligned_max:
                suggestion = 0
            elif gap > strong_min:
                suggestion = -1 if anchor < earlier else 1
        return {
            "pre_e6_value": round(earlier, 8),
            "e6_value": round(anchor, 8),
            "gap": round(gap, 8),
            "direction": direction,
            "consistency_value": consistency,
            "consistency_level": _confidence_level_from_consistency(
                consistency, manifest
            ),
            "suggested_adjustment": suggestion,
        }

    return {
        "risk_engagement": gap_payload(
            pre_e6_risk, e6_risk, risk_consistency, with_adjustment=True
        ),
        "loss_resilience": gap_payload(
            pre_e6_loss, e6_loss, loss_consistency, with_adjustment=False
        ),
        "cross_context_consistency": cross_context,
        "behavioral_confidence_base": _confidence_level_from_consistency(
            cross_context, manifest
        ),
    }


def _apply_calibration_to_baselines(
    quantitative_baselines: dict[str, dict[str, Any]],
    calibration: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    suggestion = calibration["risk_engagement"]["suggested_adjustment"]
    base_level = quantitative_baselines["risk_engagement"]["base_level"]
    if suggestion is not None and base_level is not None:
        ordered = list(manifest["revealed_profile_scoring"]["ordinal_values"])
        base_index = ordered.index(base_level)
        if base_index + suggestion < 0 or base_index + suggestion >= len(ordered):
            suggestion = 0
    quantitative_baselines["risk_engagement"][
        "suggested_adjustment"
    ] = suggestion


def _dimension_output_template(base_level: str | None) -> dict[str, Any]:
    return {
        "base_level": base_level,
        "adjustment": None,
        "confidence_level": None,
        "reason": None,
        "evidence_fields": [],
    }


def _behavioral_request(
    manifest: Mapping[str, Any],
    quantitative_baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scoring = manifest["revealed_profile_scoring"]
    dimensions = list(scoring["core_dimensions"]) + list(
        scoring["modifier_dimensions"]
    )
    return {
        "task": "Infer revealed behavioral dimensions from behavioral evidence only.",
        "rubric_reference": {
            "feature_guide": "llm_feature_guide.json",
            "rubric_version": manifest["behavioral_dimension_rubrics"][
                "rubric_version"
            ],
            "dimension_rubric_ids": {
                dimension: manifest["behavioral_dimension_rubrics"]["dimensions"][
                    dimension
                ]["rubric_id"]
                for dimension in dimensions
            },
        },
        "strict_input_scope": (
            "Use behavioral_analysis and the matching feature guide only. "
            "Stated-preference data is intentionally absent."
        ),
        "rules": [
            "Echo each immutable base_level exactly as supplied.",
            "Return only an integer adjustment within the rubric's allowed step range.",
            "Use supporting evidence and raw behavior only to justify -1, 0, or +1 relative to the Python base level.",
            "For risk_engagement, when suggested_adjustment is not null, echo that exact Python adjustment; only the middle conflict band remains an LLM judgment.",
            "Apply the matching pre-defined rubric; do not invent different thresholds.",
            "Ground each reason in named evidence_fields.",
            "Do not treat composite features and their components as independent evidence.",
            "Do not create final levels, a revealed profile, or a numerical risk score; Python calculates them.",
            "Information sensitivity is descriptive and is not a conservative/aggressive direction.",
            "Never use information sensitivity to raise or lower risk_engagement, loss_resilience, volatility_tolerance, the risk score, or the investor profile.",
            "Cross-context consistency is a confidence modifier and never changes risk direction.",
            "Use null rather than zero when evidence is insufficient.",
            "When a Python base_level is null, return null for base_level, adjustment, and confidence_level and explain the missing evidence.",
            "Return JSON only.",
        ],
        "ordinal_levels": list(scoring["ordinal_values"]),
        "confidence_levels": list(scoring["confidence_levels"]),
        "required_output_format": {
            "revealed_behavioral_dimensions": {
                dimension: _dimension_output_template(
                    quantitative_baselines[dimension]["base_level"]
                )
                for dimension in dimensions
            }
        },
    }


def _extract_dimension_result(
    revealed_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    quantitative_baselines: Mapping[str, Mapping[str, Any]],
    behavioral_analysis: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    dimensions = revealed_result.get("revealed_behavioral_dimensions")
    if not isinstance(dimensions, dict):
        raise LlmInputBuildError(
            "Revealed result must contain revealed_behavioral_dimensions"
        )
    scoring = manifest["revealed_profile_scoring"]
    expected = set(scoring["core_dimensions"]) | set(scoring["modifier_dimensions"])
    if set(dimensions) != expected:
        raise LlmInputBuildError(
            "Revealed dimensions must match manifest exactly: "
            + ", ".join(sorted(expected))
        )
    valid_levels = set(scoring["ordinal_values"])
    ordered_levels = list(scoring["ordinal_values"])
    valid_confidence = set(scoring["confidence_levels"])
    cleaned: dict[str, dict[str, Any]] = {}
    for name in expected:
        value = dimensions[name]
        if not isinstance(value, dict):
            raise LlmInputBuildError(f"{name} result must be an object")
        base_level = value.get("base_level")
        adjustment = value.get("adjustment")
        confidence = value.get("confidence_level")
        reason = value.get("reason")
        evidence = value.get("evidence_fields")
        expected_base = quantitative_baselines[name]["base_level"]
        if base_level != expected_base:
            raise LlmInputBuildError(
                f"{name} base_level must match Python baseline: {expected_base}"
            )
        if base_level is not None and base_level not in valid_levels:
            raise LlmInputBuildError(f"Invalid {name} base_level: {base_level}")
        if base_level is None and adjustment is not None:
            raise LlmInputBuildError(
                f"{name} adjustment must be null when base_level is null"
            )
        if base_level is None and confidence is not None:
            raise LlmInputBuildError(
                f"{name} confidence_level must be null when base_level is null"
            )
        if base_level is not None and confidence not in valid_confidence:
            raise LlmInputBuildError(
                f"Invalid {name} confidence_level: {confidence}"
            )
        final_level: str | None = None
        if base_level is not None:
            maximum = int(quantitative_baselines[name]["max_llm_adjustment_steps"])
            if type(adjustment) is not int or abs(adjustment) > maximum:
                raise LlmInputBuildError(
                    f"{name} adjustment must be an integer between "
                    f"{-maximum} and {maximum}"
                )
            suggested = quantitative_baselines[name].get("suggested_adjustment")
            if suggested is not None and adjustment != suggested:
                raise LlmInputBuildError(
                    f"{name} adjustment must match Python calibration: {suggested}"
                )
            final_index = ordered_levels.index(base_level) + adjustment
            if final_index < 0 or final_index >= len(ordered_levels):
                raise LlmInputBuildError(
                    f"{name} adjustment moves outside the ordinal scale"
                )
            final_level = ordered_levels[final_index]
        if not isinstance(reason, str) or not reason.strip():
            raise LlmInputBuildError(f"{name} reason is required")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(
                isinstance(field, str) and bool(field.strip())
                for field in evidence
            )
        ):
            raise LlmInputBuildError(
                f"{name} evidence_fields must contain at least one non-empty path"
            )
        # harmless normalization
        evidence = list(dict.fromkeys(evidence))
        rubric = manifest["behavioral_dimension_rubrics"]["dimensions"][name]
        allowed_evidence = set(rubric["primary_evidence"]) | set(
            rubric["supporting_evidence"]
        )
        invalid_evidence = [
            field for field in evidence if field not in allowed_evidence
        ]
        if invalid_evidence:
            raise LlmInputBuildError(
                f"{name} evidence_fields are outside its manifest rubric: "
                + ", ".join(invalid_evidence)
            )
        evidence_payload = {"behavioral_analysis": behavioral_analysis}
        for field in evidence:
            try:
                _resolve_json_path(evidence_payload, field)
            except LlmInputBuildError as exc:
                raise LlmInputBuildError(
                    f"{name} evidence field does not exist in the actual input: {field}"
                ) from exc
        cleaned[name] = {
            "base_level": base_level,
            "adjustment": adjustment,
            "final_level": final_level,
            "confidence_level": confidence,
            "reason": reason.strip(),
            "evidence_fields": evidence,
        }
    return cleaned


def calculate_revealed_profile(
    revealed_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    quantitative_baselines: Mapping[str, Mapping[str, Any]],
    behavioral_analysis: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Deterministically convert the three core ordinal dimensions to a profile."""
    dimensions = _extract_dimension_result(
        revealed_result, manifest, quantitative_baselines, behavioral_analysis
    )
    scoring = manifest["revealed_profile_scoring"]
    ordinal_values = scoring["ordinal_values"]
    core_dimensions = list(scoring["core_dimensions"])
    component_values = {
        name: (
            None
            if dimensions[name]["final_level"] is None
            else float(ordinal_values[dimensions[name]["final_level"]])
        )
        for name in core_dimensions
    }
    missing_core = [
        name for name, value in component_values.items() if value is None
    ]
    if missing_core:
        return (
            {
                "classification_status": "insufficient_evidence",
                "risk_score": None,
                "profile": None,
                "scoring_method": scoring["score_calculation"],
                "core_dimension_values": component_values,
                "core_dimension_levels": {
                    name: dimensions[name]["final_level"]
                    for name in core_dimensions
                },
                "missing_core_dimensions": missing_core,
            },
            dimensions,
        )

    risk_score = round(
        mean(float(value) for value in component_values.values()), 2
    )

    profile = None
    for band in scoring["profile_bands"]:
        lower = band["lower_exclusive"]
        upper = band["upper_inclusive"]
        if (lower is None or risk_score > float(lower)) and risk_score <= float(upper):
            profile = band["profile"]
            break
    if profile is None:
        raise LlmInputBuildError(
            f"Revealed risk score is outside manifest profile bands: {risk_score}"
        )

    return (
        {
            "classification_status": "classified",
            "risk_score": risk_score,
            "profile": profile,
            "scoring_method": scoring["score_calculation"],
            "core_dimension_values": component_values,
            "core_dimension_levels": {
                name: dimensions[name]["final_level"]
                for name in core_dimensions
            },
        },
        dimensions,
    )


def _public_behavioral_observations(
    dimensions: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> list[str]:
    dimension_phrases = {
        "risk_engagement": {
            "very_low": "여러 시장 상황에서 위험자산 참여를 매우 제한적으로 유지했어요.",
            "low": "여러 시장 상황에서 위험자산 참여를 비교적 낮게 유지했어요.",
            "medium": "여러 시장 상황에서 위험자산에 중간 수준으로 참여했어요.",
            "high": "초기 시장 상황에서 비교적 높은 위험자산 비중을 유지했어요.",
            "very_high": "여러 시장 상황에서 매우 높은 위험자산 비중을 지속했어요.",
        },
        "loss_resilience": {
            "very_low": "손실 상황에서 위험 노출을 유지하려는 정도가 매우 낮게 관찰됐어요.",
            "low": "손실 상황에서 위험 노출을 비교적 낮게 유지했어요.",
            "medium": "손실 상황에서 위험 노출을 일부 유지하면서 신중하게 조정했어요.",
            "high": "손실 상황에서도 위험 노출을 비교적 많이 유지했어요.",
            "very_high": "손실 상황에서도 위험 노출을 매우 강하게 유지했어요.",
        },
        "volatility_tolerance": {
            "very_low": "가격 변동이 커진 상황에서 위험자산 노출을 매우 낮게 유지했어요.",
            "low": "가격 변동이 커진 상황에서 위험자산 노출을 비교적 낮게 유지했어요.",
            "medium": "가격 변동이 커진 상황에서 중간 수준의 위험자산 노출을 유지했어요.",
            "high": "가격 변동이 커진 상황에서도 비교적 높은 위험자산 노출을 유지했어요.",
            "very_high": "가격 변동이 커진 상황에서도 매우 높은 위험자산 노출을 유지했어요.",
        },
        "information_sensitivity": {
            "very_low": "상충된 외부 정보가 제시된 뒤에도 기존 선택을 거의 그대로 유지했어요.",
            "low": "상충된 외부 정보가 제시된 뒤 위험자산 비중을 소폭 조정했어요.",
            "medium": "상충된 외부 정보가 제시된 뒤 위험자산 비중을 중간 정도로 조정했어요.",
            "high": "상충된 외부 정보가 제시된 뒤 위험자산 비중을 비교적 크게 조정했어요.",
            "very_high": "상충된 외부 정보가 제시된 뒤 위험자산 비중을 매우 크게 조정했어요.",
        },
    }
    observations: list[str] = []
    for name, phrases in dimension_phrases.items():
        level = dimensions[name]["final_level"]
        if level in phrases:
            observations.append(phrases[level])

    risk_direction = calibration["risk_engagement"]["direction"]
    risk_phrases = {
        "aligned": "공통 시장 조건에서도 앞서 관찰된 위험자산 참여 수준과 대체로 비슷한 행동을 보였어요.",
        "lower": "공통 시장 조건에서는 앞선 상황보다 위험자산 노출을 다소 낮췄어요.",
        "strongly_lower": "공통 시장 조건에서는 앞선 상황보다 위험자산 노출을 뚜렷하게 낮췄어요.",
        "higher": "공통 시장 조건에서는 앞선 상황보다 위험자산 노출을 다소 높였어요.",
        "strongly_higher": "공통 시장 조건에서는 앞선 상황보다 위험자산 노출을 뚜렷하게 높였어요.",
    }
    if risk_direction in risk_phrases:
        observations.append(risk_phrases[risk_direction])

    loss_direction = calibration["loss_resilience"]["direction"]
    loss_phrases = {
        "aligned": "공통 시장 조건에서도 앞서 관찰된 손실 대응과 대체로 비슷한 행동을 보였어요.",
        "lower": "공통 시장 조건에서는 앞선 손실 상황보다 위험 노출을 유지하는 정도가 다소 낮았어요.",
        "strongly_lower": "공통 시장 조건에서는 앞선 손실 상황보다 위험 노출을 유지하는 정도가 뚜렷하게 낮았어요.",
        "higher": "공통 시장 조건에서는 앞선 손실 상황보다 위험 노출을 유지하는 정도가 다소 높았어요.",
        "strongly_higher": "공통 시장 조건에서는 앞선 손실 상황보다 위험 노출을 유지하는 정도가 뚜렷하게 높았어요.",
    }
    if loss_direction in loss_phrases:
        observations.append(loss_phrases[loss_direction])

    observations.append(
        "외부 정보에 대한 반응 크기는 공격적 또는 보수적 투자 방향을 뜻하지 않아요."
    )
    return observations


def _comparison_request(
    manifest: Mapping[str, Any],
    revealed_profile: Mapping[str, Any],
    confidence_level: str,
) -> dict[str, Any]:
    scoring = manifest["revealed_profile_scoring"]
    investor_type = revealed_profile["profile"] or "분석 근거 부족"
    return {
        "task": (
            "Explain the difference between the fixed stated and revealed results "
            "and describe behavioral modifiers."
        ),
        "immutable_fields": [
            "revealed_profile.classification_status",
            "revealed_profile.risk_score",
            "revealed_profile.profile",
            "stated_preference.survey_baseline.score",
            "stated_preference.survey_baseline.profile",
            "behavioral_modifiers.cross_context_consistency",
        ],
        "rules": [
            "Do not recalculate, relabel, or override the fixed revealed profile.",
            "If classification_status is insufficient_evidence, do not invent a risk score or profile.",
            "Echo the fixed revealed profile exactly in investor_type; do not create a third profile label.",
            "Information sensitivity describes responsiveness to information and must not change the risk profile.",
            "Cross-context consistency affects confidence/representativeness, not risk direction.",
            "Do not interpret the numeric difference between stated survey score and revealed risk score as a calibrated cardinal distance; compare profiles/categories and behavioral evidence instead.",
            "Explain stated-revealed agreement or gaps using the supplied fixed evidence.",
            "Write every user-facing explanation in natural Korean 해요체; do not use -습니다/-입니다 style or plain -다/-한다 style.",
            "Avoid repeating '사용자님'; omit the subject naturally whenever possible.",
            "This service measures choices in a market simulation, not real brokerage trading. Do not say 실제 투자, 실제 거래, or 실전 투자; use 행동 분석, 시장 선택 과정, or 시뮬레이션에서의 선택 instead.",
            "Matching stated and revealed profile labels does not prove behavioral consistency. Describe consistency only from cross-context calibration and verified behavioral evidence.",
            "Use confidence_level only to control wording strength: high may use 비교적 뚜렷하게 or 여러 상황에서 비슷한 경향; medium should use 전반적으로 or 일부 차이가 관찰됐어요 and avoid 일관된, 명확한, 확실한; low should use 제한된 근거에서는 or 상황에 따른 차이가 커 조심스럽게 해석할 필요가 있어요.",
            "Do not mention experiment names, episode numbers, internal feature names, JSON paths, rubrics, levels, scores, cutoffs, or adjustments.",
            "Do not claim an increase, decrease, hold, re-entry, or other action unless it appears explicitly in verified_behavioral_observations.",
            "Information responsiveness has no aggressive or conservative direction and must only be described as response magnitude.",
            "Choose confidence_level only from the manifest confidence enum and use the Python cross-context confidence anchor as calibration evidence.",
            "Do not recommend any investment product, allocation, purchase, sale, or investment strategy.",
            "Return JSON only.",
        ],
        "ordinal_levels": list(scoring["ordinal_values"]),
        "confidence_levels": list(scoring["confidence_levels"]),
        "required_output_format": {
            "investor_type": investor_type,
            "confidence_level": confidence_level,
            "stated_preference_summary": (
                "설문에서 확인된 성향을 자연스러운 한국어 해요체로 요약해요."
            ),
            "revealed_preference_summary": (
                "시뮬레이션의 시장 선택 과정에서 관찰된 성향을 자연스러운 한국어 해요체로 요약해요."
            ),
            "stated_revealed_gap": (
                "설문 응답과 시장 선택 과정의 일치점 또는 차이를 자연스러운 한국어 해요체로 설명해요."
            ),
            "key_behavioral_evidence": [],
            "final_analysis": (
                "검증된 행동 근거만 사용해 시장 선택 과정의 행동 특성을 자연스러운 한국어 해요체로 설명해요."
            ),
        },
    }


def _schema_references(
    manifest: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    return {
        "llm_input_schema_version": manifest["input_schema_versions"][stage],
        "feature_manifest_schema_version": manifest["schema_version"],
        "feature_schema_version": manifest["feature_schema_version"],
        "feature_selection_schema_version": manifest[
            "feature_selection_schema_version"
        ],
        "analysis_stage": stage,
    }


def build_behavioral_input(
    database_path: Path,
    user_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Call 1 input without ever placing stated values in its context."""
    if not user_id or len(user_id) > 128:
        raise LlmInputBuildError("A valid user_id is required")
    with _connect_read_only(database_path) as connection:
        sessions = _completed_sessions(connection, user_id)
        behavioral = _behavioral_analysis(connection, sessions, manifest)
    quantitative_baselines = calculate_quantitative_baselines(
        behavioral, manifest
    )
    calibration = calculate_cross_context_calibration(behavioral, manifest)
    _apply_calibration_to_baselines(
        quantitative_baselines, calibration, manifest
    )
    return {
        **_schema_references(manifest, "behavioral"),
        "behavioral_analysis": behavioral,
        "quantitative_baselines": quantitative_baselines,
        "cross_context_calibration": calibration,
        "analysis_request": _behavioral_request(
            manifest, quantitative_baselines
        ),
    }


def build_comparison_input(
    database_path: Path,
    user_id: str,
    revealed_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Call 2 from stated preference and a fixed Call 1 result."""
    if not user_id or len(user_id) > 128:
        raise LlmInputBuildError("A valid user_id is required")
    with _connect_read_only(database_path) as connection:
        sessions = _completed_sessions(connection, user_id)
        stated = _stated_preference(connection, user_id, manifest)
        behavioral = _behavioral_analysis(connection, sessions, manifest)
    quantitative_baselines = calculate_quantitative_baselines(
        behavioral, manifest
    )
    calibration = calculate_cross_context_calibration(behavioral, manifest)
    _apply_calibration_to_baselines(
        quantitative_baselines, calibration, manifest
    )
    revealed_profile, dimensions = calculate_revealed_profile(
        revealed_result, manifest, quantitative_baselines, behavioral
    )
    cross_context = behavioral["episode6"]["summary_features"][
        "cross_context_consistency"
    ]
    modifier_names = manifest["revealed_profile_scoring"]["modifier_dimensions"]
    modifiers = {name: dimensions[name] for name in modifier_names}
    modifiers["cross_context_consistency"] = cross_context
    return {
        **_schema_references(manifest, "comparison"),
        "stated_preference": stated,
        "revealed_profile": revealed_profile,
        "quantitative_baselines": quantitative_baselines,
        "behavioral_evidence": {
            name: dimensions[name]
            for name in manifest["revealed_profile_scoring"]["core_dimensions"]
        },
        "behavioral_modifiers": modifiers,
        "cross_context_calibration": calibration,
        "public_behavioral_observations": _public_behavioral_observations(
            dimensions, calibration
        ),
        "analysis_request": _comparison_request(
            manifest,
            revealed_profile,
            calibration["behavioral_confidence_base"],
        ),
    }


def build_llm_input(
    database_path: Path,
    user_id: str,
    *,
    stage: str = "behavioral",
    revealed_result: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for either isolated LLM stage."""
    active_manifest = (
        load_feature_manifest() if manifest is None else dict(manifest)
    )
    if stage == "behavioral":
        if revealed_result is not None:
            raise LlmInputBuildError(
                "revealed_result is not accepted during behavioral stage"
            )
        return build_behavioral_input(database_path, user_id, active_manifest)
    if stage == "comparison":
        if revealed_result is None:
            raise LlmInputBuildError(
                "comparison stage requires a revealed_result"
            )
        return build_comparison_input(
            database_path, user_id, revealed_result, active_manifest
        )
    raise LlmInputBuildError(f"Unsupported analysis stage: {stage}")


def write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


# Backward-compatible name used by targeted tooling.
write_llm_input = write_json


def _read_revealed_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LlmInputBuildError(f"Revealed result does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LlmInputBuildError(f"Invalid revealed result JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LlmInputBuildError("Revealed result must be a JSON object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one completed user's isolated LLM analysis input"
    )
    parser.add_argument("--user-id", required=True, help="Exact experiment user_id")
    parser.add_argument(
        "--stage",
        choices=("behavioral", "comparison"),
        default="behavioral",
        help="behavioral=Call 1, comparison=Call 2",
    )
    parser.add_argument(
        "--revealed-result",
        type=Path,
        help="Call 1 JSON result; required only for comparison stage",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("EXPERIMENT_DB_PATH", str(DEFAULT_DATABASE_PATH))),
        help="SQLite experiment database path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Canonical feature schema manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path (default: backend/data/user_analysis_input.json)",
    )
    parser.add_argument(
        "--feature-guide-output",
        type=Path,
        default=DEFAULT_FEATURE_GUIDE_PATH,
        help="Reusable feature guide path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = load_feature_manifest(args.manifest)
        revealed_result = (
            None
            if args.revealed_result is None
            else _read_revealed_result(args.revealed_result)
        )
        payload = build_llm_input(
            args.database,
            args.user_id,
            stage=args.stage,
            revealed_result=revealed_result,
            manifest=manifest,
        )
        write_json(args.output, payload)
        write_json(args.feature_guide_output, build_feature_guide(manifest))
    except (LlmInputBuildError, sqlite3.DatabaseError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Created: {args.output.resolve()}")
    print(f"Feature guide: {args.feature_guide_output.resolve()}")
    print(f"Analysis stage: {args.stage}")
    print("Exported users: 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

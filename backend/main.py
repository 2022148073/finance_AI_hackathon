"""FastAPI service for the sequential, adaptive Episode 1-6 experiment."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from database import (
    connect,
    fetch_logs,
    initialize_database,
    upsert_episode1_features,
    upsert_episode2_features,
    upsert_episode3_features,
    upsert_episode4_features,
    upsert_episode5_features,
    upsert_episode6_features,
    upsert_profile_cross_context,
)
from features import (
    calculate_episode1_features,
    calculate_episode2_features,
    calculate_episode3_features,
    calculate_episode4_features,
    calculate_episode5_features,
    calculate_episode6_features,
)
from routing import route_episode3, route_episode4
from scenario_store import Scenario, load_scenarios
from schemas import (
    DecisionSubmission,
    EntryRiskShareSubmission,
    Episode5PostSubmission,
    Episode5PreSubmission,
    StartSessionRequest,
    SurveySubmission,
)
from stimulus_store import POLARITY_CYCLES, Randomizer, SOURCE_PAIRS, StimulusStore
from survey import (
    QUESTIONNAIRE_VERSION,
    SCORING_BASIS,
    SCORING_VERSION,
    SOURCE_METADATA,
    STATED_FEATURE_VERSION,
    SurveyValidationError,
    calculate_stated_features,
    calculate_survey_score,
    classify_survey_profile,
    public_questionnaire,
    validate_raw_answers,
)


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = BACKEND_DIR / "data" / "experiment.db"
DEFAULT_SCENARIO_DIR = BACKEND_DIR / "scenarios"
DEFAULT_STIMULUS_DIR = DEFAULT_SCENARIO_DIR / "episode5" / "stimuli"
ScenarioPicker = Callable[[list[str]], str]
E5_RANDOMIZATION_VERSION = "e5_randomization_v2"
E6_ASSIGNMENT_VERSION = "e6_random_assignment_v1"
EPISODE_PATHS = {
    "episode1": "E1",
    "episode2": "E2",
    "episode3": "E3",
    "episode4": "E4",
    "episode5": "E5",
    "episode6": "E6",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timer_key(decision_index: int, event_phase: str) -> str:
    return f"{decision_index}:{event_phase}"


def _expected_timer_key(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    scenario: Scenario,
) -> str | None:
    if session["episode_status"] == "completed":
        return None
    if scenario.episode == "E3" and not bool(session["entry_confirmed"]):
        return None
    logs = fetch_logs(connection, str(session["session_id"]))
    if scenario.episode == "E5":
        post_logs = [log for log in logs if log["event_phase"] == "post_information"]
        decision_index = len(post_logs) + 1
        pre_exists = any(
            int(log["decision_index"]) == decision_index
            and log["event_phase"] == "pre_information"
            for log in logs
        )
        phase = "post_information" if pre_exists else "pre_information"
        return _timer_key(decision_index, phase)
    return _timer_key(len(logs) + 1, "allocation")


def _ensure_session_timer(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    scenario: Scenario,
    now: str | None = None,
) -> sqlite3.Row:
    expected_key = _expected_timer_key(connection, session, scenario)
    current_key = session["decision_timer_key"]
    started_at = session["decision_started_at"]
    if expected_key is None:
        if current_key is not None or started_at is not None:
            connection.execute(
                "UPDATE sessions SET decision_timer_key = NULL, "
                "decision_started_at = NULL WHERE session_id = ?",
                (session["session_id"],),
            )
        else:
            return session
    elif current_key != expected_key or started_at is None:
        connection.execute(
            "UPDATE sessions SET decision_timer_key = ?, decision_started_at = ? "
            "WHERE session_id = ?",
            (expected_key, now or _utc_now(), session["session_id"]),
        )
    else:
        return session
    updated = connection.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session["session_id"],)
    ).fetchone()
    assert updated is not None
    return updated


def _elapsed_decision_time_ms(
    session: sqlite3.Row,
    expected_key: str,
    completed_at: str,
) -> int:
    if (
        session["decision_timer_key"] != expected_key
        or session["decision_started_at"] is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Decision timer is not initialized; restore the session first",
        )
    try:
        started = datetime.fromisoformat(str(session["decision_started_at"]))
        completed = datetime.fromisoformat(completed_at)
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail="Stored decision timer is invalid"
        ) from exc
    return max(0, int((completed - started).total_seconds() * 1000))


def _episode_code(episode_path: str) -> str:
    episode = EPISODE_PATHS.get(episode_path)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


def _public_scenario_id(scenario: Scenario) -> str:
    if scenario.episode in {"E3", "E4"}:
        return f"{scenario.episode}_{scenario.scenario_id.rsplit('_', 1)[-1]}"
    return scenario.scenario_id


def _e5_assignments(
    connection: sqlite3.Connection, session_id: str
) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM e5_decision_assignments WHERE session_id = ? "
        "ORDER BY decision_index",
        (session_id,),
    ).fetchall()


def _initialize_e5_assignments(
    connection: sqlite3.Connection,
    session_id: str,
    scenario: Scenario,
    stimuli: StimulusStore,
    randomizer: Randomizer,
) -> tuple[str, str]:
    pair_ids = list(SOURCE_PAIRS)
    randomizer.shuffle(pair_ids)
    polarity_cycle = randomizer.choice(list(POLARITY_CYCLES))
    for point, pair_id in zip(scenario.decision_points, pair_ids):
        first_source, second_source = SOURCE_PAIRS[pair_id]
        first_sentiment, second_sentiment = POLARITY_CYCLES[polarity_cycle][pair_id]
        pattern = "A" if first_sentiment == "positive" else "B"
        first = stimuli.choose(first_source, first_sentiment, randomizer)
        second = stimuli.choose(second_source, second_sentiment, randomizer)
        display = [first, second]
        randomizer.shuffle(display)
        price = scenario.prices[point.day - 1]
        peak = max(scenario.prices[: point.day])
        trailing_return_5d = (
            None
            if point.day <= 5
            else price / scenario.prices[point.day - 6] - 1.0
        )
        connection.execute(
            "INSERT INTO e5_decision_assignments (session_id,decision_index,"
            "decision_point,day,stimulus_pair_id,polarity_pattern,first_source,"
            "first_sentiment,first_template_id,second_source,second_sentiment,"
            "second_template_id,left_template_id,right_template_id,"
            "market_snapshot_id,normalized_price,return_from_initial,"
            "drawdown_from_peak,trailing_return_5d,rolling_volatility_20d) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_id, point.sequence, point.decision_point, point.day,
                pair_id, pattern, first.source, first.sentiment,
                first.template_id, second.source, second.sentiment,
                second.template_id, display[0].template_id,
                display[1].template_id, f"{session_id}:E5_DP{point.sequence}",
                price, price / scenario.prices[0] - 1.0,
                price / peak - 1.0, trailing_return_5d,
                scenario.volatility_for_day(point.day),
            ),
        )
    return "-".join(pair_ids), polarity_cycle


def _stimulus_pair_payload(
    assignment: sqlite3.Row, stimuli: StimulusStore
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for prefix in ("first", "second"):
        stimulus = stimuli.get(str(assignment[f"{prefix}_template_id"]))
        result.append(
            {
                "source": stimulus.source,
                "sentiment": stimulus.sentiment,
                "strength": stimulus.strength,
                "template_id": stimulus.template_id,
            }
        )
    return result


def _e5_session_state(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    scenario: Scenario,
    stimuli: StimulusStore,
) -> dict[str, object]:
    session_id = str(session["session_id"])
    logs = fetch_logs(connection, session_id)
    post_logs = [log for log in logs if log["event_phase"] == "post_information"]
    completed = session["episode_status"] == "completed"
    next_decision = None
    stimulus_cards: list[dict[str, str]] = []
    interaction_phase = "completed" if completed else "pre_information"
    reveal_day = 60 if completed else 0
    if not completed:
        next_index = len(post_logs) + 1
        point = scenario.decision_for_sequence(next_index)
        reveal_day = point.day
        next_decision = {
            "decision_point": point.decision_point,
            "sequence": point.sequence,
            "day": point.day,
        }
        pre_exists = any(
            int(log["decision_index"]) == next_index
            and log["event_phase"] == "pre_information"
            for log in logs
        )
        if pre_exists:
            interaction_phase = "post_information"
            assignment = connection.execute(
                "SELECT * FROM e5_decision_assignments "
                "WHERE session_id = ? AND decision_index = ?",
                (session_id, next_index),
            ).fetchone()
            assert assignment is not None
            stimulus_cards = [
                stimuli.public_card(str(assignment["left_template_id"]), "left"),
                stimuli.public_card(str(assignment["right_template_id"]), "right"),
            ]

    entry_share = float(session["entry_risk_share"] or 0.0)
    current_share = entry_share if not logs else float(logs[-1]["risk_share_after"])
    return {
        "session_id": session_id,
        "episode": "E5",
        "scenario_id": _public_scenario_id(scenario),
        "asset": scenario.asset,
        "episode_status": str(session["episode_status"]),
        "entry_setup_required": False,
        "interaction_phase": interaction_phase,
        "progress": {"submitted": len(post_logs), "total": 3},
        "current_risk_share": current_share,
        "current_cash_share": round(1.0 - current_share, 10),
        "next_decision": next_decision,
        "stimulus_cards": stimulus_cards,
        "price_series": [
            {
                "day": day,
                "label": f"Day {day}",
                "normalized_price": scenario.prices[day - 1],
            }
            for day in range(1, reveal_day + 1)
        ],
    }


def _session_state(
    connection: sqlite3.Connection,
    session: sqlite3.Row,
    scenario: Scenario,
    stimuli: StimulusStore | None = None,
) -> dict[str, object]:
    if scenario.episode == "E5":
        if stimuli is None:
            raise RuntimeError("Episode 5 state requires the stimulus store")
        return _e5_session_state(connection, session, scenario, stimuli)
    logs = fetch_logs(connection, str(session["session_id"]))
    submitted_count = len(logs)
    completed = session["episode_status"] == "completed"
    entry_setup_required = (
        scenario.episode == "E3" and not bool(session["entry_confirmed"])
    )
    next_decision = None
    if entry_setup_required:
        reveal_day = 0
    elif not completed:
        point = scenario.decision_for_sequence(submitted_count + 1)
        reveal_day = point.day
        next_decision = {
            "decision_point": point.decision_point,
            "sequence": point.sequence,
            "day": point.day,
        }
    else:
        reveal_day = 60

    entry_risk_share = (
        0.0
        if session["entry_risk_share"] is None
        else float(session["entry_risk_share"])
    )
    current_risk_share = (
        entry_risk_share if not logs else float(logs[-1]["risk_share_after"])
    )
    state = {
        "session_id": str(session["session_id"]),
        "episode": scenario.episode,
        "scenario_id": _public_scenario_id(scenario),
        "asset": scenario.asset,
        "episode_status": str(session["episode_status"]),
        "entry_setup_required": entry_setup_required,
        "progress": {"submitted": submitted_count, "total": len(scenario.decision_points)},
        "current_risk_share": current_risk_share,
        "current_cash_share": round(1.0 - current_risk_share, 10),
        "next_decision": next_decision,
        "price_series": [
            {
                "day": day,
                "label": f"Day {day}",
                "normalized_price": scenario.prices[day - 1],
            }
            for day in range(1, reveal_day + 1)
        ],
    }
    if scenario.episode == "E3":
        allocation_floor = float(session["allocation_floor"] or 0.0)
        state["allocation_constraints"] = {
            "allocation_floor": allocation_floor,
            "minimum_next_risk_share": allocation_floor,
        }
    return state


def _upsert_episode_features(
    connection: sqlite3.Connection,
    episode: str,
    session_id: str,
    now: str,
    episode_status: str,
    scenario: Scenario | None = None,
) -> None:
    logs = fetch_logs(connection, session_id)
    if episode == "E1":
        upsert_episode1_features(
            connection,
            session_id,
            now,
            calculate_episode1_features(logs, episode_status),
        )
    elif episode == "E2":
        e1_feature = connection.execute(
            "SELECT e1f.risk_exposure_auc "
            "FROM sessions current_session "
            "JOIN sessions e1_session "
            "ON e1_session.user_id = current_session.user_id "
            "AND e1_session.episode = 'E1' "
            "JOIN e1_features e1f ON e1f.session_id = e1_session.session_id "
            "WHERE current_session.session_id = ?",
            (session_id,),
        ).fetchone()
        e1_risk_exposure_auc = (
            None if e1_feature is None else float(e1_feature["risk_exposure_auc"])
        )
        upsert_episode2_features(
            connection,
            session_id,
            now,
            calculate_episode2_features(
                logs,
                episode_status,
                e1_risk_exposure_auc=e1_risk_exposure_auc,
            ),
        )
    elif episode == "E3":
        if scenario is None:
            raise RuntimeError("Episode 3 feature calculation requires a scenario")
        context = connection.execute(
            "SELECT allocation_floor, scenario_max_drawdown FROM sessions "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert context is not None
        upsert_episode3_features(
            connection,
            session_id,
            now,
            calculate_episode3_features(
                logs,
                episode_status,
                scenario_prices=scenario.prices,
                allocation_floor=float(context["allocation_floor"] or 0.0),
                scenario_max_drawdown=float(context["scenario_max_drawdown"]),
            ),
        )
    elif episode == "E4":
        if scenario is None or not scenario.rolling_volatility_20d:
            raise RuntimeError("Episode 4 feature calculation requires volatility data")
        context = connection.execute(
            "SELECT entry_risk_share FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert context is not None
        upsert_episode4_features(
            connection,
            session_id,
            now,
            calculate_episode4_features(
                logs,
                episode_status,
                scenario_rolling_volatility_20d=scenario.rolling_volatility_20d,
                volatility_q25=float(scenario.volatility_20d_q25),
                volatility_q75=float(scenario.volatility_20d_q75),
                entry_risk_share=float(context["entry_risk_share"]),
            ),
        )
    elif episode == "E5":
        upsert_episode5_features(
            connection,
            session_id,
            now,
            calculate_episode5_features(logs, episode_status),
        )
    elif episode == "E6":
        context = connection.execute(
            "SELECT user_id,scenario_max_drawdown,"
            "pre_e6_risk_engagement_score,"
            "pre_e6_e3_behavior_resilience_score "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert context is not None
        features = calculate_episode6_features(
            logs,
            episode_status,
            scenario_max_drawdown=(
                None
                if context["scenario_max_drawdown"] is None
                else float(context["scenario_max_drawdown"])
            ),
            pre_e6_risk_engagement_score=(
                None
                if context["pre_e6_risk_engagement_score"] is None
                else float(context["pre_e6_risk_engagement_score"])
            ),
            pre_e6_e3_behavior_resilience_score=(
                None
                if context["pre_e6_e3_behavior_resilience_score"] is None
                else float(context["pre_e6_e3_behavior_resilience_score"])
            ),
        )
        upsert_episode6_features(connection, session_id, now, features)
        if episode_status == "completed" and features.cross_context_consistency is not None:
            upsert_profile_cross_context(
                connection,
                str(context["user_id"]),
                now,
                features.cross_context_consistency,
            )


def _route_episode3_for_user(
    connection: sqlite3.Connection, user_id: str
):
    source = connection.execute(
        "SELECT e1f.risk_exposure_auc, e1f.never_entered, "
        "e2f.uptrend_risk_exposure, e2s.session_id AS e2_session_id, "
        "e2s.episode_status AS e2_status "
        "FROM sessions e1s "
        "JOIN e1_features e1f ON e1f.session_id = e1s.session_id "
        "JOIN sessions e2s ON e2s.user_id = e1s.user_id "
        "AND e2s.episode = 'E2' "
        "JOIN e2_features e2f ON e2f.session_id = e2s.session_id "
        "WHERE e1s.user_id = ? AND e1s.episode = 'E1'",
        (user_id,),
    ).fetchone()
    if (
        source is None
        or source["e2_status"] != "completed"
        or source["uptrend_risk_exposure"] is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 2 must be completed before Episode 3",
        )
    final_event = connection.execute(
        "SELECT risk_share_after FROM behavior_events "
        "WHERE session_id = ? AND decision_index = 7",
        (source["e2_session_id"],),
    ).fetchone()
    if final_event is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 2 final allocation is unavailable",
        )
    return route_episode3(
        e1_risk_exposure_auc=float(source["risk_exposure_auc"]),
        e2_uptrend_risk_exposure=float(source["uptrend_risk_exposure"]),
        e1_never_entered=bool(source["never_entered"]),
        e2_final_risk_share=float(final_event["risk_share_after"]),
    )


def _route_episode4_for_user(
    connection: sqlite3.Connection, user_id: str
):
    source = connection.execute(
        "SELECT e3s.session_id, e3s.episode_status, e3s.assigned_level, "
        "e3s.routing_score, e3f.e3_loss_resilience_score "
        "FROM sessions e3s "
        "LEFT JOIN e3_features e3f ON e3f.session_id = e3s.session_id "
        "WHERE e3s.user_id = ? AND e3s.episode = 'E3'",
        (user_id,),
    ).fetchone()
    if (
        source is None
        or source["episode_status"] != "completed"
        or source["routing_score"] is None
        or source["assigned_level"] is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 3 must be completed before Episode 4",
        )
    events = fetch_logs(connection, str(source["session_id"]))
    if len(events) != 7:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 3 behavior log is incomplete",
        )
    assigned_level = str(source["assigned_level"])
    floor_reached = assigned_level in {"L1", "L2"} and any(
        bool(event["floor_reached"]) for event in events
    )
    full_exit = assigned_level in {"L3", "L4", "L5"} and any(
        abs(float(event["risk_share_after"])) <= 1e-12 for event in events
    )
    routing = route_episode4(
        e3_routing_score=float(source["routing_score"]),
        e3_loss_resilience_score=(
            None
            if source["e3_loss_resilience_score"] is None
            else float(source["e3_loss_resilience_score"])
        ),
        e3_assigned_level=assigned_level,
        floor_reached=floor_reached,
        full_exit=full_exit,
    )
    return routing, float(events[-1]["risk_share_after"])


def _episode5_entry_for_user(
    connection: sqlite3.Connection, user_id: str
) -> float:
    source = connection.execute(
        "SELECT session_id, episode_status FROM sessions "
        "WHERE user_id = ? AND episode = 'E4'",
        (user_id,),
    ).fetchone()
    if source is None or source["episode_status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 4 must be completed before Episode 5",
        )
    final_event = connection.execute(
        "SELECT risk_share_after FROM behavior_events "
        "WHERE session_id = ? AND decision_index = 7 "
        "AND event_phase = 'allocation'",
        (source["session_id"],),
    ).fetchone()
    if final_event is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 4 final allocation is unavailable",
        )
    return float(final_event["risk_share_after"])


def _episode6_context_for_user(
    connection: sqlite3.Connection, user_id: str
) -> dict[str, object | None]:
    e5_session = connection.execute(
        "SELECT episode_status FROM sessions "
        "WHERE user_id = ? AND episode = 'E5'",
        (user_id,),
    ).fetchone()
    if e5_session is None or e5_session["episode_status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode 5 must be completed before Episode 6",
        )

    e3_context = connection.execute(
        "SELECT e3s.routing_score,e3f.behavior_resilience_score,"
        "e3f.e3_loss_resilience_score "
        "FROM sessions e3s "
        "LEFT JOIN e3_features e3f ON e3f.session_id = e3s.session_id "
        "WHERE e3s.user_id = ? AND e3s.episode = 'E3'",
        (user_id,),
    ).fetchone()
    profile = connection.execute(
        "SELECT feature_version "
        "FROM profile_features WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return {
        "pre_e6_risk_engagement_score": (
            None if e3_context is None else e3_context["routing_score"]
        ),
        "pre_e6_e3_behavior_resilience_score": (
            None if e3_context is None else e3_context["behavior_resilience_score"]
        ),
        "pre_e6_e3_loss_resilience_score": (
            None if e3_context is None else e3_context["e3_loss_resilience_score"]
        ),
        "profile_version": None if profile is None else profile["feature_version"],
    }


def _validate_e5_point(
    session: sqlite3.Row,
    scenario: Scenario,
    scenario_id: str,
    decision_point: str,
    day: int,
    expected_index: int,
) -> object:
    if session["episode_status"] == "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Episode is already completed",
        )
    point = scenario.decision_for_sequence(expected_index)
    if scenario_id != _public_scenario_id(scenario):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="scenario_id does not match the assigned scenario",
        )
    if decision_point != point.decision_point or day != point.day:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Decision must follow the assigned DP and PRE/POST order",
                "expected_decision_point": point.decision_point,
                "expected_day": point.day,
            },
        )
    return point


def _insert_e5_event(
    connection: sqlite3.Connection,
    *,
    session: sqlite3.Row,
    scenario: Scenario,
    point: object,
    assignment: sqlite3.Row,
    stimuli: StimulusStore,
    event_phase: str,
    risk_before: float,
    risk_after: float,
    risk_before_pre: float,
    risk_pre_info: float,
    risk_post_info: float | None,
    pre_information_delta: float,
    information_delta: float | None,
    aligned_source: str | None,
    decision_time_ms: int,
    now: str,
) -> None:
    display_sources = [
        stimuli.get(str(assignment["left_template_id"])).source,
        stimuli.get(str(assignment["right_template_id"])).source,
    ]
    values: dict[str, object] = {
        "session_id": str(session["session_id"]),
        "episode": "E5",
        "scenario_id": scenario.scenario_id,
        "decision_point": point.decision_point,
        "decision_index": point.sequence,
        "event_phase": event_phase,
        "day": point.day,
        "risk_share_before": risk_before,
        "risk_share_after": risk_after,
        "cash_share_after": 1.0 - risk_after,
        "delta_risk_share": risk_after - risk_before,
        "decision_time_ms": decision_time_ms,
        "normalized_price": float(assignment["normalized_price"]),
        "return_from_initial": float(assignment["return_from_initial"]),
        "drawdown_from_peak": float(assignment["drawdown_from_peak"]),
        "trailing_return_5d": assignment["trailing_return_5d"],
        "rolling_volatility_20d": assignment["rolling_volatility_20d"],
        "market_snapshot_id": str(assignment["market_snapshot_id"]),
        "risk_share_before_pre": risk_before_pre,
        "risk_share_pre_info": risk_pre_info,
        "risk_share_post_info": risk_post_info,
        "pre_information_delta": pre_information_delta,
        "information_delta": information_delta,
        "stimulus_pair_json": json.dumps(
            _stimulus_pair_payload(assignment, stimuli),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "aligned_source": aligned_source,
        "pairing_order_version": str(session["e5_pairing_order_version"]),
        "polarity_pattern": str(assignment["polarity_pattern"]),
        "display_order_json": json.dumps(display_sources, separators=(",", ":")),
        "semantic_role": point.semantic_role,
        "response_tag": point.response_tag,
        "market_phase": point.market_phase,
        "created_at": now,
    }
    columns = tuple(values)
    connection.execute(
        f"INSERT INTO behavior_events ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        tuple(values[column] for column in columns),
    )


def create_app(
    *,
    database_path: Path | None = None,
    scenario_dir: Path | None = None,
    scenario_picker: ScenarioPicker | None = None,
    stimulus_dir: Path | None = None,
    e5_randomizer: Randomizer | None = None,
) -> FastAPI:
    resolved_database_path = Path(
        database_path
        or os.getenv("EXPERIMENT_DB_PATH", str(DEFAULT_DATABASE_PATH))
    )
    resolved_scenario_dir = Path(scenario_dir or DEFAULT_SCENARIO_DIR)
    scenarios = load_scenarios(resolved_scenario_dir)
    stimuli = StimulusStore.load(Path(stimulus_dir or DEFAULT_STIMULUS_DIR))
    picker = scenario_picker or secrets.choice
    randomizer = e5_randomizer or secrets.SystemRandom()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        initialize_database(resolved_database_path)
        yield

    application = FastAPI(
        title="Sequential Risk Allocation Experiment API",
        version="0.5.0",
        lifespan=lifespan,
    )
    application.state.database_path = resolved_database_path
    application.state.scenarios = scenarios
    application.state.scenario_picker = picker
    application.state.stimuli = stimuli
    application.state.e5_randomizer = randomizer

    origins = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/survey/sessions")
    def start_survey(
        payload: StartSessionRequest,
        request: Request,
    ) -> dict[str, object]:
        """Restore completion state without disclosing any calculated result."""
        with closing(connect(Path(request.app.state.database_path))) as connection:
            completed = connection.execute(
                "SELECT 1 FROM survey_results WHERE user_id = ?",
                (payload.user_id,),
            ).fetchone() is not None
        response: dict[str, object] = {"survey_completed": completed}
        if not completed:
            response["questionnaire"] = public_questionnaire()
        return response

    @application.post("/api/survey/submissions")
    def submit_survey(
        payload: SurveySubmission,
        request: Request,
    ) -> dict[str, bool]:
        """Persist immutable raw/stated/scoring records; reveal success only."""
        try:
            answers = validate_raw_answers(payload.answers)
            features = calculate_stated_features(answers)
            score = calculate_survey_score(answers)
            profile = classify_survey_profile(score)
        except SurveyValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        database = Path(request.app.state.database_path)
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM survey_responses WHERE user_id = ?",
                    (payload.user_id,),
                ).fetchone() is not None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Survey has already been submitted",
                    )
                survey_id = str(uuid.uuid4())
                now = _utc_now()
                connection.execute(
                    "INSERT INTO survey_responses (survey_id,user_id,"
                    "questionnaire_version,source_metadata_json,raw_answers_json,"
                    "submitted_at) VALUES (?,?,?,?,?,?)",
                    (
                        survey_id,
                        payload.user_id,
                        QUESTIONNAIRE_VERSION,
                        json.dumps(SOURCE_METADATA, ensure_ascii=False),
                        json.dumps(answers, ensure_ascii=False),
                        now,
                    ),
                )
                feature_values = features.as_dict()
                feature_columns = tuple(feature_values)
                connection.execute(
                    "INSERT INTO stated_features (survey_id,user_id,feature_version,"
                    "calculated_at," + ",".join(feature_columns) + ") VALUES (" +
                    ",".join("?" for _ in range(4 + len(feature_columns))) + ")",
                    (
                        survey_id,
                        payload.user_id,
                        STATED_FEATURE_VERSION,
                        now,
                        *(feature_values[column] for column in feature_columns),
                    ),
                )
                connection.execute(
                    "INSERT INTO survey_results (survey_id,user_id,scoring_version,"
                    "scoring_basis,score,profile,calculated_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        survey_id,
                        payload.user_id,
                        SCORING_VERSION,
                        SCORING_BASIS,
                        score,
                        profile,
                        now,
                    ),
                )
                connection.commit()
            except HTTPException:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Survey has already been submitted",
                ) from exc
        return {"success": True, "survey_completed": True}

    @application.post("/api/{episode_path}/sessions")
    def start_session(
        episode_path: str,
        payload: StartSessionRequest,
        request: Request,
    ) -> dict[str, object]:
        episode = _episode_code(episode_path)
        database = Path(request.app.state.database_path)
        scenario_map: dict[str, Scenario] = request.app.state.scenarios
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND episode = ?",
                    (payload.user_id, episode),
                ).fetchone()
                if session is None:
                    routing = None
                    if episode == "E1":
                        survey = connection.execute(
                            "SELECT 1 FROM survey_results WHERE user_id = ?",
                            (payload.user_id,),
                        ).fetchone()
                        if survey is None:
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail="Stated-preference survey must be completed before Episode 1",
                            )
                    elif episode == "E2":
                        e1_session = connection.execute(
                            "SELECT episode_status FROM sessions "
                            "WHERE user_id = ? AND episode = 'E1'",
                            (payload.user_id,),
                        ).fetchone()
                        if e1_session is None or e1_session["episode_status"] != "completed":
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail="Episode 1 must be completed before Episode 2",
                            )
                    elif episode == "E3":
                        routing = _route_episode3_for_user(
                            connection, payload.user_id
                        )
                    elif episode == "E4":
                        routing, e4_entry_risk_share = _route_episode4_for_user(
                            connection, payload.user_id
                        )
                    elif episode == "E5":
                        e5_entry_risk_share = _episode5_entry_for_user(
                            connection, payload.user_id
                        )
                    elif episode == "E6":
                        e6_context = _episode6_context_for_user(
                            connection, payload.user_id
                        )
                    candidates = sorted(
                        scenario_id
                        for scenario_id, scenario in scenario_map.items()
                        if scenario.episode == episode
                        and (
                            episode not in {"E3", "E4"}
                            or scenario.level == routing.assigned_level
                        )
                    )
                    scenario_id = request.app.state.scenario_picker(candidates)
                    if scenario_id not in candidates:
                        raise HTTPException(
                            status_code=500,
                            detail="Scenario picker returned an unknown scenario",
                        )
                    now = _utc_now()
                    session_id = str(uuid.uuid4())
                    if episode == "E3":
                        assert routing is not None
                        scenario = scenario_map[scenario_id]
                        connection.execute(
                            "INSERT INTO sessions (session_id,user_id,episode,"
                            "scenario_id,episode_status,created_at,updated_at,"
                            "assigned_level,routing_score,routing_version,"
                            "scenario_max_drawdown,entry_risk_share,allocation_floor,"
                            "entry_confirmed) "
                            "VALUES (?,?,?,?, 'in_progress',?,?,?,?,?,?,?,?,?)",
                            (
                                session_id, payload.user_id, episode, scenario_id,
                                now, now, routing.assigned_level,
                                routing.routing_score, routing.routing_version,
                                scenario.max_drawdown, routing.entry_risk_share,
                                routing.allocation_floor,
                                int(routing.assigned_level in {"L1", "L2"}),
                            ),
                        )
                    elif episode == "E4":
                        assert routing is not None
                        scenario = scenario_map[scenario_id]
                        connection.execute(
                            "INSERT INTO sessions (session_id,user_id,episode,"
                            "scenario_id,episode_status,created_at,updated_at,"
                            "assigned_volatility_level,e4_routing_score,"
                            "routing_version,e4_routing_fallback,e4_context_gap,"
                            "e4_upper_level_capped,scenario_volatility_60d,"
                            "scenario_volatility_20d_min,scenario_volatility_20d_max,"
                            "scenario_volatility_20d_q25,scenario_volatility_20d_q75,"
                            "entry_risk_share,entry_confirmed) "
                            "VALUES (?,?,?,?, 'in_progress',?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                            (
                                session_id, payload.user_id, episode, scenario_id,
                                now, now, routing.assigned_level,
                                routing.routing_score, routing.routing_version,
                                int(routing.routing_fallback), routing.context_gap,
                                int(routing.upper_level_capped),
                                scenario.volatility_60d,
                                scenario.volatility_20d_min,
                                scenario.volatility_20d_max,
                                scenario.volatility_20d_q25,
                                scenario.volatility_20d_q75,
                                e4_entry_risk_share,
                            ),
                        )
                    elif episode == "E5":
                        scenario = scenario_map[scenario_id]
                        connection.execute(
                            "INSERT INTO sessions (session_id,user_id,episode,"
                            "scenario_id,episode_status,created_at,updated_at,"
                            "entry_risk_share,entry_confirmed,e5_randomization_version) "
                            "VALUES (?,?,?,?, 'in_progress',?,?,?,1,?)",
                            (
                                session_id, payload.user_id, episode, scenario_id,
                                now, now, e5_entry_risk_share,
                                E5_RANDOMIZATION_VERSION,
                            ),
                        )
                        pair_order, polarity_cycle = _initialize_e5_assignments(
                            connection,
                            session_id,
                            scenario,
                            request.app.state.stimuli,
                            request.app.state.e5_randomizer,
                        )
                        connection.execute(
                            "UPDATE sessions SET e5_pairing_order_version = ?, "
                            "e5_polarity_cycle = ? "
                            "WHERE session_id = ?",
                            (pair_order, polarity_cycle, session_id),
                        )
                    elif episode == "E6":
                        scenario = scenario_map[scenario_id]
                        connection.execute(
                            "INSERT INTO sessions (session_id,user_id,episode,"
                            "scenario_id,episode_status,created_at,updated_at,"
                            "scenario_max_drawdown,entry_risk_share,entry_confirmed,"
                            "e6_assignment_version,pre_e6_risk_engagement_score,"
                            "pre_e6_e3_behavior_resilience_score,"
                            "pre_e6_e3_loss_resilience_score,"
                            "profile_version) "
                            "VALUES (?,?,?,?, 'in_progress',?,?,?,?,1,?,?,?,?,?)",
                            (
                                session_id,
                                payload.user_id,
                                episode,
                                scenario_id,
                                now,
                                now,
                                scenario.max_drawdown,
                                0.0,
                                E6_ASSIGNMENT_VERSION,
                                e6_context["pre_e6_risk_engagement_score"],
                                e6_context[
                                    "pre_e6_e3_behavior_resilience_score"
                                ],
                                e6_context["pre_e6_e3_loss_resilience_score"],
                                e6_context["profile_version"],
                            ),
                        )
                    else:
                        connection.execute(
                            "INSERT INTO sessions (session_id,user_id,episode,scenario_id,"
                            "episode_status,created_at,updated_at) "
                            "VALUES (?,?,?,?, 'in_progress',?,?)",
                            (session_id, payload.user_id, episode, scenario_id, now, now),
                        )
                    session = connection.execute(
                        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                    ).fetchone()
                assert session is not None
                scenario = scenario_map[str(session["scenario_id"])]
                session = _ensure_session_timer(connection, session, scenario)
                connection.commit()
                return _session_state(
                    connection, session, scenario, request.app.state.stimuli
                )
            except HTTPException:
                connection.rollback()
                raise

    @application.get("/api/{episode_path}/sessions/{session_id}")
    def get_session(
        episode_path: str, session_id: str, request: Request
    ) -> dict[str, object]:
        episode = _episode_code(episode_path)
        with closing(connect(Path(request.app.state.database_path))) as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND episode = ?",
                (session_id, episode),
            ).fetchone()
            if session is None:
                connection.rollback()
                raise HTTPException(status_code=404, detail="Session not found")
            scenario = request.app.state.scenarios[str(session["scenario_id"])]
            session = _ensure_session_timer(connection, session, scenario)
            connection.commit()
            return _session_state(
                connection, session, scenario, request.app.state.stimuli
            )

    @application.post("/api/episode3/sessions/{session_id}/entry")
    def confirm_episode3_entry(
        session_id: str,
        payload: EntryRiskShareSubmission,
        request: Request,
    ) -> dict[str, object]:
        database = Path(request.app.state.database_path)
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND episode = 'E3'",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                if bool(session["entry_confirmed"]):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Episode 3 entry allocation is already confirmed",
                    )
                if fetch_logs(connection, session_id):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Episode 3 decisions have already started",
                    )
                now = _utc_now()
                connection.execute(
                    "UPDATE sessions SET entry_risk_share = ?, entry_confirmed = 1, "
                    "updated_at = ?, decision_started_at = ?, "
                    "decision_timer_key = ? WHERE session_id = ?",
                    (
                        payload.risk_share,
                        now,
                        now,
                        _timer_key(1, "allocation"),
                        session_id,
                    ),
                )
                connection.commit()
                updated = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                assert updated is not None
                scenario = request.app.state.scenarios[str(updated["scenario_id"])]
                return _session_state(
                    connection, updated, scenario, request.app.state.stimuli
                )
            except HTTPException:
                connection.rollback()
                raise

    @application.post("/api/episode5/sessions/{session_id}/pre-decisions")
    def submit_episode5_pre(
        session_id: str,
        payload: Episode5PreSubmission,
        request: Request,
    ) -> dict[str, object]:
        database = Path(request.app.state.database_path)
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND episode = 'E5'",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                scenario = request.app.state.scenarios[str(session["scenario_id"])]
                session = _ensure_session_timer(connection, session, scenario)
                logs = fetch_logs(connection, session_id)
                post_logs = [
                    log for log in logs if log["event_phase"] == "post_information"
                ]
                expected_index = len(post_logs) + 1
                if expected_index > 3 or any(
                    int(log["decision_index"]) == expected_index
                    and log["event_phase"] == "pre_information"
                    for log in logs
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="PRE decision has already been submitted",
                    )
                point = _validate_e5_point(
                    session,
                    scenario,
                    payload.scenario_id,
                    payload.decision_point,
                    payload.day,
                    expected_index,
                )
                assignment = connection.execute(
                    "SELECT * FROM e5_decision_assignments "
                    "WHERE session_id = ? AND decision_index = ?",
                    (session_id, expected_index),
                ).fetchone()
                assert assignment is not None
                risk_before = (
                    float(session["entry_risk_share"] or 0.0)
                    if not post_logs
                    else float(post_logs[-1]["risk_share_after"])
                )
                now = _utc_now()
                decision_time_ms = _elapsed_decision_time_ms(
                    session,
                    _timer_key(expected_index, "pre_information"),
                    now,
                )
                _insert_e5_event(
                    connection,
                    session=session,
                    scenario=scenario,
                    point=point,
                    assignment=assignment,
                    stimuli=request.app.state.stimuli,
                    event_phase="pre_information",
                    risk_before=risk_before,
                    risk_after=payload.risk_share_pre_info,
                    risk_before_pre=risk_before,
                    risk_pre_info=payload.risk_share_pre_info,
                    risk_post_info=None,
                    pre_information_delta=payload.risk_share_pre_info - risk_before,
                    information_delta=None,
                    aligned_source=None,
                    decision_time_ms=decision_time_ms,
                    now=now,
                )
                connection.execute(
                    "UPDATE sessions SET updated_at = ?, decision_started_at = ?, "
                    "decision_timer_key = ? WHERE session_id = ?",
                    (
                        now,
                        now,
                        _timer_key(expected_index, "post_information"),
                        session_id,
                    ),
                )
                _upsert_episode_features(
                    connection, "E5", session_id, now, "in_progress", scenario
                )
                connection.commit()
                updated = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                assert updated is not None
                return _session_state(
                    connection, updated, scenario, request.app.state.stimuli
                )
            except HTTPException:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="PRE decision has already been submitted",
                ) from exc

    @application.post("/api/episode5/sessions/{session_id}/post-decisions")
    def submit_episode5_post(
        session_id: str,
        payload: Episode5PostSubmission,
        request: Request,
    ) -> dict[str, object]:
        database = Path(request.app.state.database_path)
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND episode = 'E5'",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                scenario = request.app.state.scenarios[str(session["scenario_id"])]
                session = _ensure_session_timer(connection, session, scenario)
                logs = fetch_logs(connection, session_id)
                post_logs = [
                    log for log in logs if log["event_phase"] == "post_information"
                ]
                expected_index = len(post_logs) + 1
                pre_event = next(
                    (
                        log
                        for log in logs
                        if int(log["decision_index"]) == expected_index
                        and log["event_phase"] == "pre_information"
                    ),
                    None,
                )
                if pre_event is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="PRE decision must be submitted before POST",
                    )
                point = _validate_e5_point(
                    session,
                    scenario,
                    payload.scenario_id,
                    payload.decision_point,
                    payload.day,
                    expected_index,
                )
                assignment = connection.execute(
                    "SELECT * FROM e5_decision_assignments "
                    "WHERE session_id = ? AND decision_index = ?",
                    (session_id, expected_index),
                ).fetchone()
                assert assignment is not None
                risk_pre = float(pre_event["risk_share_pre_info"])
                information_delta = payload.risk_share_post_info - risk_pre
                aligned_source = None
                if information_delta > 1e-12:
                    target_sentiment = "positive"
                elif information_delta < -1e-12:
                    target_sentiment = "negative"
                else:
                    target_sentiment = None
                if target_sentiment is not None:
                    aligned_source = (
                        str(assignment["first_source"])
                        if assignment["first_sentiment"] == target_sentiment
                        else str(assignment["second_source"])
                )
                now = _utc_now()
                decision_time_ms = _elapsed_decision_time_ms(
                    session,
                    _timer_key(expected_index, "post_information"),
                    now,
                )
                _insert_e5_event(
                    connection,
                    session=session,
                    scenario=scenario,
                    point=point,
                    assignment=assignment,
                    stimuli=request.app.state.stimuli,
                    event_phase="post_information",
                    risk_before=risk_pre,
                    risk_after=payload.risk_share_post_info,
                    risk_before_pre=float(pre_event["risk_share_before_pre"]),
                    risk_pre_info=risk_pre,
                    risk_post_info=payload.risk_share_post_info,
                    pre_information_delta=float(pre_event["pre_information_delta"]),
                    information_delta=information_delta,
                    aligned_source=aligned_source,
                    decision_time_ms=decision_time_ms,
                    now=now,
                )
                completed = expected_index == 3
                episode_status = "completed" if completed else "in_progress"
                next_timer_key = (
                    None
                    if completed
                    else _timer_key(expected_index + 1, "pre_information")
                )
                connection.execute(
                    "UPDATE sessions SET episode_status = ?, updated_at = ?, "
                    "completed_at = ?, decision_started_at = ?, "
                    "decision_timer_key = ? WHERE session_id = ?",
                    (
                        episode_status,
                        now,
                        now if completed else None,
                        None if completed else now,
                        next_timer_key,
                        session_id,
                    ),
                )
                _upsert_episode_features(
                    connection,
                    "E5",
                    session_id,
                    now,
                    episode_status,
                    scenario,
                )
                connection.commit()
                updated = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                assert updated is not None
                return _session_state(
                    connection, updated, scenario, request.app.state.stimuli
                )
            except HTTPException:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="POST decision has already been submitted",
                ) from exc

    @application.post("/api/{episode_path}/sessions/{session_id}/decisions")
    def submit_decision(
        episode_path: str,
        session_id: str,
        payload: DecisionSubmission,
        request: Request,
    ) -> dict[str, object]:
        episode = _episode_code(episode_path)
        database = Path(request.app.state.database_path)
        scenario_map: dict[str, Scenario] = request.app.state.scenarios
        with closing(connect(database)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND episode = ?",
                    (session_id, episode),
                ).fetchone()
                if session is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                if session["episode_status"] == "completed":
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Episode is already completed",
                    )
                if episode == "E3" and not bool(session["entry_confirmed"]):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Episode 3 entry allocation must be confirmed first",
                    )

                scenario = scenario_map[str(session["scenario_id"])]
                session = _ensure_session_timer(connection, session, scenario)
                logs = fetch_logs(connection, session_id)
                expected = scenario.decision_for_sequence(len(logs) + 1)
                if payload.scenario_id != _public_scenario_id(scenario):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="scenario_id does not match the assigned scenario",
                    )
                if (
                    payload.decision_point != expected.decision_point
                    or payload.day != expected.day
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "message": "Decision must follow the assigned DP order",
                            "expected_decision_point": expected.decision_point,
                            "expected_day": expected.day,
                        },
                    )

                entry_risk_share = (
                    0.0
                    if session["entry_risk_share"] is None
                    else float(session["entry_risk_share"])
                )
                risk_before = (
                    entry_risk_share
                    if not logs
                    else float(logs[-1]["risk_share_after"])
                )
                risk_after = payload.risk_share_after
                allocation_floor = (
                    None
                    if episode != "E3"
                    else float(session["allocation_floor"] or 0.0)
                )
                if allocation_floor is not None and risk_after < allocation_floor:
                    raise HTTPException(
                        status_code=422,
                        detail="risk_share_after is below the Episode 3 allocation floor",
                    )
                price = scenario.prices[expected.day - 1]
                peak = max(scenario.prices[: expected.day])
                trailing_return_5d = (
                    None
                    if expected.day <= 5
                    else price / scenario.prices[expected.day - 6] - 1.0
                )
                return_since_previous_dp = (
                    None
                    if not logs
                    else price / float(logs[-1]["normalized_price"]) - 1.0
                )
                abs_return_since_previous_dp = (
                    None
                    if return_since_previous_dp is None
                    else abs(return_since_previous_dp)
                )
                max_abs_daily_return_since_previous_dp = None
                if logs:
                    previous_day = int(logs[-1]["day"])
                    interval_returns = [
                        scenario.prices[index] / scenario.prices[index - 1] - 1.0
                        for index in range(previous_day, expected.day)
                    ]
                    if interval_returns:
                        max_abs_daily_return_since_previous_dp = max(
                            abs(value) for value in interval_returns
                        )
                rolling_volatility_20d = (
                    None
                    if episode != "E4"
                    else scenario.volatility_for_day(expected.day)
                )
                previous_dp_volatility_20d = (
                    None
                    if episode != "E4" or not logs
                    else scenario.volatility_for_day(int(logs[-1]["day"]))
                )
                delta_volatility_20d = (
                    None
                    if (
                        rolling_volatility_20d is None
                        or previous_dp_volatility_20d is None
                    )
                    else rolling_volatility_20d - previous_dp_volatility_20d
                )
                volatility_direction = None
                if delta_volatility_20d is not None:
                    if delta_volatility_20d >= 0.01:
                        volatility_direction = "rising"
                    elif delta_volatility_20d <= -0.01:
                        volatility_direction = "falling"
                    else:
                        volatility_direction = "stable"
                volatility_percentile = (
                    None
                    if episode != "E4"
                    else scenario.volatility_percentile_for_day(expected.day)
                )
                now = _utc_now()
                decision_time_ms = _elapsed_decision_time_ms(
                    session,
                    _timer_key(expected.sequence, "allocation"),
                    now,
                )
                connection.execute(
                    "INSERT INTO behavior_events (session_id,episode,scenario_id,"
                    "decision_point,decision_index,day,risk_share_before,"
                    "risk_share_after,cash_share_after,delta_risk_share,"
                    "decision_time_ms,normalized_price,return_from_initial,"
                    "drawdown_from_peak,trailing_return_5d,"
                    "return_since_previous_dp,abs_return_since_previous_dp,"
                    "max_abs_daily_return_since_previous_dp,"
                    "rolling_volatility_20d,previous_dp_volatility_20d,"
                    "delta_volatility_20d,volatility_percentile,"
                    "volatility_direction,allocation_floor,floor_reached,"
                    "initial_preallocated_risk_share,semantic_role,"
                    "response_tag,market_phase,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session_id, episode, scenario.scenario_id,
                        expected.decision_point, expected.sequence, expected.day,
                        risk_before, risk_after, 1.0 - risk_after,
                        risk_after - risk_before, decision_time_ms, price,
                        price / scenario.prices[0] - 1.0, price / peak - 1.0,
                        trailing_return_5d, return_since_previous_dp,
                        abs_return_since_previous_dp,
                        max_abs_daily_return_since_previous_dp,
                        rolling_volatility_20d,
                        previous_dp_volatility_20d,
                        delta_volatility_20d,
                        volatility_percentile,
                        volatility_direction,
                        allocation_floor,
                        (
                            None
                            if allocation_floor is None
                            else int(abs(risk_after - allocation_floor) <= 1e-12)
                        ),
                        (entry_risk_share if episode == "E3" else None),
                        expected.semantic_role,
                        expected.response_tag, expected.market_phase, now,
                    ),
                )

                completed = expected.sequence == len(scenario.decision_points)
                episode_status = "completed" if completed else "in_progress"
                next_timer_key = (
                    None
                    if completed
                    else _timer_key(expected.sequence + 1, "allocation")
                )
                connection.execute(
                    "UPDATE sessions SET episode_status = ?, updated_at = ?, "
                    "completed_at = ?, decision_started_at = ?, "
                    "decision_timer_key = ? WHERE session_id = ?",
                    (
                        episode_status,
                        now,
                        now if completed else None,
                        None if completed else now,
                        next_timer_key,
                        session_id,
                    ),
                )
                _upsert_episode_features(
                    connection,
                    episode,
                    session_id,
                    now,
                    episode_status,
                    scenario=scenario,
                )
                connection.commit()

                updated_session = connection.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                assert updated_session is not None
                return _session_state(
                    connection,
                    updated_session,
                    scenario,
                    request.app.state.stimuli,
                )
            except HTTPException:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Decision has already been submitted",
                ) from exc

    return application


app = create_app()

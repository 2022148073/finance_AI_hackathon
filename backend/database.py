"""SQLite storage for experiment sessions, append-only events, and features."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from features import (
    Episode1Features,
    Episode2Features,
    Episode3Features,
    Episode4Features,
    Episode5Features,
    Episode6Features,
)


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    episode TEXT NOT NULL CHECK (episode IN ('E1','E2','E3','E4','E5','E6')),
    scenario_id TEXT NOT NULL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    assigned_level TEXT,
    routing_score REAL,
    routing_version TEXT,
    scenario_max_drawdown REAL,
    entry_risk_share REAL,
    allocation_floor REAL,
    entry_confirmed INTEGER CHECK (entry_confirmed IN (0,1)),
    assigned_volatility_level TEXT,
    e4_routing_score REAL,
    e4_routing_fallback INTEGER CHECK (e4_routing_fallback IN (0,1)),
    e4_context_gap REAL,
    e4_upper_level_capped INTEGER CHECK (e4_upper_level_capped IN (0,1)),
    scenario_volatility_60d REAL,
    scenario_volatility_20d_min REAL,
    scenario_volatility_20d_max REAL,
    scenario_volatility_20d_q25 REAL,
    scenario_volatility_20d_q75 REAL,
    e5_pairing_order_version TEXT,
    e5_randomization_version TEXT,
    e5_polarity_cycle TEXT,
    e6_assignment_version TEXT,
    pre_e6_risk_engagement_score REAL,
    pre_e6_e3_behavior_resilience_score REAL,
    pre_e6_e3_loss_resilience_score REAL,
    profile_version TEXT,
    decision_started_at TEXT,
    decision_timer_key TEXT,
    UNIQUE (user_id, episode)
);

CREATE TABLE IF NOT EXISTS behavior_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    episode TEXT NOT NULL CHECK (episode IN ('E1','E2','E3','E4','E5','E6')),
    scenario_id TEXT NOT NULL,
    decision_point TEXT NOT NULL,
    decision_index INTEGER NOT NULL CHECK (decision_index BETWEEN 1 AND 7),
    event_phase TEXT NOT NULL DEFAULT 'allocation' CHECK (
        event_phase IN ('allocation','pre_information','post_information')
    ),
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 60),
    risk_share_before REAL NOT NULL CHECK (risk_share_before BETWEEN 0 AND 1),
    risk_share_after REAL NOT NULL CHECK (risk_share_after BETWEEN 0 AND 1),
    cash_share_after REAL NOT NULL CHECK (cash_share_after BETWEEN 0 AND 1),
    delta_risk_share REAL NOT NULL,
    decision_time_ms INTEGER NOT NULL CHECK (decision_time_ms >= 0),
    normalized_price REAL NOT NULL,
    return_from_initial REAL NOT NULL,
    drawdown_from_peak REAL NOT NULL,
    trailing_return_5d REAL,
    return_since_previous_dp REAL,
    abs_return_since_previous_dp REAL,
    max_abs_daily_return_since_previous_dp REAL,
    rolling_volatility_20d REAL,
    previous_dp_volatility_20d REAL,
    delta_volatility_20d REAL,
    volatility_percentile REAL,
    volatility_direction TEXT,
    market_snapshot_id TEXT,
    risk_share_before_pre REAL,
    risk_share_pre_info REAL,
    risk_share_post_info REAL,
    pre_information_delta REAL,
    information_delta REAL,
    stimulus_pair_json TEXT,
    aligned_source TEXT,
    pairing_order_version TEXT,
    polarity_pattern TEXT,
    display_order_json TEXT,
    allocation_floor REAL,
    floor_reached INTEGER CHECK (floor_reached IN (0,1)),
    initial_preallocated_risk_share REAL,
    semantic_role TEXT NOT NULL,
    response_tag TEXT,
    market_phase TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, decision_index, event_phase),
    UNIQUE (session_id, decision_point, event_phase)
);

CREATE TABLE IF NOT EXISTS e5_decision_assignments (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    decision_index INTEGER NOT NULL CHECK (decision_index BETWEEN 1 AND 3),
    decision_point TEXT NOT NULL,
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 60),
    stimulus_pair_id TEXT NOT NULL,
    polarity_pattern TEXT NOT NULL CHECK (polarity_pattern IN ('A','B')),
    first_source TEXT NOT NULL,
    first_sentiment TEXT NOT NULL,
    first_template_id TEXT NOT NULL,
    second_source TEXT NOT NULL,
    second_sentiment TEXT NOT NULL,
    second_template_id TEXT NOT NULL,
    left_template_id TEXT NOT NULL,
    right_template_id TEXT NOT NULL,
    market_snapshot_id TEXT NOT NULL UNIQUE,
    normalized_price REAL NOT NULL,
    return_from_initial REAL NOT NULL,
    drawdown_from_peak REAL NOT NULL,
    trailing_return_5d REAL,
    rolling_volatility_20d REAL,
    PRIMARY KEY (session_id, decision_index),
    UNIQUE (session_id, stimulus_pair_id)
);

CREATE TABLE IF NOT EXISTS e1_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL, initial_risk_share REAL,
    risk_exposure_auc REAL NOT NULL, mean_risk_share REAL,
    market_participation_rate REAL NOT NULL, time_to_first_entry INTEGER,
    never_entered INTEGER NOT NULL CHECK (never_entered IN (0,1)),
    adjustment_frequency INTEGER NOT NULL, mean_abs_allocation_change REAL,
    hold_rate REAL, decision_time_median REAL, mild_gain_response REAL,
    mild_drawdown_response REAL, recovery_response REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);

CREATE TABLE IF NOT EXISTS e2_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL, recent_return_sensitivity REAL,
    gain_period_risk_escalation REAL, uptrend_risk_exposure REAL,
    e2_vs_e1_risk_shift REAL,
    strong_gain_response REAL, pullback_response_after_gain REAL,
    renewed_rise_response REAL, uptrend_risk_increase_count INTEGER NOT NULL,
    gain_period_hold_rate REAL, gain_adjustment_intensity REAL,
    decision_time_median REAL, strong_gain_decision_time REAL,
    correction_decision_time REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);

CREATE TABLE IF NOT EXISTS e3_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL,
    loss_period_risk_change REAL,
    drawdown_sensitivity REAL,
    first_meaningful_reduction_drawdown REAL,
    loss_period_risk_exposure REAL,
    max_loss_period_reduction REAL,
    recovery_reentry REAL,
    drawdown_period_risk_increase_count INTEGER NOT NULL,
    drawdown_reduction_consistency REAL,
    reference_point_crossing_response REAL,
    trough_response REAL,
    early_recovery_response REAL,
    late_recovery_response REAL,
    post_loss_risk_persistence REAL,
    recovery_reentry_ratio REAL,
    retention_score REAL,
    reduction_score REAL,
    threshold_score REAL,
    recovery_score REAL,
    severity_factor REAL,
    behavior_resilience_score REAL,
    e3_loss_resilience_score REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);
CREATE TABLE IF NOT EXISTS e4_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL,
    volatility_sensitivity REAL,
    high_vol_risk_exposure REAL,
    low_vol_risk_exposure REAL,
    high_vs_low_vol_risk_shift REAL,
    volatility_increase_response_mean REAL,
    volatility_decrease_response_mean REAL,
    peak_volatility_response REAL,
    volatility_derisking_consistency REAL,
    volatility_risk_increase_count INTEGER NOT NULL,
    volatility_adjustment_intensity REAL,
    high_vol_hold_rate REAL,
    volatility_compression_reentry REAL,
    final_vs_entry_risk_change REAL,
    decision_time_volatility_median REAL,
    peak_volatility_decision_time REAL,
    volatility_shift_decision_time REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);
CREATE TABLE IF NOT EXISTS e5_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL,
    external_information_sensitivity REAL,
    information_adjustment_rate REAL,
    information_hold_rate REAL,
    news_alignment_score INTEGER NOT NULL,
    expert_alignment_score INTEGER NOT NULL,
    community_alignment_score INTEGER NOT NULL,
    news_alignment_magnitude REAL NOT NULL,
    expert_alignment_magnitude REAL NOT NULL,
    community_alignment_magnitude REAL NOT NULL,
    dp1_information_delta REAL,
    dp2_information_delta REAL,
    dp3_information_delta REAL,
    positive_alignment_count INTEGER NOT NULL,
    negative_alignment_count INTEGER NOT NULL,
    information_counter_adjustment_count INTEGER NOT NULL,
    conflict_hold_count INTEGER NOT NULL,
    pre_information_decision_time_median REAL,
    post_information_decision_time_median REAL,
    information_decision_time_change REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);
CREATE TABLE IF NOT EXISTS e6_features (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    feature_version TEXT NOT NULL, computed_at TEXT NOT NULL,
    decision_count INTEGER NOT NULL,
    anchor_risk_exposure_auc REAL,
    anchor_mean_risk_share REAL,
    anchor_drawdown_risk_change REAL,
    anchor_drawdown_sensitivity REAL,
    anchor_loss_risk_exposure REAL,
    anchor_recovery_reentry REAL,
    anchor_recovery_reentry_ratio REAL,
    e6_retention_score REAL,
    e6_reduction_score REAL,
    e6_threshold_score REAL,
    e6_recovery_score REAL,
    e6_behavior_resilience_score REAL,
    risk_engagement_consistency REAL,
    loss_response_consistency REAL,
    cross_context_consistency REAL,
    anchor_adjustment_frequency INTEGER NOT NULL,
    anchor_hold_rate REAL,
    anchor_adjustment_intensity REAL,
    anchor_peak_response REAL,
    anchor_trough_response REAL,
    anchor_early_recovery_response REAL,
    anchor_late_recovery_response REAL,
    anchor_final_vs_entry_change REAL,
    anchor_decision_time_median REAL,
    anchor_max_drawdown_decision_time REAL,
    anchor_recovery_decision_time REAL,
    episode_status TEXT NOT NULL CHECK (episode_status IN ('in_progress','completed'))
);

CREATE TABLE IF NOT EXISTS profile_features (
    user_id TEXT PRIMARY KEY,
    feature_version TEXT,
    computed_at TEXT,
    risk_engagement REAL,
    loss_resilience REAL,
    volatility_tolerance REAL,
    information_sensitivity REAL,
    cross_context_consistency REAL
);

CREATE TABLE IF NOT EXISTS survey_responses (
    survey_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    questionnaire_version TEXT NOT NULL,
    source_metadata_json TEXT NOT NULL,
    raw_answers_json TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stated_features (
    survey_id TEXT PRIMARY KEY REFERENCES survey_responses(survey_id),
    user_id TEXT NOT NULL UNIQUE,
    feature_version TEXT NOT NULL,
    calculated_at TEXT NOT NULL,
    risk_capacity_age REAL NOT NULL CHECK (risk_capacity_age BETWEEN 0 AND 1),
    investment_horizon REAL NOT NULL CHECK (investment_horizon BETWEEN 0 AND 1),
    risky_asset_experience REAL NOT NULL CHECK (risky_asset_experience BETWEEN 0 AND 1),
    experience_breadth REAL NOT NULL CHECK (experience_breadth BETWEEN 0 AND 1),
    derivative_experience REAL NOT NULL CHECK (derivative_experience BETWEEN 0 AND 1),
    stated_loss_tolerance REAL NOT NULL CHECK (stated_loss_tolerance BETWEEN 0 AND 1),
    investment_exposure REAL NOT NULL CHECK (investment_exposure BETWEEN 0 AND 1),
    financial_capacity REAL NOT NULL CHECK (financial_capacity BETWEEN 0 AND 1),
    return_seeking REAL NOT NULL CHECK (return_seeking BETWEEN 0 AND 1),
    financial_literacy REAL NOT NULL CHECK (financial_literacy BETWEEN 0 AND 1),
    vulnerability_flag INTEGER NOT NULL CHECK (vulnerability_flag IN (0,1))
);

CREATE TABLE IF NOT EXISTS survey_results (
    survey_id TEXT PRIMARY KEY REFERENCES survey_responses(survey_id),
    user_id TEXT NOT NULL UNIQUE,
    scoring_version TEXT NOT NULL,
    scoring_basis TEXT NOT NULL,
    score REAL NOT NULL CHECK (score BETWEEN 0 AND 100),
    profile TEXT NOT NULL CHECK (
        profile IN ('안정형','안정추구형','위험중립형','적극투자형','공격투자형')
    ),
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_analysis_runs (
    analysis_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued','processing','completed','failed')
    ),
    model TEXT NOT NULL,
    analysis_config_version TEXT NOT NULL,
    manifest_schema_version TEXT NOT NULL,
    behavioral_input_schema_version TEXT NOT NULL,
    comparison_input_schema_version TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    error_code TEXT,
    internal_error TEXT
);

CREATE INDEX IF NOT EXISTS llm_analysis_runs_user_created
ON llm_analysis_runs (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS assessment_attempts (
    assessment_id TEXT PRIMARY KEY,
    participant_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    previous_assessment_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (participant_id, attempt_number),
    UNIQUE (participant_id, previous_assessment_id)
);

CREATE INDEX IF NOT EXISTS assessment_attempts_participant_number
ON assessment_attempts (participant_id, attempt_number DESC);

CREATE TABLE IF NOT EXISTS llm_analysis_artifacts (
    analysis_id TEXT PRIMARY KEY REFERENCES llm_analysis_runs(analysis_id),
    behavioral_input_json TEXT,
    call1_raw_response_json TEXT,
    revealed_result_json TEXT,
    comparison_input_json TEXT,
    call2_raw_response_json TEXT,
    public_result_json TEXT
);

CREATE TRIGGER IF NOT EXISTS survey_responses_no_update
BEFORE UPDATE ON survey_responses
BEGIN SELECT RAISE(ABORT, 'survey_responses are append-only'); END;

CREATE TRIGGER IF NOT EXISTS assessment_attempts_no_update
BEFORE UPDATE ON assessment_attempts
BEGIN SELECT RAISE(ABORT, 'assessment_attempts are immutable'); END;

CREATE TRIGGER IF NOT EXISTS assessment_attempts_no_delete
BEFORE DELETE ON assessment_attempts
BEGIN SELECT RAISE(ABORT, 'assessment_attempts are append-only'); END;

CREATE TRIGGER IF NOT EXISTS survey_responses_no_delete
BEFORE DELETE ON survey_responses
BEGIN SELECT RAISE(ABORT, 'survey_responses are append-only'); END;

CREATE TRIGGER IF NOT EXISTS stated_features_no_update
BEFORE UPDATE ON stated_features
BEGIN SELECT RAISE(ABORT, 'stated_features are immutable'); END;

CREATE TRIGGER IF NOT EXISTS survey_results_no_update
BEFORE UPDATE ON survey_results
BEGIN SELECT RAISE(ABORT, 'survey_results are immutable'); END;

CREATE TRIGGER IF NOT EXISTS behavior_events_no_update
BEFORE UPDATE ON behavior_events
BEGIN SELECT RAISE(ABORT, 'behavior_events are append-only'); END;

CREATE TRIGGER IF NOT EXISTS behavior_events_no_delete
BEFORE DELETE ON behavior_events
BEGIN SELECT RAISE(ABORT, 'behavior_events are append-only'); END;
"""


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _migrate_behavior_events_to_phases(connection: sqlite3.Connection) -> None:
    """Replace the legacy one-event-per-DP table without changing raw rows."""
    connection.execute("DROP TRIGGER IF EXISTS behavior_events_no_update")
    connection.execute("DROP TRIGGER IF EXISTS behavior_events_no_delete")
    connection.execute(
        "ALTER TABLE behavior_events RENAME TO behavior_events_legacy_e5"
    )
    connection.executescript(SCHEMA)
    old_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(behavior_events_legacy_e5)")
    }
    new_columns = [
        row["name"]
        for row in connection.execute("PRAGMA table_info(behavior_events)")
    ]
    common_columns = [
        column
        for column in new_columns
        if column in old_columns and column != "event_phase"
    ]
    column_list = ", ".join(common_columns)
    connection.execute(
        f"INSERT INTO behavior_events ({column_list}, event_phase) "
        f"SELECT {column_list}, 'allocation' FROM behavior_events_legacy_e5"
    )
    connection.execute("DROP TABLE behavior_events_legacy_e5")


def initialize_database(database_path: Path) -> None:
    with closing(connect(database_path)) as connection:
        connection.executescript(SCHEMA)
        stated_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(stated_features)")
        }
        if "stated_risk_tolerance" in stated_columns:
            connection.execute(
                "ALTER TABLE stated_features DROP COLUMN stated_risk_tolerance"
            )
        event_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(behavior_events)")
        }
        if "return_since_previous_dp" not in event_columns:
            connection.execute(
                "ALTER TABLE behavior_events ADD COLUMN "
                "return_since_previous_dp REAL"
            )
        for column, definition in (
            ("allocation_floor", "REAL"),
            ("floor_reached", "INTEGER"),
            ("initial_preallocated_risk_share", "REAL"),
            ("abs_return_since_previous_dp", "REAL"),
            ("max_abs_daily_return_since_previous_dp", "REAL"),
            ("rolling_volatility_20d", "REAL"),
            ("previous_dp_volatility_20d", "REAL"),
            ("delta_volatility_20d", "REAL"),
            ("volatility_percentile", "REAL"),
            ("volatility_direction", "TEXT"),
        ):
            if column not in event_columns:
                connection.execute(
                    f"ALTER TABLE behavior_events ADD COLUMN {column} {definition}"
                )
        event_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(behavior_events)")
        }
        if "event_phase" not in event_columns:
            _migrate_behavior_events_to_phases(connection)

        session_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "pre_e6_volatility_tolerance" in session_columns:
            connection.execute(
                "ALTER TABLE sessions DROP COLUMN pre_e6_volatility_tolerance"
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(sessions)")
            }
        for column, definition in (
            ("assigned_level", "TEXT"),
            ("routing_score", "REAL"),
            ("routing_version", "TEXT"),
            ("scenario_max_drawdown", "REAL"),
            ("entry_risk_share", "REAL"),
            ("allocation_floor", "REAL"),
            ("entry_confirmed", "INTEGER"),
            ("assigned_volatility_level", "TEXT"),
            ("e4_routing_score", "REAL"),
            ("e4_routing_fallback", "INTEGER"),
            ("e4_context_gap", "REAL"),
            ("e4_upper_level_capped", "INTEGER"),
            ("scenario_volatility_60d", "REAL"),
            ("scenario_volatility_20d_min", "REAL"),
            ("scenario_volatility_20d_max", "REAL"),
            ("scenario_volatility_20d_q25", "REAL"),
            ("scenario_volatility_20d_q75", "REAL"),
            ("e5_pairing_order_version", "TEXT"),
            ("e5_randomization_version", "TEXT"),
            ("e5_polarity_cycle", "TEXT"),
            ("e6_assignment_version", "TEXT"),
            ("pre_e6_risk_engagement_score", "REAL"),
            ("pre_e6_e3_behavior_resilience_score", "REAL"),
            ("pre_e6_e3_loss_resilience_score", "REAL"),
            ("profile_version", "TEXT"),
            ("decision_started_at", "TEXT"),
            ("decision_timer_key", "TEXT"),
        ):
            if column not in session_columns:
                connection.execute(
                    f"ALTER TABLE sessions ADD COLUMN {column} {definition}"
                )
        connection.execute(
            "UPDATE sessions SET entry_confirmed = CASE "
            "WHEN episode != 'E3' THEN 1 "
            "WHEN assigned_level IN ('L1','L2') THEN 1 "
            "WHEN EXISTS (SELECT 1 FROM behavior_events event "
            "WHERE event.session_id = sessions.session_id) THEN 1 "
            "ELSE 0 END WHERE entry_confirmed IS NULL"
        )

        e2_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e2_features)")
        }
        if (
            "return_chasing_consistency" in e2_columns
            and "uptrend_risk_increase_count" not in e2_columns
        ):
            connection.execute(
                "ALTER TABLE e2_features RENAME COLUMN "
                "return_chasing_consistency TO uptrend_risk_increase_count"
            )
        e2_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e2_features)")
        }
        if "e2_vs_e1_risk_shift" not in e2_columns:
            connection.execute(
                "ALTER TABLE e2_features ADD COLUMN e2_vs_e1_risk_shift REAL"
            )

        e3_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e3_features)")
        }
        if (
            "loss_domain_risk_increase_count" in e3_columns
            and "drawdown_period_risk_increase_count" not in e3_columns
        ):
            connection.execute(
                "ALTER TABLE e3_features RENAME COLUMN "
                "loss_domain_risk_increase_count "
                "TO drawdown_period_risk_increase_count"
            )
            e3_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(e3_features)")
            }
        required_e3_columns = {
            "decision_count",
            "loss_period_risk_change",
            "drawdown_sensitivity",
            "first_meaningful_reduction_drawdown",
            "loss_period_risk_exposure",
            "max_loss_period_reduction",
            "recovery_reentry",
            "drawdown_period_risk_increase_count",
            "drawdown_reduction_consistency",
            "reference_point_crossing_response",
            "trough_response",
            "early_recovery_response",
            "late_recovery_response",
            "post_loss_risk_persistence",
            "recovery_reentry_ratio",
            "retention_score",
            "reduction_score",
            "threshold_score",
            "recovery_score",
            "severity_factor",
            "behavior_resilience_score",
            "e3_loss_resilience_score",
        }
        for column in sorted(required_e3_columns - e3_columns):
            column_type = (
                "INTEGER"
                if column in {
                    "decision_count",
                    "drawdown_period_risk_increase_count",
                }
                else "REAL"
            )
            connection.execute(
                f"ALTER TABLE e3_features ADD COLUMN {column} {column_type}"
            )

        e4_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e4_features)")
        }
        required_e4_columns = {
            "decision_count",
            "volatility_sensitivity",
            "high_vol_risk_exposure",
            "low_vol_risk_exposure",
            "high_vs_low_vol_risk_shift",
            "volatility_increase_response_mean",
            "volatility_decrease_response_mean",
            "peak_volatility_response",
            "volatility_derisking_consistency",
            "volatility_risk_increase_count",
            "volatility_adjustment_intensity",
            "high_vol_hold_rate",
            "volatility_compression_reentry",
            "final_vs_entry_risk_change",
            "decision_time_volatility_median",
            "peak_volatility_decision_time",
            "volatility_shift_decision_time",
        }
        for column in sorted(required_e4_columns - e4_columns):
            column_type = (
                "INTEGER"
                if column in {"decision_count", "volatility_risk_increase_count"}
                else "REAL"
            )
            connection.execute(
                f"ALTER TABLE e4_features ADD COLUMN {column} {column_type}"
            )

        e5_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e5_features)")
        }
        if (
            "direction_reversal_count" in e5_columns
            and "information_counter_adjustment_count" not in e5_columns
        ):
            connection.execute(
                "ALTER TABLE e5_features RENAME COLUMN "
                "direction_reversal_count TO information_counter_adjustment_count"
            )
            e5_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(e5_features)")
            }
        required_e5_columns = {
            "decision_count",
            "external_information_sensitivity",
            "information_adjustment_rate",
            "information_hold_rate",
            "news_alignment_score",
            "expert_alignment_score",
            "community_alignment_score",
            "news_alignment_magnitude",
            "expert_alignment_magnitude",
            "community_alignment_magnitude",
            "dp1_information_delta",
            "dp2_information_delta",
            "dp3_information_delta",
            "positive_alignment_count",
            "negative_alignment_count",
            "information_counter_adjustment_count",
            "conflict_hold_count",
            "pre_information_decision_time_median",
            "post_information_decision_time_median",
            "information_decision_time_change",
        }
        integer_e5_columns = {
            "decision_count",
            "news_alignment_score",
            "expert_alignment_score",
            "community_alignment_score",
            "positive_alignment_count",
            "negative_alignment_count",
            "information_counter_adjustment_count",
            "conflict_hold_count",
        }
        for column in sorted(required_e5_columns - e5_columns):
            column_type = "INTEGER" if column in integer_e5_columns else "REAL"
            connection.execute(
                f"ALTER TABLE e5_features ADD COLUMN {column} {column_type}"
            )

        e6_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(e6_features)")
        }
        required_e6_columns = {
            "decision_count",
            "anchor_risk_exposure_auc",
            "anchor_mean_risk_share",
            "anchor_drawdown_risk_change",
            "anchor_drawdown_sensitivity",
            "anchor_loss_risk_exposure",
            "anchor_recovery_reentry",
            "anchor_recovery_reentry_ratio",
            "e6_retention_score",
            "e6_reduction_score",
            "e6_threshold_score",
            "e6_recovery_score",
            "e6_behavior_resilience_score",
            "risk_engagement_consistency",
            "loss_response_consistency",
            "cross_context_consistency",
            "anchor_adjustment_frequency",
            "anchor_hold_rate",
            "anchor_adjustment_intensity",
            "anchor_peak_response",
            "anchor_trough_response",
            "anchor_early_recovery_response",
            "anchor_late_recovery_response",
            "anchor_final_vs_entry_change",
            "anchor_decision_time_median",
            "anchor_max_drawdown_decision_time",
            "anchor_recovery_decision_time",
        }
        for column in sorted(required_e6_columns - e6_columns):
            column_type = (
                "INTEGER"
                if column in {"decision_count", "anchor_adjustment_frequency"}
                else "REAL"
            )
            connection.execute(
                f"ALTER TABLE e6_features ADD COLUMN {column} {column_type}"
            )

        profile_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(profile_features)")
        }
        if (
            "cross_contest_consistency" in profile_columns
            and "cross_context_consistency" not in profile_columns
        ):
            connection.execute(
                "ALTER TABLE profile_features RENAME COLUMN "
                "cross_contest_consistency TO cross_context_consistency"
            )

        run_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(llm_analysis_runs)")
        }
        if "analysis_config_version" not in run_columns:
            connection.execute(
                "ALTER TABLE llm_analysis_runs ADD COLUMN "
                "analysis_config_version TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "input_fingerprint" not in run_columns:
            connection.execute(
                "ALTER TABLE llm_analysis_runs ADD COLUMN "
                "input_fingerprint TEXT NOT NULL DEFAULT 'legacy'"
            )

        artifact_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(llm_analysis_artifacts)")
        }
        for column in ("behavioral_input_json", "comparison_input_json"):
            if column not in artifact_columns:
                connection.execute(
                    f"ALTER TABLE llm_analysis_artifacts ADD COLUMN {column} TEXT"
                )
        connection.commit()


def fetch_logs(connection: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM behavior_events WHERE session_id = ? "
        "ORDER BY decision_index, CASE event_phase "
        "WHEN 'pre_information' THEN 0 WHEN 'post_information' THEN 1 ELSE 0 END",
        (session_id,),
    ).fetchall()


E1_COLUMNS = (
    "feature_version", "decision_count", "initial_risk_share",
    "risk_exposure_auc", "mean_risk_share", "market_participation_rate",
    "time_to_first_entry", "never_entered", "adjustment_frequency",
    "mean_abs_allocation_change", "hold_rate", "decision_time_median",
    "mild_gain_response", "mild_drawdown_response", "recovery_response",
    "episode_status",
)

E2_COLUMNS = (
    "feature_version", "decision_count", "recent_return_sensitivity",
    "gain_period_risk_escalation", "uptrend_risk_exposure",
    "e2_vs_e1_risk_shift",
    "strong_gain_response", "pullback_response_after_gain",
    "renewed_rise_response", "uptrend_risk_increase_count",
    "gain_period_hold_rate", "gain_adjustment_intensity",
    "decision_time_median", "strong_gain_decision_time",
    "correction_decision_time", "episode_status",
)

E3_COLUMNS = (
    "feature_version", "decision_count", "loss_period_risk_change",
    "drawdown_sensitivity", "first_meaningful_reduction_drawdown",
    "loss_period_risk_exposure", "max_loss_period_reduction",
    "recovery_reentry", "drawdown_period_risk_increase_count",
    "drawdown_reduction_consistency", "reference_point_crossing_response",
    "trough_response", "early_recovery_response", "late_recovery_response",
    "post_loss_risk_persistence", "recovery_reentry_ratio",
    "retention_score", "reduction_score", "threshold_score",
    "recovery_score", "severity_factor", "behavior_resilience_score",
    "e3_loss_resilience_score", "episode_status",
)

E4_COLUMNS = (
    "feature_version", "decision_count", "volatility_sensitivity",
    "high_vol_risk_exposure", "low_vol_risk_exposure",
    "high_vs_low_vol_risk_shift", "volatility_increase_response_mean",
    "volatility_decrease_response_mean", "peak_volatility_response",
    "volatility_derisking_consistency", "volatility_risk_increase_count",
    "volatility_adjustment_intensity", "high_vol_hold_rate",
    "volatility_compression_reentry", "final_vs_entry_risk_change",
    "decision_time_volatility_median", "peak_volatility_decision_time",
    "volatility_shift_decision_time", "episode_status",
)

E5_COLUMNS = (
    "feature_version", "decision_count", "external_information_sensitivity",
    "information_adjustment_rate", "information_hold_rate",
    "news_alignment_score", "expert_alignment_score",
    "community_alignment_score", "news_alignment_magnitude",
    "expert_alignment_magnitude", "community_alignment_magnitude",
    "dp1_information_delta",
    "dp2_information_delta", "dp3_information_delta",
    "positive_alignment_count", "negative_alignment_count",
    "information_counter_adjustment_count", "conflict_hold_count",
    "pre_information_decision_time_median",
    "post_information_decision_time_median",
    "information_decision_time_change", "episode_status",
)

E6_COLUMNS = (
    "feature_version", "decision_count", "anchor_risk_exposure_auc",
    "anchor_mean_risk_share", "anchor_drawdown_risk_change",
    "anchor_drawdown_sensitivity", "anchor_loss_risk_exposure",
    "anchor_recovery_reentry", "anchor_recovery_reentry_ratio",
    "e6_retention_score", "e6_reduction_score", "e6_threshold_score",
    "e6_recovery_score", "e6_behavior_resilience_score",
    "risk_engagement_consistency", "loss_response_consistency",
    "cross_context_consistency", "anchor_adjustment_frequency",
    "anchor_hold_rate", "anchor_adjustment_intensity",
    "anchor_peak_response", "anchor_trough_response",
    "anchor_early_recovery_response", "anchor_late_recovery_response",
    "anchor_final_vs_entry_change", "anchor_decision_time_median",
    "anchor_max_drawdown_decision_time", "anchor_recovery_decision_time",
    "episode_status",
)


def _upsert(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    session_id: str,
    computed_at: str,
    values: dict[str, object],
) -> None:
    all_columns = ("session_id", "computed_at", *columns)
    placeholders = ", ".join("?" for _ in all_columns)
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in all_columns[1:]
    )
    connection.execute(
        f"INSERT INTO {table} ({', '.join(all_columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
        (session_id, computed_at, *(values[column] for column in columns)),
    )


def upsert_episode1_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode1Features,
) -> None:
    values = features.as_dict()
    values["never_entered"] = int(bool(values["never_entered"]))
    _upsert(connection, "e1_features", E1_COLUMNS, session_id, computed_at, values)


def upsert_episode2_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode2Features,
) -> None:
    _upsert(
        connection,
        "e2_features",
        E2_COLUMNS,
        session_id,
        computed_at,
        features.as_dict(),
    )


def upsert_episode3_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode3Features,
) -> None:
    _upsert(
        connection,
        "e3_features",
        E3_COLUMNS,
        session_id,
        computed_at,
        features.as_dict(),
    )


def upsert_episode4_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode4Features,
) -> None:
    _upsert(
        connection,
        "e4_features",
        E4_COLUMNS,
        session_id,
        computed_at,
        features.as_dict(),
    )


def upsert_episode5_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode5Features,
) -> None:
    _upsert(
        connection,
        "e5_features",
        E5_COLUMNS,
        session_id,
        computed_at,
        features.as_dict(),
    )


def upsert_episode6_features(
    connection: sqlite3.Connection,
    session_id: str,
    computed_at: str,
    features: Episode6Features,
) -> None:
    _upsert(
        connection,
        "e6_features",
        E6_COLUMNS,
        session_id,
        computed_at,
        features.as_dict(),
    )


def upsert_profile_cross_context(
    connection: sqlite3.Connection,
    user_id: str,
    computed_at: str,
    cross_context_consistency: float | None,
) -> None:
    connection.execute(
        "INSERT INTO profile_features (user_id,feature_version,computed_at,"
        "cross_context_consistency) VALUES (?,'profile_v1',?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "feature_version=excluded.feature_version, "
        "computed_at=excluded.computed_at, "
        "cross_context_consistency=excluded.cross_context_consistency",
        (user_id, computed_at, cross_context_consistency),
    )

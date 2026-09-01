from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from build_llm_input import (  # noqa: E402
    LlmInputBuildError,
    _extract_dimension_result,
    load_feature_manifest,
)
from database import connect, initialize_database  # noqa: E402
from llm_pipeline import (  # noqa: E402
    AnalysisPipelineError,
    _comparison_response_schema,
    _runtime_settings,
    _update_artifact,
    _validate_response_shape,
    create_or_restore_analysis_run,
)


class LlmPipelineTargetedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "llm-test.db"
        initialize_database(self.database_path)
        self.manifest = load_feature_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _insert_eligible_user(self, user_id: str) -> None:
        now = "2026-09-02T00:00:00+00:00"
        survey_id = f"survey-{user_id}"
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO survey_responses VALUES (?,?,?,?,?,?)",
                (survey_id, user_id, "test", "{}", "{}", now),
            )
            connection.execute(
                "INSERT INTO survey_results VALUES (?,?,?,?,?,?,?)",
                (
                    survey_id,
                    user_id,
                    "test",
                    "test",
                    50.0,
                    "위험중립형",
                    now,
                ),
            )
            for index in range(1, 7):
                connection.execute(
                    "INSERT INTO sessions "
                    "(session_id,user_id,episode,scenario_id,episode_status,"
                    "created_at,updated_at,completed_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        f"session-{user_id}-{index}",
                        user_id,
                        f"E{index}",
                        f"scenario-{index}",
                        "completed",
                        now,
                        now,
                        now,
                    ),
                )
            connection.commit()

    def test_reasoning_effort_creates_a_distinct_analysis_config(self) -> None:
        self._insert_eligible_user("cache-user")
        environment = {
            "KIMI_MODEL": "moonshotai/kimi-k3",
            "KIMI_ANALYSIS_REVISION": "v1",
        }
        with patch.dict(os.environ, {**environment, "KIMI_REASONING_EFFORT": "low"}):
            low = create_or_restore_analysis_run(self.database_path, "cache-user")
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "UPDATE llm_analysis_runs SET status = 'completed' "
                "WHERE analysis_id = ?",
                (low["analysis_id"],),
            )
            connection.commit()

        with patch.dict(os.environ, {**environment, "KIMI_REASONING_EFFORT": "max"}):
            maximum = create_or_restore_analysis_run(
                self.database_path, "cache-user"
            )

        self.assertNotEqual(low["analysis_id"], maximum["analysis_id"])
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT analysis_config_version FROM llm_analysis_runs "
                "ORDER BY created_at, rowid"
            ).fetchall()
        self.assertEqual(
            [row["analysis_config_version"] for row in rows],
            ["kimi_k3_low_v1", "kimi_k3_max_v1"],
        )

    def test_private_input_snapshot_columns_are_migrated_and_writable(self) -> None:
        legacy_path = Path(self.temporary.name) / "legacy-llm.db"
        legacy = sqlite3.connect(legacy_path)
        legacy.executescript(
            """
            CREATE TABLE llm_analysis_runs (
                analysis_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                status TEXT NOT NULL, model TEXT NOT NULL,
                manifest_schema_version TEXT NOT NULL,
                behavioral_input_schema_version TEXT NOT NULL,
                comparison_input_schema_version TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                completed_at TEXT, error_code TEXT, internal_error TEXT
            );
            CREATE TABLE llm_analysis_artifacts (
                analysis_id TEXT PRIMARY KEY,
                call1_raw_response_json TEXT, revealed_result_json TEXT,
                call2_raw_response_json TEXT, public_result_json TEXT
            );
            """
        )
        legacy.close()
        initialize_database(legacy_path)
        with closing(connect(legacy_path)) as connection:
            run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(llm_analysis_runs)")
            }
            artifact_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(llm_analysis_artifacts)"
                )
            }
        self.assertIn("analysis_config_version", run_columns)
        self.assertIn("behavioral_input_json", artifact_columns)
        self.assertIn("comparison_input_json", artifact_columns)

        self._insert_eligible_user("snapshot-user")
        run = create_or_restore_analysis_run(self.database_path, "snapshot-user")
        _update_artifact(
            self.database_path,
            run["analysis_id"],
            "behavioral_input_json",
            {"stage": "call1"},
        )
        _update_artifact(
            self.database_path,
            run["analysis_id"],
            "comparison_input_json",
            {"stage": "call2"},
        )
        with closing(connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT behavioral_input_json,comparison_input_json "
                "FROM llm_analysis_artifacts WHERE analysis_id = ?",
                (run["analysis_id"],),
            ).fetchone()
        self.assertEqual(row["behavioral_input_json"], '{"stage": "call1"}')
        self.assertEqual(row["comparison_input_json"], '{"stage": "call2"}')

    def test_evidence_must_be_manifest_allowed_and_exist_in_actual_input(self) -> None:
        rubrics = self.manifest["behavioral_dimension_rubrics"]["dimensions"]
        baselines = {
            name: {"base_level": "medium", "max_llm_adjustment_steps": 1}
            for name in rubrics
        }
        behavioral: dict[str, object] = {}
        for rubric in rubrics.values():
            path = str(rubric["primary_evidence"][0]).split(".")[1:]
            cursor = behavioral
            for segment in path[:-1]:
                cursor = cursor.setdefault(segment, {})  # type: ignore[assignment]
            cursor[path[-1]] = 0.5

        dimensions = {
            name: {
                "base_level": "medium",
                "adjustment": 0,
                "confidence_level": "medium",
                "reason": "test",
                "evidence_fields": [rubric["primary_evidence"][0]],
            }
            for name, rubric in rubrics.items()
        }
        result = {"revealed_behavioral_dimensions": dimensions}
        cleaned = _extract_dimension_result(
            result, self.manifest, baselines, behavioral
        )
        self.assertEqual(cleaned["risk_engagement"]["final_level"], "medium")

        dimensions["risk_engagement"]["evidence_fields"] = [
            "behavioral_analysis.episode3.fake_feature"
        ]
        with self.assertRaisesRegex(LlmInputBuildError, "outside its manifest"):
            _extract_dimension_result(result, self.manifest, baselines, behavioral)

        allowed_but_missing = rubrics["risk_engagement"]["supporting_evidence"][4]
        dimensions["risk_engagement"]["evidence_fields"] = [allowed_but_missing]
        with self.assertRaisesRegex(LlmInputBuildError, "actual input"):
            _extract_dimension_result(result, self.manifest, baselines, behavioral)

    def test_comparison_confidence_is_manifest_ordinal(self) -> None:
        template = {
            "investor_type": "위험중립형",
            "confidence_level": None,
            "stated_preference_summary": None,
            "revealed_preference_summary": None,
            "stated_revealed_gap": None,
            "key_behavioral_evidence": [],
            "final_analysis": None,
        }
        comparison_input = {
            "analysis_request": {"required_output_format": template}
        }
        schema = _comparison_response_schema(comparison_input, self.manifest)
        self.assertEqual(
            schema["properties"]["confidence_level"]["enum"],
            self.manifest["revealed_profile_scoring"]["confidence_levels"],
        )
        response = {
            **template,
            "confidence_level": "high",
            "stated_preference_summary": "설문 요약",
            "revealed_preference_summary": "행동 요약",
            "stated_revealed_gap": "차이",
            "key_behavioral_evidence": ["행동 근거"],
            "final_analysis": "종합 해석",
        }
        _validate_response_shape(response, template, self.manifest)
        response["confidence_level"] = "very_high"
        with self.assertRaisesRegex(AnalysisPipelineError, "confidence"):
            _validate_response_shape(response, template, self.manifest)


if __name__ == "__main__":
    unittest.main()

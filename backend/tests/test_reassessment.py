from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import connect  # noqa: E402
from main import create_app  # noqa: E402


class ReassessmentTargetedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "reassessment-test.db"
        app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            scenario_picker=lambda candidates: candidates[0],
        )
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def _insert_completed_measurement(self) -> None:
        now = "2026-09-04T00:00:00+00:00"
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO survey_responses VALUES (?,?,?,?,?,?)",
                ("survey-old", "assessment-old", "test", "{}", "{}", now),
            )
            connection.execute(
                "INSERT INTO survey_results VALUES (?,?,?,?,?,?,?)",
                (
                    "survey-old",
                    "assessment-old",
                    "test",
                    "test",
                    50.0,
                    "위험중립형",
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO llm_analysis_runs "
                "(analysis_id,user_id,status,model,analysis_config_version,"
                "manifest_schema_version,behavioral_input_schema_version,"
                "comparison_input_schema_version,input_fingerprint,created_at,"
                "updated_at,completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "analysis-old",
                    "assessment-old",
                    "completed",
                    "test-model",
                    "test-config",
                    "test-manifest",
                    "test-behavioral",
                    "test-comparison",
                    "old-fingerprint",
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO llm_analysis_artifacts (analysis_id,public_result_json) "
                "VALUES (?,?)",
                ("analysis-old", '{"result":"preserved"}'),
            )
            connection.commit()

    def test_restart_creates_idempotent_separate_attempt_without_deleting_old_data(self) -> None:
        self._insert_completed_measurement()
        payload = {
            "participant_id": "participant-one",
            "previous_assessment_id": "assessment-old",
        }
        response = self.client.post("/api/assessment-attempts", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        first = response.json()
        self.assertTrue(first["created"])
        self.assertEqual(first["attempt_number"], 2)
        self.assertNotEqual(first["assessment_id"], "assessment-old")

        retry = self.client.post("/api/assessment-attempts", json=payload)
        self.assertEqual(retry.status_code, 201, retry.text)
        self.assertEqual(retry.json()["assessment_id"], first["assessment_id"])
        self.assertFalse(retry.json()["created"])

        survey = self.client.post(
            "/api/survey/sessions", json={"user_id": first["assessment_id"]}
        )
        self.assertEqual(survey.status_code, 200, survey.text)
        self.assertFalse(survey.json()["survey_completed"])
        self.assertIn("questionnaire", survey.json())

        with closing(connect(self.database_path)) as connection:
            self.assertIsNotNone(
                connection.execute(
                    "SELECT 1 FROM survey_responses WHERE user_id = 'assessment-old'"
                ).fetchone()
            )
            artifact = connection.execute(
                "SELECT public_result_json FROM llm_analysis_artifacts "
                "WHERE analysis_id = 'analysis-old'"
            ).fetchone()
            attempts = connection.execute(
                "SELECT assessment_id,attempt_number FROM assessment_attempts "
                "WHERE participant_id = ? ORDER BY attempt_number",
                ("participant-one",),
            ).fetchall()
        self.assertEqual(artifact["public_result_json"], '{"result":"preserved"}')
        self.assertEqual(
            [(row["assessment_id"], row["attempt_number"]) for row in attempts],
            [("assessment-old", 1), (first["assessment_id"], 2)],
        )

    def test_restart_requires_a_completed_previous_analysis(self) -> None:
        response = self.client.post(
            "/api/assessment-attempts",
            json={
                "participant_id": "participant-one",
                "previous_assessment_id": "unfinished-assessment",
            },
        )
        self.assertEqual(response.status_code, 409, response.text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import connect
from main import create_app
from survey import (
    SCORING_BASIS,
    calculate_stated_features,
    calculate_survey_score,
    classify_survey_profile,
)


def conservative_answers() -> dict[str, str | list[str]]:
    return {
        "age_group": "age_61_plus",
        "investment_horizon": "under_6_months",
        "investment_experience": ["conservative"],
        "derivative_experience": "none_or_under_1_year",
        "loss_tolerance": "principal_preservation",
        "investment_asset_ratio": "up_to_10",
        "monthly_income": "up_to_1m",
        "investment_purpose": "living_or_short_term",
        "financial_knowledge": "none",
        "vulnerability": "yes",
    }


def aggressive_answers() -> dict[str, str | list[str]]:
    return {
        "age_group": "age_20_40",
        "investment_horizon": "years_3_plus",
        "investment_experience": ["aggressive"],
        "derivative_experience": "years_3_plus",
        "loss_tolerance": "high_risk_for_return",
        "investment_asset_ratio": "over_70",
        "monthly_income": "over_5m",
        "investment_purpose": "active_wealth_growth",
        "financial_knowledge": "including_derivatives",
        "vulnerability": "no",
    }


def middle_answers() -> dict[str, str | list[str]]:
    return {
        "age_group": "age_41_50",
        "investment_horizon": "years_1_2",
        "investment_experience": ["neutral"],
        "derivative_experience": "years_1_3",
        "loss_tolerance": "partial_principal_loss",
        "investment_asset_ratio": "up_to_50",
        "monthly_income": "up_to_3m",
        "investment_purpose": "above_market_return",
        "financial_knowledge": "deep",
        "vulnerability": "no",
    }


class SurveyCalculationTests(unittest.TestCase):
    def test_most_conservative_response(self) -> None:
        answers = conservative_answers()
        features = calculate_stated_features(answers).as_dict()
        self.assertEqual(classify_survey_profile(calculate_survey_score(answers)), "안정형")
        self.assertEqual(features["investment_horizon"], 0.0)
        self.assertEqual(features["stated_loss_tolerance"], 0.0)
        self.assertEqual(features["vulnerability_flag"], 1)

    def test_most_aggressive_response(self) -> None:
        answers = aggressive_answers()
        features = calculate_stated_features(answers).as_dict()
        self.assertEqual(classify_survey_profile(calculate_survey_score(answers)), "공격투자형")
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in features.values()))
        self.assertEqual(features["stated_risk_tolerance"], 1.0)

    def test_middle_response(self) -> None:
        answers = middle_answers()
        features = calculate_stated_features(answers).as_dict()
        self.assertEqual(classify_survey_profile(calculate_survey_score(answers)), "위험중립형")
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in features.values()))

    def test_multiple_experience_uses_max_and_breadth(self) -> None:
        answers = conservative_answers()
        answers["investment_experience"] = ["conservative", "active"]
        features = calculate_stated_features(answers)
        self.assertEqual(features.risky_asset_experience, 0.75)
        self.assertEqual(features.experience_breadth, 0.4)


class SurveyApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "survey-test.db"
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

    def submit(self, user_id: str, answers=None):
        return self.client.post(
            "/api/survey/submissions",
            json={"user_id": user_id, "answers": answers or middle_answers()},
        )

    def test_results_are_persisted_but_not_disclosed(self) -> None:
        initial = self.client.post(
            "/api/survey/sessions", json={"user_id": "private_result"}
        )
        public_payload = json.dumps(initial.json(), ensure_ascii=False)
        for hidden in ("survey_score", "survey_profile", "scoring_basis", "risk_capacity_age"):
            self.assertNotIn(hidden, public_payload)

        response = self.submit("private_result")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(), {"success": True, "survey_completed": True}
        )
        serialized = json.dumps(response.json(), ensure_ascii=False)
        for hidden in ("score", "profile", "feature", "위험중립형"):
            self.assertNotIn(hidden, serialized)

        state = self.client.post(
            "/api/survey/sessions", json={"user_id": "private_result"}
        )
        self.assertEqual(state.json(), {"survey_completed": True})
        with closing(connect(self.database_path)) as connection:
            raw = connection.execute(
                "SELECT * FROM survey_responses WHERE user_id = 'private_result'"
            ).fetchone()
            stated = connection.execute(
                "SELECT * FROM stated_features WHERE user_id = 'private_result'"
            ).fetchone()
            result = connection.execute(
                "SELECT * FROM survey_results WHERE user_id = 'private_result'"
            ).fetchone()
            self.assertIsNotNone(raw)
            self.assertIsNotNone(stated)
            self.assertIsNotNone(result)
            self.assertEqual(result["scoring_basis"], SCORING_BASIS)

        frontend = (BACKEND_DIR.parent / "frontend" / "src" / "App.jsx").read_text(
            encoding="utf-8"
        ) + (BACKEND_DIR.parent / "frontend" / "src" / "Survey.jsx").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("survey_profile", frontend)
        self.assertNotIn("survey_score", frontend)

    def test_validation_duplicate_submission_and_e1_gate(self) -> None:
        blocked = self.client.post(
            "/api/episode1/sessions", json={"user_id": "needs_survey"}
        )
        self.assertEqual(blocked.status_code, 409)

        invalid = middle_answers()
        invalid["age_group"] = ["age_41_50"]
        self.assertEqual(self.submit("invalid_user", invalid).status_code, 422)

        accepted = self.submit("needs_survey")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(self.submit("needs_survey").status_code, 409)
        started = self.client.post(
            "/api/episode1/sessions", json={"user_id": "needs_survey"}
        )
        self.assertEqual(started.status_code, 200, started.text)

    def test_survey_and_behavioral_storage_are_independent(self) -> None:
        self.assertEqual(self.submit("independent").status_code, 200)
        with closing(connect(self.database_path)) as connection:
            before = tuple(
                connection.execute(
                    "SELECT * FROM stated_features WHERE user_id = 'independent'"
                ).fetchone()
            )

        session = self.client.post(
            "/api/episode1/sessions", json={"user_id": "independent"}
        ).json()
        decision = self.client.post(
            f"/api/episode1/sessions/{session['session_id']}/decisions",
            json={
                "scenario_id": session["scenario_id"],
                "decision_point": session["next_decision"]["decision_point"],
                "day": session["next_decision"]["day"],
                "risk_share_after": 0.5,
            },
        )
        self.assertEqual(decision.status_code, 200, decision.text)
        with closing(connect(self.database_path)) as connection:
            after = tuple(
                connection.execute(
                    "SELECT * FROM stated_features WHERE user_id = 'independent'"
                ).fetchone()
            )
            self.assertEqual(before, after)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM e1_features").fetchone()[0],
                1,
            )


if __name__ == "__main__":
    unittest.main()

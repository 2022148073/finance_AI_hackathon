from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import connect
from main import create_app
from scenario_store import load_scenarios


class RecordingE6Picker:
    def __init__(self, selected: str = "E6_02") -> None:
        self.selected = selected
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, candidates: list[str]) -> str:
        self.calls.append(tuple(candidates))
        return self.selected


class Episode6ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "e6-test.db"
        self.scenarios = load_scenarios(BACKEND_DIR / "scenarios")
        self.picker = RecordingE6Picker()
        self.app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            scenario_picker=self.picker,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def seed_prior_context(self, user_id: str) -> None:
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO sessions (session_id,user_id,episode,scenario_id,"
                "episode_status,created_at,updated_at,completed_at,routing_score) "
                "VALUES (?,?,'E3','E3_L3_01','completed','now','now','now',0.5)",
                (f"e3-{user_id}", user_id),
            )
            connection.execute(
                "INSERT INTO e3_features (session_id,feature_version,computed_at,"
                "decision_count,drawdown_period_risk_increase_count,"
                "behavior_resilience_score,e3_loss_resilience_score,episode_status) "
                "VALUES (?,'e3_v3','now',7,0,0.7,0.42,'completed')",
                (f"e3-{user_id}",),
            )
            connection.execute(
                "INSERT INTO sessions (session_id,user_id,episode,scenario_id,"
                "episode_status,created_at,updated_at,completed_at) "
                "VALUES (?,?,'E5','E5_01','completed','now','now','now')",
                (f"e5-{user_id}", user_id),
            )
            connection.execute(
                "INSERT INTO profile_features (user_id,feature_version,computed_at,"
                "volatility_tolerance) VALUES (?,'pre_profile_v1','now',0.65)",
                (user_id,),
            )
            connection.commit()

    def start_e6(self, user_id: str) -> dict[str, object]:
        self.seed_prior_context(user_id)
        response = self.client.post(
            "/api/episode6/sessions", json={"user_id": user_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_configured_scenarios_are_assigned_once_and_context_is_snapshotted(self) -> None:
        expected = {
            "E6_01": [1, 16, 25, 35, 40, 49, 60],
            "E6_02": [1, 22, 34, 42, 48, 56, 60],
            "E6_03": [1, 18, 26, 30, 35, 46, 60],
        }
        self.assertEqual(
            {
                scenario_id: [point.day for point in scenario.decision_points]
                for scenario_id, scenario in self.scenarios.items()
                if scenario.episode == "E6"
            },
            expected,
        )
        for scenario_id in expected:
            self.assertEqual(
                [
                    point.semantic_role
                    for point in self.scenarios[scenario_id].decision_points
                ],
                [
                    "anchor_entry",
                    "pre_drawdown_anchor",
                    "drawdown_progression",
                    "max_drawdown_anchor",
                    "early_recovery_anchor",
                    "recovered_state_anchor",
                    "final_anchor",
                ],
            )

        state = self.start_e6("assignment_user")
        self.assertEqual(state["scenario_id"], "E6_02")
        self.assertEqual(state["current_risk_share"], 0.0)
        self.assertEqual(state["next_decision"]["day"], 1)
        self.assertEqual(len(state["price_series"]), 1)
        self.assertEqual(self.picker.calls, [("E6_01", "E6_02", "E6_03")])

        resumed = self.client.post(
            "/api/episode6/sessions", json={"user_id": "assignment_user"}
        )
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["session_id"], state["session_id"])
        self.assertEqual(resumed.json()["scenario_id"], "E6_02")
        self.assertEqual(len(self.picker.calls), 1)

        with closing(connect(self.database_path)) as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (state["session_id"],)
            ).fetchone()
            assert session is not None
            self.assertEqual(session["scenario_id"], "E6_02")
            self.assertEqual(session["e6_assignment_version"], "e6_random_assignment_v1")
            self.assertEqual(session["entry_risk_share"], 0.0)
            self.assertAlmostEqual(session["pre_e6_risk_engagement_score"], 0.5)
            self.assertAlmostEqual(
                session["pre_e6_e3_behavior_resilience_score"], 0.7
            )
            self.assertAlmostEqual(
                session["pre_e6_e3_loss_resilience_score"], 0.42
            )
            self.assertAlmostEqual(session["pre_e6_volatility_tolerance"], 0.65)
            self.assertEqual(session["profile_version"], "pre_profile_v1")

    def test_anchor_logs_features_and_profile_consistency(self) -> None:
        self.picker.selected = "E6_01"
        state = self.start_e6("feature_user")
        scenario = self.scenarios["E6_01"]
        shares = [0.60, 0.60, 0.50, 0.35, 0.40, 0.50, 0.55]
        for point, share in zip(scenario.decision_points, shares):
            response = self.client.post(
                f"/api/episode6/sessions/{state['session_id']}/decisions",
                json={
                    "scenario_id": state["scenario_id"],
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": share,
                    "decision_time_ms": point.sequence * 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            state = response.json()
        self.assertEqual(state["episode_status"], "completed")

        days = [point.day for point in scenario.decision_points]
        expected_auc = sum(
            shares[index] * (days[index + 1] - days[index])
            for index in range(6)
        ) / 59.0
        expected_loss_exposure = (
            shares[1] * (days[2] - days[1])
            + shares[2] * (days[3] - days[2])
        ) / (days[3] - days[1])

        with closing(connect(self.database_path)) as connection:
            events = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? "
                "ORDER BY decision_index",
                (state["session_id"],),
            ).fetchall()
            self.assertEqual(len(events), 7)
            self.assertEqual(events[0]["risk_share_before"], 0.0)
            self.assertEqual(events[0]["market_phase"], "anchor_entry")
            self.assertEqual(events[3]["market_phase"], "max_drawdown_anchor")

            features = connection.execute(
                "SELECT * FROM e6_features WHERE session_id = ?",
                (state["session_id"],),
            ).fetchone()
            assert features is not None
            self.assertEqual(features["feature_version"], "e6_v1")
            self.assertAlmostEqual(features["anchor_risk_exposure_auc"], expected_auc)
            self.assertAlmostEqual(features["anchor_mean_risk_share"], 0.50)
            self.assertAlmostEqual(features["anchor_drawdown_risk_change"], -0.25)
            self.assertAlmostEqual(
                features["anchor_loss_risk_exposure"], expected_loss_exposure
            )
            self.assertAlmostEqual(features["anchor_recovery_reentry"], 0.15)
            self.assertAlmostEqual(features["anchor_recovery_reentry_ratio"], 0.60)

            expected_retention = min(1.0, expected_loss_exposure / 0.60)
            expected_reduction = 1.0 - min(1.0, (0.60 - 0.35) / 0.60)
            expected_threshold = min(
                1.0,
                abs(events[2]["drawdown_from_peak"])
                / abs(scenario.max_drawdown),
            )
            expected_behavior = (
                0.40 * expected_retention
                + 0.30 * expected_reduction
                + 0.20 * expected_threshold
                + 0.10 * 0.60
            )
            self.assertAlmostEqual(features["e6_retention_score"], expected_retention)
            self.assertAlmostEqual(features["e6_reduction_score"], expected_reduction)
            self.assertAlmostEqual(features["e6_threshold_score"], expected_threshold)
            self.assertAlmostEqual(features["e6_recovery_score"], 0.60)
            self.assertAlmostEqual(
                features["e6_behavior_resilience_score"], expected_behavior
            )

            expected_risk_consistency = 1.0 - abs(expected_auc - 0.5)
            expected_loss_consistency = 1.0 - abs(expected_behavior - 0.7)
            expected_cross = (
                expected_risk_consistency + expected_loss_consistency
            ) / 2.0
            self.assertAlmostEqual(
                features["risk_engagement_consistency"], expected_risk_consistency
            )
            self.assertAlmostEqual(
                features["loss_response_consistency"], expected_loss_consistency
            )
            self.assertAlmostEqual(features["cross_context_consistency"], expected_cross)
            self.assertEqual(features["anchor_adjustment_frequency"], 5)
            self.assertAlmostEqual(features["anchor_hold_rate"], 1 / 6)
            self.assertAlmostEqual(features["anchor_adjustment_intensity"], 0.09)
            self.assertAlmostEqual(features["anchor_peak_response"], 0.0)
            self.assertAlmostEqual(features["anchor_trough_response"], -0.15)
            self.assertAlmostEqual(features["anchor_early_recovery_response"], 0.05)
            self.assertAlmostEqual(features["anchor_late_recovery_response"], 0.10)
            self.assertAlmostEqual(features["anchor_final_vs_entry_change"], -0.05)
            self.assertEqual(features["anchor_decision_time_median"], 4000)
            self.assertEqual(features["anchor_max_drawdown_decision_time"], 4000)
            self.assertEqual(features["anchor_recovery_decision_time"], 5500)

            profile = connection.execute(
                "SELECT * FROM profile_features WHERE user_id = 'feature_user'"
            ).fetchone()
            assert profile is not None
            self.assertAlmostEqual(profile["cross_context_consistency"], expected_cross)


if __name__ == "__main__":
    unittest.main()

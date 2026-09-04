from __future__ import annotations

import sqlite3
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
from scenario_store import load_scenarios
from test_support import TEST_ACCESS_CODE, authorize_access


class SequentialEpisodeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "experiment-test.db"
        self.scenarios = load_scenarios(BACKEND_DIR / "scenarios")
        self.app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            scenario_picker=lambda candidates: (
                "E1_01"
                if "E1_01" in candidates
                else ("E2_01" if "E2_01" in candidates else candidates[0])
            ),
            access_code=TEST_ACCESS_CODE,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        authorize_access(self.client)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def start(self, user_id: str = "test_user") -> dict[str, object]:
        survey_state = self.client.post(
            "/api/survey/sessions", json={"user_id": user_id}
        )
        self.assertEqual(survey_state.status_code, 200, survey_state.text)
        if not survey_state.json()["survey_completed"]:
            survey = self.client.post(
                "/api/survey/submissions",
                json={
                    "user_id": user_id,
                    "answers": {
                        "age_group": "age_41_50",
                        "investment_horizon": "years_1_2",
                        "investment_experience": ["neutral"],
                        "derivative_experience": "none_or_under_1_year",
                        "loss_tolerance": "partial_principal_loss",
                        "investment_asset_ratio": "up_to_50",
                        "monthly_income": "up_to_3m",
                        "investment_purpose": "above_market_return",
                        "financial_knowledge": "deep",
                        "vulnerability": "no",
                    },
                },
            )
            self.assertEqual(survey.status_code, 200, survey.text)
        response = self.client.post(
            "/api/episode1/sessions", json={"user_id": user_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def complete_state(
        self, state: dict[str, object], shares: list[float]
    ) -> dict[str, object]:
        episode = str(state["episode"])
        public_scenario_id = str(state["scenario_id"])
        scenario_id = public_scenario_id
        if episode == "E3":
            with closing(connect(self.database_path)) as connection:
                row = connection.execute(
                    "SELECT scenario_id FROM sessions WHERE session_id = ?",
                    (state["session_id"],),
                ).fetchone()
                scenario_id = str(row["scenario_id"])
        scenario = self.scenarios[scenario_id]
        current = state
        for point, share in zip(scenario.decision_points, shares):
            response = self.client.post(
                f"/api/episode{episode[1:]}/sessions/{state['session_id']}/decisions",
                json={
                    "scenario_id": public_scenario_id,
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": share,
                    "decision_time_ms": point.sequence * 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            current = response.json()
        return current

    def test_random_assignment_is_persisted_and_future_prices_are_hidden(self) -> None:
        first = self.start()
        resumed = self.start()
        self.assertEqual(first["session_id"], resumed["session_id"])
        self.assertEqual(first["scenario_id"], "E1_01")
        self.assertEqual(len(first["price_series"]), 1)
        self.assertEqual(first["next_decision"]["decision_point"], "E1_DP1")
        serialized = str(first)
        self.assertNotIn("market_phase", serialized)
        self.assertNotIn("response_tag", serialized)

    def test_episode3_l5_03_corrected_decision_days(self) -> None:
        days = [point.day for point in self.scenarios["E3_L5_03"].decision_points]
        self.assertEqual(days, [12, 17, 23, 38, 45, 52, 60])

    def test_invalid_increment_order_and_duplicate_are_rejected(self) -> None:
        session = self.start()
        session_id = session["session_id"]
        invalid_step = self.client.post(
            f"/api/episode1/sessions/{session_id}/decisions",
            json={
                "scenario_id": "E1_01",
                "decision_point": "E1_DP1",
                "day": 1,
                "risk_share_after": 0.33,
                "decision_time_ms": 1000,
            },
        )
        self.assertEqual(invalid_step.status_code, 422)

        wrong_order = self.client.post(
            f"/api/episode1/sessions/{session_id}/decisions",
            json={
                "scenario_id": "E1_01",
                "decision_point": "E1_DP2",
                "day": 4,
                "risk_share_after": 0.5,
                "decision_time_ms": 1000,
            },
        )
        self.assertEqual(wrong_order.status_code, 409)

        valid_payload = {
            "scenario_id": "E1_01",
            "decision_point": "E1_DP1",
            "day": 1,
            "risk_share_after": 0.5,
            "decision_time_ms": 1000,
        }
        accepted = self.client.post(
            f"/api/episode1/sessions/{session_id}/decisions",
            json=valid_payload,
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(len(accepted.json()["price_series"]), 4)

        duplicate = self.client.post(
            f"/api/episode1/sessions/{session_id}/decisions",
            json=valid_payload,
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_full_sequence_completes_and_raw_logs_are_immutable(self) -> None:
        state = self.start("complete_user")
        session_id = state["session_id"]
        shares = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        scenario = self.scenarios["E1_01"]
        for point, share in zip(scenario.decision_points, shares):
            response = self.client.post(
                f"/api/episode1/sessions/{session_id}/decisions",
                json={
                    "scenario_id": "E1_01",
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": share,
                    "decision_time_ms": point.sequence * 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            state = response.json()

        self.assertEqual(state["episode_status"], "completed")
        self.assertIsNone(state["next_decision"])
        self.assertEqual(len(state["price_series"]), 60)

        with closing(connect(self.database_path)) as connection:
            logs = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? ORDER BY decision_index",
                (session_id,),
            ).fetchall()
            features = connection.execute(
                "SELECT * FROM e1_features WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self.assertEqual(len(logs), 7)
            self.assertIsNotNone(features)
            assert features is not None
            self.assertEqual(features["feature_version"], "e1_v2")
            self.assertEqual(features["episode_status"], "completed")
            self.assertAlmostEqual(features["mean_risk_share"], 0.3)
            self.assertAlmostEqual(features["market_participation_rate"], 56 / 59)
            self.assertEqual(logs[0]["risk_share_before"], 0.0)
            self.assertEqual(logs[1]["risk_share_before"], 0.0)
            self.assertEqual(logs[1]["delta_risk_share"], 0.1)
            self.assertEqual(logs[1]["market_phase"], "early_gain")

            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE behavior_events SET risk_share_after = 1 WHERE event_id = ?",
                    (logs[0]["event_id"],),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM behavior_events WHERE event_id = ?",
                    (logs[0]["event_id"],),
                )

    def test_episode2_requires_e1_and_uses_common_log(self) -> None:
        blocked = self.client.post(
            "/api/episode2/sessions", json={"user_id": "sequential_user"}
        )
        self.assertEqual(blocked.status_code, 409)

        e1 = self.start("sequential_user")
        e1_scenario = self.scenarios["E1_01"]
        for point in e1_scenario.decision_points:
            response = self.client.post(
                f"/api/episode1/sessions/{e1['session_id']}/decisions",
                json={
                    "scenario_id": "E1_01",
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": 0.5,
                    "decision_time_ms": 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)

        e2_response = self.client.post(
            "/api/episode2/sessions", json={"user_id": "sequential_user"}
        )
        self.assertEqual(e2_response.status_code, 200, e2_response.text)
        e2 = e2_response.json()
        self.assertEqual(e2["scenario_id"], "E2_01")
        self.assertEqual(e2["next_decision"]["decision_point"], "E2_DP1")
        self.assertEqual(len(e2["price_series"]), 1)

        scenario = self.scenarios["E2_01"]
        for point, share in zip(
            scenario.decision_points, [0.4, 0.4, 0.5, 0.6, 0.7, 0.5, 0.65]
        ):
            response = self.client.post(
                f"/api/episode2/sessions/{e2['session_id']}/decisions",
                json={
                    "scenario_id": "E2_01",
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": share,
                    "decision_time_ms": point.sequence * 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["episode_status"], "completed")

        with closing(connect(self.database_path)) as connection:
            episode_counts = connection.execute(
                "SELECT episode, COUNT(*) AS count FROM behavior_events "
                "GROUP BY episode ORDER BY episode"
            ).fetchall()
            self.assertEqual(
                [(row["episode"], row["count"]) for row in episode_counts],
                [("E1", 7), ("E2", 7)],
            )
            e2_features = connection.execute(
                "SELECT * FROM e2_features WHERE session_id = ?",
                (e2["session_id"],),
            ).fetchone()
            self.assertIsNotNone(e2_features)
            assert e2_features is not None
            self.assertEqual(e2_features["feature_version"], "e2_v3")
            self.assertEqual(e2_features["uptrend_risk_increase_count"], 3)
            self.assertAlmostEqual(
                e2_features["e2_vs_e1_risk_shift"],
                e2_features["uptrend_risk_exposure"] - 0.5,
            )
            e2_events = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? "
                "ORDER BY decision_index",
                (e2["session_id"],),
            ).fetchall()
            self.assertIsNone(e2_events[0]["return_since_previous_dp"])
            self.assertAlmostEqual(
                e2_events[1]["return_since_previous_dp"],
                e2_events[1]["normalized_price"]
                / e2_events[0]["normalized_price"]
                - 1.0,
            )
            self.assertIsNotNone(e2_events[1]["trailing_return_5d"])
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertTrue(
                {"sessions", "behavior_events", "e1_features", "e2_features",
                 "e3_features", "e4_features", "e5_features", "e6_features",
                 "profile_features"}.issubset(tables)
            )
            profile_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(profile_features)"
                ).fetchall()
            }
            self.assertIn("cross_context_consistency", profile_columns)
            self.assertNotIn("cross_contest_consistency", profile_columns)

    def test_episode3_routing_context_and_features(self) -> None:
        user_id = "adaptive_user"
        e1 = self.start(user_id)
        self.complete_state(e1, [0.5] * 7)
        e2_response = self.client.post(
            "/api/episode2/sessions", json={"user_id": user_id}
        )
        self.assertEqual(e2_response.status_code, 200, e2_response.text)
        e2 = e2_response.json()
        self.complete_state(e2, [0.4, 0.4, 0.5, 0.6, 0.7, 0.5, 0.65])

        e3_response = self.client.post(
            "/api/episode3/sessions", json={"user_id": user_id}
        )
        self.assertEqual(e3_response.status_code, 200, e3_response.text)
        e3 = e3_response.json()
        self.assertEqual(e3["scenario_id"], "E3_01")
        self.assertEqual(e3["current_risk_share"], 0.65)
        self.assertTrue(e3["entry_setup_required"])
        self.assertIsNone(e3["next_decision"])
        self.assertEqual(e3["price_series"], [])
        self.assertNotIn("assigned_level", e3)
        self.assertNotIn("routing_score", e3)
        entry_response = self.client.post(
            f"/api/episode3/sessions/{e3['session_id']}/entry",
            json={"risk_share": 0.55},
        )
        self.assertEqual(entry_response.status_code, 200, entry_response.text)
        e3 = entry_response.json()
        self.assertFalse(e3["entry_setup_required"])
        self.assertEqual(e3["current_risk_share"], 0.55)
        self.assertEqual(e3["next_decision"]["day"], 24)
        self.assertEqual(len(e3["price_series"]), 24)

        completed = self.complete_state(
            e3, [0.60, 0.55, 0.50, 0.45, 0.40, 0.45, 0.50]
        )
        self.assertEqual(completed["episode_status"], "completed")
        with closing(connect(self.database_path)) as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (e3["session_id"],)
            ).fetchone()
            self.assertEqual(session["assigned_level"], "L3")
            self.assertEqual(session["scenario_id"], "E3_L3_01")
            self.assertEqual(session["routing_version"], "e3_routing_v1")
            self.assertAlmostEqual(session["routing_score"], 0.7 * 0.5 + 0.3 * (16.4 / 31))
            self.assertLess(session["scenario_max_drawdown"], 0.0)
            self.assertEqual(session["entry_risk_share"], 0.55)
            self.assertEqual(session["entry_confirmed"], 1)
            events = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? ORDER BY decision_index",
                (e3["session_id"],),
            ).fetchall()
            self.assertEqual(len(events), 7)
            self.assertEqual(events[0]["risk_share_before"], 0.55)
            self.assertEqual(events[0]["initial_preallocated_risk_share"], 0.55)
            self.assertEqual(events[0]["allocation_floor"], 0.0)
            features = connection.execute(
                "SELECT * FROM e3_features WHERE session_id = ?", (e3["session_id"],)
            ).fetchone()
            self.assertEqual(features["feature_version"], "e3_v3")
            self.assertEqual(features["episode_status"], "completed")
            self.assertAlmostEqual(features["loss_period_risk_change"], -0.20)
            self.assertAlmostEqual(features["recovery_reentry"], 0.10)
            self.assertIsNotNone(features["e3_loss_resilience_score"])

    def test_episode3_l1_floor_without_initial_reduction_limit(self) -> None:
        user_id = "never_entered_user"
        e1 = self.start(user_id)
        self.complete_state(e1, [0.0] * 7)
        e2_response = self.client.post(
            "/api/episode2/sessions", json={"user_id": user_id}
        )
        self.assertEqual(e2_response.status_code, 200, e2_response.text)
        self.complete_state(e2_response.json(), [0.0] * 7)
        e3_response = self.client.post(
            "/api/episode3/sessions", json={"user_id": user_id}
        )
        self.assertEqual(e3_response.status_code, 200, e3_response.text)
        e3 = e3_response.json()
        self.assertEqual(e3["scenario_id"], "E3_01")
        self.assertEqual(e3["current_risk_share"], 0.30)
        self.assertEqual(
            e3["allocation_constraints"]["minimum_next_risk_share"], 0.10
        )
        point = e3["next_decision"]
        rejected = self.client.post(
            f"/api/episode3/sessions/{e3['session_id']}/decisions",
            json={
                "scenario_id": e3["scenario_id"],
                "decision_point": point["decision_point"],
                "day": point["day"],
                "risk_share_after": 0.05,
                "decision_time_ms": 1000,
            },
        )
        self.assertEqual(rejected.status_code, 422)
        accepted_payload = {
            "scenario_id": e3["scenario_id"],
            "decision_point": point["decision_point"],
            "day": point["day"],
            "risk_share_after": 0.10,
            "decision_time_ms": 1000,
        }
        accepted = self.client.post(
            f"/api/episode3/sessions/{e3['session_id']}/decisions",
            json=accepted_payload,
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        with closing(connect(self.database_path)) as connection:
            event = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ?",
                (e3["session_id"],),
            ).fetchone()
            self.assertEqual(event["initial_preallocated_risk_share"], 0.30)
            self.assertEqual(event["allocation_floor"], 0.10)
            self.assertEqual(event["floor_reached"], 1)


if __name__ == "__main__":
    unittest.main()

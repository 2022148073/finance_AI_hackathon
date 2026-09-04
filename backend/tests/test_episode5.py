from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import connect
from main import create_app
from scenario_store import load_scenarios
from stimulus_store import POLARITY_CYCLES, SOURCE_PAIRS
from test_support import TEST_ACCESS_CODE, authorize_access, complete_survey


class Episode5ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "e5-test.db"
        self.scenarios = load_scenarios(BACKEND_DIR / "scenarios")
        self.app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            stimulus_dir=BACKEND_DIR / "scenarios" / "episode5" / "stimuli",
            scenario_picker=lambda candidates: candidates[0],
            e5_randomizer=random.Random(20260831),
            access_code=TEST_ACCESS_CODE,
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        authorize_access(self.client)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def seed_completed_e4(self, user_id: str, final_share: float = 0.60) -> None:
        complete_survey(self.client, user_id)
        session_id = f"e4-{user_id}"
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO sessions (session_id,user_id,episode,scenario_id,"
                "episode_status,created_at,updated_at,completed_at,entry_risk_share,"
                "entry_confirmed) VALUES (?,?,'E4','E4_V1_01','completed',"
                "'now','now','now',?,1)",
                (session_id, user_id, final_share),
            )
            connection.execute(
                "INSERT INTO behavior_events (session_id,episode,scenario_id,"
                "decision_point,decision_index,event_phase,day,risk_share_before,"
                "risk_share_after,cash_share_after,delta_risk_share,decision_time_ms,"
                "normalized_price,return_from_initial,drawdown_from_peak,"
                "semantic_role,market_phase,created_at) VALUES "
                "(?,'E4','E4_V1_01','E4_DP7',7,'allocation',60,?,?,?,?,1000,"
                "100,0,0,'seed','seed','now')",
                (
                    session_id,
                    final_share,
                    final_share,
                    1.0 - final_share,
                    0.0,
                ),
            )
            connection.commit()

    def start_e5(self, user_id: str) -> dict[str, object]:
        self.seed_completed_e4(user_id)
        response = self.client.post(
            "/api/episode5/sessions", json={"user_id": user_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_scenario_days_and_randomization_are_persisted_once(self) -> None:
        expected_days = {
            "E5_01": [20, 31, 43],
            "E5_02": [28, 39, 49],
            "E5_03": [16, 30, 48],
        }
        self.assertEqual(
            {
                scenario_id: [point.day for point in scenario.decision_points]
                for scenario_id, scenario in self.scenarios.items()
                if scenario.episode == "E5"
            },
            expected_days,
        )
        state = self.start_e5("randomization_user")
        with closing(connect(self.database_path)) as connection:
            session = connection.execute(
                "SELECT e5_polarity_cycle,e5_randomization_version FROM sessions "
                "WHERE session_id = ?",
                (state["session_id"],),
            ).fetchone()
            assert session is not None
            polarity_cycle = session["e5_polarity_cycle"]
            self.assertIn(polarity_cycle, POLARITY_CYCLES)
            self.assertEqual(
                session["e5_randomization_version"], "e5_randomization_v2"
            )
            before = connection.execute(
                "SELECT * FROM e5_decision_assignments WHERE session_id = ? "
                "ORDER BY decision_index",
                (state["session_id"],),
            ).fetchall()
            self.assertEqual(len(before), 3)
            self.assertEqual({row["stimulus_pair_id"] for row in before}, set(SOURCE_PAIRS))
            source_sentiments = {source: [] for source in ("news", "expert", "community")}
            for row in before:
                self.assertEqual(
                    (row["first_source"], row["second_source"]),
                    SOURCE_PAIRS[row["stimulus_pair_id"]],
                )
                self.assertEqual(
                    {row["first_sentiment"], row["second_sentiment"]},
                    {"positive", "negative"},
                )
                self.assertEqual(
                    {row["left_template_id"], row["right_template_id"]},
                    {row["first_template_id"], row["second_template_id"]},
                )
                self.assertEqual(
                    (row["first_sentiment"], row["second_sentiment"]),
                    POLARITY_CYCLES[polarity_cycle][row["stimulus_pair_id"]],
                )
                source_sentiments[row["first_source"]].append(row["first_sentiment"])
                source_sentiments[row["second_source"]].append(row["second_sentiment"])
            for sentiments in source_sentiments.values():
                self.assertEqual(sorted(sentiments), ["negative", "positive"])
            before_values = [tuple(row) for row in before]

        resumed = self.client.post(
            "/api/episode5/sessions", json={"user_id": "randomization_user"}
        )
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["session_id"], state["session_id"])
        with closing(connect(self.database_path)) as connection:
            after = connection.execute(
                "SELECT * FROM e5_decision_assignments WHERE session_id = ? "
                "ORDER BY decision_index",
                (state["session_id"],),
            ).fetchall()
            self.assertEqual([tuple(row) for row in after], before_values)

    def test_pre_post_flow_snapshot_privacy_logs_and_features(self) -> None:
        state = self.start_e5("flow_user")
        self.assertEqual(state["scenario_id"], "E5_01")
        self.assertEqual(state["interaction_phase"], "pre_information")
        self.assertEqual(state["stimulus_cards"], [])
        decisions = [(0.70, 0.50), (0.40, 0.40), (0.45, 0.55)]
        pre_times = [1000, 3000, 5000]
        post_times = [2000, 4000, 6000]

        for index, ((pre_share, post_share), pre_time, post_time) in enumerate(
            zip(decisions, pre_times, post_times), start=1
        ):
            point = state["next_decision"]
            reveal_before = list(state["price_series"])
            with closing(connect(self.database_path)) as connection:
                connection.execute(
                    "UPDATE sessions SET decision_started_at = ? WHERE session_id = ?",
                    (
                        (
                            datetime.now(timezone.utc)
                            - timedelta(milliseconds=pre_time)
                        ).isoformat(),
                        state["session_id"],
                    ),
                )
                connection.commit()
            pre_response = self.client.post(
                f"/api/episode5/sessions/{state['session_id']}/pre-decisions",
                json={
                    "scenario_id": state["scenario_id"],
                    "decision_point": point["decision_point"],
                    "day": point["day"],
                    "risk_share_pre_info": pre_share,
                    "decision_time_ms": 1,
                },
            )
            self.assertEqual(pre_response.status_code, 200, pre_response.text)
            post_state = pre_response.json()
            self.assertEqual(post_state["interaction_phase"], "post_information")
            self.assertEqual(post_state["price_series"], reveal_before)
            self.assertEqual(len(post_state["stimulus_cards"]), 2)
            for card in post_state["stimulus_cards"]:
                self.assertEqual(
                    set(card), {"position", "source_label", "title", "content"}
                )
            restored = self.client.get(
                f"/api/episode5/sessions/{state['session_id']}"
            )
            self.assertEqual(restored.status_code, 200, restored.text)
            self.assertEqual(
                restored.json()["stimulus_cards"], post_state["stimulus_cards"]
            )

            with closing(connect(self.database_path)) as connection:
                connection.execute(
                    "UPDATE sessions SET decision_started_at = ? WHERE session_id = ?",
                    (
                        (
                            datetime.now(timezone.utc)
                            - timedelta(milliseconds=post_time)
                        ).isoformat(),
                        state["session_id"],
                    ),
                )
                connection.commit()

            post_response = self.client.post(
                f"/api/episode5/sessions/{state['session_id']}/post-decisions",
                json={
                    "scenario_id": state["scenario_id"],
                    "decision_point": point["decision_point"],
                    "day": point["day"],
                    "risk_share_post_info": post_share,
                    "decision_time_ms": 1,
                },
            )
            self.assertEqual(post_response.status_code, 200, post_response.text)
            state = post_response.json()
            if index < 3:
                self.assertEqual(state["interaction_phase"], "pre_information")
                self.assertGreater(len(state["price_series"]), len(reveal_before))

        self.assertEqual(state["episode_status"], "completed")
        self.assertEqual(state["progress"], {"submitted": 3, "total": 3})
        with closing(connect(self.database_path)) as connection:
            events = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? "
                "ORDER BY decision_index, event_phase DESC",
                (state["session_id"],),
            ).fetchall()
            self.assertEqual(len(events), 6)
            for index in range(3):
                pair = [
                    event for event in events if event["decision_index"] == index + 1
                ]
                self.assertEqual(
                    {event["event_phase"] for event in pair},
                    {"pre_information", "post_information"},
                )
                snapshot_fields = (
                    "market_snapshot_id", "normalized_price",
                    "return_from_initial", "drawdown_from_peak",
                    "trailing_return_5d", "rolling_volatility_20d",
                )
                for field in snapshot_fields:
                    self.assertEqual(pair[0][field], pair[1][field])
            post_events = [
                event for event in events if event["event_phase"] == "post_information"
            ]
            self.assertAlmostEqual(post_events[0]["information_delta"], -0.20)
            self.assertIsNotNone(post_events[0]["aligned_source"])
            self.assertIsNone(post_events[1]["aligned_source"])
            features = connection.execute(
                "SELECT * FROM e5_features WHERE session_id = ?",
                (state["session_id"],),
            ).fetchone()
            assert features is not None
            self.assertEqual(features["feature_version"], "e5_v2")
            self.assertAlmostEqual(features["external_information_sensitivity"], 0.10)
            self.assertAlmostEqual(features["information_adjustment_rate"], 2 / 3)
            self.assertAlmostEqual(features["information_hold_rate"], 1 / 3)
            self.assertEqual(features["positive_alignment_count"], 1)
            self.assertEqual(features["negative_alignment_count"], 1)
            self.assertEqual(features["information_counter_adjustment_count"], 1)
            self.assertEqual(features["conflict_hold_count"], 1)
            self.assertEqual(
                features["news_alignment_score"]
                + features["expert_alignment_score"]
                + features["community_alignment_score"],
                2,
            )
            expected_magnitudes = {
                "news": 0.0,
                "expert": 0.0,
                "community": 0.0,
            }
            for event in post_events:
                if event["aligned_source"] is not None:
                    expected_magnitudes[event["aligned_source"]] += abs(
                        event["information_delta"]
                    )
            for source, expected in expected_magnitudes.items():
                self.assertAlmostEqual(
                    features[f"{source}_alignment_magnitude"], expected
                )
            self.assertAlmostEqual(sum(expected_magnitudes.values()), 0.30)
            self.assertAlmostEqual(
                features["pre_information_decision_time_median"], 3000, delta=150
            )
            self.assertAlmostEqual(
                features["post_information_decision_time_median"], 4000, delta=150
            )
            self.assertAlmostEqual(
                features["information_decision_time_change"], 1000, delta=300
            )

    def test_post_before_pre_and_duplicate_pre_are_rejected(self) -> None:
        state = self.start_e5("order_user")
        point = state["next_decision"]
        common = {
            "scenario_id": state["scenario_id"],
            "decision_point": point["decision_point"],
            "day": point["day"],
            "decision_time_ms": 1000,
        }
        post_first = self.client.post(
            f"/api/episode5/sessions/{state['session_id']}/post-decisions",
            json={**common, "risk_share_post_info": 0.50},
        )
        self.assertEqual(post_first.status_code, 409)
        accepted = self.client.post(
            f"/api/episode5/sessions/{state['session_id']}/pre-decisions",
            json={**common, "risk_share_pre_info": 0.50},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        duplicate = self.client.post(
            f"/api/episode5/sessions/{state['session_id']}/pre-decisions",
            json={**common, "risk_share_pre_info": 0.50},
        )
        self.assertEqual(duplicate.status_code, 409)


if __name__ == "__main__":
    unittest.main()

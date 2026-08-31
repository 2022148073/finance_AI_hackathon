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
from features import calculate_episode4_features
from main import create_app
from routing import route_episode4
from scenario_store import load_scenarios


class Episode4RoutingTests(unittest.TestCase):
    def test_score_bands_fallback_conflict_and_final_cap(self) -> None:
        normal = route_episode4(
            e3_routing_score=0.50,
            e3_loss_resilience_score=0.72,
            e3_assigned_level="L3",
            floor_reached=False,
            full_exit=False,
        )
        self.assertEqual(normal.assigned_level, "V4")
        self.assertFalse(normal.routing_fallback)
        self.assertAlmostEqual(normal.routing_score, 0.4 * 0.50 + 0.6 * 0.72)

        fallback = route_episode4(
            e3_routing_score=0.86,
            e3_loss_resilience_score=None,
            e3_assigned_level="L4",
            floor_reached=False,
            full_exit=False,
        )
        self.assertEqual(fallback.assigned_level, "V5")
        self.assertTrue(fallback.routing_fallback)
        self.assertIsNone(fallback.context_gap)

        conflict = route_episode4(
            e3_routing_score=0.10,
            e3_loss_resilience_score=0.45,
            e3_assigned_level="L3",
            floor_reached=False,
            full_exit=False,
        )
        self.assertEqual(conflict.assigned_level, "V3")
        self.assertAlmostEqual(conflict.context_gap or 0.0, 0.35)

        capped = route_episode4(
            e3_routing_score=0.90,
            e3_loss_resilience_score=None,
            e3_assigned_level="L1",
            floor_reached=True,
            full_exit=False,
        )
        self.assertEqual(capped.assigned_level, "V2")
        self.assertTrue(capped.upper_level_capped)


class Episode4FeatureTests(unittest.TestCase):
    def test_volatility_features_and_hold_exclusion(self) -> None:
        days = [1, 21, 28, 35, 42, 51, 60]
        volatilities = [None, 0.10, 0.12, 0.16, 0.14, 0.09, 0.11]
        shares = [0.50, 0.50, 0.40, 0.40, 0.45, 0.55, 0.55]
        logs: list[dict[str, object]] = []
        previous_share = 0.50
        previous_volatility: float | None = None
        for index, (day, volatility, share) in enumerate(
            zip(days, volatilities, shares), start=1
        ):
            delta_volatility = (
                None
                if volatility is None or previous_volatility is None
                else volatility - previous_volatility
            )
            direction = None
            if delta_volatility is not None:
                direction = (
                    "rising"
                    if delta_volatility >= 0.01
                    else ("falling" if delta_volatility <= -0.01 else "stable")
                )
            logs.append(
                {
                    "decision_index": index,
                    "day": day,
                    "risk_share_after": share,
                    "delta_risk_share": share - previous_share,
                    "decision_time_ms": index * 1000,
                    "rolling_volatility_20d": volatility,
                    "delta_volatility_20d": delta_volatility,
                    "volatility_direction": direction,
                }
            )
            previous_share = share
            if volatility is not None:
                previous_volatility = volatility

        rolling = [None] * 20 + [0.09 + index * 0.002 for index in range(40)]
        result = calculate_episode4_features(
            logs,
            "completed",
            scenario_rolling_volatility_20d=rolling,
            volatility_q25=0.1095,
            volatility_q75=0.1485,
            entry_risk_share=0.50,
        )
        self.assertEqual(result.decision_count, 7)
        self.assertIsNotNone(result.volatility_sensitivity)
        self.assertIsNotNone(result.high_vol_risk_exposure)
        self.assertIsNotNone(result.low_vol_risk_exposure)
        self.assertEqual(result.volatility_risk_increase_count, 0)
        # Non-stable transitions have changes [-.1, 0, +.05, +.1, 0].
        self.assertAlmostEqual(
            result.volatility_adjustment_intensity or 0.0,
            (0.10 + 0.05 + 0.10) / 3,
        )
        self.assertAlmostEqual(result.final_vs_entry_risk_change or 0.0, 0.05)
        self.assertEqual(result.peak_volatility_decision_time, 4000.0)


class Episode4ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "e4-test.db"
        self.scenarios = load_scenarios(BACKEND_DIR / "scenarios")
        self.app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            scenario_picker=lambda candidates: candidates[0],
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def seed_completed_e3(self, user_id: str) -> None:
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO sessions (session_id,user_id,episode,scenario_id,"
                "episode_status,created_at,updated_at,completed_at,assigned_level,"
                "routing_score,routing_version,entry_risk_share,allocation_floor,"
                "entry_confirmed) VALUES (?,?, 'E3','E3_L3_01','completed',"
                "'now','now','now','L3',.50,'e3_routing_v1',.60,0,1)",
                (f"e3-{user_id}", user_id),
            )
            connection.execute(
                "INSERT INTO e3_features (session_id,feature_version,computed_at,"
                "decision_count,drawdown_period_risk_increase_count,"
                "e3_loss_resilience_score,episode_status) "
                "VALUES (?,'e3_v2','now',7,0,.50,'completed')",
                (f"e3-{user_id}",),
            )
            for index, day in enumerate([1, 10, 20, 30, 40, 50, 60], start=1):
                connection.execute(
                    "INSERT INTO behavior_events (session_id,episode,scenario_id,"
                    "decision_point,decision_index,day,risk_share_before,"
                    "risk_share_after,cash_share_after,delta_risk_share,"
                    "decision_time_ms,normalized_price,return_from_initial,"
                    "drawdown_from_peak,semantic_role,market_phase,created_at) "
                    "VALUES (?, 'E3','E3_L3_01',?,?,?,?,.60,.40,0,1000,100,0,0,"
                    "'seed','seed','now')",
                    (
                        f"e3-{user_id}", f"E3_DP{index}", index, day, 0.60,
                    ),
                )
            connection.commit()

    def test_episode4_decision_point_configuration(self) -> None:
        expected = {
            "E4_V1_01": [1, 21, 28, 31, 40, 51, 60],
            "E4_V1_02": [1, 21, 30, 37, 49, 55, 60],
            "E4_V1_03": [1, 21, 26, 40, 45, 52, 60],
            "E4_V2_01": [1, 21, 26, 35, 43, 51, 60],
            "E4_V2_02": [1, 21, 33, 40, 44, 50, 60],
            "E4_V2_03": [1, 21, 28, 34, 41, 48, 60],
            "E4_V3_01": [1, 21, 22, 34, 40, 55, 60],
            "E4_V3_02": [1, 21, 27, 35, 41, 54, 60],
            "E4_V3_03": [1, 21, 31, 39, 49, 54, 60],
            "E4_V4_01": [1, 21, 29, 34, 40, 48, 60],
            "E4_V4_02": [1, 21, 31, 36, 47, 52, 60],
            "E4_V4_03": [1, 21, 30, 39, 44, 55, 60],
            "E4_V5_01": [1, 21, 29, 35, 47, 53, 60],
            "E4_V5_02": [1, 21, 27, 32, 35, 52, 60],
            "E4_V5_03": [1, 21, 28, 33, 43, 52, 60],
        }
        actual = {
            scenario_id: [point.day for point in scenario.decision_points]
            for scenario_id, scenario in self.scenarios.items()
            if scenario.episode == "E4"
        }
        self.assertEqual(actual, expected)

        expected_sources = {
            "E4_V2_01": "V2/E4_V2_01.json",
            "E4_V2_02": "V2/E4_V2_02.json",
            "E4_V2_03": "V2/E4_V2_03.json",
            "E4_V4_01": "V4/E4_V4_01.json",
            "E4_V4_02": "V4/E4_V4_02.json",
            "E4_V4_03": "V4/E4_V4_03.json",
            "E4_V5_01": "V5/E4_V5_01_extension.json",
            "E4_V5_02": "V5/E4_V5_02_extension.json",
            "E4_V5_03": "V5/E4_V5_03.json",
        }
        frontend_root = BACKEND_DIR.parent / "frontend" / "scenarios" / "episode4"
        for scenario_id, relative_source in expected_sources.items():
            source = json.loads(
                (frontend_root / relative_source).read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(self.scenarios[scenario_id].prices),
                [float(value) for value in source["series"]["normalized_prices"]],
            )

    def test_episode4_market_phase_configuration_only_changes_dp3_to_dp7(self) -> None:
        expected = {
            "E4_V1_01": ["volatility_expansion", "stable_elevated_volatility", "local_volatility_peak", "volatility_compression", "final_low_volatility"],
            "E4_V1_02": ["volatility_expansion", "local_volatility_peak", "volatility_compression", "renewed_volatility", "final_local_volatility_peak"],
            "E4_V1_03": ["local_volatility_trough", "volatility_expansion", "local_volatility_peak", "volatility_compression", "continued_compression"],
            "E4_V2_01": ["local_volatility_trough", "local_volatility_peak", "sustained_elevated_volatility", "strong_volatility_compression", "renewed_volatility"],
            "E4_V2_02": ["strong_volatility_compression", "stable_low_volatility", "volatility_spike", "sustained_elevated_volatility", "final_elevated_volatility"],
            "E4_V2_03": ["stable_volatility", "volatility_expansion", "mild_volatility_compression", "continued_compression", "renewed_volatility_peak"],
            "E4_V3_01": ["local_volatility_peak", "mild_volatility_compression", "continued_compression", "local_volatility_trough", "renewed_volatility"],
            "E4_V3_02": ["strong_volatility_compression", "stable_volatility", "local_volatility_trough", "renewed_volatility", "continued_expansion"],
            "E4_V3_03": ["strong_volatility_compression", "continued_compression", "local_volatility_trough", "renewed_volatility_spike", "sustained_elevated_volatility"],
            "E4_V4_01": ["strong_volatility_compression", "renewed_volatility", "strong_volatility_compression", "mild_volatility_expansion", "final_low_volatility"],
            "E4_V4_02": ["local_volatility_peak", "strong_volatility_compression", "continued_compression", "local_volatility_trough", "renewed_volatility"],
            "E4_V4_03": ["volatility_expansion", "local_volatility_peak", "strong_volatility_compression", "local_volatility_trough", "renewed_volatility"],
            "E4_V5_01": ["local_volatility_peak", "strong_volatility_compression", "local_volatility_trough", "mild_volatility_expansion", "final_compressed_volatility"],
            "E4_V5_02": ["mild_volatility_expansion", "local_volatility_peak", "strong_volatility_compression", "local_volatility_trough", "renewed_volatility"],
            "E4_V5_03": ["local_volatility_trough", "renewed_volatility", "local_volatility_peak", "strong_volatility_compression", "final_low_volatility"],
        }
        expected_roles = [
            "entry_allocation", "initial_vol_anchor", "first_volatility_shift",
            "established_vol_regime", "volatility_extreme",
            "volatility_reversal", "final_vol_state",
        ]
        for scenario_id, phases in expected.items():
            points = self.scenarios[scenario_id].decision_points
            self.assertEqual(
                [point.semantic_role for point in points], expected_roles
            )
            self.assertEqual(
                [point.market_phase for point in points[:2]], expected_roles[:2]
            )
            self.assertEqual(
                [point.market_phase for point in points[2:]], phases
            )

    def test_e4_session_logs_context_and_features(self) -> None:
        user_id = "episode4_user"
        self.seed_completed_e3(user_id)
        response = self.client.post(
            "/api/episode4/sessions", json={"user_id": user_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["episode"], "E4")
        self.assertEqual(state["scenario_id"], "E4_01")
        self.assertEqual(state["asset"], "Asset 4")
        self.assertNotIn("assigned_volatility_level", state)
        self.assertNotIn("rolling_volatility_20d", str(state))

        internal_id = "E4_V3_01"
        scenario = self.scenarios[internal_id]
        for point, share in zip(
            scenario.decision_points, [0.60, 0.60, 0.50, 0.45, 0.40, 0.50, 0.55]
        ):
            response = self.client.post(
                f"/api/episode4/sessions/{state['session_id']}/decisions",
                json={
                    "scenario_id": "E4_01",
                    "decision_point": point.decision_point,
                    "day": point.day,
                    "risk_share_after": share,
                    "decision_time_ms": point.sequence * 1000,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["episode_status"], "completed")

        with closing(connect(self.database_path)) as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (state["session_id"],)
            ).fetchone()
            assert session is not None
            self.assertEqual(session["assigned_volatility_level"], "V3")
            self.assertEqual(session["routing_version"], "e4_routing_v1")
            self.assertEqual(session["e4_routing_fallback"], 0)
            self.assertAlmostEqual(session["e4_routing_score"], 0.50)
            self.assertIsNotNone(session["scenario_volatility_20d_q75"])
            events = connection.execute(
                "SELECT * FROM behavior_events WHERE session_id = ? "
                "ORDER BY decision_index",
                (state["session_id"],),
            ).fetchall()
            self.assertEqual(len(events), 7)
            self.assertIsNone(events[0]["rolling_volatility_20d"])
            self.assertIsNotNone(events[1]["rolling_volatility_20d"])
            self.assertIsNone(events[1]["delta_volatility_20d"])
            self.assertIsNotNone(events[2]["delta_volatility_20d"])
            self.assertIsNotNone(events[2]["max_abs_daily_return_since_previous_dp"])
            features = connection.execute(
                "SELECT * FROM e4_features WHERE session_id = ?",
                (state["session_id"],),
            ).fetchone()
            assert features is not None
            self.assertEqual(features["feature_version"], "e4_v1")
            self.assertEqual(features["decision_count"], 7)
            self.assertEqual(features["episode_status"], "completed")


if __name__ == "__main__":
    unittest.main()

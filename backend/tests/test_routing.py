from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routing import route_episode3


class Episode3RoutingTests(unittest.TestCase):
    def test_weighted_score_levels(self) -> None:
        cases = [
            (0.10, "L1"),
            (0.30, "L2"),
            (0.50, "L3"),
            (0.70, "L4"),
            (0.90, "L5"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                result = route_episode3(
                    e1_risk_exposure_auc=score,
                    e2_uptrend_risk_exposure=score,
                    e1_never_entered=False,
                    e2_final_risk_share=0.55,
                )
                self.assertAlmostEqual(result.routing_score, score)
                self.assertEqual(result.assigned_level, expected)

    def test_context_gap_routes_to_l3(self) -> None:
        result = route_episode3(
            e1_risk_exposure_auc=0.10,
            e2_uptrend_risk_exposure=0.60,
            e1_never_entered=False,
            e2_final_risk_share=0.50,
        )
        self.assertEqual(result.assigned_level, "L3")

    def test_never_entered_override_has_priority(self) -> None:
        result = route_episode3(
            e1_risk_exposure_auc=0.0,
            e2_uptrend_risk_exposure=0.80,
            e1_never_entered=True,
            e2_final_risk_share=0.80,
        )
        self.assertEqual(result.assigned_level, "L1")
        self.assertEqual(result.entry_risk_share, 0.30)
        self.assertEqual(result.allocation_floor, 0.10)

    def test_l2_preallocation_rule(self) -> None:
        result = route_episode3(
            e1_risk_exposure_auc=0.25,
            e2_uptrend_risk_exposure=0.25,
            e1_never_entered=False,
            e2_final_risk_share=0.60,
        )
        self.assertEqual(result.assigned_level, "L2")
        self.assertEqual(result.entry_risk_share, 0.20)
        self.assertEqual(result.allocation_floor, 0.10)


if __name__ == "__main__":
    unittest.main()

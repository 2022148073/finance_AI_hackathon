from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from features import (
    calculate_episode1_features,
    calculate_episode2_features,
    calculate_episode3_features,
)


DAYS = [1, 4, 11, 21, 28, 42, 60]
TAGS = [
    None,
    "mild_gain",
    "mild_drawdown",
    "recovery",
    "mild_drawdown",
    None,
    "mild_gain",
]


def make_logs(shares: list[float]) -> list[dict[str, object]]:
    logs: list[dict[str, object]] = []
    previous = 0.0
    for index, (day, share, tag) in enumerate(
        zip(DAYS, shares, TAGS), start=1
    ):
        logs.append(
            {
                "decision_index": index,
                "day": day,
                "risk_share_after": share,
                "delta_risk_share": share - previous,
                "decision_time_ms": index * 1000,
                "response_tag": tag,
            }
        )
        previous = share
    return logs


class Episode1FeatureTests(unittest.TestCase):
    def test_always_zero_never_enters(self) -> None:
        result = calculate_episode1_features(make_logs([0.0] * 7), "completed")
        self.assertEqual(result.initial_risk_share, 0.0)
        self.assertEqual(result.risk_exposure_auc, 0.0)
        self.assertEqual(result.mean_risk_share, 0.0)
        self.assertEqual(result.market_participation_rate, 0.0)
        self.assertIsNone(result.time_to_first_entry)
        self.assertTrue(result.never_entered)
        self.assertEqual(result.adjustment_frequency, 0)
        self.assertEqual(result.mean_abs_allocation_change, 0.0)
        self.assertEqual(result.hold_rate, 1.0)

    def test_always_eighty_percent(self) -> None:
        result = calculate_episode1_features(make_logs([0.8] * 7), "completed")
        self.assertAlmostEqual(result.risk_exposure_auc, 0.8)
        self.assertAlmostEqual(result.mean_risk_share, 0.8)
        self.assertAlmostEqual(result.market_participation_rate, 1.0)
        self.assertEqual(result.time_to_first_entry, 0)
        self.assertFalse(result.never_entered)
        self.assertEqual(result.adjustment_frequency, 0)
        self.assertAlmostEqual(result.hold_rate or 0.0, 1.0)

    def test_gradual_entry_and_increase(self) -> None:
        shares = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        result = calculate_episode1_features(make_logs(shares), "completed")
        expected_auc = sum(
            shares[index] * (DAYS[index + 1] - DAYS[index])
            for index in range(6)
        ) / 59
        self.assertAlmostEqual(result.risk_exposure_auc, expected_auc)
        self.assertAlmostEqual(result.mean_risk_share or 0.0, 0.3)
        self.assertAlmostEqual(result.market_participation_rate, 56 / 59)
        self.assertEqual(result.time_to_first_entry, 3)
        self.assertEqual(result.adjustment_frequency, 6)
        self.assertAlmostEqual(result.mean_abs_allocation_change or 0.0, 0.1)
        self.assertEqual(result.hold_rate, 0.0)

    def test_abrupt_adjustments_and_tagged_responses(self) -> None:
        shares = [0.8, 0.8, 0.8, 0.2, 0.2, 0.9, 0.9]
        result = calculate_episode1_features(make_logs(shares), "completed")
        self.assertEqual(result.adjustment_frequency, 2)
        self.assertAlmostEqual(
            result.mean_abs_allocation_change or 0.0,
            (0.6 + 0.7) / 2,
        )
        self.assertAlmostEqual(result.hold_rate or 0.0, 4 / 6)
        # Tagged responses are generic means of allocation changes at those DPs.
        self.assertAlmostEqual(result.mild_gain_response or 0.0, 0.0)
        self.assertAlmostEqual(result.mild_drawdown_response or 0.0, 0.0)
        self.assertAlmostEqual(result.recovery_response or 0.0, -0.6)
        self.assertEqual(result.decision_time_median, 4000.0)


class Episode2FeatureTests(unittest.TestCase):
    def make_e2_logs(
        self,
        shares: list[float],
        trailing_returns: list[float | None] | None = None,
    ) -> list[dict[str, object]]:
        days = [1, 19, 25, 35, 50, 56, 60]
        returns = trailing_returns or [None, 0.01, 0.02, 0.04, 0.06, -0.02, 0.03]
        logs: list[dict[str, object]] = []
        previous = 0.0
        for index, (day, share, trailing) in enumerate(
            zip(days, shares, returns), start=1
        ):
            logs.append(
                {
                    "decision_index": index,
                    "day": day,
                    "risk_share_after": share,
                    "delta_risk_share": share - previous,
                    "decision_time_ms": index * 1000,
                    "trailing_return_5d": trailing,
                    "return_since_previous_dp": trailing,
                }
            )
            previous = share
        return logs

    def test_gain_features_and_time_weighted_exposure(self) -> None:
        shares = [0.2, 0.3, 0.4, 0.6, 0.7, 0.5, 0.65]
        result = calculate_episode2_features(
            self.make_e2_logs(shares),
            "completed",
            e1_risk_exposure_auc=0.25,
        )
        expected_exposure = (
            0.3 * (25 - 19) + 0.4 * (35 - 25) + 0.6 * (50 - 35)
        ) / (50 - 19)
        self.assertAlmostEqual(result.gain_period_risk_escalation or 0.0, 0.4)
        self.assertAlmostEqual(result.uptrend_risk_exposure or 0.0, expected_exposure)
        self.assertAlmostEqual(
            result.e2_vs_e1_risk_shift or 0.0,
            expected_exposure - 0.25,
        )
        self.assertEqual(result.uptrend_risk_increase_count, 3)
        self.assertEqual(result.gain_period_hold_rate, 0.0)
        self.assertAlmostEqual(
            result.gain_adjustment_intensity or 0.0, (0.1 + 0.2 + 0.1) / 3
        )
        self.assertAlmostEqual(result.strong_gain_response or 0.0, 0.1)
        self.assertAlmostEqual(result.pullback_response_after_gain or 0.0, -0.2)
        self.assertAlmostEqual(result.renewed_rise_response or 0.0, 0.15)
        self.assertEqual(result.decision_time_median, 4000.0)
        self.assertEqual(result.strong_gain_decision_time, 5000.0)
        self.assertEqual(result.correction_decision_time, 6000.0)

    def test_sensitivity_is_ols_slope_and_zero_variance_is_null(self) -> None:
        # DP3-DP5 x=[.01,.02,.03], y=[.05,.10,.15] gives slope 5.
        result = calculate_episode2_features(
            self.make_e2_logs(
                [0.2, 0.2, 0.25, 0.35, 0.5, 0.5, 0.5],
                [None, 0.0, 0.01, 0.02, 0.03, 0.0, 0.0],
            ),
            "completed",
        )
        self.assertAlmostEqual(result.recent_return_sensitivity or 0.0, 5.0)

        zero_variance = calculate_episode2_features(
            self.make_e2_logs(
                [0.2, 0.2, 0.25, 0.35, 0.5, 0.5, 0.5],
                [None, 0.0, 0.02, 0.02, 0.02, 0.0, 0.0],
            ),
            "completed",
        )
        self.assertIsNone(zero_variance.recent_return_sensitivity)

        only_through_dp4 = calculate_episode2_features(
            self.make_e2_logs(
                [0.2, 0.2, 0.25, 0.35, 0.5, 0.5, 0.5],
                [None, 0.0, 0.01, 0.02, 0.03, 0.0, 0.0],
            )[:4],
            "in_progress",
        )
        self.assertIsNone(only_through_dp4.recent_return_sensitivity)
        self.assertIsNone(only_through_dp4.e2_vs_e1_risk_shift)

    def test_gain_adjustment_intensity_excludes_holds(self) -> None:
        result = calculate_episode2_features(
            self.make_e2_logs([0.2, 0.3, 0.3, 0.5, 0.5, 0.4, 0.4]),
            "completed",
        )
        self.assertAlmostEqual(result.gain_period_hold_rate or 0.0, 2 / 3)
        self.assertAlmostEqual(result.gain_adjustment_intensity or 0.0, 0.2)


class Episode3FeatureTests(unittest.TestCase):
    def test_loss_and_recovery_features(self) -> None:
        days = [10, 15, 20, 30, 40, 50, 60]
        shares = [0.60, 0.55, 0.50, 0.45, 0.35, 0.40, 0.50]
        drawdowns = [-0.01, -0.03, -0.05, -0.08, -0.12, -0.08, -0.04]
        prices = [101.0, 101.0, 97.0, 94.0, 90.0, 94.0, 98.0]
        scenario_prices = [100.0] + [101.0] * 10 + [99.0, 98.0, 100.0, 101.0]
        scenario_prices.extend([97.0] * (60 - len(scenario_prices)))
        logs: list[dict[str, object]] = []
        previous_share = 0.70
        for index, (day, share, drawdown, price) in enumerate(
            zip(days, shares, drawdowns, prices), start=1
        ):
            logs.append(
                {
                    "decision_index": index,
                    "day": day,
                    "risk_share_after": share,
                    "delta_risk_share": share - previous_share,
                    "drawdown_from_peak": drawdown,
                    "normalized_price": price,
                }
            )
            previous_share = share

        result = calculate_episode3_features(
            logs,
            "completed",
            scenario_prices=scenario_prices,
            allocation_floor=0.10,
            scenario_max_drawdown=-0.15,
        )
        expected_exposure = (
            0.60 * 5 + 0.55 * 5 + 0.50 * 10 + 0.45 * 10
        ) / 30
        self.assertAlmostEqual(result.loss_period_risk_change or 0.0, -0.25)
        self.assertAlmostEqual(result.drawdown_sensitivity or 0.0, 0.25 / 0.11)
        self.assertAlmostEqual(
            result.first_meaningful_reduction_drawdown or 0.0, 0.05
        )
        self.assertAlmostEqual(result.loss_period_risk_exposure or 0.0, expected_exposure)
        self.assertAlmostEqual(result.max_loss_period_reduction or 0.0, -0.25)
        self.assertAlmostEqual(result.recovery_reentry or 0.0, 0.15)
        self.assertEqual(result.drawdown_period_risk_increase_count, 0)
        self.assertEqual(result.drawdown_reduction_consistency, 1.0)
        self.assertAlmostEqual(result.reference_point_crossing_response or 0.0, -0.05)
        self.assertAlmostEqual(result.trough_response or 0.0, -0.10)
        self.assertAlmostEqual(result.early_recovery_response or 0.0, 0.05)
        self.assertAlmostEqual(result.late_recovery_response or 0.0, 0.10)
        self.assertAlmostEqual(result.post_loss_risk_persistence or 0.0, -0.10)
        self.assertAlmostEqual(result.recovery_reentry_ratio or 0.0, 0.60)
        expected_retention = expected_exposure / 0.60
        expected_reduction = 1.0 - (0.60 - 0.35) / (0.60 - 0.10)
        expected_threshold = 0.05 / 0.15
        expected_behavior = (
            0.40 * expected_retention
            + 0.30 * expected_reduction
            + 0.20 * expected_threshold
            + 0.10 * 0.60
        )
        self.assertAlmostEqual(result.retention_score or 0.0, expected_retention)
        self.assertAlmostEqual(result.reduction_score or 0.0, expected_reduction)
        self.assertAlmostEqual(result.threshold_score or 0.0, expected_threshold)
        self.assertAlmostEqual(result.recovery_score or 0.0, 0.60)
        self.assertAlmostEqual(result.severity_factor or 0.0, 0.50)
        self.assertAlmostEqual(
            result.behavior_resilience_score or 0.0, expected_behavior
        )
        self.assertAlmostEqual(
            result.e3_loss_resilience_score or 0.0, expected_behavior * 0.50
        )

    def test_reentry_ratio_is_null_without_pre_trough_reduction(self) -> None:
        logs = [
            {
                "decision_index": index,
                "day": index * 5,
                "risk_share_after": 0.40,
                "delta_risk_share": 0.0,
                "drawdown_from_peak": -0.01 * index,
                "normalized_price": 100.0 - index,
            }
            for index in range(1, 8)
        ]
        result = calculate_episode3_features(logs, "completed")
        self.assertIsNone(result.recovery_reentry_ratio)
        self.assertEqual(result.recovery_score, 1.0)

    def test_no_meaningful_reduction_is_observed_resilience_only_after_dp5(self) -> None:
        logs = [
            {
                "decision_index": index,
                "day": index * 5,
                "risk_share_after": 0.40,
                "delta_risk_share": 0.0,
                "drawdown_from_peak": -0.01 * index,
                "normalized_price": 100.0 - index,
            }
            for index in range(1, 8)
        ]
        before_dp5 = calculate_episode3_features(
            logs[:4],
            "in_progress",
            allocation_floor=0.0,
            scenario_max_drawdown=-0.15,
        )
        self.assertIsNone(before_dp5.threshold_score)

        through_dp5 = calculate_episode3_features(
            logs[:5],
            "in_progress",
            allocation_floor=0.0,
            scenario_max_drawdown=-0.15,
        )
        self.assertEqual(through_dp5.threshold_score, 1.0)
        self.assertIsNone(through_dp5.recovery_score)

        completed = calculate_episode3_features(
            logs,
            "completed",
            allocation_floor=0.0,
            scenario_max_drawdown=-0.15,
        )
        self.assertIsNone(completed.first_meaningful_reduction_drawdown)
        self.assertEqual(completed.threshold_score, 1.0)
        self.assertIsNone(completed.recovery_reentry_ratio)
        self.assertEqual(completed.recovery_score, 1.0)
        self.assertAlmostEqual(completed.behavior_resilience_score or 0.0, 1.0)
        self.assertAlmostEqual(completed.e3_loss_resilience_score or 0.0, 0.5)

        at_floor = calculate_episode3_features(
            [dict(log, risk_share_after=0.10) for log in logs[:5]],
            "in_progress",
            allocation_floor=0.10,
            scenario_max_drawdown=-0.15,
        )
        self.assertIsNone(at_floor.threshold_score)


if __name__ == "__main__":
    unittest.main()

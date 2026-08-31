"""Behavioral feature calculations for Episodes 1 through 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Mapping, Sequence


E1_FEATURE_VERSION = "e1_v2"
E2_FEATURE_VERSION = "e2_v3"
E3_FEATURE_VERSION = "e3_v3"
E4_FEATURE_VERSION = "e4_v1"
E5_FEATURE_VERSION = "e5_v2"
E6_FEATURE_VERSION = "e6_v1"
EPISODE_DURATION = 59
EPSILON = 1e-12


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class Episode1Features:
    feature_version: str
    decision_count: int
    initial_risk_share: float | None
    risk_exposure_auc: float
    mean_risk_share: float | None
    market_participation_rate: float
    time_to_first_entry: int | None
    never_entered: bool
    adjustment_frequency: int
    mean_abs_allocation_change: float | None
    hold_rate: float | None
    decision_time_median: float | None
    mild_gain_response: float | None
    mild_drawdown_response: float | None
    recovery_response: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode2Features:
    feature_version: str
    decision_count: int
    recent_return_sensitivity: float | None
    gain_period_risk_escalation: float | None
    uptrend_risk_exposure: float | None
    e2_vs_e1_risk_shift: float | None
    strong_gain_response: float | None
    pullback_response_after_gain: float | None
    renewed_rise_response: float | None
    uptrend_risk_increase_count: int
    gain_period_hold_rate: float | None
    gain_adjustment_intensity: float | None
    decision_time_median: float | None
    strong_gain_decision_time: float | None
    correction_decision_time: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode3Features:
    feature_version: str
    decision_count: int
    loss_period_risk_change: float | None
    drawdown_sensitivity: float | None
    first_meaningful_reduction_drawdown: float | None
    loss_period_risk_exposure: float | None
    max_loss_period_reduction: float | None
    recovery_reentry: float | None
    drawdown_period_risk_increase_count: int
    drawdown_reduction_consistency: float | None
    reference_point_crossing_response: float | None
    trough_response: float | None
    early_recovery_response: float | None
    late_recovery_response: float | None
    post_loss_risk_persistence: float | None
    recovery_reentry_ratio: float | None
    retention_score: float | None
    reduction_score: float | None
    threshold_score: float | None
    recovery_score: float | None
    severity_factor: float | None
    behavior_resilience_score: float | None
    e3_loss_resilience_score: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode4Features:
    feature_version: str
    decision_count: int
    volatility_sensitivity: float | None
    high_vol_risk_exposure: float | None
    low_vol_risk_exposure: float | None
    high_vs_low_vol_risk_shift: float | None
    volatility_increase_response_mean: float | None
    volatility_decrease_response_mean: float | None
    peak_volatility_response: float | None
    volatility_derisking_consistency: float | None
    volatility_risk_increase_count: int
    volatility_adjustment_intensity: float | None
    high_vol_hold_rate: float | None
    volatility_compression_reentry: float | None
    final_vs_entry_risk_change: float | None
    decision_time_volatility_median: float | None
    peak_volatility_decision_time: float | None
    volatility_shift_decision_time: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode5Features:
    feature_version: str
    decision_count: int
    external_information_sensitivity: float | None
    information_adjustment_rate: float | None
    information_hold_rate: float | None
    news_alignment_score: int
    expert_alignment_score: int
    community_alignment_score: int
    news_alignment_magnitude: float
    expert_alignment_magnitude: float
    community_alignment_magnitude: float
    dp1_information_delta: float | None
    dp2_information_delta: float | None
    dp3_information_delta: float | None
    positive_alignment_count: int
    negative_alignment_count: int
    information_counter_adjustment_count: int
    conflict_hold_count: int
    pre_information_decision_time_median: float | None
    post_information_decision_time_median: float | None
    information_decision_time_change: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Episode6Features:
    feature_version: str
    decision_count: int
    anchor_risk_exposure_auc: float | None
    anchor_mean_risk_share: float | None
    anchor_drawdown_risk_change: float | None
    anchor_drawdown_sensitivity: float | None
    anchor_loss_risk_exposure: float | None
    anchor_recovery_reentry: float | None
    anchor_recovery_reentry_ratio: float | None
    e6_retention_score: float | None
    e6_reduction_score: float | None
    e6_threshold_score: float | None
    e6_recovery_score: float | None
    e6_behavior_resilience_score: float | None
    risk_engagement_consistency: float | None
    loss_response_consistency: float | None
    cross_context_consistency: float | None
    anchor_adjustment_frequency: int
    anchor_hold_rate: float | None
    anchor_adjustment_intensity: float | None
    anchor_peak_response: float | None
    anchor_trough_response: float | None
    anchor_early_recovery_response: float | None
    anchor_late_recovery_response: float | None
    anchor_final_vs_entry_change: float | None
    anchor_decision_time_median: float | None
    anchor_max_drawdown_decision_time: float | None
    anchor_recovery_decision_time: float | None
    episode_status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _BehaviorResilienceComponents:
    recovery_reentry_ratio: float | None
    retention_score: float | None
    reduction_score: float | None
    threshold_score: float | None
    recovery_score: float | None
    behavior_resilience_score: float | None


def _behavior_resilience_components(
    *,
    baseline_risk_share: float | None,
    loss_period_risk_exposure: float | None,
    min_loss_period_risk_share: float | None,
    allocation_floor: float,
    loss_phase_complete: bool,
    first_meaningful_reduction_drawdown: float | None,
    scenario_max_drawdown: float | None,
    trough_risk_share: float | None,
    recovery_risk_share: float | None,
    recovery_complete: bool,
) -> _BehaviorResilienceComponents:
    """Shared E3/E6 resilience components; no scenario severity weighting."""
    recovery_reentry_ratio: float | None = None
    if (
        baseline_risk_share is not None
        and trough_risk_share is not None
        and recovery_risk_share is not None
        and baseline_risk_share > trough_risk_share + EPSILON
    ):
        recovery_reentry_ratio = float(
            (recovery_risk_share - trough_risk_share)
            / (baseline_risk_share - trough_risk_share)
        )

    retention_score = (
        None
        if (
            baseline_risk_share is None
            or baseline_risk_share <= EPSILON
            or loss_period_risk_exposure is None
        )
        else _clip(loss_period_risk_exposure / baseline_risk_share)
    )
    available_reduction = (
        None
        if baseline_risk_share is None
        else baseline_risk_share - allocation_floor
    )
    reduction_score = (
        None
        if (
            baseline_risk_share is None
            or min_loss_period_risk_share is None
            or available_reduction is None
            or available_reduction <= EPSILON
        )
        else 1.0
        - _clip(
            (baseline_risk_share - min_loss_period_risk_share)
            / available_reduction
        )
    )
    valid_threshold_exposure = (
        baseline_risk_share is not None
        and baseline_risk_share > allocation_floor + EPSILON
    )
    if (
        not loss_phase_complete
        or scenario_max_drawdown is None
        or abs(scenario_max_drawdown) <= EPSILON
        or not valid_threshold_exposure
    ):
        threshold_score = None
    elif first_meaningful_reduction_drawdown is None:
        threshold_score = 1.0
    else:
        threshold_score = _clip(
            abs(first_meaningful_reduction_drawdown)
            / abs(scenario_max_drawdown)
        )

    if (
        not recovery_complete
        or baseline_risk_share is None
        or trough_risk_share is None
        or recovery_risk_share is None
    ):
        recovery_score = None
    elif baseline_risk_share > trough_risk_share + EPSILON:
        recovery_score = (
            None
            if recovery_reentry_ratio is None
            else _clip(recovery_reentry_ratio)
        )
    else:
        recovery_score = 1.0

    component_scores = (
        retention_score,
        reduction_score,
        threshold_score,
        recovery_score,
    )
    behavior_resilience_score = (
        None
        if any(score is None for score in component_scores)
        else (
            0.40 * float(retention_score)
            + 0.30 * float(reduction_score)
            + 0.20 * float(threshold_score)
            + 0.10 * float(recovery_score)
        )
    )
    return _BehaviorResilienceComponents(
        recovery_reentry_ratio=recovery_reentry_ratio,
        retention_score=retention_score,
        reduction_score=reduction_score,
        threshold_score=threshold_score,
        recovery_score=recovery_score,
        behavior_resilience_score=behavior_resilience_score,
    )


def _tagged_mean(
    logs: Sequence[Mapping[str, object]], response_tag: str
) -> float | None:
    changes = [
        float(log["delta_risk_share"])
        for log in logs
        if log["response_tag"] == response_tag
    ]
    return None if not changes else float(mean(changes))


def calculate_episode1_features(
    logs: Sequence[Mapping[str, object]], episode_status: str
) -> Episode1Features:
    """Recompute Episode 1 features from append-only logs."""
    ordered = sorted(logs, key=lambda log: int(log["decision_index"]))
    shares = [float(log["risk_share_after"]) for log in ordered]
    days = [int(log["day"]) for log in ordered]

    exposure_days = sum(
        shares[index] * (days[index + 1] - days[index])
        for index in range(max(0, len(ordered) - 1))
    )
    participation_days = sum(
        days[index + 1] - days[index]
        for index in range(max(0, len(ordered) - 1))
        if shares[index] > 0
    )

    changes = [shares[index] - shares[index - 1] for index in range(1, len(shares))]
    actual_changes = [abs(change) for change in changes if abs(change) > EPSILON]
    mean_abs_change = (
        None
        if not changes
        else (0.0 if not actual_changes else float(mean(actual_changes)))
    )
    hold_rate = (
        None
        if not changes
        else sum(abs(change) <= EPSILON for change in changes) / len(changes)
    )
    first_entry_day = next(
        (day for day, share in zip(days, shares) if share > 0), None
    )

    return Episode1Features(
        feature_version=E1_FEATURE_VERSION,
        decision_count=len(ordered),
        initial_risk_share=None if not shares else shares[0],
        risk_exposure_auc=float(exposure_days / EPISODE_DURATION),
        mean_risk_share=None if not shares else float(mean(shares)),
        market_participation_rate=float(participation_days / EPISODE_DURATION),
        time_to_first_entry=(None if first_entry_day is None else first_entry_day - 1),
        never_entered=episode_status == "completed" and first_entry_day is None,
        adjustment_frequency=len(actual_changes),
        mean_abs_allocation_change=mean_abs_change,
        hold_rate=hold_rate,
        decision_time_median=(
            None
            if not ordered
            else float(median(float(log["decision_time_ms"]) for log in ordered))
        ),
        mild_gain_response=_tagged_mean(ordered, "mild_gain"),
        mild_drawdown_response=_tagged_mean(ordered, "mild_drawdown"),
        recovery_response=_tagged_mean(ordered, "recovery"),
        episode_status=episode_status,
    )


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return the OLS slope with an intercept, or None for zero x variance."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= EPSILON:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(numerator / denominator)


def calculate_episode2_features(
    logs: Sequence[Mapping[str, object]],
    episode_status: str,
    e1_risk_exposure_auc: float | None = None,
) -> Episode2Features:
    """Recompute Episode 2 features from the common append-only event log.

    DP3-DP5 recent-return sensitivity is the OLS slope from regressing each
    allocation change on the return since the preceding decision point. It is
    not calculated until all three DP3-DP5 observations exist. Uptrend exposure
    is time-weighted over the closed intervals DP2->DP3->DP4->DP5.
    """
    ordered = sorted(logs, key=lambda log: int(log["decision_index"]))
    by_index = {int(log["decision_index"]): log for log in ordered}
    gain_logs = [by_index[index] for index in (3, 4, 5) if index in by_index]
    valid_sensitivity_logs = [
        log
        for log in gain_logs
        if log["return_since_previous_dp"] is not None
    ]
    if len(valid_sensitivity_logs) != 3:
        sensitivity = None
    else:
        sensitivity = _ols_slope(
            [
                float(log["return_since_previous_dp"])
                for log in valid_sensitivity_logs
            ],
            [float(log["delta_risk_share"]) for log in valid_sensitivity_logs],
        )

    dp2 = by_index.get(2)
    dp5 = by_index.get(5)
    gain_period_risk_escalation = (
        None
        if dp2 is None or dp5 is None
        else float(dp5["risk_share_after"]) - float(dp2["risk_share_after"])
    )

    uptrend_risk_exposure: float | None = None
    if all(index in by_index for index in (2, 3, 4, 5)):
        interval_starts = [by_index[index] for index in (2, 3, 4)]
        interval_ends = [by_index[index] for index in (3, 4, 5)]
        total_duration = int(by_index[5]["day"]) - int(by_index[2]["day"])
        if total_duration > 0:
            weighted = sum(
                float(start["risk_share_after"])
                * (int(end["day"]) - int(start["day"]))
                for start, end in zip(interval_starts, interval_ends)
            )
            uptrend_risk_exposure = float(weighted / total_duration)

    gain_changes = [float(log["delta_risk_share"]) for log in gain_logs]
    actual_gain_changes = [
        abs(change) for change in gain_changes if abs(change) > EPSILON
    ]

    def delta_at(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["delta_risk_share"])

    def time_at(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["decision_time_ms"])

    return Episode2Features(
        feature_version=E2_FEATURE_VERSION,
        decision_count=len(ordered),
        recent_return_sensitivity=sensitivity,
        gain_period_risk_escalation=gain_period_risk_escalation,
        uptrend_risk_exposure=uptrend_risk_exposure,
        e2_vs_e1_risk_shift=(
            None
            if (
                episode_status != "completed"
                or uptrend_risk_exposure is None
                or e1_risk_exposure_auc is None
            )
            else uptrend_risk_exposure - e1_risk_exposure_auc
        ),
        strong_gain_response=delta_at(5),
        pullback_response_after_gain=delta_at(6),
        renewed_rise_response=delta_at(7),
        uptrend_risk_increase_count=sum(
            change > EPSILON for change in gain_changes
        ),
        gain_period_hold_rate=(
            None
            if not gain_changes
            else sum(abs(change) <= EPSILON for change in gain_changes) / len(gain_changes)
        ),
        gain_adjustment_intensity=(
            None
            if not gain_changes
            else (0.0 if not actual_gain_changes else float(mean(actual_gain_changes)))
        ),
        decision_time_median=(
            None
            if not ordered
            else float(median(float(log["decision_time_ms"]) for log in ordered))
        ),
        strong_gain_decision_time=time_at(5),
        correction_decision_time=time_at(6),
        episode_status=episode_status,
    )


def calculate_episode3_features(
    logs: Sequence[Mapping[str, object]],
    episode_status: str,
    scenario_prices: Sequence[float] | None = None,
    allocation_floor: float = 0.0,
    scenario_max_drawdown: float | None = None,
) -> Episode3Features:
    """Recompute Episode 3 loss and recovery features from raw events."""
    ordered = sorted(logs, key=lambda log: int(log["decision_index"]))
    by_index = {int(log["decision_index"]): log for log in ordered}

    def share(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["risk_share_after"])

    def delta(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["delta_risk_share"])

    def severity(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else -float(log["drawdown_from_peak"])

    risk_dp1 = share(1)
    risk_dp5 = share(5)
    risk_dp7 = share(7)
    severity_dp1 = severity(1)
    severity_dp5 = severity(5)
    loss_period_risk_change = (
        None if risk_dp1 is None or risk_dp5 is None else risk_dp5 - risk_dp1
    )
    drawdown_sensitivity: float | None = None
    if (
        loss_period_risk_change is not None
        and severity_dp1 is not None
        and severity_dp5 is not None
        and severity_dp5 - severity_dp1 > EPSILON
    ):
        drawdown_sensitivity = float(
            -loss_period_risk_change / (severity_dp5 - severity_dp1)
        )

    first_meaningful_reduction_drawdown: float | None = None
    if risk_dp1 is not None:
        for index in (2, 3, 4, 5):
            candidate_share = share(index)
            candidate_severity = severity(index)
            if (
                candidate_share is not None
                and candidate_severity is not None
                and candidate_severity > EPSILON
                and risk_dp1 - candidate_share >= 0.10 - EPSILON
            ):
                first_meaningful_reduction_drawdown = candidate_severity
                break

    loss_period_risk_exposure: float | None = None
    if all(index in by_index for index in (1, 2, 3, 4, 5)):
        total_duration = int(by_index[5]["day"]) - int(by_index[1]["day"])
        if total_duration > 0:
            weighted = sum(
                float(by_index[index]["risk_share_after"])
                * (int(by_index[index + 1]["day"]) - int(by_index[index]["day"]))
                for index in (1, 2, 3, 4)
            )
            loss_period_risk_exposure = float(weighted / total_duration)

    loss_logs = [by_index[index] for index in (2, 3, 4, 5) if index in by_index]
    loss_changes = [float(log["delta_risk_share"]) for log in loss_logs]
    max_loss_period_reduction = (
        None
        if risk_dp1 is None or not loss_logs
        else min(float(log["risk_share_after"]) for log in loss_logs) - risk_dp1
    )
    recovery_reentry = (
        None if risk_dp5 is None or risk_dp7 is None else risk_dp7 - risk_dp5
    )

    crossing_response: float | None = None
    previous_day = 1
    if scenario_prices is not None:
        for log in ordered:
            current_day = int(log["day"])
            crossing_exists = any(
                scenario_prices[day - 2] >= 100.0
                and scenario_prices[day - 1] < 100.0
                for day in range(max(2, previous_day + 1), current_day + 1)
            )
            if crossing_exists:
                crossing_response = float(log["delta_risk_share"])
                break
            previous_day = current_day
    else:
        previous_price = 100.0
        for log in ordered:
            current_price = float(log["normalized_price"])
            if previous_price >= 100.0 and current_price < 100.0:
                crossing_response = float(log["delta_risk_share"])
                break
            previous_price = current_price

    min_loss_period_share = (
        None
        if not loss_logs
        else min(float(log["risk_share_after"]) for log in loss_logs)
    )
    loss_phase_complete = all(index in by_index for index in (1, 2, 3, 4, 5))
    resilience = _behavior_resilience_components(
        baseline_risk_share=risk_dp1,
        loss_period_risk_exposure=loss_period_risk_exposure,
        min_loss_period_risk_share=min_loss_period_share,
        allocation_floor=allocation_floor,
        loss_phase_complete=loss_phase_complete,
        first_meaningful_reduction_drawdown=first_meaningful_reduction_drawdown,
        scenario_max_drawdown=scenario_max_drawdown,
        trough_risk_share=risk_dp5,
        recovery_risk_share=risk_dp7,
        recovery_complete=7 in by_index,
    )
    severity_factor = (
        None
        if scenario_max_drawdown is None
        else _clip(abs(scenario_max_drawdown) / 0.30)
    )
    e3_loss_resilience_score = (
        None
        if resilience.behavior_resilience_score is None or severity_factor is None
        else resilience.behavior_resilience_score * severity_factor
    )

    return Episode3Features(
        feature_version=E3_FEATURE_VERSION,
        decision_count=len(ordered),
        loss_period_risk_change=loss_period_risk_change,
        drawdown_sensitivity=drawdown_sensitivity,
        first_meaningful_reduction_drawdown=first_meaningful_reduction_drawdown,
        loss_period_risk_exposure=loss_period_risk_exposure,
        max_loss_period_reduction=max_loss_period_reduction,
        recovery_reentry=recovery_reentry,
        drawdown_period_risk_increase_count=sum(
            change > EPSILON for change in loss_changes
        ),
        drawdown_reduction_consistency=(
            None
            if 5 not in by_index
            else sum(change < -EPSILON for change in loss_changes) / 4.0
        ),
        reference_point_crossing_response=crossing_response,
        trough_response=delta(5),
        early_recovery_response=delta(6),
        late_recovery_response=delta(7),
        post_loss_risk_persistence=(
            None
            if risk_dp1 is None or risk_dp7 is None
            else risk_dp7 - risk_dp1
        ),
        recovery_reentry_ratio=resilience.recovery_reentry_ratio,
        retention_score=resilience.retention_score,
        reduction_score=resilience.reduction_score,
        threshold_score=resilience.threshold_score,
        recovery_score=resilience.recovery_score,
        severity_factor=severity_factor,
        behavior_resilience_score=resilience.behavior_resilience_score,
        e3_loss_resilience_score=e3_loss_resilience_score,
        episode_status=episode_status,
    )


def calculate_episode4_features(
    logs: Sequence[Mapping[str, object]],
    episode_status: str,
    *,
    scenario_rolling_volatility_20d: Sequence[float | None],
    volatility_q25: float,
    volatility_q75: float,
    entry_risk_share: float,
) -> Episode4Features:
    """Recompute E4 responses using server-side scenario volatility data.

    High/low exposure treats the selected allocation as piecewise constant and
    weights each one-day interval from Day 21 through Day 59 whose starting-day
    volatility is in the scenario-specific top/bottom quartile.
    """
    ordered = sorted(logs, key=lambda log: int(log["decision_index"]))
    by_index = {int(log["decision_index"]): log for log in ordered}
    vol_logs = [
        log
        for log in ordered
        if int(log["decision_index"]) >= 2
        and log["delta_volatility_20d"] is not None
    ]
    sensitivity = _ols_slope(
        [float(log["delta_volatility_20d"]) for log in vol_logs],
        [float(log["delta_risk_share"]) for log in vol_logs],
    )

    def response_mean(direction: str) -> float | None:
        values = [
            float(log["delta_risk_share"])
            for log in vol_logs
            if log["volatility_direction"] == direction
        ]
        return None if not values else float(mean(values))

    high_exposure: float | None = None
    low_exposure: float | None = None
    if ordered:
        last_observed_day = int(ordered[-1]["day"])
        weighted_high = weighted_low = 0.0
        high_duration = low_duration = 0
        active_index = 0
        for day in range(1, last_observed_day):
            while (
                active_index + 1 < len(ordered)
                and int(ordered[active_index + 1]["day"]) <= day
            ):
                active_index += 1
            if int(ordered[active_index]["day"]) > day:
                continue
            daily_volatility = scenario_rolling_volatility_20d[day - 1]
            if daily_volatility is None:
                continue
            active_share = float(ordered[active_index]["risk_share_after"])
            if daily_volatility >= volatility_q75:
                weighted_high += active_share
                high_duration += 1
            if daily_volatility <= volatility_q25:
                weighted_low += active_share
                low_duration += 1
        if high_duration:
            high_exposure = weighted_high / high_duration
        if low_duration:
            low_exposure = weighted_low / low_duration

    peak_log = max(
        (
            log
            for log in ordered
            if int(log["decision_index"]) >= 2
            and log["rolling_volatility_20d"] is not None
        ),
        key=lambda log: float(log["rolling_volatility_20d"]),
        default=None,
    )
    changing_vol_logs = [
        log
        for log in vol_logs
        if log["volatility_direction"] in {"rising", "falling"}
    ]
    actual_adjustments = [
        abs(float(log["delta_risk_share"]))
        for log in changing_vol_logs
        if abs(float(log["delta_risk_share"])) > EPSILON
    ]
    rising_logs = [
        log for log in vol_logs if log["volatility_direction"] == "rising"
    ]
    high_dp_logs = [
        log
        for log in ordered
        if int(log["decision_index"]) >= 2
        and log["rolling_volatility_20d"] is not None
        and float(log["rolling_volatility_20d"]) >= volatility_q75
    ]
    falling_logs = [
        log for log in vol_logs if float(log["delta_volatility_20d"]) < 0.0
    ]
    compression_log = min(
        falling_logs,
        key=lambda log: float(log["delta_volatility_20d"]),
        default=None,
    )
    shift_log = max(
        vol_logs,
        key=lambda log: abs(float(log["delta_volatility_20d"])),
        default=None,
    )
    dp7 = by_index.get(7)

    return Episode4Features(
        feature_version=E4_FEATURE_VERSION,
        decision_count=len(ordered),
        volatility_sensitivity=sensitivity,
        high_vol_risk_exposure=high_exposure,
        low_vol_risk_exposure=low_exposure,
        high_vs_low_vol_risk_shift=(
            None
            if high_exposure is None or low_exposure is None
            else high_exposure - low_exposure
        ),
        volatility_increase_response_mean=response_mean("rising"),
        volatility_decrease_response_mean=response_mean("falling"),
        peak_volatility_response=(
            None if peak_log is None else float(peak_log["delta_risk_share"])
        ),
        volatility_derisking_consistency=(
            None
            if not rising_logs
            else sum(
                float(log["delta_risk_share"]) < -EPSILON
                for log in rising_logs
            )
            / len(rising_logs)
        ),
        volatility_risk_increase_count=sum(
            float(log["delta_risk_share"]) > EPSILON for log in rising_logs
        ),
        volatility_adjustment_intensity=(
            None
            if not changing_vol_logs
            else (
                0.0
                if not actual_adjustments
                else float(mean(actual_adjustments))
            )
        ),
        high_vol_hold_rate=(
            None
            if not high_dp_logs
            else sum(
                abs(float(log["delta_risk_share"])) <= EPSILON
                for log in high_dp_logs
            )
            / len(high_dp_logs)
        ),
        volatility_compression_reentry=(
            None
            if compression_log is None
            else float(compression_log["delta_risk_share"])
        ),
        final_vs_entry_risk_change=(
            None
            if dp7 is None
            else float(dp7["risk_share_after"]) - entry_risk_share
        ),
        decision_time_volatility_median=(
            None
            if not [log for log in ordered if int(log["decision_index"]) >= 2]
            else float(
                median(
                    float(log["decision_time_ms"])
                    for log in ordered
                    if int(log["decision_index"]) >= 2
                )
            )
        ),
        peak_volatility_decision_time=(
            None if peak_log is None else float(peak_log["decision_time_ms"])
        ),
        volatility_shift_decision_time=(
            None if shift_log is None else float(shift_log["decision_time_ms"])
        ),
        episode_status=episode_status,
    )


def calculate_episode5_features(
    logs: Sequence[Mapping[str, object]], episode_status: str
) -> Episode5Features:
    """Recompute E5 information responses from append-only PRE/POST events."""
    pre_logs = sorted(
        (log for log in logs if log["event_phase"] == "pre_information"),
        key=lambda log: int(log["decision_index"]),
    )
    post_logs = sorted(
        (log for log in logs if log["event_phase"] == "post_information"),
        key=lambda log: int(log["decision_index"]),
    )
    post_by_index = {int(log["decision_index"]): log for log in post_logs}
    deltas = [float(log["information_delta"]) for log in post_logs]
    complete_observation = (
        episode_status == "completed"
        and set(post_by_index) == {1, 2, 3}
    )
    alignment_scores = {"news": 0, "expert": 0, "community": 0}
    alignment_magnitudes = {"news": 0.0, "expert": 0.0, "community": 0.0}
    for log in post_logs:
        aligned_source = log["aligned_source"]
        if aligned_source in alignment_scores:
            alignment_scores[str(aligned_source)] += 1
            alignment_magnitudes[str(aligned_source)] += abs(
                float(log["information_delta"])
            )

    pre_median = (
        None
        if not pre_logs
        else float(median(float(log["decision_time_ms"]) for log in pre_logs))
    )
    post_median = (
        None
        if not post_logs
        else float(median(float(log["decision_time_ms"]) for log in post_logs))
    )

    def delta_at(index: int) -> float | None:
        log = post_by_index.get(index)
        return None if log is None else float(log["information_delta"])

    return Episode5Features(
        feature_version=E5_FEATURE_VERSION,
        decision_count=len(post_logs),
        external_information_sensitivity=(
            None
            if not complete_observation
            else float(mean(abs(delta) for delta in deltas))
        ),
        information_adjustment_rate=(
            None
            if not complete_observation
            else sum(abs(delta) > EPSILON for delta in deltas) / 3.0
        ),
        information_hold_rate=(
            None
            if not complete_observation
            else sum(abs(delta) <= EPSILON for delta in deltas) / 3.0
        ),
        news_alignment_score=alignment_scores["news"],
        expert_alignment_score=alignment_scores["expert"],
        community_alignment_score=alignment_scores["community"],
        news_alignment_magnitude=alignment_magnitudes["news"],
        expert_alignment_magnitude=alignment_magnitudes["expert"],
        community_alignment_magnitude=alignment_magnitudes["community"],
        dp1_information_delta=delta_at(1),
        dp2_information_delta=delta_at(2),
        dp3_information_delta=delta_at(3),
        positive_alignment_count=sum(delta > EPSILON for delta in deltas),
        negative_alignment_count=sum(delta < -EPSILON for delta in deltas),
        information_counter_adjustment_count=sum(
            abs(float(log["pre_information_delta"])) > EPSILON
            and abs(float(log["information_delta"])) > EPSILON
            and (
                float(log["pre_information_delta"])
                * float(log["information_delta"])
                < 0.0
            )
            for log in post_logs
        ),
        conflict_hold_count=sum(abs(delta) <= EPSILON for delta in deltas),
        pre_information_decision_time_median=pre_median,
        post_information_decision_time_median=post_median,
        information_decision_time_change=(
            None
            if pre_median is None or post_median is None
            else post_median - pre_median
        ),
        episode_status=episode_status,
    )


def calculate_episode6_features(
    logs: Sequence[Mapping[str, object]],
    episode_status: str,
    *,
    scenario_max_drawdown: float | None,
    pre_e6_risk_engagement_score: float | None,
    pre_e6_e3_behavior_resilience_score: float | None,
) -> Episode6Features:
    """Calculate anchor/calibration features from the shared allocation log."""
    ordered = sorted(logs, key=lambda log: int(log["decision_index"]))
    by_index = {int(log["decision_index"]): log for log in ordered}
    complete_observation = (
        episode_status == "completed" and set(by_index) == set(range(1, 8))
    )

    def share(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["risk_share_after"])

    def delta(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else float(log["delta_risk_share"])

    def severity(index: int) -> float | None:
        log = by_index.get(index)
        return None if log is None else -float(log["drawdown_from_peak"])

    risk_dp1 = share(1)
    risk_dp2 = share(2)
    risk_dp4 = share(4)
    risk_dp6 = share(6)
    risk_dp7 = share(7)

    anchor_risk_exposure_auc: float | None = None
    anchor_mean_risk_share: float | None = None
    if complete_observation:
        anchor_risk_exposure_auc = float(
            sum(
                float(by_index[index]["risk_share_after"])
                * (
                    int(by_index[index + 1]["day"])
                    - int(by_index[index]["day"])
                )
                for index in range(1, 7)
            )
            / EPISODE_DURATION
        )
        anchor_mean_risk_share = float(
            mean(float(by_index[index]["risk_share_after"]) for index in range(1, 8))
        )

    anchor_drawdown_risk_change = (
        None if risk_dp2 is None or risk_dp4 is None else risk_dp4 - risk_dp2
    )
    severity_dp2 = severity(2)
    severity_dp4 = severity(4)
    severity_change = (
        None
        if severity_dp2 is None or severity_dp4 is None
        else severity_dp4 - severity_dp2
    )
    anchor_drawdown_sensitivity = (
        None
        if (
            anchor_drawdown_risk_change is None
            or severity_change is None
            or severity_change <= EPSILON
        )
        else float(-anchor_drawdown_risk_change / severity_change)
    )

    anchor_loss_risk_exposure: float | None = None
    if all(index in by_index for index in (2, 3, 4)):
        loss_duration = int(by_index[4]["day"]) - int(by_index[2]["day"])
        if loss_duration > 0:
            anchor_loss_risk_exposure = float(
                sum(
                    float(by_index[index]["risk_share_after"])
                    * (
                        int(by_index[index + 1]["day"])
                        - int(by_index[index]["day"])
                    )
                    for index in (2, 3)
                )
                / loss_duration
            )

    first_meaningful_reduction_drawdown: float | None = None
    if risk_dp2 is not None:
        for index in (3, 4):
            candidate_share = share(index)
            candidate_severity = severity(index)
            if (
                candidate_share is not None
                and candidate_severity is not None
                and candidate_severity > EPSILON
                and risk_dp2 - candidate_share >= 0.10 - EPSILON
            ):
                first_meaningful_reduction_drawdown = candidate_severity
                break

    loss_logs = [by_index[index] for index in (3, 4) if index in by_index]
    min_loss_period_share = (
        None
        if not loss_logs
        else min(float(log["risk_share_after"]) for log in loss_logs)
    )
    resilience = _behavior_resilience_components(
        baseline_risk_share=risk_dp2,
        loss_period_risk_exposure=anchor_loss_risk_exposure,
        min_loss_period_risk_share=min_loss_period_share,
        allocation_floor=0.0,
        loss_phase_complete=all(index in by_index for index in (2, 3, 4)),
        first_meaningful_reduction_drawdown=first_meaningful_reduction_drawdown,
        scenario_max_drawdown=scenario_max_drawdown,
        trough_risk_share=risk_dp4,
        recovery_risk_share=risk_dp6,
        recovery_complete=6 in by_index,
    )

    risk_engagement_consistency = (
        None
        if (
            not complete_observation
            or anchor_risk_exposure_auc is None
            or pre_e6_risk_engagement_score is None
        )
        else _clip(
            1.0
            - abs(anchor_risk_exposure_auc - pre_e6_risk_engagement_score)
        )
    )
    loss_response_consistency = (
        None
        if (
            not complete_observation
            or resilience.behavior_resilience_score is None
            or pre_e6_e3_behavior_resilience_score is None
        )
        else _clip(
            1.0
            - abs(
                resilience.behavior_resilience_score
                - pre_e6_e3_behavior_resilience_score
            )
        )
    )
    cross_context_consistency = (
        None
        if risk_engagement_consistency is None or loss_response_consistency is None
        else float((risk_engagement_consistency + loss_response_consistency) / 2.0)
    )

    changes = [
        float(ordered[index]["risk_share_after"])
        - float(ordered[index - 1]["risk_share_after"])
        for index in range(1, len(ordered))
    ]
    actual_changes = [abs(change) for change in changes if abs(change) > EPSILON]
    recovery_times = [
        float(by_index[index]["decision_time_ms"])
        for index in (5, 6)
        if index in by_index
    ]

    return Episode6Features(
        feature_version=E6_FEATURE_VERSION,
        decision_count=len(ordered),
        anchor_risk_exposure_auc=anchor_risk_exposure_auc,
        anchor_mean_risk_share=anchor_mean_risk_share,
        anchor_drawdown_risk_change=anchor_drawdown_risk_change,
        anchor_drawdown_sensitivity=anchor_drawdown_sensitivity,
        anchor_loss_risk_exposure=anchor_loss_risk_exposure,
        anchor_recovery_reentry=(
            None if risk_dp4 is None or risk_dp6 is None else risk_dp6 - risk_dp4
        ),
        anchor_recovery_reentry_ratio=resilience.recovery_reentry_ratio,
        e6_retention_score=resilience.retention_score,
        e6_reduction_score=resilience.reduction_score,
        e6_threshold_score=resilience.threshold_score,
        e6_recovery_score=resilience.recovery_score,
        e6_behavior_resilience_score=resilience.behavior_resilience_score,
        risk_engagement_consistency=risk_engagement_consistency,
        loss_response_consistency=loss_response_consistency,
        cross_context_consistency=cross_context_consistency,
        anchor_adjustment_frequency=len(actual_changes),
        anchor_hold_rate=(
            None
            if not changes
            else sum(abs(change) <= EPSILON for change in changes) / len(changes)
        ),
        anchor_adjustment_intensity=(
            None
            if not changes
            else (0.0 if not actual_changes else float(mean(actual_changes)))
        ),
        anchor_peak_response=delta(2),
        anchor_trough_response=delta(4),
        anchor_early_recovery_response=delta(5),
        anchor_late_recovery_response=delta(6),
        anchor_final_vs_entry_change=(
            None if risk_dp1 is None or risk_dp7 is None else risk_dp7 - risk_dp1
        ),
        anchor_decision_time_median=(
            None
            if not ordered
            else float(median(float(log["decision_time_ms"]) for log in ordered))
        ),
        anchor_max_drawdown_decision_time=(
            None
            if 4 not in by_index
            else float(by_index[4]["decision_time_ms"])
        ),
        anchor_recovery_decision_time=(
            None if len(recovery_times) != 2 else float(median(recovery_times))
        ),
        episode_status=episode_status,
    )

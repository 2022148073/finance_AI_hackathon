"""Versioned adaptive routing rules for Episodes 3 and 4."""

from __future__ import annotations

from dataclasses import dataclass


ROUTING_VERSION = "e3_routing_v1"
E4_ROUTING_VERSION = "e4_routing_v1"
E1_WEIGHT = 0.7
E2_WEIGHT = 0.3
CONTEXT_GAP_THRESHOLD = 0.30
LEVEL_ENTRY_RULES = {
    "L1": (0.30, 0.10),
    "L2": (0.20, 0.10),
}


@dataclass(frozen=True)
class Episode3Routing:
    assigned_level: str
    routing_score: float
    routing_version: str
    context_gap: float
    entry_risk_share: float
    allocation_floor: float


@dataclass(frozen=True)
class Episode4Routing:
    assigned_level: str
    routing_score: float
    routing_version: str
    routing_fallback: bool
    context_gap: float | None
    upper_level_capped: bool


def _score_level(score: float) -> str:
    if score < 0.20:
        return "L1"
    if score < 0.40:
        return "L2"
    if score < 0.60:
        return "L3"
    if score < 0.80:
        return "L4"
    return "L5"


def route_episode3(
    *,
    e1_risk_exposure_auc: float,
    e2_uptrend_risk_exposure: float,
    e1_never_entered: bool,
    e2_final_risk_share: float,
) -> Episode3Routing:
    """Apply the documented MVP weighted-score and exception rules."""
    score = min(
        1.0,
        max(
            0.0,
            E1_WEIGHT * e1_risk_exposure_auc
            + E2_WEIGHT * e2_uptrend_risk_exposure,
        ),
    )
    context_gap = abs(e1_risk_exposure_auc - e2_uptrend_risk_exposure)
    if e1_never_entered:
        level = "L1"
    elif context_gap > CONTEXT_GAP_THRESHOLD:
        level = "L3"
    else:
        level = _score_level(score)

    if level in LEVEL_ENTRY_RULES:
        entry_share, floor = LEVEL_ENTRY_RULES[level]
    else:
        entry_share = e2_final_risk_share
        floor = 0.0

    return Episode3Routing(
        assigned_level=level,
        routing_score=score,
        routing_version=ROUTING_VERSION,
        context_gap=context_gap,
        entry_risk_share=entry_share,
        allocation_floor=floor,
    )


def _volatility_level(score: float) -> str:
    if score < 0.20:
        return "V1"
    if score < 0.40:
        return "V2"
    if score < 0.60:
        return "V3"
    if score < 0.80:
        return "V4"
    return "V5"


def route_episode4(
    *,
    e3_routing_score: float,
    e3_loss_resilience_score: float | None,
    e3_assigned_level: str,
    floor_reached: bool,
    full_exit: bool,
) -> Episode4Routing:
    """Apply the fixed MVP E4 routing, conflict override, and final V2 cap."""
    fallback = e3_loss_resilience_score is None
    if fallback:
        score = min(1.0, max(0.0, e3_routing_score))
        context_gap = None
        level = _volatility_level(score)
    else:
        score = min(
            1.0,
            max(
                0.0,
                0.4 * e3_routing_score + 0.6 * e3_loss_resilience_score,
            ),
        )
        context_gap = abs(e3_routing_score - e3_loss_resilience_score)
        level = "V3" if context_gap >= 0.35 else _volatility_level(score)

    should_cap = (
        e3_assigned_level in {"L1", "L2"} and floor_reached
    ) or (
        e3_assigned_level in {"L3", "L4", "L5"} and full_exit
    )
    capped = should_cap and level in {"V3", "V4", "V5"}
    if capped:
        level = "V2"

    return Episode4Routing(
        assigned_level=level,
        routing_score=score,
        routing_version=E4_ROUTING_VERSION,
        routing_fallback=fallback,
        context_gap=context_gap,
        upper_level_capped=capped,
    )

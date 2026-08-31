"""Shared market-state calculations for private scenario builders."""

from __future__ import annotations

import math
from statistics import stdev


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def market_data(prices: list[float]) -> dict[str, object]:
    """Return annualized volatility metrics using only the 60-day path."""
    returns = [
        prices[index] / prices[index - 1] - 1.0
        for index in range(1, len(prices))
    ]
    rolling: list[float | None] = [None] * 20
    for day in range(21, 61):
        window_returns = returns[day - 21 : day - 1]
        rolling.append(stdev(window_returns) * math.sqrt(252.0))
    valid = [value for value in rolling if value is not None]
    return {
        "rolling_volatility_20d": rolling,
        "volatility_60d": stdev(returns) * math.sqrt(252.0),
        "volatility_20d_min": min(valid),
        "volatility_20d_max": max(valid),
        "volatility_20d_q25": percentile(valid, 0.25),
        "volatility_20d_q75": percentile(valid, 0.75),
    }

"""Private server-side scenario loader for Episodes 1 through 6.

These files are server-side only. API responses are constructed separately and
never include market_phase, response_tag, or unrevealed prices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


TOTAL_DAYS = 60
TOTAL_DECISIONS = 7


class ScenarioConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionPoint:
    decision_point: str
    sequence: int
    day: int
    semantic_role: str
    response_tag: str | None
    market_phase: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    episode: str
    asset: str
    level: str | None
    prices: tuple[float, ...]
    decision_points: tuple[DecisionPoint, ...]
    rolling_volatility_20d: tuple[float | None, ...] = ()
    volatility_60d: float | None = None
    volatility_20d_min: float | None = None
    volatility_20d_max: float | None = None
    volatility_20d_q25: float | None = None
    volatility_20d_q75: float | None = None

    def decision_for_sequence(self, sequence: int) -> DecisionPoint:
        if not 1 <= sequence <= len(self.decision_points):
            raise IndexError(f"Decision sequence out of range: {sequence}")
        return self.decision_points[sequence - 1]

    @property
    def max_drawdown(self) -> float:
        peak = self.prices[0]
        result = 0.0
        for price in self.prices:
            peak = max(peak, price)
            result = min(result, price / peak - 1.0)
        return result

    def volatility_for_day(self, day: int) -> float | None:
        if not self.rolling_volatility_20d:
            return None
        return self.rolling_volatility_20d[day - 1]

    def volatility_percentile_for_day(self, day: int) -> float | None:
        current = self.volatility_for_day(day)
        if current is None:
            return None
        valid = [
            value for value in self.rolling_volatility_20d if value is not None
        ]
        return sum(value <= current for value in valid) / len(valid)


def _validate_scenario(scenario: Scenario, source: Path) -> None:
    if scenario.episode not in {"E1", "E2", "E3", "E4", "E5", "E6"}:
        raise ScenarioConfigurationError(f"{source.name}: unsupported episode")
    expected_asset = f"Asset {scenario.episode[1:]}"
    if scenario.asset != expected_asset:
        raise ScenarioConfigurationError(
            f"{source.name}: asset must be {expected_asset}"
        )
    expected_levels = {
        "E3": {"L1", "L2", "L3", "L4", "L5"},
        "E4": {"V1", "V2", "V3", "V4", "V5"},
    }
    if scenario.episode in expected_levels:
        if scenario.level not in expected_levels[scenario.episode]:
            raise ScenarioConfigurationError(
                f"{source.name}: invalid {scenario.episode} level"
            )
        if not scenario.scenario_id.startswith(
            f"{scenario.episode}_{scenario.level}_"
        ):
            raise ScenarioConfigurationError(
                f"{source.name}: scenario_id and level do not match"
            )
    elif scenario.level is not None:
        raise ScenarioConfigurationError(
            f"{source.name}: level is only valid for adaptive episodes"
        )
    if len(scenario.prices) != TOTAL_DAYS:
        raise ScenarioConfigurationError(
            f"{source.name}: expected {TOTAL_DAYS} prices"
        )
    if scenario.prices[0] != 100.0 or any(price <= 0 for price in scenario.prices):
        raise ScenarioConfigurationError(
            f"{source.name}: prices must be positive and Day 1 must equal 100"
        )
    expected_decisions = 3 if scenario.episode == "E5" else TOTAL_DECISIONS
    if len(scenario.decision_points) != expected_decisions:
        raise ScenarioConfigurationError(
            f"{source.name}: expected {expected_decisions} decision points"
        )

    sequences = [point.sequence for point in scenario.decision_points]
    days = [point.day for point in scenario.decision_points]
    identifiers = [point.decision_point for point in scenario.decision_points]
    if sequences != list(range(1, expected_decisions + 1)):
        raise ScenarioConfigurationError(f"{source.name}: invalid DP sequence")
    if identifiers != [
        f"{scenario.episode}_DP{i}" for i in range(1, expected_decisions + 1)
    ]:
        raise ScenarioConfigurationError(f"{source.name}: invalid DP identifiers")
    invalid_start = scenario.episode in {"E1", "E2", "E6"} and days[0] != 1
    invalid_end = (
        days[-1] > TOTAL_DAYS
        if scenario.episode == "E5"
        else days[-1] != TOTAL_DAYS
    )
    if days != sorted(set(days)) or invalid_start or days[0] < 1 or invalid_end:
        raise ScenarioConfigurationError(f"{source.name}: invalid DP days")
    if scenario.episode in {"E4", "E5"}:
        if len(scenario.rolling_volatility_20d) != TOTAL_DAYS:
            raise ScenarioConfigurationError(
                f"{source.name}: expected {TOTAL_DAYS} rolling volatility values"
            )
        if any(
            scenario.rolling_volatility_20d[index] is not None
            for index in range(20)
        ) or any(
            scenario.rolling_volatility_20d[index] is None
            for index in range(20, TOTAL_DAYS)
        ):
            raise ScenarioConfigurationError(
                f"{source.name}: invalid rolling volatility warm-up"
            )
        metrics = (
            scenario.volatility_60d,
            scenario.volatility_20d_min,
            scenario.volatility_20d_max,
            scenario.volatility_20d_q25,
            scenario.volatility_20d_q75,
        )
        if any(value is None or value < 0 for value in metrics):
            raise ScenarioConfigurationError(
                f"{source.name}: invalid volatility metrics"
            )


def load_scenarios(directory: Path) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for path in sorted(directory.glob("E[123456]_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        market_data = raw.get("market_data", {})
        decision_points = tuple(
            DecisionPoint(
                decision_point=str(point["decision_point"]),
                sequence=int(point["sequence"]),
                day=int(point["day"]),
                semantic_role=str(point["semantic_role"]),
                response_tag=(
                    None
                    if point.get("response_tag") is None
                    else str(point["response_tag"])
                ),
                market_phase=str(point["market_phase"]),
            )
            for point in raw["decision_points"]
        )
        scenario = Scenario(
            scenario_id=str(raw["scenario_id"]),
            episode=str(raw["episode"]),
            asset=str(raw["asset"]),
            level=(None if raw.get("level") is None else str(raw["level"])),
            prices=tuple(float(price) for price in raw["prices"]),
            decision_points=decision_points,
            rolling_volatility_20d=tuple(
                None if value is None else float(value)
                for value in market_data.get("rolling_volatility_20d", [])
            ),
            volatility_60d=(
                None
                if market_data.get("volatility_60d") is None
                else float(market_data["volatility_60d"])
            ),
            volatility_20d_min=(
                None
                if market_data.get("volatility_20d_min") is None
                else float(market_data["volatility_20d_min"])
            ),
            volatility_20d_max=(
                None
                if market_data.get("volatility_20d_max") is None
                else float(market_data["volatility_20d_max"])
            ),
            volatility_20d_q25=(
                None
                if market_data.get("volatility_20d_q25") is None
                else float(market_data["volatility_20d_q25"])
            ),
            volatility_20d_q75=(
                None
                if market_data.get("volatility_20d_q75") is None
                else float(market_data["volatility_20d_q75"])
            ),
        )
        _validate_scenario(scenario, path)
        if scenario.scenario_id in scenarios:
            raise ScenarioConfigurationError(
                f"Duplicate scenario_id: {scenario.scenario_id}"
            )
        scenarios[scenario.scenario_id] = scenario

    for episode in ("E1", "E2"):
        count = sum(scenario.episode == episode for scenario in scenarios.values())
        if count != 3:
            raise ScenarioConfigurationError(
                f"Expected exactly 3 {episode} scenarios; found {count}"
            )
    for level in ("L1", "L2", "L3", "L4", "L5"):
        count = sum(
            scenario.episode == "E3" and scenario.level == level
            for scenario in scenarios.values()
        )
        if count != 3:
            raise ScenarioConfigurationError(
                f"Expected exactly 3 E3 {level} scenarios; found {count}"
            )
    for level in ("V1", "V2", "V3", "V4", "V5"):
        count = sum(
            scenario.episode == "E4" and scenario.level == level
            for scenario in scenarios.values()
        )
        if count != 3:
            raise ScenarioConfigurationError(
                f"Expected exactly 3 E4 {level} scenarios; found {count}"
            )
    e5_count = sum(scenario.episode == "E5" for scenario in scenarios.values())
    if e5_count != 3:
        raise ScenarioConfigurationError(
            f"Expected exactly 3 E5 scenarios; found {e5_count}"
        )
    e6_count = sum(scenario.episode == "E6" for scenario in scenarios.values())
    if e6_count != 3:
        raise ScenarioConfigurationError(
            f"Expected exactly 3 E6 scenarios; found {e6_count}"
        )
    return scenarios

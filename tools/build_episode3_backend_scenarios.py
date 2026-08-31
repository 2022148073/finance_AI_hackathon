"""Build private Episode 3 scenarios from privacy-safe selected price JSON."""

from __future__ import annotations

import json
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = WEB_DIR / "frontend" / "scenarios" / "episode3"
OUTPUT_DIR = WEB_DIR / "backend" / "scenarios"

DECISION_DAYS = {
    "E3_L1_01": [11, 15, 21, 31, 37, 50, 60],
    "E3_L1_02": [23, 26, 28, 31, 33, 40, 60],
    "E3_L1_03": [25, 28, 31, 34, 37, 47, 60],
    "E3_L2_01": [20, 24, 27, 32, 37, 50, 60],
    "E3_L2_02": [12, 17, 21, 25, 28, 38, 60],
    "E3_L2_03": [16, 22, 30, 39, 45, 54, 60],
    "E3_L3_01": [24, 31, 33, 36, 40, 48, 60],
    "E3_L3_02": [8, 14, 19, 24, 32, 42, 60],
    "E3_L3_03": [13, 28, 32, 35, 38, 50, 60],
    "E3_L4_01": [9, 20, 25, 31, 39, 49, 60],
    "E3_L4_02": [15, 18, 24, 31, 38, 45, 60],
    "E3_L4_03": [16, 22, 30, 36, 44, 50, 60],
    "E3_L5_01": [9, 15, 22, 28, 34, 40, 60],
    "E3_L5_02": [16, 21, 25, 34, 41, 49, 60],
    "E3_L5_03": [12, 17, 23, 38, 45, 52, 60],
}

ROLES = [
    "pre_loss_anchor",
    "early_drawdown",
    "mid_drawdown",
    "deep_drawdown",
    "max_stress",
    "early_recovery",
    "late_recovery_final",
]


def point_metadata(scenario_id: str, sequence: int) -> tuple[str, str, str]:
    role = ROLES[sequence - 1]
    market_phase = role
    response_tag = role
    if scenario_id == "E3_L4_02" and sequence == 2:
        market_phase = "shock_drawdown"
        response_tag = "shock_drawdown"
    if scenario_id == "E3_L5_03" and sequence == 3:
        market_phase = "interim_rebound"
        response_tag = "interim_rebound"
    return role, response_tag, market_phase


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id, days in DECISION_DAYS.items():
        level = scenario_id.split("_")[1]
        source_path = SOURCE_DIR / level / f"{scenario_id}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        prices = source["series"]["normalized_prices"]
        if len(prices) != 60 or prices[0] != 100.0:
            raise ValueError(f"Invalid price series: {scenario_id}")
        decision_points = []
        for sequence, day in enumerate(days, start=1):
            semantic_role, response_tag, market_phase = point_metadata(
                scenario_id, sequence
            )
            decision_points.append(
                {
                    "decision_point": f"E3_DP{sequence}",
                    "sequence": sequence,
                    "day": day,
                    "semantic_role": semantic_role,
                    "response_tag": response_tag,
                    "market_phase": market_phase,
                }
            )
        output = {
            "scenario_id": scenario_id,
            "episode": "E3",
            "level": level,
            "asset": "Asset 3",
            "prices": prices,
            "decision_points": decision_points,
        }
        (OUTPUT_DIR / f"{scenario_id}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

"""Build private Episode 6 anchor scenarios from normalized-price JSON."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_market_data import market_data


WEB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = WEB_DIR / "frontend" / "scenarios" / "episode6"
OUTPUT_DIR = WEB_DIR / "backend" / "scenarios"

SCENARIOS = {
    "E6_01": ("E6_01.json", [1, 16, 25, 35, 40, 49, 60]),
    "E6_02": ("E6_02.json", [1, 22, 34, 42, 48, 56, 60]),
    "E6_03": ("E6_03.json", [1, 18, 26, 30, 35, 46, 60]),
}

ROLES = (
    "anchor_entry",
    "pre_drawdown_anchor",
    "drawdown_progression",
    "max_drawdown_anchor",
    "early_recovery_anchor",
    "recovered_state_anchor",
    "final_anchor",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id, (source_name, days) in SCENARIOS.items():
        source = json.loads((SOURCE_DIR / source_name).read_text(encoding="utf-8"))
        prices = [float(value) for value in source["series"]["normalized_prices"]]
        if len(prices) != 60 or prices[0] != 100.0:
            raise ValueError(f"Invalid price series: {source_name}")
        points = [
            {
                "decision_point": f"E6_DP{sequence}",
                "sequence": sequence,
                "day": day,
                "semantic_role": role,
                "response_tag": role,
                "market_phase": role,
            }
            for sequence, (day, role) in enumerate(zip(days, ROLES), start=1)
        ]
        output = {
            "scenario_id": scenario_id,
            "episode": "E6",
            "level": None,
            "asset": "Asset 6",
            "prices": prices,
            "market_data": market_data(prices),
            "decision_points": points,
        }
        (OUTPUT_DIR / f"{scenario_id}.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

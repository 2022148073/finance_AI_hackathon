"""Build private Episode 5 scenarios from selected normalized-price JSON."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_market_data import market_data


WEB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = WEB_DIR / "frontend" / "scenarios" / "episode5"
OUTPUT_DIR = WEB_DIR / "backend" / "scenarios"

SCENARIOS = {
    "E5_01": ("E5_01.json", [20, 31, 43]),
    "E5_02": ("E5_02.json", [28, 39, 49]),
    "E5_03": ("E5_03.json", [16, 30, 48]),
}

ROLES = (
    "early_information_conflict",
    "mid_information_conflict",
    "late_information_conflict",
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
                "decision_point": f"E5_DP{sequence}",
                "sequence": sequence,
                "day": day,
                "semantic_role": role,
                "response_tag": None,
                "market_phase": role,
            }
            for sequence, (day, role) in enumerate(zip(days, ROLES), start=1)
        ]
        output = {
            "scenario_id": scenario_id,
            "episode": "E5",
            "level": None,
            "asset": "Asset 5",
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

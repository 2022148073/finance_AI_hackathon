"""Rebuild Episode 2 metadata around backend-only private price series."""

from __future__ import annotations

import json
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WEB_DIR / "backend" / "scenarios"

DECISION_DAYS = {
    "E2_01": [1, 19, 25, 35, 50, 56, 60],
    "E2_02": [1, 12, 20, 33, 44, 48, 60],
    "E2_03": [1, 20, 24, 35, 44, 49, 60],
}
POINT_METADATA = [
    ("initial", None, "initial"),
    ("pre_trend_anchor", None, "pre_trend"),
    ("early_trend_confirmation", "early_trend", "early_uptrend"),
    ("established_uptrend", "established_uptrend", "established_uptrend"),
    ("strong_gain", "strong_gain", "strong_gain"),
    ("correction_after_gain", "pullback_after_gain", "correction_after_gain"),
    ("renewed_high_final", "renewed_rise", "renewed_rise"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id, days in DECISION_DAYS.items():
        output_path = OUTPUT_DIR / f"{scenario_id}.json"
        source = json.loads(output_path.read_text(encoding="utf-8"))
        prices = source["prices"]
        if len(prices) != 60 or prices[0] != 100.0:
            raise ValueError(f"Invalid price series: {scenario_id}")
        points = []
        for sequence, (day, metadata) in enumerate(
            zip(days, POINT_METADATA), start=1
        ):
            semantic_role, response_tag, market_phase = metadata
            points.append(
                {
                    "decision_point": f"E2_DP{sequence}",
                    "sequence": sequence,
                    "day": day,
                    "semantic_role": semantic_role,
                    "response_tag": response_tag,
                    "market_phase": market_phase,
                }
            )
        output = {
            "scenario_id": scenario_id,
            "episode": "E2",
            "asset": "Asset 2",
            "prices": prices,
            "decision_points": points,
        }
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

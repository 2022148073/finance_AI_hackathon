"""Rebuild Episode 4 metadata around backend-only private price series."""

from __future__ import annotations

import json
from pathlib import Path

from scenario_market_data import market_data


WEB_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WEB_DIR / "backend" / "scenarios"

# Runtime scenario IDs are independent of source rank. Legacy source filenames
# are retained only as inert provenance labels; price data is read from backend.
SCENARIOS = {
    "E4_V1_01": ("V1/E4_V1_01.json", [1, 21, 28, 31, 40, 51, 60]),
    "E4_V1_02": ("V1/E4_V1_02.json", [1, 21, 30, 37, 49, 55, 60]),
    "E4_V1_03": ("V1/E4_V1_03.json", [1, 21, 26, 40, 45, 52, 60]),
    "E4_V2_01": ("V2/E4_V2_01.json", [1, 21, 26, 35, 43, 51, 60]),
    "E4_V2_02": ("V2/E4_V2_02.json", [1, 21, 33, 40, 44, 50, 60]),
    "E4_V2_03": ("V2/E4_V2_03.json", [1, 21, 28, 34, 41, 48, 60]),
    "E4_V3_01": ("V3/E4_V3_01.json", [1, 21, 22, 34, 40, 55, 60]),
    "E4_V3_02": ("V3/E4_V3_02.json", [1, 21, 27, 35, 41, 54, 60]),
    "E4_V3_03": ("V3/E4_V3_03.json", [1, 21, 31, 39, 49, 54, 60]),
    "E4_V4_01": ("V4/E4_V4_01.json", [1, 21, 29, 34, 40, 48, 60]),
    "E4_V4_02": ("V4/E4_V4_02.json", [1, 21, 31, 36, 47, 52, 60]),
    "E4_V4_03": ("V4/E4_V4_03.json", [1, 21, 30, 39, 44, 55, 60]),
    "E4_V5_01": ("V5/E4_V5_01_extension.json", [1, 21, 29, 35, 47, 53, 60]),
    "E4_V5_02": ("V5/E4_V5_02_extension.json", [1, 21, 27, 32, 35, 52, 60]),
    "E4_V5_03": ("V5/E4_V5_03.json", [1, 21, 28, 33, 43, 52, 60]),
}

ROLES = [
    "entry_allocation",
    "initial_vol_anchor",
    "first_volatility_shift",
    "established_vol_regime",
    "volatility_extreme",
    "volatility_reversal",
    "final_vol_state",
]

# DP3-DP7 phases are scenario metadata. DP1/DP2 continue to use their shared
# semantic roles, and runtime behavior never branches on these scenario IDs.
MARKET_PHASES = {
    "E4_V1_01": (
        "volatility_expansion", "stable_elevated_volatility",
        "local_volatility_peak", "volatility_compression",
        "final_low_volatility",
    ),
    "E4_V1_02": (
        "volatility_expansion", "local_volatility_peak",
        "volatility_compression", "renewed_volatility",
        "final_local_volatility_peak",
    ),
    "E4_V1_03": (
        "local_volatility_trough", "volatility_expansion",
        "local_volatility_peak", "volatility_compression",
        "continued_compression",
    ),
    "E4_V2_01": (
        "local_volatility_trough", "local_volatility_peak",
        "sustained_elevated_volatility", "strong_volatility_compression",
        "renewed_volatility",
    ),
    "E4_V2_02": (
        "strong_volatility_compression", "stable_low_volatility",
        "volatility_spike", "sustained_elevated_volatility",
        "final_elevated_volatility",
    ),
    "E4_V2_03": (
        "stable_volatility", "volatility_expansion",
        "mild_volatility_compression", "continued_compression",
        "renewed_volatility_peak",
    ),
    "E4_V3_01": (
        "local_volatility_peak", "mild_volatility_compression",
        "continued_compression", "local_volatility_trough",
        "renewed_volatility",
    ),
    "E4_V3_02": (
        "strong_volatility_compression", "stable_volatility",
        "local_volatility_trough", "renewed_volatility",
        "continued_expansion",
    ),
    "E4_V3_03": (
        "strong_volatility_compression", "continued_compression",
        "local_volatility_trough", "renewed_volatility_spike",
        "sustained_elevated_volatility",
    ),
    "E4_V4_01": (
        "strong_volatility_compression", "renewed_volatility",
        "strong_volatility_compression", "mild_volatility_expansion",
        "final_low_volatility",
    ),
    "E4_V4_02": (
        "local_volatility_peak", "strong_volatility_compression",
        "continued_compression", "local_volatility_trough",
        "renewed_volatility",
    ),
    "E4_V4_03": (
        "volatility_expansion", "local_volatility_peak",
        "strong_volatility_compression", "local_volatility_trough",
        "renewed_volatility",
    ),
    "E4_V5_01": (
        "local_volatility_peak", "strong_volatility_compression",
        "local_volatility_trough", "mild_volatility_expansion",
        "final_compressed_volatility",
    ),
    "E4_V5_02": (
        "mild_volatility_expansion", "local_volatility_peak",
        "strong_volatility_compression", "local_volatility_trough",
        "renewed_volatility",
    ),
    "E4_V5_03": (
        "local_volatility_trough", "renewed_volatility",
        "local_volatility_peak", "strong_volatility_compression",
        "final_low_volatility",
    ),
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario_id, (relative_source, days) in SCENARIOS.items():
        output_path = OUTPUT_DIR / f"{scenario_id}.json"
        source = json.loads(output_path.read_text(encoding="utf-8"))
        prices = source["prices"]
        if len(prices) != 60 or prices[0] != 100.0:
            raise ValueError(f"Invalid price series: {relative_source}")
        level = scenario_id.split("_")[1]
        points = [
            {
                "decision_point": f"E4_DP{sequence}",
                "sequence": sequence,
                "day": day,
                "semantic_role": role,
                "response_tag": role,
                "market_phase": (
                    role
                    if sequence <= 2
                    else MARKET_PHASES[scenario_id][sequence - 3]
                ),
            }
            for sequence, (day, role) in enumerate(zip(days, ROLES), start=1)
        ]
        output = {
            "scenario_id": scenario_id,
            "episode": "E4",
            "level": level,
            "asset": "Asset 4",
            "prices": prices,
            "market_data": market_data(prices),
            "decision_points": points,
        }
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

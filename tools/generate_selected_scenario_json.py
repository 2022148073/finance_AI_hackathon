"""Generate privacy-safe web JSON for selected Episodes 2 through 6.

Inputs outside web/ are read-only. All generated files stay under
web/frontend/scenarios/ and contain no ticker, date, or source path.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEB_ROOT.parent
SELECTED_CSV = (
    PROJECT_ROOT / "data" / "scenario_selected" / "selected_scenarios.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = WEB_ROOT / "frontend" / "scenarios"

TARGET_EPISODES = {"E2", "E3", "E4", "E5", "E6"}
EXPECTED_EPISODE_COUNTS = {"E2": 3, "E3": 15, "E4": 15, "E5": 3, "E6": 3}
EXPECTED_LEVEL_COUNTS = {
    **{("E3", f"L{level}"): 3 for level in range(1, 6)},
    **{("E4", f"V{level}"): 3 for level in range(1, 6)},
}
EXPECTED_FILE_COUNT = 39


class ScenarioJsonError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_window(row: dict[str, str]) -> list[dict[str, str]]:
    path = PROCESSED_DIR / f"processed_{row['ticker']}.csv"
    window = [
        item
        for item in read_csv(path)
        if row["start_date"] <= item["date"] <= row["end_date"]
    ]
    if len(window) != 60:
        raise ScenarioJsonError(
            f"{row['episode']} {row['level']} rank {row['rank']}: "
            f"expected 60 rows, found {len(window)}"
        )
    if (
        window[0]["date"] != row["start_date"]
        or window[-1]["date"] != row["end_date"]
    ):
        raise ScenarioJsonError("Selected dates do not match processed data")
    return window


def output_path_for(row: dict[str, str]) -> Path:
    episode = row["episode"]
    rank = int(row["rank"])
    episode_folder = OUTPUT_ROOT / f"episode{episode.removeprefix('E')}"
    if episode in {"E3", "E4"}:
        level = row["level"]
        filename = f"{episode}_{level}_{rank:02d}"
        if (
            episode == "E4"
            and level == "V5"
            and "V5_extension" in row["source_path"]
        ):
            filename += "_extension"
        return episode_folder / level / f"{filename}.json"
    return episode_folder / f"{episode}_{rank:02d}.json"


def build_payload(
    row: dict[str, str], window: list[dict[str, str]], output_path: Path
) -> dict[str, object]:
    closes = [float(item["close"]) for item in window]
    first_close = closes[0]
    payload: dict[str, object] = {
        "scenario_id": output_path.stem,
        "episode": int(row["episode"].removeprefix("E")),
        "rank": int(row["rank"]),
        "asset": f"Asset {row['episode'].removeprefix('E')}",
        "period": {
            "start": "Day 1",
            "end": "Day 60",
            "trading_days": 60,
        },
        "normalization": {"base_day": "Day 1", "base_value": 100},
        "series": {
            "days": list(range(1, 61)),
            "normalized_prices": [
                round(close / first_close * 100, 6) for close in closes
            ],
        },
    }
    if row["level"]:
        payload["level"] = row["level"]
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_payload(
    row: dict[str, str],
    path: Path,
    expected_prices: list[float],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload["series"]
    if payload["asset"] != f"Asset {row['episode'].removeprefix('E')}":
        raise ScenarioJsonError(f"Invalid asset label: {path}")
    if payload["period"] != {
        "start": "Day 1",
        "end": "Day 60",
        "trading_days": 60,
    }:
        raise ScenarioJsonError(f"Invalid period: {path}")
    if series["days"] != list(range(1, 61)):
        raise ScenarioJsonError(f"Invalid day series: {path}")
    if series["normalized_prices"] != expected_prices:
        raise ScenarioJsonError(f"Normalized prices do not match: {path}")
    if series["normalized_prices"][0] != 100.0:
        raise ScenarioJsonError(f"Day 1 is not normalized to 100: {path}")

    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden_values = (
        row["ticker"],
        row["start_date"],
        row["end_date"],
        row["filename"],
        row["source_path"],
    )
    if any(value and value in serialized for value in forbidden_values):
        raise ScenarioJsonError(f"Private source metadata leaked into {path}")
    if any(
        key in payload
        for key in ("ticker", "start_date", "end_date", "source_path", "filename")
    ):
        raise ScenarioJsonError(f"Private source key leaked into {path}")


def main() -> int:
    selected = [
        row for row in read_csv(SELECTED_CSV) if row["episode"] in TARGET_EPISODES
    ]
    selected.sort(
        key=lambda row: (
            int(row["episode"].removeprefix("E")),
            int(row["level"][1:]) if row["level"] else 0,
            int(row["rank"]),
        )
    )

    episode_counts = Counter(row["episode"] for row in selected)
    level_counts = Counter(
        (row["episode"], row["level"])
        for row in selected
        if row["episode"] in {"E3", "E4"}
    )
    if dict(episode_counts) != EXPECTED_EPISODE_COUNTS:
        raise ScenarioJsonError(f"Unexpected episode counts: {episode_counts}")
    if dict(level_counts) != EXPECTED_LEVEL_COUNTS:
        raise ScenarioJsonError(f"Unexpected level counts: {level_counts}")

    generated: list[Path] = []
    for row in selected:
        window = load_window(row)
        output_path = output_path_for(row)
        payload = build_payload(row, window, output_path)
        write_json(output_path, payload)
        expected_prices = payload["series"]["normalized_prices"]
        validate_payload(row, output_path, expected_prices)
        generated.append(output_path)

    if len(generated) != EXPECTED_FILE_COUNT or len(set(generated)) != len(generated):
        raise ScenarioJsonError("Generated file count or uniqueness validation failed")

    print(f"Generated files: {len(generated)}")
    for episode in sorted(EXPECTED_EPISODE_COUNTS):
        print(f"{episode}: {episode_counts[episode]}")
    print("E3 levels: " + ", ".join(f"L{i}=3" for i in range(1, 6)))
    print("E4 levels: " + ", ".join(f"V{i}=3" for i in range(1, 6)))
    print("Private metadata leaks: 0")
    print("Validation passed: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

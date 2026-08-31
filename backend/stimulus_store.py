"""Private Episode 5 stimulus templates and canonical source-pair config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, TypeVar


SOURCE_PAIRS: dict[str, tuple[str, str]] = {
    "NE": ("news", "expert"),
    "NC": ("news", "community"),
    "EC": ("expert", "community"),
}
POLARITY_CYCLES: dict[str, dict[str, tuple[str, str]]] = {
    "A": {
        "NE": ("positive", "negative"),
        "NC": ("negative", "positive"),
        "EC": ("positive", "negative"),
    },
    "B": {
        "NE": ("negative", "positive"),
        "NC": ("positive", "negative"),
        "EC": ("negative", "positive"),
    },
}
SOURCE_LABELS = {
    "news": "뉴스 기사",
    "expert": "전문가 의견",
    "community": "투자자 커뮤니티",
}
SENTIMENTS = ("positive", "negative")
EXPECTED_TEMPLATE_COUNT = 5
T = TypeVar("T")


class Randomizer(Protocol):
    def choice(self, sequence: Sequence[T]) -> T: ...
    def shuffle(self, sequence: list[T]) -> None: ...


class StimulusConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stimulus:
    template_id: str
    source: str
    sentiment: str
    strength: int
    title: str
    content: str


class StimulusStore:
    def __init__(self, stimuli: dict[tuple[str, str], tuple[Stimulus, ...]]):
        self._stimuli = stimuli
        self._by_id = {
            stimulus.template_id: stimulus
            for values in stimuli.values()
            for stimulus in values
        }

    @classmethod
    def load(cls, directory: Path) -> "StimulusStore":
        result: dict[tuple[str, str], tuple[Stimulus, ...]] = {}
        seen_ids: set[str] = set()
        for source in SOURCE_LABELS:
            for sentiment in SENTIMENTS:
                suffix = "pos" if sentiment == "positive" else "neg"
                path = directory / f"{source}_{suffix}.json"
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, list) or len(raw) != EXPECTED_TEMPLATE_COUNT:
                    raise StimulusConfigurationError(
                        f"{path.name}: expected {EXPECTED_TEMPLATE_COUNT} templates"
                    )
                values: list[Stimulus] = []
                for item in raw:
                    stimulus = Stimulus(
                        template_id=str(item["template_id"]),
                        source=str(item["source"]),
                        sentiment=str(item["sentiment"]),
                        strength=int(item["strength"]),
                        title=str(item["title"]),
                        content=str(item["content"]),
                    )
                    if stimulus.source != source or stimulus.sentiment != sentiment:
                        raise StimulusConfigurationError(
                            f"{path.name}: source/sentiment mismatch"
                        )
                    if (
                        stimulus.template_id in seen_ids
                        or not stimulus.title
                        or not stimulus.content
                    ):
                        raise StimulusConfigurationError(
                            f"{path.name}: duplicate or incomplete template"
                        )
                    seen_ids.add(stimulus.template_id)
                    values.append(stimulus)
                result[(source, sentiment)] = tuple(values)
        return cls(result)

    def choose(
        self, source: str, sentiment: str, randomizer: Randomizer
    ) -> Stimulus:
        return randomizer.choice(self._stimuli[(source, sentiment)])

    def get(self, template_id: str) -> Stimulus:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise StimulusConfigurationError(
                f"Unknown stimulus template: {template_id}"
            ) from exc

    def public_card(self, template_id: str, position: str) -> dict[str, str]:
        stimulus = self.get(template_id)
        return {
            "position": position,
            "source_label": SOURCE_LABELS[stimulus.source],
            "title": stimulus.title,
            "content": stimulus.content,
        }

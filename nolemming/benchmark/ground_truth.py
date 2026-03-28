"""Historical ground truth data loading for benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nolemming.analysis.sentiment import SentimentTrajectory


@dataclass
class GroundTruthEvent:
    """A historical event with known outcomes for benchmarking."""

    event_id: str
    name: str
    stimulus_path: Path
    actual_sentiment: SentimentTrajectory
    actual_keywords: list[str] = field(default_factory=list)
    actual_engagement_volumes: list[int] = field(default_factory=list)


def _parse_sentiment(data: dict[str, list[float] | list[int]]) -> SentimentTrajectory:
    """Parse a sentiment trajectory from JSON data."""
    return SentimentTrajectory(
        timestamps=list(data.get("timestamps", [])),
        scores=[float(s) for s in data.get("scores", [])],
        volumes=[int(v) for v in data.get("volumes", [])],
    )


def _load_event_json(path: Path) -> dict[str, object]:
    """Load and parse a single event JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _build_event(
    event_id: str, data: dict[str, object], data_dir: Path,
) -> GroundTruthEvent:
    """Construct a GroundTruthEvent from parsed JSON data."""
    stimulus_rel = str(data.get("stimulus_path", ""))
    sentiment_data = data.get("actual_sentiment", {})
    keywords = data.get("actual_keywords", [])
    volumes = data.get("actual_engagement_volumes", [])

    return GroundTruthEvent(
        event_id=event_id,
        name=str(data.get("name", event_id)),
        stimulus_path=data_dir / stimulus_rel,
        actual_sentiment=_parse_sentiment(sentiment_data),  # type: ignore[arg-type]
        actual_keywords=list(keywords),  # type: ignore[arg-type]
        actual_engagement_volumes=[int(v) for v in volumes],  # type: ignore[union-attr]
    )


class GroundTruthLoader:
    """Loads ground truth events from the benchmarks/data/ directory."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load_event(self, event_id: str) -> GroundTruthEvent:
        """Load a single ground truth event by ID."""
        path = self._data_dir / f"{event_id}.json"
        if not path.exists():
            msg = f"Ground truth event not found: {path}"
            raise FileNotFoundError(msg)
        data = _load_event_json(path)
        return _build_event(event_id, data, self._data_dir)

    def list_events(self) -> list[str]:
        """List all available ground truth event IDs."""
        if not self._data_dir.exists():
            return []
        return sorted(
            p.stem for p in self._data_dir.glob("*.json")
        )

"""Sentiment trajectory extraction from simulation results."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from nolemming.core.types import SimulationResult

POSITIVE_WORDS: set[str] = {
    "bullish", "great", "amazing", "growth", "beat",
    "outperform", "upgrade", "buy", "strong", "excellent",
}

NEGATIVE_WORDS: set[str] = {
    "bearish", "terrible", "crash", "miss", "underperform",
    "downgrade", "sell", "weak", "disappointing", "decline",
}


@dataclass
class SentimentTrajectory:
    """Timestamped sentiment scores from a simulation."""

    timestamps: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    volumes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "timestamps": self.timestamps,
            "scores": self.scores,
            "volumes": self.volumes,
        }


def _score_text(text: str) -> float:
    """Score a single text string using keyword matching."""
    words = text.lower().split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    return (pos - neg) / max(total, 1)


def _load_posts_by_round(db_path: str) -> dict[int, list[str]]:
    """Load post content grouped by round from the simulation DB."""
    posts: dict[int, list[str]] = defaultdict(list)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT round, content FROM posts ORDER BY round"
        )
        for round_num, content in cursor:
            posts[round_num].append(content)
    return dict(posts)


def _build_trajectory(
    posts_by_round: dict[int, list[str]],
    minutes_per_round: int,
) -> SentimentTrajectory:
    """Build a sentiment trajectory from posts grouped by round."""
    timestamps: list[float] = []
    scores: list[float] = []
    volumes: list[int] = []

    for round_num in sorted(posts_by_round.keys()):
        posts = posts_by_round[round_num]
        round_scores = [_score_text(p) for p in posts]
        avg = sum(round_scores) / max(len(round_scores), 1)
        timestamps.append(round_num * minutes_per_round / 60.0)
        scores.append(avg)
        volumes.append(len(posts))

    return SentimentTrajectory(
        timestamps=timestamps, scores=scores, volumes=volumes,
    )


def _load_posts_by_round_and_archetype(
    db_path: str,
) -> dict[int, dict[int, list[str]]]:
    """Load posts grouped by (archetype_id, round)."""
    result: dict[int, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with sqlite3.connect(db_path) as conn:
        try:
            cursor = conn.execute(
                "SELECT p.round, u.archetype_id, p.content "
                "FROM posts p JOIN users u ON p.user_id = u.user_id "
                "ORDER BY p.round"
            )
        except sqlite3.OperationalError:
            return dict(result)
        for round_num, archetype_id, content in cursor:
            result[archetype_id][round_num].append(content)
    return dict(result)


class SentimentAnalyzer:
    """Extracts sentiment trajectories from simulation results."""

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        self._llm_model = llm_model

    def extract_trajectory(
        self, result: SimulationResult,
    ) -> SentimentTrajectory:
        """Extract aggregate sentiment over time from simulation."""
        posts = _load_posts_by_round(str(result.db_path))
        return _build_trajectory(
            posts, result.config.minutes_per_round,
        )

    def extract_per_archetype(
        self, result: SimulationResult,
    ) -> dict[int, SentimentTrajectory]:
        """Break down sentiment by neural archetype."""
        by_arch = _load_posts_by_round_and_archetype(
            str(result.db_path),
        )
        return {
            arch_id: _build_trajectory(
                rounds, result.config.minutes_per_round,
            )
            for arch_id, rounds in by_arch.items()
        }

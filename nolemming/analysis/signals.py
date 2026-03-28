"""Quantitative signal extraction from simulation results."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nolemming.analysis.sentiment import SentimentAnalyzer, SentimentTrajectory
from nolemming.core.types import SimulationResult

TOP_KEYWORDS_COUNT = 20
MOMENTUM_WINDOW = 3
STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "this", "that", "it",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "its", "what", "which", "who",
}


@dataclass
class PredictionSignals:
    """Quantitative trading signals extracted from simulation."""

    sentiment_score: float
    sentiment_momentum: float
    consensus_strength: float
    volatility_estimate: float
    dominant_archetype: str
    archetype_dominance: dict[str, float] = field(default_factory=dict)
    narrative_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment_score": self.sentiment_score,
            "sentiment_momentum": self.sentiment_momentum,
            "consensus_strength": self.consensus_strength,
            "volatility_estimate": self.volatility_estimate,
            "dominant_archetype": self.dominant_archetype,
            "archetype_dominance": self.archetype_dominance,
            "narrative_keywords": self.narrative_keywords,
        }


def _compute_momentum(
    scores: list[float], window: int = MOMENTUM_WINDOW,
) -> float:
    """Rate of change of sentiment over the last N timesteps."""
    if len(scores) < 2:
        return 0.0
    recent = scores[-window:]
    if len(recent) < 2:
        return 0.0
    return (recent[-1] - recent[0]) / len(recent)


def _compute_consensus(
    per_archetype: dict[int, SentimentTrajectory],
) -> float:
    """Consensus strength: 1 - normalized variance of final sentiments."""
    if not per_archetype:
        return 0.0
    finals = [
        t.scores[-1]
        for t in per_archetype.values()
        if t.scores
    ]
    if len(finals) < 2:
        return 1.0
    mean = sum(finals) / len(finals)
    variance = sum((s - mean) ** 2 for s in finals) / len(finals)
    return max(0.0, 1.0 - variance)


def _compute_volatility(trajectory: SentimentTrajectory) -> float:
    """Estimate discussion volatility from sentiment score variance."""
    if len(trajectory.scores) < 2:
        return 0.0
    diffs = [
        abs(trajectory.scores[i] - trajectory.scores[i - 1])
        for i in range(1, len(trajectory.scores))
    ]
    return sum(diffs) / len(diffs)


def _extract_keywords(db_path: str) -> list[str]:
    """Extract top keywords from all post content."""
    word_counts: Counter[str] = Counter()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT content FROM posts")
        for (content,) in cursor:
            words = content.lower().split()
            filtered = [w for w in words if _is_keyword(w)]
            word_counts.update(filtered)
    return [w for w, _ in word_counts.most_common(TOP_KEYWORDS_COUNT)]


def _is_keyword(word: str) -> bool:
    """Check if a word qualifies as a keyword."""
    return len(word) > 2 and word.isalpha() and word not in STOP_WORDS


def _compute_archetype_dominance(
    result: SimulationResult,
    per_archetype: dict[int, SentimentTrajectory],
) -> dict[str, float]:
    """Fraction of total discourse volume per archetype."""
    total_volume = sum(
        sum(t.volumes) for t in per_archetype.values()
    )
    if total_volume == 0:
        return {}
    arch_labels = {a.archetype_id: a.label for a in result.archetypes}
    return {
        arch_labels.get(arch_id, str(arch_id)): (
            sum(t.volumes) / total_volume
        )
        for arch_id, t in per_archetype.items()
    }


def _find_dominant_archetype(
    dominance: dict[str, float],
) -> str:
    """Return the archetype label with highest discourse fraction."""
    if not dominance:
        return "unknown"
    return max(dominance, key=dominance.get)  # type: ignore[arg-type]


class SignalExtractor:
    """Extracts quantitative prediction signals from simulations."""

    def extract(self, result: SimulationResult) -> PredictionSignals:
        """Extract all trading signals from a simulation result."""
        analyzer = SentimentAnalyzer()
        trajectory = analyzer.extract_trajectory(result)
        per_archetype = analyzer.extract_per_archetype(result)

        dominance = _compute_archetype_dominance(
            result, per_archetype,
        )
        final_score = trajectory.scores[-1] if trajectory.scores else 0.0

        return PredictionSignals(
            sentiment_score=final_score,
            sentiment_momentum=_compute_momentum(trajectory.scores),
            consensus_strength=_compute_consensus(per_archetype),
            volatility_estimate=_compute_volatility(trajectory),
            dominant_archetype=_find_dominant_archetype(dominance),
            archetype_dominance=dominance,
            narrative_keywords=_extract_keywords(str(result.db_path)),
        )

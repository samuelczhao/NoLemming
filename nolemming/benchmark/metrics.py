"""Evaluation metrics for benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import kl_div
from scipy.stats import pearsonr

from nolemming.analysis.sentiment import SentimentTrajectory


@dataclass
class BenchmarkMetrics:
    """Evaluation metrics comparing predicted vs actual outcomes."""

    sentiment_correlation: float
    engagement_kl_divergence: float
    narrative_overlap: float

    def summary(self) -> str:
        return (
            f"Sentiment r={self.sentiment_correlation:.3f}, "
            f"Engagement KL={self.engagement_kl_divergence:.3f}, "
            f"Narrative overlap={self.narrative_overlap:.3f}"
        )


def _align_series(
    predicted: list[float], actual: list[float],
) -> tuple[list[float], list[float]]:
    """Truncate both series to the shorter length."""
    min_len = min(len(predicted), len(actual))
    return predicted[:min_len], actual[:min_len]


def compute_sentiment_correlation(
    predicted: SentimentTrajectory,
    actual: SentimentTrajectory,
) -> float:
    """Pearson correlation between predicted and actual sentiment."""
    p_scores, a_scores = _align_series(
        predicted.scores, actual.scores,
    )
    if len(p_scores) < 3:
        return 0.0
    if np.std(p_scores) == 0 or np.std(a_scores) == 0:
        return 0.0
    r, _ = pearsonr(p_scores, a_scores)
    return float(r) if np.isfinite(r) else 0.0


def _normalize_distribution(
    values: list[int],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Convert counts to a probability distribution with smoothing."""
    arr = np.array(values, dtype=np.float64)
    arr = arr + 1e-10  # Laplace smoothing to avoid log(0)
    return arr / arr.sum()


def compute_engagement_divergence(
    predicted_volumes: list[int],
    actual_volumes: list[int],
) -> float:
    """KL divergence between predicted and actual engagement."""
    min_len = min(len(predicted_volumes), len(actual_volumes))
    if min_len == 0:
        return float("inf")

    p = _normalize_distribution(predicted_volumes[:min_len])
    q = _normalize_distribution(actual_volumes[:min_len])
    divergence = float(np.sum(kl_div(p, q)))
    return divergence if np.isfinite(divergence) else float("inf")


def compute_narrative_overlap(
    predicted_keywords: list[str],
    actual_keywords: list[str],
) -> float:
    """Jaccard similarity between predicted and actual keyword sets."""
    pred_set = set(k.lower() for k in predicted_keywords)
    actual_set = set(k.lower() for k in actual_keywords)
    if not pred_set and not actual_set:
        return 1.0
    union = pred_set | actual_set
    if not union:
        return 0.0
    return len(pred_set & actual_set) / len(union)

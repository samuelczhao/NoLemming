"""Feed algorithms and engagement modulation."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nolemming.core.types import AgentProfile, EngagementTemplate

logger = logging.getLogger(__name__)

MIN_ACTIVATION = 0.01
MAX_ACTIVATION = 0.95


class EmbeddingModel(Protocol):
    """Protocol for sentence embedding models."""

    def encode(self, sentences: list[str]) -> NDArray[np.float32]: ...


class EngagementDynamics:
    """Modulates agent activation based on content similarity to stimulus."""

    def __init__(
        self,
        template: EngagementTemplate,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._template = template
        self._encoder = embedding_model

    def compute_content_similarity(
        self,
        content: list[str],
        stimulus_embedding: NDArray[np.float32],
    ) -> float:
        """Average cosine similarity of current content to original stimulus."""
        if not content or self._encoder is None:
            return 0.5
        content_embeddings = self._encoder.encode(content)
        return self._mean_cosine_similarity(content_embeddings, stimulus_embedding)

    def modulate_activation(
        self,
        agent: AgentProfile,
        content_similarity: float,
    ) -> float:
        """Adjusted activation probability for agent given current content."""
        engagement = self._template.get_engagement(
            agent.archetype.archetype_id,
            content_similarity,
        )
        activation = agent.activity_level * engagement
        return float(np.clip(activation, MIN_ACTIVATION, MAX_ACTIVATION))

    def _mean_cosine_similarity(
        self,
        embeddings: NDArray[np.float32],
        reference: NDArray[np.float32],
    ) -> float:
        """Compute mean cosine similarity between embeddings and reference."""
        ref_norm = reference / (np.linalg.norm(reference) + 1e-8)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms
        similarities = normalized @ ref_norm
        return float(np.mean(similarities))

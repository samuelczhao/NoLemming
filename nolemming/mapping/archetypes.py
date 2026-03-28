"""Clustering compressed neural responses into behavioral archetypes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from nolemming.core.types import NeuralArchetype
from nolemming.mapping.brain_atlas import BrainAtlas

REGION_BEHAVIOR: dict[str, str] = {
    "amygdala_proxy": "Responds to threats with urgency, likely to share alarming content",
    "reward_circuit_proxy": "Optimistic, attracted to positive narratives, early adopter of trends",
    "prefrontal_cortex": "Analytical, deliberative, prefers data-driven discussion",
    "temporal_pole": "Socially attuned, influenced by group dynamics",
    "insula_proxy": "Risk-averse, skeptical, quick to disengage from uncertain information",
    "acc": "Contrarian, notices inconsistencies, likely to challenge narratives",
    "visual_cortex": "Visually oriented, responds strongly to imagery and aesthetics",
    "auditory_cortex": "Attuned to tone and vocal cues, sensitive to rhetoric",
    "motor_cortex": "Action-oriented, inclined to participate and engage physically",
    "somatosensory": "Empathetic, bodily aware, responds to visceral content",
    "parietal": "Attention-driven, integrates multiple information streams",
    "language_areas": "Verbal, articulate, engages deeply with written arguments",
}

REGION_LABEL_PREFIX: dict[str, str] = {
    "amygdala_proxy": "fear-dominant",
    "reward_circuit_proxy": "reward-seeking",
    "prefrontal_cortex": "analytical",
    "temporal_pole": "social-attuned",
    "insula_proxy": "risk-averse",
    "acc": "contrarian",
    "visual_cortex": "visual-driven",
    "auditory_cortex": "auditory-tuned",
    "motor_cortex": "action-oriented",
    "somatosensory": "empathetic",
    "parietal": "attention-focused",
    "language_areas": "verbal-analytical",
}


class ArchetypeClusterer:
    """Groups compressed neural responses into distinct behavioral archetypes."""

    def __init__(self, n_archetypes: int = 8) -> None:
        self.n_archetypes = n_archetypes

    def cluster(
        self,
        compressed_timesteps: NDArray[np.float32],
        atlas: BrainAtlas,
        original_activations: NDArray[np.float32],
    ) -> list[NeuralArchetype]:
        """Cluster compressed vectors and map back to brain regions."""
        effective_k = min(self.n_archetypes, compressed_timesteps.shape[0])
        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(compressed_timesteps)
        return [
            self._build_archetype(i, kmeans, labels, atlas, original_activations)
            for i in range(effective_k)
        ]

    def _build_archetype(
        self,
        cluster_idx: int,
        kmeans: KMeans,
        labels: NDArray[np.int32],
        atlas: BrainAtlas,
        original_activations: NDArray[np.float32],
    ) -> NeuralArchetype:
        """Construct a single NeuralArchetype from cluster results."""
        mask = labels == cluster_idx
        pop_fraction = float(np.sum(mask)) / len(labels)
        cluster_mean = np.mean(original_activations[mask], axis=0)
        dominant = atlas.get_dominant_regions(cluster_mean.astype(np.float32))
        return NeuralArchetype(
            archetype_id=cluster_idx,
            centroid=kmeans.cluster_centers_[cluster_idx].astype(np.float32),
            label=_make_label(dominant),
            description=_make_description(dominant),
            population_fraction=pop_fraction,
            dominant_regions=dominant,
        )


def _make_label(dominant_regions: list[str]) -> str:
    """Generate a short label from the top dominant region."""
    if not dominant_regions:
        return "mixed-profile"
    top = dominant_regions[0]
    return REGION_LABEL_PREFIX.get(top, top.replace("_", "-"))


def _make_description(dominant_regions: list[str]) -> str:
    """Generate a behavioral description from dominant brain regions."""
    traits = [
        REGION_BEHAVIOR[r] for r in dominant_regions if r in REGION_BEHAVIOR
    ]
    if not traits:
        return "Mixed neural profile with no single dominant behavioral tendency."
    return ". ".join(traits) + "."

"""Engagement template builder: bridges neural archetypes to simulation behavior."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nolemming.core.types import EngagementTemplate, NeuralArchetype, NeuralResponse
from nolemming.mapping.brain_atlas import BrainAtlas

MIN_ENGAGEMENT = 0.1
MAX_ENGAGEMENT = 0.9
BASE_DECAY = 2.0
INTENSITY_DECAY_SCALE = 3.0


class EngagementTemplateBuilder:
    """Computes engagement probabilities from archetype-stimulus alignment."""

    def build(
        self,
        archetypes: list[NeuralArchetype],
        stimulus_response: NeuralResponse,
        atlas: BrainAtlas,
    ) -> EngagementTemplate:
        """Build an engagement template for the given stimulus and archetypes."""
        stimulus_rois = _extract_roi_vector(stimulus_response, atlas)
        engagements = np.array([
            _compute_engagement(arch, stimulus_rois, atlas)
            for arch in archetypes
        ], dtype=np.float32)
        decay = _compute_decay(stimulus_response)
        return EngagementTemplate(
            archetype_engagement=engagements,
            similarity_decay=decay,
            archetypes=archetypes,
        )


def _extract_roi_vector(
    response: NeuralResponse, atlas: BrainAtlas,
) -> NDArray[np.float32]:
    """Extract ROI activation vector from a neural response."""
    rois = atlas.extract_all_rois(response.mean_activation())
    return np.array(list(rois.values()), dtype=np.float32)


def _archetype_roi_vector(
    archetype: NeuralArchetype, atlas: BrainAtlas,
) -> NDArray[np.float32]:
    """Score each ROI based on whether it's a dominant region."""
    regions = list(atlas.REGIONS.keys())
    scores = np.zeros(len(regions), dtype=np.float32)
    for i, region in enumerate(regions):
        if region in archetype.dominant_regions:
            scores[i] = 1.0
    return scores


def _compute_engagement(
    archetype: NeuralArchetype,
    stimulus_rois: NDArray[np.float32],
    atlas: BrainAtlas,
) -> float:
    """Normalized dot product of archetype profile with stimulus ROIs."""
    arch_vector = _archetype_roi_vector(archetype, atlas)
    dot = float(np.dot(arch_vector, stimulus_rois))
    norm = float(np.linalg.norm(arch_vector) * np.linalg.norm(stimulus_rois))
    raw = dot / norm if norm > 0 else 0.5
    return _scale_engagement(raw)


def _scale_engagement(raw: float) -> float:
    """Scale raw similarity to [MIN_ENGAGEMENT, MAX_ENGAGEMENT]."""
    return float(
        np.clip(
            MIN_ENGAGEMENT + raw * (MAX_ENGAGEMENT - MIN_ENGAGEMENT),
            MIN_ENGAGEMENT,
            MAX_ENGAGEMENT,
        )
    )


def _compute_decay(stimulus_response: NeuralResponse) -> float:
    """Higher stimulus intensity → slower decay (lower value)."""
    intensity = float(np.mean(np.abs(stimulus_response.mean_activation())))
    max_intensity = float(np.max(np.abs(stimulus_response.mean_activation())))
    normalized = intensity / max_intensity if max_intensity > 0 else 0.5
    return float(BASE_DECAY - normalized * INTENSITY_DECAY_SCALE)

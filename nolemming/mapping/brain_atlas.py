"""Brain region ROI definitions on the fsaverage5 mesh."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

REGIONS: dict[str, tuple[int, int]] = {
    "visual_cortex": (0, 2000),
    "auditory_cortex": (2000, 3000),
    "motor_cortex": (3000, 4000),
    "prefrontal_cortex": (4000, 6000),
    "amygdala_proxy": (6000, 6500),
    "insula_proxy": (6500, 7000),
    "temporal_pole": (7000, 7500),
    "reward_circuit_proxy": (7500, 8000),
    "acc": (8000, 8500),
    "somatosensory": (8500, 9500),
    "parietal": (9500, 11000),
    "language_areas": (11000, 12500),
}

REGION_PSYCH_LABELS: dict[str, str] = {
    "visual_cortex": "visual processing",
    "auditory_cortex": "auditory processing",
    "motor_cortex": "motor planning and execution",
    "prefrontal_cortex": "analytical reasoning and deliberation",
    "amygdala_proxy": "fear/threat detection",
    "insula_proxy": "risk aversion and disgust sensitivity",
    "temporal_pole": "social cognition and group dynamics",
    "reward_circuit_proxy": "reward seeking and optimism",
    "acc": "conflict monitoring and error detection",
    "somatosensory": "bodily awareness and empathy",
    "parietal": "spatial attention and integration",
    "language_areas": "language comprehension and production",
}


class BrainAtlas:
    """Extracts region-of-interest activations from fsaverage5 vertex data."""

    REGIONS = REGIONS
    REGION_PSYCH_LABELS = REGION_PSYCH_LABELS

    def extract_roi(
        self, activations: NDArray[np.float32], region: str,
    ) -> NDArray[np.float32]:
        """Extract mean activation for a single brain region."""
        if region not in self.REGIONS:
            msg = f"Unknown region: {region}"
            raise ValueError(msg)
        start, end = self.REGIONS[region]
        return np.mean(activations[start:end]).astype(np.float32)

    def extract_all_rois(
        self, activations: NDArray[np.float32],
    ) -> dict[str, float]:
        """Compute mean activation for every defined brain region."""
        return {
            region: float(self.extract_roi(activations, region))
            for region in self.REGIONS
        }

    def get_dominant_regions(
        self,
        activations: NDArray[np.float32],
        top_k: int = 3,
    ) -> list[str]:
        """Return the top_k regions with highest mean activation."""
        roi_values = self.extract_all_rois(activations)
        sorted_regions = sorted(
            roi_values, key=roi_values.__getitem__, reverse=True,
        )
        return sorted_regions[:top_k]

"""Mock brain encoder for testing and development."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from farswarm.core.types import (
    FSAVERAGE5_VERTICES,
    NeuralResponse,
    Stimulus,
    StimulusType,
)
from farswarm.encoders.base import BrainEncoder

TIMESTEPS_TEXT = 60
TIMESTEPS_AUDIO_DEFAULT = 120
TIMESTEPS_VIDEO = 300

# Simulated cortical region boundaries (vertex index ranges)
VISUAL_CORTEX = slice(0, 4000)
AUDITORY_CORTEX = slice(4000, 7000)
LANGUAGE_NETWORK = slice(7000, 12000)
DEFAULT_MOTOR = slice(12000, 15000)

DEFAULT_SEED = 42


class MockEncoder(BrainEncoder):
    """Generates reproducible synthetic neural responses for testing."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        n_timesteps = _resolve_timesteps(stimulus)
        rng = np.random.default_rng(self._seed)
        base = _generate_base_noise(rng, n_timesteps)
        _add_region_structure(rng, base, stimulus.stimulus_type)
        _add_temporal_dynamics(base, n_timesteps)
        return NeuralResponse(activations=base.astype(np.float32))


def _resolve_timesteps(stimulus: Stimulus) -> int:
    """Determine number of timesteps based on stimulus type."""
    if stimulus.stimulus_type == StimulusType.TEXT:
        return TIMESTEPS_TEXT
    if stimulus.stimulus_type == StimulusType.VIDEO:
        return TIMESTEPS_VIDEO
    return _audio_timesteps(stimulus.path)


def _audio_timesteps(path: Path) -> int:
    """Attempt to read audio duration, fall back to default."""
    try:
        import soundfile as sf  # type: ignore[import-untyped]
        info = sf.info(str(path))
        return max(1, int(info.duration))
    except (ImportError, RuntimeError, FileNotFoundError):
        return TIMESTEPS_AUDIO_DEFAULT


def _generate_base_noise(
    rng: np.random.Generator,
    n_timesteps: int,
) -> NDArray[np.float64]:
    """Low-amplitude gaussian noise as the baseline signal."""
    return rng.normal(loc=0.0, scale=0.1, size=(n_timesteps, FSAVERAGE5_VERTICES))


def _add_region_structure(
    rng: np.random.Generator,
    activations: NDArray[np.float64],
    stimulus_type: StimulusType,
) -> None:
    """Boost specific cortical regions based on stimulus modality."""
    region_gains = _region_gains_for(stimulus_type)
    for region, gain in region_gains:
        n_verts = len(range(*region.indices(FSAVERAGE5_VERTICES)))
        activations[:, region] += rng.normal(
            loc=gain, scale=gain * 0.2, size=(activations.shape[0], n_verts),
        )


def _region_gains_for(
    stimulus_type: StimulusType,
) -> list[tuple[slice, float]]:
    """Map stimulus type to (region, activation gain) pairs."""
    if stimulus_type == StimulusType.TEXT:
        return [(LANGUAGE_NETWORK, 0.8), (AUDITORY_CORTEX, 0.3)]
    if stimulus_type == StimulusType.AUDIO:
        return [(AUDITORY_CORTEX, 0.9), (LANGUAGE_NETWORK, 0.5)]
    return [
        (VISUAL_CORTEX, 1.0),
        (AUDITORY_CORTEX, 0.6),
        (LANGUAGE_NETWORK, 0.4),
    ]


def _add_temporal_dynamics(
    activations: NDArray[np.float64],
    n_timesteps: int,
) -> None:
    """Apply slow temporal envelope so activations vary over time."""
    t = np.linspace(0, 2 * np.pi, n_timesteps)
    envelope = 0.5 + 0.5 * np.sin(t)
    activations *= envelope[:, np.newaxis]

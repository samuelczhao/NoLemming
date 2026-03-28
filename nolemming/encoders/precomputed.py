"""Encoder that loads pre-computed neural responses from .npy files.

Use this with TRIBE v2 responses generated in Google Colab:
    1. Run notebooks/tribe_v2_encoder.ipynb in Colab (free GPU)
    2. Download the .npy files
    3. Use this encoder to load them locally

    encoder = PrecomputedEncoder(responses_dir="./neural_responses")
    response = encoder.encode(stimulus)  # loads {stimulus_stem}_neural.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nolemming.core.types import NeuralResponse, Stimulus
from nolemming.encoders.base import BrainEncoder


class PrecomputedEncoder(BrainEncoder):
    """Loads pre-computed neural responses from .npy files."""

    def __init__(self, responses_dir: str = "./neural_responses") -> None:
        self._dir = Path(responses_dir)

    @property
    def name(self) -> str:
        return "precomputed"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        stem = stimulus.path.stem
        npy_path = self._dir / f"{stem}_neural.npy"
        if not npy_path.exists():
            msg = (
                f"Pre-computed response not found: {npy_path}\n"
                f"Run notebooks/tribe_v2_encoder.ipynb in Google Colab "
                f"to generate .npy files for your stimuli."
            )
            raise FileNotFoundError(msg)
        activations = np.load(npy_path).astype(np.float32)
        return NeuralResponse(activations=activations)

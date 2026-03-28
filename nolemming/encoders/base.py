"""Abstract base class for brain encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nolemming.core.types import NeuralResponse, Stimulus


class BrainEncoder(ABC):
    """Maps a stimulus to predicted cortical activations on fsaverage5."""

    @abstractmethod
    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        """Predict neural response to a stimulus."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this encoder."""

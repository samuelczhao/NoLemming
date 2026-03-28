"""TRIBE v2 brain encoder adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nolemming.core.types import NeuralResponse, Stimulus, StimulusType
from nolemming.encoders.base import BrainEncoder

if TYPE_CHECKING:
    from types import ModuleType

DEFAULT_MODEL_NAME = "facebook/tribev2"
DEFAULT_CACHE_FOLDER = Path.home() / ".cache" / "nolemming" / "tribev2"


class TribeV2Encoder(BrainEncoder):
    """Wraps the tribev2 package to predict cortical activations."""

    def __init__(self, cache_folder: Path | None = None) -> None:
        self._tribev2 = _import_tribev2()
        self._cache_folder = cache_folder or DEFAULT_CACHE_FOLDER
        self._model: object | None = None

    @property
    def name(self) -> str:
        return "tribe_v2"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        model = self._get_model()
        events_df = _get_events(model, stimulus)
        predictions = model.predict(events=events_df)
        return NeuralResponse(activations=np.asarray(predictions, dtype=np.float32))

    def _get_model(self) -> object:
        """Lazy-load and cache the TRIBE v2 model."""
        if self._model is None:
            tribe_model_cls = self._tribev2.TribeModel
            self._model = tribe_model_cls.from_pretrained(
                DEFAULT_MODEL_NAME,
                cache_folder=str(self._cache_folder),
            )
        return self._model


def _import_tribev2() -> ModuleType:
    """Import tribev2, raising a clear error if missing."""
    try:
        import tribev2  # type: ignore[import-untyped]
        return tribev2
    except ImportError as exc:
        msg = (
            "tribev2 is required for TribeV2Encoder but not installed. "
            "Install it with: pip install tribev2"
        )
        raise ImportError(msg) from exc


def _get_events(model: object, stimulus: Stimulus) -> object:
    """Dispatch to the correct events method based on stimulus type."""
    path_str = str(stimulus.path)
    dispatch = {
        StimulusType.TEXT: lambda: model.get_events_dataframe(text_path=path_str),
        StimulusType.AUDIO: lambda: model.get_events_dataframe(audio_path=path_str),
        StimulusType.VIDEO: lambda: model.get_events_dataframe(video_path=path_str),
    }
    return dispatch[stimulus.stimulus_type]()

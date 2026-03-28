"""Tests for core type definitions."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from nolemming.core.types import (
    FSAVERAGE5_VERTICES,
    CompressedResponse,
    EngagementTemplate,
    NeuralArchetype,
    NeuralResponse,
    Stimulus,
    StimulusType,
)


class TestStimulus:
    def test_from_video_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.mp4"
        path.touch()
        s = Stimulus.from_path(path)
        assert s.stimulus_type == StimulusType.VIDEO

    def test_from_audio_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        path.touch()
        s = Stimulus.from_path(path)
        assert s.stimulus_type == StimulusType.AUDIO

    def test_from_text_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.txt"
        path.touch()
        s = Stimulus.from_path(path)
        assert s.stimulus_type == StimulusType.TEXT

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "test.xyz"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            Stimulus.from_path(path)


class TestNeuralResponse:
    def test_valid_construction(self) -> None:
        data = np.random.randn(10, FSAVERAGE5_VERTICES).astype(np.float32)
        resp = NeuralResponse(activations=data)
        assert resp.n_timesteps == 10
        assert resp.vertex_count == FSAVERAGE5_VERTICES

    def test_wrong_shape_raises(self) -> None:
        data = np.random.randn(10, 100).astype(np.float32)
        with pytest.raises(ValueError, match="Expected"):
            NeuralResponse(activations=data)

    def test_wrong_dims_raises(self) -> None:
        data = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2D"):
            NeuralResponse(activations=data)

    def test_mean_activation(self) -> None:
        data = np.ones((5, FSAVERAGE5_VERTICES), dtype=np.float32) * 2.0
        resp = NeuralResponse(activations=data)
        mean = resp.mean_activation()
        assert mean.shape == (FSAVERAGE5_VERTICES,)
        np.testing.assert_allclose(mean, 2.0)


class TestCompressedResponse:
    def test_n_dims(self) -> None:
        latent = np.random.randn(64).astype(np.float32)
        cr = CompressedResponse(latent=latent)
        assert cr.n_dims == 64


class TestEngagementTemplate:
    def _make_archetype(self, idx: int) -> NeuralArchetype:
        return NeuralArchetype(
            archetype_id=idx,
            centroid=np.zeros(64, dtype=np.float32),
            label=f"type_{idx}",
            description=f"Archetype {idx}",
            population_fraction=0.5,
        )

    def test_get_engagement(self) -> None:
        archetypes = [self._make_archetype(0), self._make_archetype(1)]
        template = EngagementTemplate(
            archetype_engagement=np.array([0.8, 0.3], dtype=np.float32),
            similarity_decay=2.0,
            archetypes=archetypes,
        )
        high = template.get_engagement(0, content_similarity=1.0)
        low = template.get_engagement(0, content_similarity=0.0)
        assert high > low
        assert 0.0 <= high <= 1.0
        assert 0.0 <= low <= 1.0

    def test_engagement_clipped(self) -> None:
        archetypes = [self._make_archetype(0)]
        template = EngagementTemplate(
            archetype_engagement=np.array([1.5], dtype=np.float32),
            similarity_decay=0.0,
            archetypes=archetypes,
        )
        result = template.get_engagement(0, content_similarity=1.0)
        assert result <= 1.0

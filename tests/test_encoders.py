"""Tests for MockEncoder and EncoderRegistry."""

from pathlib import Path

import numpy as np
import pytest

from nolemming.core.types import (
    FSAVERAGE5_VERTICES,
    NeuralResponse,
    Stimulus,
    StimulusType,
)
from nolemming.encoders.mock import (
    TIMESTEPS_TEXT,
    TIMESTEPS_VIDEO,
    MockEncoder,
)
from nolemming.encoders.registry import EncoderRegistry, encoder_registry


# --- MockEncoder ---


class TestMockEncoder:
    def _make_text_stimulus(self, tmp_path: Path) -> Stimulus:
        p = tmp_path / "test.txt"
        p.write_text("earnings call transcript")
        return Stimulus(path=p, stimulus_type=StimulusType.TEXT)

    def _make_video_stimulus(self, tmp_path: Path) -> Stimulus:
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00")
        return Stimulus(path=p, stimulus_type=StimulusType.VIDEO)

    def test_text_response_shape(self, tmp_path: Path) -> None:
        stimulus = self._make_text_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        assert response.activations.shape == (TIMESTEPS_TEXT, FSAVERAGE5_VERTICES)

    def test_video_response_shape(self, tmp_path: Path) -> None:
        stimulus = self._make_video_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        assert response.activations.shape == (TIMESTEPS_VIDEO, FSAVERAGE5_VERTICES)

    def test_response_dtype(self, tmp_path: Path) -> None:
        stimulus = self._make_text_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        assert response.activations.dtype == np.float32

    def test_reproducibility_same_seed(self, tmp_path: Path) -> None:
        stimulus = self._make_text_stimulus(tmp_path)
        r1 = MockEncoder(seed=42).encode(stimulus)
        r2 = MockEncoder(seed=42).encode(stimulus)
        np.testing.assert_array_equal(r1.activations, r2.activations)

    def test_different_seed_gives_different_output(self, tmp_path: Path) -> None:
        stimulus = self._make_text_stimulus(tmp_path)
        r1 = MockEncoder(seed=42).encode(stimulus)
        r2 = MockEncoder(seed=99).encode(stimulus)
        assert not np.array_equal(r1.activations, r2.activations)

    def test_text_boosts_language_network(self, tmp_path: Path) -> None:
        """Text stimuli should have higher activation in language areas (7000-12000)."""
        stimulus = self._make_text_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        mean_act = response.mean_activation()
        language_mean = float(np.mean(mean_act[7000:12000]))
        motor_mean = float(np.mean(mean_act[12000:15000]))
        assert language_mean > motor_mean

    def test_video_boosts_visual_cortex(self, tmp_path: Path) -> None:
        """Video stimuli should have higher activation in visual cortex (0-4000)."""
        stimulus = self._make_video_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        mean_act = response.mean_activation()
        visual_mean = float(np.mean(mean_act[0:4000]))
        motor_mean = float(np.mean(mean_act[12000:15000]))
        assert visual_mean > motor_mean

    def test_encoder_name(self) -> None:
        assert MockEncoder().name == "mock"

    def test_temporal_dynamics_vary(self, tmp_path: Path) -> None:
        """Activations should vary across timesteps due to temporal envelope."""
        stimulus = self._make_text_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        first_step_mean = float(np.mean(np.abs(response.activations[0])))
        mid_step_mean = float(np.mean(np.abs(response.activations[30])))
        assert first_step_mean != pytest.approx(mid_step_mean, abs=1e-6)

    def test_n_timesteps_property(self, tmp_path: Path) -> None:
        stimulus = self._make_text_stimulus(tmp_path)
        response = MockEncoder(seed=42).encode(stimulus)
        assert response.n_timesteps == TIMESTEPS_TEXT


# --- EncoderRegistry ---


class TestEncoderRegistry:
    def test_get_mock_encoder(self) -> None:
        encoder = encoder_registry.get("mock")
        assert encoder.name == "mock"

    def test_list_encoders_contains_mock(self) -> None:
        names = encoder_registry.list_encoders()
        assert "mock" in names

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown encoder"):
            encoder_registry.get("nonexistent_encoder")

    def test_register_custom_encoder(self) -> None:
        registry = EncoderRegistry()
        registry.register("mock", MockEncoder)
        encoder = registry.get("mock", seed=123)
        assert encoder.name == "mock"

    def test_list_empty_registry(self) -> None:
        registry = EncoderRegistry()
        assert registry.list_encoders() == []

    def test_register_overwrites(self) -> None:
        registry = EncoderRegistry()
        registry.register("test", MockEncoder)
        registry.register("test", MockEncoder)
        assert registry.list_encoders() == ["test"]

    def test_get_passes_kwargs(self) -> None:
        registry = EncoderRegistry()
        registry.register("mock", MockEncoder)
        encoder = registry.get("mock", seed=99)
        assert isinstance(encoder, MockEncoder)

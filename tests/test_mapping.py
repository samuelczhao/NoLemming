"""Tests for VoxelCompressor, ArchetypeClusterer, EngagementTemplateBuilder, BrainAtlas."""

from pathlib import Path

import numpy as np
import pytest

from farswarm.core.types import (
    FSAVERAGE5_VERTICES,
    CompressedResponse,
    EngagementTemplate,
    NeuralArchetype,
    NeuralResponse,
    Stimulus,
    StimulusType,
)
from farswarm.encoders.mock import MockEncoder
from farswarm.mapping.archetypes import ArchetypeClusterer
from farswarm.mapping.brain_atlas import BrainAtlas, REGIONS
from farswarm.mapping.compressor import VoxelCompressor
from farswarm.mapping.engagement import EngagementTemplateBuilder


@pytest.fixture()
def text_response(tmp_path: Path) -> NeuralResponse:
    p = tmp_path / "test.txt"
    p.write_text("test stimulus content")
    stimulus = Stimulus(path=p, stimulus_type=StimulusType.TEXT)
    return MockEncoder(seed=42).encode(stimulus)


@pytest.fixture()
def atlas() -> BrainAtlas:
    return BrainAtlas()


# --- BrainAtlas ---


class TestBrainAtlas:
    def test_regions_defined(self, atlas: BrainAtlas) -> None:
        assert len(atlas.REGIONS) > 0
        assert "visual_cortex" in atlas.REGIONS
        assert "language_areas" in atlas.REGIONS

    def test_extract_roi_returns_scalar(self, atlas: BrainAtlas) -> None:
        activations = np.random.default_rng(42).standard_normal(FSAVERAGE5_VERTICES).astype(np.float32)
        result = atlas.extract_roi(activations, "visual_cortex")
        assert result.shape == ()

    def test_extract_roi_unknown_region_raises(self, atlas: BrainAtlas) -> None:
        activations = np.zeros(FSAVERAGE5_VERTICES, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown region"):
            atlas.extract_roi(activations, "nonexistent_region")

    def test_extract_all_rois_keys(self, atlas: BrainAtlas) -> None:
        activations = np.random.default_rng(42).standard_normal(FSAVERAGE5_VERTICES).astype(np.float32)
        rois = atlas.extract_all_rois(activations)
        assert set(rois.keys()) == set(REGIONS.keys())
        assert all(isinstance(v, float) for v in rois.values())

    def test_extract_roi_respects_region_boundaries(self, atlas: BrainAtlas) -> None:
        """Setting high values only in visual_cortex should give high ROI for that region."""
        activations = np.zeros(FSAVERAGE5_VERTICES, dtype=np.float32)
        start, end = REGIONS["visual_cortex"]
        activations[start:end] = 5.0
        visual_val = float(atlas.extract_roi(activations, "visual_cortex"))
        motor_val = float(atlas.extract_roi(activations, "motor_cortex"))
        assert visual_val == pytest.approx(5.0)
        assert motor_val == pytest.approx(0.0)

    def test_get_dominant_regions_default_top_3(self, atlas: BrainAtlas) -> None:
        activations = np.zeros(FSAVERAGE5_VERTICES, dtype=np.float32)
        start, end = REGIONS["language_areas"]
        activations[start:end] = 10.0
        dominant = atlas.get_dominant_regions(activations)
        assert len(dominant) == 3
        assert dominant[0] == "language_areas"

    def test_get_dominant_regions_custom_k(self, atlas: BrainAtlas) -> None:
        activations = np.random.default_rng(42).standard_normal(FSAVERAGE5_VERTICES).astype(np.float32)
        dominant = atlas.get_dominant_regions(activations, top_k=5)
        assert len(dominant) == 5

    def test_get_dominant_regions_returns_strings(self, atlas: BrainAtlas) -> None:
        activations = np.random.default_rng(42).standard_normal(FSAVERAGE5_VERTICES).astype(np.float32)
        dominant = atlas.get_dominant_regions(activations)
        assert all(isinstance(r, str) for r in dominant)


# --- VoxelCompressor ---


class TestVoxelCompressor:
    def test_fit_single_and_compress(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=64)
        compressor.fit_single(text_response)
        compressed = compressor.compress(text_response)
        assert isinstance(compressed, CompressedResponse)
        assert compressed.latent.ndim == 1

    def test_compress_before_fit_raises(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=64)
        with pytest.raises(RuntimeError, match="not fitted"):
            compressor.compress(text_response)

    def test_compress_timesteps_shape(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=64)
        compressor.fit_single(text_response)
        result = compressor.compress_timesteps(text_response)
        n_timesteps = text_response.n_timesteps
        # Effective dims = min(64, 60, 20484) = 60 for text
        effective_dims = min(64, n_timesteps, FSAVERAGE5_VERTICES)
        assert result.shape == (n_timesteps, effective_dims)

    def test_compress_timesteps_dtype(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=64)
        compressor.fit_single(text_response)
        result = compressor.compress_timesteps(text_response)
        assert result.dtype == np.float32

    def test_fewer_dims_when_timesteps_less_than_n_dims(self, text_response: NeuralResponse) -> None:
        """Text has 60 timesteps, so requesting 128 dims yields only 60."""
        compressor = VoxelCompressor(n_dims=128)
        compressor.fit_single(text_response)
        compressed = compressor.compress(text_response)
        assert compressed.n_dims == text_response.n_timesteps

    def test_fit_with_multiple_responses(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=8)
        compressor.fit([text_response, text_response])
        compressed = compressor.compress(text_response)
        # With only 2 responses, effective dims = min(8, 2, 20484) = 2
        assert compressed.n_dims == 2

    def test_fit_empty_raises(self) -> None:
        compressor = VoxelCompressor(n_dims=64)
        with pytest.raises(ValueError, match="at least one"):
            compressor.fit([])

    def test_latent_is_finite(self, text_response: NeuralResponse) -> None:
        compressor = VoxelCompressor(n_dims=32)
        compressor.fit_single(text_response)
        compressed = compressor.compress(text_response)
        assert np.all(np.isfinite(compressed.latent))


# --- ArchetypeClusterer ---


class TestArchetypeClusterer:
    def _get_compressed_and_activations(
        self, response: NeuralResponse,
    ) -> tuple[np.ndarray, np.ndarray]:
        compressor = VoxelCompressor(n_dims=32)
        compressor.fit_single(response)
        compressed = compressor.compress_timesteps(response)
        return compressed, response.activations

    def test_cluster_returns_correct_count(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        compressed, activations = self._get_compressed_and_activations(text_response)
        clusterer = ArchetypeClusterer(n_archetypes=4)
        archetypes = clusterer.cluster(compressed, atlas, activations)
        assert len(archetypes) == 4

    def test_cluster_caps_at_n_samples(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        """Requesting more archetypes than timesteps caps at n_timesteps."""
        compressed, activations = self._get_compressed_and_activations(text_response)
        clusterer = ArchetypeClusterer(n_archetypes=200)
        archetypes = clusterer.cluster(compressed, atlas, activations)
        assert len(archetypes) <= text_response.n_timesteps

    def test_archetype_fields(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        compressed, activations = self._get_compressed_and_activations(text_response)
        clusterer = ArchetypeClusterer(n_archetypes=4)
        archetypes = clusterer.cluster(compressed, atlas, activations)
        for arch in archetypes:
            assert isinstance(arch, NeuralArchetype)
            assert isinstance(arch.archetype_id, int)
            assert isinstance(arch.label, str)
            assert len(arch.label) > 0
            assert isinstance(arch.description, str)
            assert 0.0 < arch.population_fraction <= 1.0
            assert isinstance(arch.dominant_regions, list)
            assert len(arch.dominant_regions) > 0

    def test_population_fractions_sum_to_one(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        compressed, activations = self._get_compressed_and_activations(text_response)
        clusterer = ArchetypeClusterer(n_archetypes=8)
        archetypes = clusterer.cluster(compressed, atlas, activations)
        total = sum(a.population_fraction for a in archetypes)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_centroid_shape_matches_compressed_dims(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        compressed, activations = self._get_compressed_and_activations(text_response)
        clusterer = ArchetypeClusterer(n_archetypes=4)
        archetypes = clusterer.cluster(compressed, atlas, activations)
        for arch in archetypes:
            assert arch.centroid.shape == (compressed.shape[1],)


# --- EngagementTemplateBuilder ---


class TestEngagementTemplateBuilder:
    def _build_template(
        self, response: NeuralResponse, atlas: BrainAtlas,
    ) -> EngagementTemplate:
        compressor = VoxelCompressor(n_dims=32)
        compressor.fit_single(response)
        compressed = compressor.compress_timesteps(response)
        clusterer = ArchetypeClusterer(n_archetypes=4)
        archetypes = clusterer.cluster(compressed, atlas, response.activations)
        builder = EngagementTemplateBuilder()
        return builder.build(archetypes, response, atlas)

    def test_build_returns_engagement_template(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        assert isinstance(template, EngagementTemplate)

    def test_engagement_array_length_matches_archetypes(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        assert len(template.archetype_engagement) == len(template.archetypes)

    def test_engagement_values_in_range(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        for val in template.archetype_engagement:
            assert 0.0 <= float(val) <= 1.0

    def test_get_engagement_returns_float(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        result = template.get_engagement(archetype_id=0, content_similarity=0.8)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_get_engagement_higher_similarity_higher_engagement(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        low = template.get_engagement(archetype_id=0, content_similarity=0.1)
        high = template.get_engagement(archetype_id=0, content_similarity=0.9)
        assert high >= low

    def test_get_engagement_at_boundaries(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        zero = template.get_engagement(archetype_id=0, content_similarity=0.0)
        one = template.get_engagement(archetype_id=0, content_similarity=1.0)
        assert 0.0 <= zero <= 1.0
        assert 0.0 <= one <= 1.0

    def test_similarity_decay_is_finite(
        self, text_response: NeuralResponse, atlas: BrainAtlas,
    ) -> None:
        template = self._build_template(text_response, atlas)
        assert np.isfinite(template.similarity_decay)

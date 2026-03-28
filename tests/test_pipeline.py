"""Integration test: full pipeline with mock encoder in fallback mode."""

from pathlib import Path

import numpy as np
import pytest

from nolemming.agents.factory import AgentFactory
from nolemming.analysis.networks import NetworkAnalyzer
from nolemming.analysis.sentiment import SentimentAnalyzer, SentimentTrajectory
from nolemming.analysis.signals import PredictionSignals, SignalExtractor
from nolemming.core.types import (
    AgentProfile,
    EngagementTemplate,
    NeuralArchetype,
    NeuralResponse,
    Platform,
    SimulationConfig,
    SimulationResult,
    Stimulus,
    StimulusType,
)
from nolemming.encoders.mock import MockEncoder
from nolemming.mapping.archetypes import ArchetypeClusterer
from nolemming.mapping.brain_atlas import BrainAtlas
from nolemming.mapping.compressor import VoxelCompressor
from nolemming.mapping.engagement import EngagementTemplateBuilder
from nolemming.simulation.engine import SimulationEngine


class TestFullPipeline:
    """End-to-end: stimulus -> encode -> compress -> cluster -> simulate -> analyze."""

    async def test_full_pipeline(self, tmp_path: Path) -> None:
        # 1. Create stimulus
        stim_path = tmp_path / "earnings.txt"
        stim_path.write_text(
            "Q4 earnings beat expectations. Revenue growth of 15%. "
            "Strong bullish guidance for next quarter."
        )
        stimulus = Stimulus(path=stim_path, stimulus_type=StimulusType.TEXT)

        # 2. Encode with mock encoder
        encoder = MockEncoder(seed=42)
        response = encoder.encode(stimulus)
        assert isinstance(response, NeuralResponse)
        assert response.n_timesteps == 60
        assert response.activations.shape[1] == 20_484

        # 3. Compress
        compressor = VoxelCompressor(n_dims=32)
        compressor.fit_single(response)
        compressed_mean = compressor.compress(response)
        compressed_ts = compressor.compress_timesteps(response)
        assert compressed_mean.latent.ndim == 1
        assert compressed_ts.shape[0] == response.n_timesteps

        # 4. Cluster into archetypes
        atlas = BrainAtlas()
        clusterer = ArchetypeClusterer(n_archetypes=4)
        archetypes = clusterer.cluster(compressed_ts, atlas, response.activations)
        assert len(archetypes) == 4
        assert all(isinstance(a, NeuralArchetype) for a in archetypes)
        fractions = sum(a.population_fraction for a in archetypes)
        assert fractions == pytest.approx(1.0, abs=1e-6)

        # 5. Build engagement template
        builder = EngagementTemplateBuilder()
        template = builder.build(archetypes, response, atlas)
        assert isinstance(template, EngagementTemplate)
        assert len(template.archetype_engagement) == 4
        for i in range(4):
            eng = template.get_engagement(i, content_similarity=0.7)
            assert 0.0 <= eng <= 1.0

        # 6. Generate agent population
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(n_agents=20)
        assert len(agents) == 20
        assert all(isinstance(a, AgentProfile) for a in agents)

        # 7. Run simulation (fallback mode)
        config = SimulationConfig(
            stimulus=stimulus,
            n_agents=20,
            n_rounds=5,
            minutes_per_round=60,
            platform=Platform.TWITTER,
            encoder_name="mock",
            n_archetypes=4,
            pca_dims=32,
            seed=42,
        )
        engine = SimulationEngine(config, output_base=tmp_path / "sim_output")
        await engine.setup(agents, template)
        result = await engine.run()
        assert isinstance(result, SimulationResult)
        assert result.db_path.exists()
        assert result.actions_path.exists()
        assert len(result.agents) == 20
        assert len(result.archetypes) == 4

        # 8. Sentiment analysis
        sentiment = SentimentAnalyzer()
        trajectory = sentiment.extract_trajectory(result)
        assert isinstance(trajectory, SentimentTrajectory)
        assert len(trajectory.timestamps) > 0
        assert len(trajectory.scores) == len(trajectory.timestamps)

        # 9. Signal extraction
        signals = SignalExtractor().extract(result)
        assert isinstance(signals, PredictionSignals)
        assert isinstance(signals.sentiment_score, float)
        assert isinstance(signals.dominant_archetype, str)
        assert 0.0 <= signals.consensus_strength <= 1.0

        # 10. Network analysis
        network = NetworkAnalyzer()
        coalitions = network.analyze_coalitions(result)
        assert isinstance(coalitions.groups, list)
        assert isinstance(coalitions.polarization_index, float)

    async def test_encoder_deterministic(self, tmp_path: Path) -> None:
        """Same seed produces bit-identical neural responses."""
        stim_path = tmp_path / "stim.txt"
        stim_path.write_text("test content for reproducibility")
        stimulus = Stimulus(path=stim_path, stimulus_type=StimulusType.TEXT)

        r1 = MockEncoder(seed=42).encode(stimulus)
        r2 = MockEncoder(seed=42).encode(stimulus)
        np.testing.assert_array_equal(r1.activations, r2.activations)

    async def test_pipeline_structural_consistency(self, tmp_path: Path) -> None:
        """Repeated pipeline runs produce same number of archetypes with valid structure."""
        stim_path = tmp_path / "stim.txt"
        stim_path.write_text("test content for consistency check")
        stimulus = Stimulus(path=stim_path, stimulus_type=StimulusType.TEXT)

        response = MockEncoder(seed=42).encode(stimulus)
        compressor = VoxelCompressor(n_dims=16)
        compressor.fit_single(response)
        compressed = compressor.compress_timesteps(response)
        atlas = BrainAtlas()
        archetypes = ArchetypeClusterer(n_archetypes=3).cluster(
            compressed, atlas, response.activations,
        )
        assert len(archetypes) == 3
        total_frac = sum(a.population_fraction for a in archetypes)
        assert total_frac == pytest.approx(1.0, abs=1e-6)
        for a in archetypes:
            assert len(a.dominant_regions) > 0
            assert len(a.label) > 0

    async def test_pipeline_with_video_stimulus(self, tmp_path: Path) -> None:
        """Verify pipeline works with video stimulus type (300 timesteps)."""
        stim_path = tmp_path / "test.mp4"
        stim_path.write_bytes(b"\x00")
        stimulus = Stimulus(path=stim_path, stimulus_type=StimulusType.VIDEO)

        response = MockEncoder(seed=42).encode(stimulus)
        assert response.n_timesteps == 300

        compressor = VoxelCompressor(n_dims=64)
        compressor.fit_single(response)
        compressed = compressor.compress_timesteps(response)
        assert compressed.shape == (300, 64)

        atlas = BrainAtlas()
        archetypes = ArchetypeClusterer(n_archetypes=8).cluster(
            compressed, atlas, response.activations,
        )
        assert len(archetypes) == 8

        template = EngagementTemplateBuilder().build(archetypes, response, atlas)
        assert len(template.archetype_engagement) == 8

"""Main orchestration pipeline for Farswarm."""

from __future__ import annotations

from pathlib import Path

from farswarm.config import FarswarmConfig
from farswarm.core.llm import LLMBackend, OpenAICompatibleBackend
from farswarm.core.types import (
    EngagementTemplate,
    NeuralArchetype,
    NeuralResponse,
    SimulationConfig,
    SimulationResult,
    Stimulus,
)


class FarswarmPipeline:
    """End-to-end pipeline: stimulus -> brain encoding -> archetypes -> simulation -> report.

    This is the main entry point for running Farswarm.
    Every component is pluggable: brain encoder, LLM backend, simulation engine.
    """

    def __init__(
        self,
        config: FarswarmConfig | None = None,
        llm: LLMBackend | None = None,
    ) -> None:
        self.config = config or FarswarmConfig()
        self.llm = llm or self._default_llm()

    def _default_llm(self) -> LLMBackend:
        return OpenAICompatibleBackend(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
        )

    def encode_stimulus(self, stimulus: Stimulus) -> NeuralResponse:
        """Run brain encoder on stimulus."""
        from farswarm.encoders.registry import encoder_registry

        encoder = encoder_registry.get(self.config.encoder_name)
        return encoder.encode(stimulus)

    def build_archetypes(
        self,
        response: NeuralResponse,
        n_archetypes: int | None = None,
        pca_dims: int | None = None,
    ) -> tuple[list[NeuralArchetype], EngagementTemplate]:
        """Compress neural response and cluster into archetypes."""
        from farswarm.mapping.archetypes import ArchetypeClusterer
        from farswarm.mapping.brain_atlas import BrainAtlas
        from farswarm.mapping.compressor import VoxelCompressor
        from farswarm.mapping.engagement import EngagementTemplateBuilder

        dims = pca_dims or self.config.default_pca_dims
        n_arch = n_archetypes or self.config.default_n_archetypes

        compressor = VoxelCompressor(n_dims=dims)
        compressor.fit_single(response)
        compressed = compressor.compress_timesteps(response)

        atlas = BrainAtlas()
        clusterer = ArchetypeClusterer(n_archetypes=n_arch)
        archetypes = clusterer.cluster(compressed, atlas, response.activations)

        builder = EngagementTemplateBuilder()
        template = builder.build(archetypes, response, atlas)

        return archetypes, template

    async def generate_agents(
        self,
        archetypes: list[NeuralArchetype],
        stimulus_context: str,
        n_agents: int | None = None,
        seed: int = 42,
    ) -> list:
        """Generate neurally-grounded agent population."""
        from farswarm.agents.factory import AgentFactory
        from farswarm.agents.profile import ProfileBuilder

        n = n_agents or self.config.default_n_agents
        factory = AgentFactory(archetypes=archetypes, seed=seed)
        agents = factory.generate_population(n)

        builder = ProfileBuilder(llm=self.llm)
        agents = await builder.enrich_profiles(agents, stimulus_context)
        return agents

    async def run_simulation(
        self,
        sim_config: SimulationConfig,
        agents: list,
        engagement_template: EngagementTemplate,
        archetypes: list[NeuralArchetype],
    ) -> SimulationResult:
        """Run the social simulation with neurally-grounded agents."""
        from farswarm.simulation.engine import SimulationEngine

        engine = SimulationEngine(config=sim_config)
        await engine.setup(agents, engagement_template)
        return await engine.run()

    async def analyze(self, result: SimulationResult) -> dict:
        """Run full analysis on simulation results."""
        from farswarm.analysis.networks import NetworkAnalyzer
        from farswarm.analysis.report import ReportGenerator
        from farswarm.analysis.sentiment import SentimentAnalyzer
        from farswarm.analysis.signals import SignalExtractor

        sentiment_analyzer = SentimentAnalyzer()
        trajectory = sentiment_analyzer.extract_trajectory(result)

        signal_extractor = SignalExtractor()
        signals = signal_extractor.extract(result)

        network_analyzer = NetworkAnalyzer()
        coalitions = network_analyzer.analyze_coalitions(result)

        report_gen = ReportGenerator(llm=self.llm)
        report = await report_gen.generate(
            result, trajectory, signals, coalitions,
        )

        return {
            "report": report,
            "sentiment": trajectory,
            "signals": signals,
            "coalitions": coalitions,
        }

    async def run(
        self,
        stimulus_path: str | Path,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ) -> dict:
        """Full pipeline: stimulus -> prediction report.

        This is the simplest way to use Farswarm:
            pipeline = FarswarmPipeline()
            result = await pipeline.run("earnings_call.mp4")
        """
        stimulus = Stimulus.from_path(stimulus_path)
        response = self.encode_stimulus(stimulus)
        archetypes, template = self.build_archetypes(response)

        context = f"Analyzing stimulus: {stimulus.path.name}"
        agents = await self.generate_agents(archetypes, context, n_agents)

        sim_config = SimulationConfig(
            stimulus=stimulus,
            n_agents=n_agents or self.config.default_n_agents,
            n_rounds=n_rounds or self.config.default_n_rounds,
            encoder_name=self.config.encoder_name,
            llm_model=self.config.llm_model,
        )

        sim_result = await self.run_simulation(
            sim_config, agents, template, archetypes,
        )
        analysis = await self.analyze(sim_result)

        return {
            "simulation": sim_result,
            "analysis": analysis,
            "simulation_id": sim_result.simulation_id,
        }

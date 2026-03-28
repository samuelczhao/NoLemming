"""3-way benchmark comparison framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nolemming.analysis.sentiment import SentimentAnalyzer
from nolemming.analysis.signals import SignalExtractor
from nolemming.benchmark.ground_truth import GroundTruthLoader
from nolemming.benchmark.metrics import (
    BenchmarkMetrics,
    compute_engagement_divergence,
    compute_narrative_overlap,
    compute_sentiment_correlation,
)
from nolemming.core.types import SimulationConfig, SimulationResult


@dataclass
class BenchmarkCondition:
    """A named simulation configuration for benchmarking."""

    name: str
    config: SimulationConfig


@dataclass
class BenchmarkResult:
    """Results of benchmarking multiple conditions against ground truth."""

    event_id: str
    conditions: dict[str, BenchmarkMetrics] = field(default_factory=dict)

    def winner(self) -> str:
        """Return the condition with highest sentiment correlation."""
        if not self.conditions:
            return "none"
        return max(
            self.conditions,
            key=lambda k: self.conditions[k].sentiment_correlation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "conditions": {
                k: {
                    "sentiment_correlation": v.sentiment_correlation,
                    "engagement_kl_divergence": v.engagement_kl_divergence,
                    "narrative_overlap": v.narrative_overlap,
                }
                for k, v in self.conditions.items()
            },
            "winner": self.winner(),
        }


async def _run_simulation(
    config: SimulationConfig,
) -> SimulationResult:
    """Run a full simulation pipeline for a benchmark condition."""
    from nolemming.agents.factory import AgentFactory
    from nolemming.encoders.registry import encoder_registry
    from nolemming.mapping.archetypes import ArchetypeClusterer
    from nolemming.mapping.brain_atlas import BrainAtlas
    from nolemming.mapping.compressor import VoxelCompressor
    from nolemming.mapping.engagement import EngagementTemplateBuilder
    from nolemming.simulation.engine import SimulationEngine

    encoder = encoder_registry.get(config.encoder_name)
    response = encoder.encode(config.stimulus)

    compressor = VoxelCompressor(n_dims=config.pca_dims)
    compressor.fit_single(response)
    compressed = compressor.compress_timesteps(response)

    atlas = BrainAtlas()
    archetypes = ArchetypeClusterer(
        n_archetypes=config.n_archetypes,
    ).cluster(compressed, atlas, response.activations)

    template = EngagementTemplateBuilder().build(
        archetypes, response, atlas,
    )

    agents = AgentFactory(
        archetypes=archetypes, seed=config.seed,
    ).generate_population(config.n_agents)

    engine = SimulationEngine(config=config)
    await engine.setup(agents, template)
    return await engine.run()


def _evaluate_condition(
    result: SimulationResult,
    event_id: str,
    loader: GroundTruthLoader,
) -> BenchmarkMetrics:
    """Evaluate a single simulation result against ground truth."""
    ground_truth = loader.load_event(event_id)
    analyzer = SentimentAnalyzer()
    trajectory = analyzer.extract_trajectory(result)
    extractor = SignalExtractor()
    signals = extractor.extract(result)

    return BenchmarkMetrics(
        sentiment_correlation=compute_sentiment_correlation(
            trajectory, ground_truth.actual_sentiment,
        ),
        engagement_kl_divergence=compute_engagement_divergence(
            trajectory.volumes,
            ground_truth.actual_engagement_volumes,
        ),
        narrative_overlap=compute_narrative_overlap(
            signals.narrative_keywords,
            ground_truth.actual_keywords,
        ),
    )


class BenchmarkRunner:
    """Runs benchmark comparisons across multiple conditions and events."""

    def __init__(self, data_dir: Path) -> None:
        self.loader = GroundTruthLoader(data_dir)

    async def run_event(
        self,
        event_id: str,
        conditions: list[BenchmarkCondition],
    ) -> BenchmarkResult:
        """Run all conditions for a single event."""
        result = BenchmarkResult(event_id=event_id)
        for condition in conditions:
            sim_result = await _run_simulation(condition.config)
            metrics = _evaluate_condition(
                sim_result, event_id, self.loader,
            )
            result.conditions[condition.name] = metrics
        return result

    async def run_all(
        self,
        conditions: list[BenchmarkCondition],
    ) -> list[BenchmarkResult]:
        """Run all events with all conditions."""
        events = self.loader.list_events()
        results: list[BenchmarkResult] = []
        for event_id in events:
            bench_result = await self.run_event(event_id, conditions)
            results.append(bench_result)
        return results

    def summary(self, results: list[BenchmarkResult]) -> str:
        """Generate a text summary of benchmark results."""
        if not results:
            return "No benchmark results."
        lines = [_format_result_line(r) for r in results]
        wins = _count_wins(results)
        win_summary = ", ".join(
            f"{name}: {count}" for name, count in wins.items()
        )
        lines.append(f"\nWins: {win_summary}")
        return "\n".join(lines)


def _format_result_line(result: BenchmarkResult) -> str:
    """Format a single benchmark result as a summary line."""
    condition_parts = [
        f"  {name}: {m.summary()}"
        for name, m in result.conditions.items()
    ]
    header = f"Event: {result.event_id} (winner: {result.winner()})"
    return header + "\n" + "\n".join(condition_parts)


def _count_wins(
    results: list[BenchmarkResult],
) -> dict[str, int]:
    """Count how many events each condition won."""
    wins: dict[str, int] = {}
    for r in results:
        w = r.winner()
        wins[w] = wins.get(w, 0) + 1
    return wins

"""Phase 0 Validation: Does neural grounding improve simulation predictions?

3-way comparison on 5 earnings call events:
  - neural: agents grounded in neural archetypes with engagement templates
  - vanilla: agents with random archetypes (same pipeline, random seed)
  - random: agents with shuffled archetype assignments

Designed for low-compute environments (2018 Mac, Ollama with qwen2.5:3b).
Uses small agent counts and few rounds to keep runtime manageable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farswarm.agents.factory import AgentFactory
from farswarm.analysis.networks import NetworkAnalyzer
from farswarm.analysis.sentiment import SentimentAnalyzer
from farswarm.analysis.signals import SignalExtractor
from farswarm.benchmark.ground_truth import GroundTruthLoader
from farswarm.benchmark.metrics import (
    compute_engagement_divergence,
    compute_narrative_overlap,
    compute_sentiment_correlation,
)
from farswarm.core.types import AgentProfile, SimulationConfig, Stimulus
from farswarm.encoders.mock import MockEncoder
from farswarm.mapping.archetypes import ArchetypeClusterer
from farswarm.mapping.brain_atlas import BrainAtlas
from farswarm.mapping.compressor import VoxelCompressor
from farswarm.mapping.engagement import EngagementTemplateBuilder
from farswarm.simulation.engine import SimulationEngine

N_AGENTS = 15
N_ROUNDS = 3
N_ARCHETYPES = 5
PCA_DIMS = 32
LLM_MODEL = "qwen2.5:3b"
LLM_BASE_URL = "http://localhost:11434/v1"
DATA_DIR = Path("benchmarks/data")

# Set True to use LLM for post generation (slower but meaningful content)
# Set False for fast synthetic posts (for debugging)
USE_LLM = True


async def run_condition(
    name: str,
    stimulus: Stimulus,
    seed: int,
    shuffle_archetypes: bool = False,
) -> dict:
    """Run a single benchmark condition."""
    print(f"    [{name}] Encoding stimulus...")
    encoder = MockEncoder(seed=seed)
    response = encoder.encode(stimulus)

    compressor = VoxelCompressor(n_dims=PCA_DIMS)
    compressor.fit_single(response)
    compressed = compressor.compress_timesteps(response)

    atlas = BrainAtlas()
    archetypes = ArchetypeClusterer(n_archetypes=N_ARCHETYPES).cluster(
        compressed, atlas, response.activations,
    )
    template = EngagementTemplateBuilder().build(archetypes, response, atlas)

    factory = AgentFactory(archetypes=archetypes, seed=seed)
    agents = factory.generate_population(N_AGENTS)

    if shuffle_archetypes:
        agents = _shuffle_archetype_assignments(agents, archetypes, seed)

    config = SimulationConfig(
        stimulus=stimulus,
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        n_archetypes=N_ARCHETYPES,
        pca_dims=PCA_DIMS,
        llm_model=LLM_MODEL,
        seed=seed,
    )

    llm = None
    if USE_LLM:
        from farswarm.core.llm import OpenAICompatibleBackend
        llm = OpenAICompatibleBackend(
            model=LLM_MODEL, base_url=LLM_BASE_URL,
        )

    # Read stimulus text for context
    stimulus_text = stimulus.path.read_text(encoding="utf-8")[:500]

    print(f"    [{name}] Running simulation ({N_AGENTS} agents, {N_ROUNDS} rounds)...")
    engine = SimulationEngine(config=config, llm=llm)
    await engine.setup(agents, template)
    result = await engine.run()

    print(f"    [{name}] Analyzing results...")
    trajectory = SentimentAnalyzer().extract_trajectory(result)
    signals = SignalExtractor().extract(result)

    return {
        "result": result,
        "trajectory": trajectory,
        "signals": signals,
    }


def _shuffle_archetype_assignments(
    agents: list[AgentProfile],
    archetypes: list,
    seed: int,
) -> list[AgentProfile]:
    """Randomly reassign archetypes to agents (null baseline)."""
    import numpy as np
    rng = np.random.default_rng(seed + 999)
    shuffled = []
    for agent in agents:
        random_arch = archetypes[rng.integers(0, len(archetypes))]
        shuffled.append(AgentProfile(
            agent_id=agent.agent_id,
            archetype=random_arch,
            name=agent.name,
            username=agent.username,
            bio=agent.bio,
            persona=agent.persona,
            activity_level=agent.activity_level,
            stance=agent.stance,
        ))
    return shuffled


async def benchmark_event(event_id: str, loader: GroundTruthLoader) -> dict:
    """Run 3-way benchmark for a single event."""
    gt = loader.load_event(event_id)
    stimulus = Stimulus.from_path(gt.stimulus_path)

    print(f"\n  Event: {gt.name}")

    conditions = {}
    for name, seed, shuffle in [
        ("neural", 42, False),
        ("vanilla", 99, False),
        ("random", 42, True),
    ]:
        start = time.time()
        data = await run_condition(name, stimulus, seed, shuffle)
        elapsed = time.time() - start
        print(f"    [{name}] Done in {elapsed:.1f}s")

        traj = data["trajectory"]
        signals = data["signals"]

        corr = compute_sentiment_correlation(traj, gt.actual_sentiment)
        kl = compute_engagement_divergence(traj.volumes, gt.actual_engagement_volumes)
        overlap = compute_narrative_overlap(
            signals.narrative_keywords, gt.actual_keywords,
        )

        conditions[name] = {
            "sentiment_correlation": corr,
            "engagement_kl_divergence": kl,
            "narrative_overlap": overlap,
            "sentiment_score": signals.sentiment_score,
            "dominant_archetype": signals.dominant_archetype,
        }

    winner = max(conditions, key=lambda k: conditions[k]["sentiment_correlation"])
    return {"event_id": event_id, "conditions": conditions, "winner": winner}


async def main() -> None:
    print("=" * 60)
    print("FARSWARM PHASE 0 VALIDATION")
    print("=" * 60)
    print(f"Config: {N_AGENTS} agents, {N_ROUNDS} rounds, {N_ARCHETYPES} archetypes")
    print(f"LLM: {LLM_MODEL} via {LLM_BASE_URL}")
    print(f"Events: 5 earnings calls (AAPL, TSLA, NVDA, META, GOOGL)")

    loader = GroundTruthLoader(data_dir=DATA_DIR)
    events = [e for e in loader.list_events() if e != "sample_event"]
    print(f"Found {len(events)} events: {events}")

    results = []
    total_start = time.time()

    for event_id in events:
        result = await benchmark_event(event_id, loader)
        results.append(result)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    wins = {"neural": 0, "vanilla": 0, "random": 0}
    for r in results:
        print(f"\n{r['event_id']} (winner: {r['winner']})")
        for name, metrics in r["conditions"].items():
            print(
                f"  {name:8s}: corr={metrics['sentiment_correlation']:+.3f} "
                f"kl={metrics['engagement_kl_divergence']:.3f} "
                f"overlap={metrics['narrative_overlap']:.3f} "
                f"dominant={metrics['dominant_archetype']}"
            )
        wins[r["winner"]] += 1

    print("\n" + "-" * 60)
    print("SUMMARY")
    print(f"  Neural wins: {wins['neural']}/5")
    print(f"  Vanilla wins: {wins['vanilla']}/5")
    print(f"  Random wins: {wins['random']}/5")
    print(f"  Total time: {total_elapsed:.0f}s")

    if wins["neural"] >= 3:
        print("\n  PHASE 0 PASSED: Neural grounding shows improvement")
    elif wins["neural"] >= wins["vanilla"]:
        print("\n  PHASE 0 MARGINAL: Neural slightly better, needs investigation")
    else:
        print("\n  PHASE 0 FAILED: Neural grounding does not improve predictions")

    output_path = Path("benchmarks/results/phase0_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

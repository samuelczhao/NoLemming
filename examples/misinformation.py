"""Example: Model misinformation spread through neurally-diverse population.

Simulate how a piece of misinformation would spread, identifying which
neural archetypes are most susceptible and where intervention is most effective.

Usage:
    python examples/misinformation.py misinfo_article.txt
"""

from __future__ import annotations

import argparse
import asyncio


async def main(
    stimulus_path: str,
    encoder: str = "mock",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    n_agents: int = 1000,
) -> None:
    from nolemming.config import NoLemmingConfig
    from nolemming.core.llm import OpenAICompatibleBackend
    from nolemming.core.pipeline import NoLemmingPipeline

    config = NoLemmingConfig(
        encoder_name=encoder,
        llm_model=model,
        llm_base_url=base_url,
    )
    llm = OpenAICompatibleBackend(model=model, base_url=base_url)
    pipeline = NoLemmingPipeline(config=config, llm=llm)

    print(f"Simulating misinformation spread: {stimulus_path}")
    result = await pipeline.run(stimulus_path, n_agents=n_agents, n_rounds=72)

    signals = result["analysis"]["signals"]
    coalitions = result["analysis"]["coalitions"]
    report = result["analysis"]["report"]

    print("\n" + "=" * 60)
    print("MISINFORMATION SPREAD ANALYSIS")
    print("=" * 60)
    print(report.to_markdown())

    print("\nArchetype Susceptibility:")
    for label, fraction in signals.archetype_dominance.items():
        print(f"  {label}: {fraction:.1%} of discourse")

    print(f"\nPolarization Index: {coalitions.polarization_index:.3f}")
    print(f"Coalition Count: {len(coalitions.groups)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model misinformation spread")
    parser.add_argument("stimulus", help="Path to misinformation content")
    parser.add_argument("--encoder", default="mock", help="Brain encoder")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--agents", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(main(
        args.stimulus,
        encoder=args.encoder,
        model=args.model,
        base_url=args.base_url,
        n_agents=args.agents,
    ))

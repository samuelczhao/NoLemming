"""Example: A/B test content variants without real users.

Compare how different versions of content (ads, headlines, announcements)
would spread through a neurally-grounded simulated population.

Usage:
    python examples/ab_testing.py variant_a.txt variant_b.txt

    # With local LLM:
    python examples/ab_testing.py a.txt b.txt --model llama3 --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def main(
    stimuli: list[str],
    encoder: str = "mock",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    n_agents: int = 500,
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

    results = {}
    for path in stimuli:
        print(f"\nRunning variant: {Path(path).name}")
        result = await pipeline.run(path, n_agents=n_agents, n_rounds=48)
        results[Path(path).name] = result

    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)

    for name, result in results.items():
        signals = result["analysis"]["signals"]
        print(f"\n[{name}]")
        print(f"  Sentiment: {signals.sentiment_score:+.3f}")
        print(f"  Consensus: {signals.consensus_strength:.3f}")
        print(f"  Volatility: {signals.volatility_estimate:.3f}")
        print(f"  Dominant archetype: {signals.dominant_archetype}")

    names = list(results.keys())
    if len(names) == 2:
        s1 = results[names[0]]["analysis"]["signals"]
        s2 = results[names[1]]["analysis"]["signals"]
        delta = s1.sentiment_score - s2.sentiment_score
        winner = names[0] if delta > 0 else names[1]
        print(f"\nWinner: {winner} (sentiment delta: {abs(delta):.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test content variants")
    parser.add_argument("stimuli", nargs="+", help="Stimulus files to compare")
    parser.add_argument("--encoder", default="mock", help="Brain encoder")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--agents", type=int, default=500, help="Agents per variant")
    args = parser.parse_args()

    asyncio.run(main(
        args.stimuli,
        encoder=args.encoder,
        model=args.model,
        base_url=args.base_url,
        n_agents=args.agents,
    ))

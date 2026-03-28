"""Example: Predict post-earnings social sentiment trajectory.

Feed an earnings call through NoLemming to predict how social media
will react over the next 72 hours.

Usage:
    python examples/market_sentiment.py path/to/earnings_call.mp4

    # With Ollama (local LLM):
    python examples/market_sentiment.py earnings.mp4 --model llama3 --base-url http://localhost:11434/v1

    # With TRIBE v2 brain encoder:
    python examples/market_sentiment.py earnings.mp4 --encoder tribe_v2
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def main(
    stimulus_path: str,
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

    print(f"Encoding stimulus: {stimulus_path}")
    result = await pipeline.run(
        stimulus_path,
        n_agents=n_agents,
        n_rounds=72,  # 72 hours
    )

    report = result["analysis"]["report"]
    signals = result["analysis"]["signals"]

    print("\n" + "=" * 60)
    print(report.to_markdown())
    print("=" * 60)
    print(f"\nSentiment Score: {signals.sentiment_score:+.3f}")
    print(f"Consensus Strength: {signals.consensus_strength:.3f}")
    print(f"Dominant Archetype: {signals.dominant_archetype}")
    print(f"Predicted Volatility: {signals.volatility_estimate:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict post-earnings sentiment")
    parser.add_argument("stimulus", help="Path to earnings call file")
    parser.add_argument("--encoder", default="mock", help="Brain encoder")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    parser.add_argument("--agents", type=int, default=500, help="Number of agents")
    args = parser.parse_args()

    asyncio.run(main(
        args.stimulus,
        encoder=args.encoder,
        model=args.model,
        base_url=args.base_url,
        n_agents=args.agents,
    ))

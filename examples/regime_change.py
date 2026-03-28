"""Example: Detect market regime change precursors.

Monitor information environment by analyzing stimuli through brain encoding,
tracking shifts in population-level neural response distributions that may
precede market regime changes.

Usage:
    python examples/regime_change.py stimulus1.mp4 stimulus2.mp4 stimulus3.mp4
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np


async def main(
    stimuli: list[str],
    encoder: str = "mock",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
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

    print("Analyzing neural response distribution shifts...")
    print(f"Processing {len(stimuli)} stimuli\n")

    archetype_distributions: list[dict[str, float]] = []

    for path in stimuli:
        name = Path(path).name
        from nolemming.core.types import Stimulus

        stimulus = Stimulus.from_path(path)
        response = pipeline.encode_stimulus(stimulus)
        archetypes, template = pipeline.build_archetypes(response)

        dist = {a.label: a.population_fraction for a in archetypes}
        archetype_distributions.append(dist)
        print(f"  {name}: {len(archetypes)} archetypes detected")

    if len(archetype_distributions) >= 2:
        print("\n" + "=" * 60)
        print("REGIME CHANGE ANALYSIS")
        print("=" * 60)
        _analyze_distribution_shift(archetype_distributions, stimuli)


def _analyze_distribution_shift(
    distributions: list[dict[str, float]],
    stimuli: list[str],
) -> None:
    """Compare archetype distributions across stimuli for regime shifts."""
    all_labels = set()
    for d in distributions:
        all_labels.update(d.keys())

    for i in range(1, len(distributions)):
        prev = distributions[i - 1]
        curr = distributions[i]

        shifts: list[tuple[str, float]] = []
        for label in all_labels:
            delta = curr.get(label, 0.0) - prev.get(label, 0.0)
            if abs(delta) > 0.05:
                shifts.append((label, delta))

        name_prev = Path(stimuli[i - 1]).name
        name_curr = Path(stimuli[i]).name
        print(f"\n{name_prev} -> {name_curr}:")

        if shifts:
            shifts.sort(key=lambda x: abs(x[1]), reverse=True)
            for label, delta in shifts:
                direction = "+" if delta > 0 else ""
                print(f"  {label}: {direction}{delta:.1%}")

            max_shift = max(abs(s[1]) for s in shifts)
            if max_shift > 0.15:
                print("  [WARNING] Significant distribution shift detected")
            else:
                print("  Distribution shift within normal range")
        else:
            print("  No significant shifts detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect regime change precursors")
    parser.add_argument("stimuli", nargs="+", help="Stimulus files in chronological order")
    parser.add_argument("--encoder", default="mock", help="Brain encoder")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--base-url", default=None, help="LLM API base URL")
    args = parser.parse_args()

    asyncio.run(main(
        args.stimuli,
        encoder=args.encoder,
        model=args.model,
        base_url=args.base_url,
    ))

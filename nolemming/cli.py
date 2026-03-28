"""NoLemming CLI — brain-encoded swarm social simulation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="nolemming",
    help="Brain-encoded swarm social simulation engine.",
    no_args_is_help=True,
)
console = Console()


def _run_async(coro: object) -> object:
    """Run an async coroutine from sync CLI context."""
    return asyncio.run(coro)  # type: ignore[arg-type]


@app.command()
def run(
    stimulus: Path = typer.Argument(..., help="Path to stimulus file (video/audio/text)"),
    agents: int = typer.Option(1000, "--agents", "-a", help="Number of agents"),
    rounds: int = typer.Option(168, "--rounds", "-r", help="Simulation rounds"),
    encoder: str = typer.Option("mock", "--encoder", "-e", help="Brain encoder name"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model name"),
    base_url: str | None = typer.Option(None, "--base-url", help="LLM API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="LLM API key"),
    output: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
) -> None:
    """Run a brain-encoded swarm simulation on a stimulus."""
    console.print(Panel.fit(
        "[bold]NoLemming[/bold] — Brain-Encoded Swarm Social Simulation",
        border_style="cyan",
    ))

    if not stimulus.exists():
        console.print(f"[red]Error: stimulus file not found: {stimulus}[/red]")
        raise typer.Exit(1)

    from nolemming.config import NoLemmingConfig
    from nolemming.core.llm import OpenAICompatibleBackend
    from nolemming.core.pipeline import NoLemmingPipeline

    config = NoLemmingConfig(
        encoder_name=encoder,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        output_dir=output,
    )
    llm = OpenAICompatibleBackend(
        model=model, api_key=api_key, base_url=base_url,
    )
    pipeline = NoLemmingPipeline(config=config, llm=llm)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running simulation...", total=None)
        result = _run_async(pipeline.run(stimulus, n_agents=agents, n_rounds=rounds))

    report = result["analysis"]["report"]  # type: ignore[index]
    console.print(Panel(report.to_markdown(), title="Prediction Report", border_style="green"))
    console.print(f"\n[dim]Simulation ID: {result['simulation_id']}[/dim]")  # type: ignore[index]


@app.command()
def benchmark(
    event: str = typer.Argument(..., help="Event ID to benchmark"),
    data_dir: Path = typer.Option("./benchmarks/data", "--data-dir", help="Ground truth data"),
    encoder: str = typer.Option("mock", "--encoder", "-e", help="Brain encoder"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model"),
    agents: int = typer.Option(500, "--agents", "-a", help="Agents per condition"),
) -> None:
    """Run 3-way benchmark: neural vs vanilla vs random."""
    console.print(f"[bold]Benchmarking event:[/bold] {event}")
    console.print("[dim]Conditions: neural, vanilla, random[/dim]")

    from nolemming.benchmark.runner import BenchmarkRunner
    from nolemming.core.types import SimulationConfig, Stimulus

    runner = BenchmarkRunner(data_dir=data_dir)
    events = runner.loader.list_events()
    if event not in events:
        console.print(f"[red]Event not found. Available: {events}[/red]")
        raise typer.Exit(1)

    gt = runner.loader.load_event(event)

    from nolemming.benchmark.runner import BenchmarkCondition

    conditions = [
        BenchmarkCondition(
            name="neural",
            config=SimulationConfig(
                stimulus=Stimulus.from_path(gt.stimulus_path),
                n_agents=agents,
                encoder_name=encoder,
            ),
        ),
        BenchmarkCondition(
            name="vanilla",
            config=SimulationConfig(
                stimulus=Stimulus.from_path(gt.stimulus_path),
                n_agents=agents,
                encoder_name="mock",
            ),
        ),
        BenchmarkCondition(
            name="random",
            config=SimulationConfig(
                stimulus=Stimulus.from_path(gt.stimulus_path),
                n_agents=agents,
                encoder_name="mock",
            ),
        ),
    ]

    result = _run_async(runner.run_event(event, conditions))
    console.print(Panel(
        result.summary(),
        title=f"Benchmark: {event}",
        border_style="yellow",
    ))


@app.command()
def compare(
    stimuli: list[Path] = typer.Argument(..., help="Stimulus files to compare"),
    agents: int = typer.Option(1000, "--agents", "-a", help="Agents per variant"),
    encoder: str = typer.Option("mock", "--encoder", "-e", help="Brain encoder"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model"),
) -> None:
    """Compare multiple stimulus variants (A/B testing)."""
    console.print(f"[bold]Comparing {len(stimuli)} variants[/bold]")

    for path in stimuli:
        if not path.exists():
            console.print(f"[red]Error: file not found: {path}[/red]")
            raise typer.Exit(1)

    from nolemming.config import NoLemmingConfig
    from nolemming.core.pipeline import NoLemmingPipeline

    config = NoLemmingConfig(encoder_name=encoder, llm_model=model)
    pipeline = NoLemmingPipeline(config=config)

    results = {}
    for path in stimuli:
        console.print(f"  Running: {path.name}")
        result = _run_async(pipeline.run(path, n_agents=agents))
        results[path.name] = result

    console.print("\n[bold]Comparison Summary:[/bold]")
    for name, result in results.items():
        signals = result["analysis"]["signals"]  # type: ignore[index]
        console.print(
            f"  {name}: sentiment={signals.sentiment_score:+.2f} "
            f"consensus={signals.consensus_strength:.2f} "
            f"dominant={signals.dominant_archetype}",
        )


@app.command()
def encoders() -> None:
    """List available brain encoders."""
    from nolemming.encoders.registry import encoder_registry

    console.print("[bold]Available Brain Encoders:[/bold]")
    for name in encoder_registry.list_encoders():
        console.print(f"  - {name}")


if __name__ == "__main__":
    app()

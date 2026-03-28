"""Benchmark result visualization using Plotly."""

from __future__ import annotations

import plotly.graph_objects as go

from nolemming.analysis.sentiment import SentimentTrajectory
from nolemming.benchmark.runner import BenchmarkResult


def _sentiment_trace(
    name: str,
    trajectory: SentimentTrajectory,
    dash: str | None = None,
) -> go.Scatter:
    """Create a plotly scatter trace for a sentiment trajectory."""
    line_config = {"dash": dash} if dash else {}
    return go.Scatter(
        x=trajectory.timestamps,
        y=trajectory.scores,
        mode="lines+markers",
        name=name,
        line=line_config,
    )


def _sentiment_layout(title: str) -> go.Layout:
    """Create layout for sentiment comparison chart."""
    return go.Layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1.1, 1.1],
        template="plotly_white",
    )


def _metrics_bar_traces(
    results: list[BenchmarkResult],
) -> tuple[list[str], list[go.Bar]]:
    """Build bar chart traces for benchmark metrics summary."""
    if not results:
        return [], []

    condition_names = list(results[0].conditions.keys())
    event_ids = [r.event_id for r in results]
    traces: list[go.Bar] = []

    for cond_name in condition_names:
        correlations = [
            r.conditions[cond_name].sentiment_correlation
            if cond_name in r.conditions else 0.0
            for r in results
        ]
        traces.append(go.Bar(
            name=cond_name,
            x=event_ids,
            y=correlations,
        ))

    return event_ids, traces


class BenchmarkVisualizer:
    """Generates Plotly HTML charts for benchmark results."""

    def plot_sentiment_comparison(
        self,
        predicted: dict[str, SentimentTrajectory],
        actual: SentimentTrajectory,
        title: str,
    ) -> str:
        """Compare predicted vs actual sentiment trajectories."""
        fig = go.Figure()
        fig.add_trace(_sentiment_trace(
            "Actual", actual, dash="dash",
        ))
        for name, traj in predicted.items():
            fig.add_trace(_sentiment_trace(name, traj))
        fig.update_layout(_sentiment_layout(title))
        return fig.to_html(include_plotlyjs="cdn")

    def plot_benchmark_summary(
        self, results: list[BenchmarkResult],
    ) -> str:
        """Summary chart of sentiment correlation across all events."""
        _, traces = _metrics_bar_traces(results)
        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Benchmark: Sentiment Correlation by Event",
            xaxis_title="Event",
            yaxis_title="Pearson r",
            barmode="group",
            template="plotly_white",
        )
        return fig.to_html(include_plotlyjs="cdn")

"""Structured prediction report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nolemming.analysis.networks import CoalitionReport
from nolemming.analysis.sentiment import SentimentTrajectory
from nolemming.analysis.signals import PredictionSignals
from nolemming.core.llm import LLMBackend
from nolemming.core.types import SimulationResult


@dataclass
class PredictionReport:
    """Structured prediction report from simulation analysis."""

    title: str
    summary: str
    sentiment_analysis: str
    archetype_dynamics: str
    key_predictions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    signals: PredictionSignals = field(default_factory=lambda: PredictionSignals(
        sentiment_score=0.0,
        sentiment_momentum=0.0,
        consensus_strength=0.0,
        volatility_estimate=0.0,
        dominant_archetype="unknown",
    ))
    generated_at: str = ""

    def to_markdown(self) -> str:
        predictions = "\n".join(f"- {p}" for p in self.key_predictions)
        return (
            f"# {self.title}\n\n"
            f"**Generated:** {self.generated_at}\n"
            f"**Confidence:** {self.confidence:.0%}\n\n"
            f"## Summary\n{self.summary}\n\n"
            f"## Sentiment Analysis\n{self.sentiment_analysis}\n\n"
            f"## Archetype Dynamics\n{self.archetype_dynamics}\n\n"
            f"## Key Predictions\n{predictions}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sentiment_analysis": self.sentiment_analysis,
            "archetype_dynamics": self.archetype_dynamics,
            "key_predictions": self.key_predictions,
            "confidence": self.confidence,
            "signals": self.signals.to_dict(),
            "generated_at": self.generated_at,
        }


REPORT_SYSTEM_PROMPT = (
    "You are a quantitative analyst. Given simulation data about "
    "social media reactions to a financial event, produce a structured "
    "prediction report. Respond with valid JSON only."
)

REPORT_USER_TEMPLATE = (
    "Analyze this simulation data and produce a prediction report.\n\n"
    "Data:\n{data}\n\n"
    "Respond with JSON containing these fields:\n"
    "- title: string\n"
    "- summary: 2-3 sentence executive summary\n"
    "- sentiment_analysis: paragraph on sentiment dynamics\n"
    "- archetype_dynamics: paragraph on which brain types drove narrative\n"
    "- key_predictions: list of 3-5 bullet point predictions\n"
    "- confidence: float 0-1"
)


def _build_analysis_prompt(
    result: SimulationResult,
    sentiment: SentimentTrajectory,
    signals: PredictionSignals,
    coalitions: CoalitionReport,
) -> str:
    return json.dumps({
        "simulation_id": result.simulation_id,
        "n_agents": result.config.n_agents,
        "n_rounds": result.config.n_rounds,
        "sentiment_trajectory": sentiment.to_dict(),
        "signals": signals.to_dict(),
        "polarization_index": coalitions.polarization_index,
        "archetype_affinity": coalitions.archetype_affinity,
        "n_coalitions": len(coalitions.groups),
    })


def _parse_llm_response(
    raw: str, signals: PredictionSignals,
) -> PredictionReport:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return PredictionReport(
        title=data.get("title", "Prediction Report"),
        summary=data.get("summary", ""),
        sentiment_analysis=data.get("sentiment_analysis", ""),
        archetype_dynamics=data.get("archetype_dynamics", ""),
        key_predictions=data.get("key_predictions", []),
        confidence=float(data.get("confidence", 0.5)),
        signals=signals,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _build_fallback_report(
    signals: PredictionSignals,
    sentiment: SentimentTrajectory,
    coalitions: CoalitionReport,
) -> PredictionReport:
    """Generate a report without LLM when none is available."""
    direction = "positive" if signals.sentiment_score > 0 else "negative"
    return PredictionReport(
        title="Simulation Prediction Report",
        summary=(
            f"Sentiment is {direction} ({signals.sentiment_score:+.2f}) "
            f"with {signals.consensus_strength:.0%} consensus. "
            f"Dominant archetype: {signals.dominant_archetype}."
        ),
        sentiment_analysis=(
            f"The simulation produced {len(sentiment.timestamps)} "
            f"data points. Sentiment momentum: {signals.sentiment_momentum:+.3f}. "
            f"Predicted volatility: {signals.volatility_estimate:.3f}."
        ),
        archetype_dynamics=(
            f"The {signals.dominant_archetype} archetype dominated discourse. "
            f"Population split into {len(coalitions.groups)} coalitions "
            f"with polarization index {coalitions.polarization_index:.2f}."
        ),
        key_predictions=[
            f"Sentiment direction: {direction}",
            f"Consensus: {'strong' if signals.consensus_strength > 0.7 else 'weak'}",
            f"Dominant narrative: {signals.dominant_archetype}-driven",
        ],
        confidence=0.3,
        signals=signals,
        generated_at=datetime.now(UTC).isoformat(),
    )


class ReportGenerator:
    """Generates structured prediction reports using LLM synthesis."""

    def __init__(self, llm: LLMBackend | None = None) -> None:
        self._llm = llm

    async def generate(
        self,
        result: SimulationResult,
        sentiment: SentimentTrajectory,
        signals: PredictionSignals,
        coalitions: CoalitionReport,
    ) -> PredictionReport:
        """Synthesize all analysis into a structured prediction report."""
        if self._llm is None:
            return _build_fallback_report(signals, sentiment, coalitions)

        prompt_data = _build_analysis_prompt(
            result, sentiment, signals, coalitions,
        )
        user_msg = REPORT_USER_TEMPLATE.format(data=prompt_data)
        response = await self._llm.generate(
            system_prompt=REPORT_SYSTEM_PROMPT,
            user_prompt=user_msg,
            temperature=0.3,
        )
        return _parse_llm_response(response.content, signals)

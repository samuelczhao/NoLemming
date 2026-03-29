"""Agent profile builder with LLM or template-based persona generation."""

from __future__ import annotations

import asyncio

from nolemming.core.llm import LLMBackend
from nolemming.core.types import AgentProfile, NeuralArchetype

MAX_BIO_CHARS = 500
MAX_PERSONA_CHARS = 2000
BATCH_CONCURRENCY = 5

SYSTEM_PROMPT = (
    "Generate a social media persona for someone with this cognitive profile. "
    "Return EXACTLY two sections separated by '---':\n"
    "1. A short bio (max 500 chars) for their social media profile.\n"
    "2. A detailed persona description (max 2000 chars) describing how they think, "
    "what they care about, their communication style, and likely reactions.\n"
    "Keep it authentic and distinctive. No generic filler."
)

# Template-based personas for when no LLM is available
ARCHETYPE_TEMPLATES: dict[str, tuple[str, str]] = {
    "fear-dominant": (
        "Risk analyst. Cautious investor. First to spot trouble.",
        "Hypervigilant about market threats. Shares alarming news quickly. "
        "Focuses on downside risk and worst-case scenarios. Skeptical of "
        "bullish narratives. Communication style: urgent, warning-heavy. "
        "Tends to amplify negative signals and question positive ones.",
    ),
    "reward-seeking": (
        "Growth investor. Optimist. Always looking for the next big thing.",
        "Excited by positive earnings surprises and growth metrics. Quick to "
        "celebrate wins and share bullish takes. Drawn to momentum plays. "
        "Communication style: enthusiastic, forward-looking. May underweight "
        "risks in favor of upside narratives.",
    ),
    "analytical": (
        "Data-driven analyst. Numbers first, narratives second.",
        "Focuses on margins, revenue growth rates, and guidance vs estimates. "
        "Skeptical of hype, prefers to dig into the details. Communication "
        "style: measured, data-heavy, often contrarian to popular sentiment. "
        "Will challenge both bulls and bears with specific numbers.",
    ),
    "social-attuned": (
        "Community pulse reader. Follows the conversation.",
        "Highly responsive to what others are saying. Picks up on trending "
        "narratives and amplifies them. Communication style: conversational, "
        "reactive. Influenced by group consensus but can also shift it. "
        "Often the bridge between different opinion clusters.",
    ),
    "risk-averse": (
        "Conservative investor. Capital preservation above all.",
        "Focuses on red flags, spending concerns, and valuation risks. Quick "
        "to raise skepticism about aggressive guidance or spending plans. "
        "Communication style: cautious, questioning. Tends to highlight what "
        "could go wrong rather than what went right.",
    ),
    "contrarian": (
        "Contrarian thinker. If everyone agrees, something's wrong.",
        "Notices inconsistencies between earnings and market reaction. "
        "Challenges the dominant narrative regardless of direction. "
        "Communication style: provocative, questioning. Provides the "
        "counterargument that keeps discussion honest.",
    ),
    "verbal-analytical": (
        "Detailed writer. Deep analysis, long-form takes.",
        "Engages deeply with transcript content and management commentary. "
        "Produces thoughtful analysis connecting multiple data points. "
        "Communication style: articulate, thorough. Prefers nuance over "
        "hot takes. Often referenced by others for analysis depth.",
    ),
}

DEFAULT_TEMPLATE = (
    "Market participant. Follows earnings closely.",
    "Engages with financial content based on personal analysis. "
    "Communication style varies by topic. Responds to both data "
    "and narrative. Participates actively in post-earnings discussion.",
)


class ProfileBuilder:
    """Enriches agent profiles with LLM or template-based personas."""

    def __init__(self, llm: LLMBackend | None = None) -> None:
        self._llm = llm

    async def enrich_profiles(
        self,
        agents: list[AgentProfile],
        stimulus_context: str,
    ) -> list[AgentProfile]:
        """Generate detailed persona and bio for each agent."""
        if self._llm is None:
            return self._enrich_from_templates(agents)
        return await self._enrich_from_llm(agents, stimulus_context)

    def _enrich_from_templates(
        self, agents: list[AgentProfile],
    ) -> list[AgentProfile]:
        """Generate personas from archetype templates (no LLM needed)."""
        for agent in agents:
            bio, persona = _template_for_archetype(agent.archetype)
            agent.bio = bio
            agent.persona = persona
            agent.stance = _stance_for_archetype(agent.archetype)
        return agents

    async def _enrich_from_llm(
        self,
        agents: list[AgentProfile],
        stimulus_context: str,
    ) -> list[AgentProfile]:
        """Generate personas via LLM calls."""
        sem = asyncio.Semaphore(BATCH_CONCURRENCY)
        tasks = [
            self._enrich_one(agent, stimulus_context, sem)
            for agent in agents
        ]
        return await asyncio.gather(*tasks)

    async def _enrich_one(
        self,
        agent: AgentProfile,
        stimulus_context: str,
        sem: asyncio.Semaphore,
    ) -> AgentProfile:
        async with sem:
            bio, persona = await self._generate_persona(
                agent.archetype, agent.name, stimulus_context,
            )
            agent.bio = bio[:MAX_BIO_CHARS]
            agent.persona = persona[:MAX_PERSONA_CHARS]
            agent.stance = _stance_for_archetype(agent.archetype)
            return agent

    async def _generate_persona(
        self,
        archetype: NeuralArchetype,
        name: str,
        stimulus_context: str,
    ) -> tuple[str, str]:
        regions = ", ".join(archetype.dominant_regions[:3])
        user_msg = (
            f"Name: {name}\n"
            f"Archetype: {archetype.label}\n"
            f"Cognitive profile: {archetype.description}\n"
            f"Dominant brain regions: {regions}\n"
            f"Context: {stimulus_context[:300]}"
        )
        response = await self._llm.generate(  # type: ignore[union-attr]
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_msg,
            temperature=0.9,
            max_tokens=512,
        )
        return _parse_response(response.content)


def _template_for_archetype(
    archetype: NeuralArchetype,
) -> tuple[str, str]:
    """Get (bio, persona) template matching the archetype label."""
    label = archetype.label
    if label in ARCHETYPE_TEMPLATES:
        return ARCHETYPE_TEMPLATES[label]
    for key, template in ARCHETYPE_TEMPLATES.items():
        if key in label:
            return template
    return DEFAULT_TEMPLATE


def _stance_for_archetype(archetype: NeuralArchetype) -> str:
    """Derive stance from archetype's dominant regions."""
    label = archetype.label
    if "fear" in label or "risk" in label:
        return "bearish"
    if "reward" in label:
        return "bullish"
    if "contrarian" in label:
        return "contrarian"
    if "analytical" in label:
        return "neutral-analytical"
    return "neutral"


def _parse_response(text: str) -> tuple[str, str]:
    if "---" in text:
        parts = text.split("---", 1)
        return parts[0].strip(), parts[1].strip()
    mid = len(text) // 3
    return text[:mid].strip(), text[mid:].strip()

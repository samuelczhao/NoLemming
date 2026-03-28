"""Agent profile builder with LLM persona generation."""

from __future__ import annotations

import asyncio
from typing import Protocol

from nolemming.core.types import AgentProfile, NeuralArchetype

MAX_BIO_CHARS = 500
MAX_PERSONA_CHARS = 2000
BATCH_CONCURRENCY = 10

SYSTEM_PROMPT = (
    "Generate a social media persona for someone with this cognitive profile. "
    "You will receive their archetype description, name, and a stimulus context. "
    "Return EXACTLY two sections separated by '---':\n"
    "1. A short bio (max 500 chars) for their social media profile.\n"
    "2. A detailed persona description (max 2000 chars) describing how they think, "
    "what they care about, their communication style, and likely reactions.\n"
    "Keep it authentic and distinctive. No generic filler."
)


class LLMClient(Protocol):
    """Protocol for async OpenAI-compatible chat client."""

    @property
    def chat(self) -> ChatCompletions: ...


class ChatCompletions(Protocol):
    @property
    def completions(self) -> Completions: ...


class Completions(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class ProfileBuilder:
    """Enriches agent profiles with LLM-generated personas."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._client = llm_client
        self._model = llm_model

    async def enrich_profiles(
        self,
        agents: list[AgentProfile],
        stimulus_context: str,
    ) -> list[AgentProfile]:
        """Generate detailed persona and bio for each agent."""
        if self._client is None:
            self._client = self._create_default_client()
        semaphore = asyncio.Semaphore(BATCH_CONCURRENCY)
        tasks = [
            self._enrich_one(agent, stimulus_context, semaphore)
            for agent in agents
        ]
        return await asyncio.gather(*tasks)

    async def _enrich_one(
        self,
        agent: AgentProfile,
        stimulus_context: str,
        semaphore: asyncio.Semaphore,
    ) -> AgentProfile:
        async with semaphore:
            bio, persona = await self._generate_persona(
                agent.archetype, agent.name, stimulus_context,
            )
            agent.bio = bio[:MAX_BIO_CHARS]
            agent.persona = persona[:MAX_PERSONA_CHARS]
            return agent

    async def _generate_persona(
        self,
        archetype: NeuralArchetype,
        name: str,
        stimulus_context: str,
    ) -> tuple[str, str]:
        """Generate (bio, persona) via LLM call."""
        user_msg = self._build_user_prompt(archetype, name, stimulus_context)
        response = await self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.9,
        )
        return self._parse_response(response.choices[0].message.content)  # type: ignore[attr-defined]

    def _build_user_prompt(
        self,
        archetype: NeuralArchetype,
        name: str,
        stimulus_context: str,
    ) -> str:
        regions = ", ".join(archetype.dominant_regions) if archetype.dominant_regions else "general"
        return (
            f"Name: {name}\n"
            f"Archetype: {archetype.label}\n"
            f"Cognitive profile: {archetype.description}\n"
            f"Dominant brain regions: {regions}\n"
            f"Stimulus context: {stimulus_context}"
        )

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Split LLM output into (bio, persona)."""
        if "---" in text:
            parts = text.split("---", 1)
            return parts[0].strip(), parts[1].strip()
        mid = len(text) // 3
        return text[:mid].strip(), text[mid:].strip()

    def _create_default_client(self) -> LLMClient:
        from openai import AsyncOpenAI
        return AsyncOpenAI()  # type: ignore[return-value]

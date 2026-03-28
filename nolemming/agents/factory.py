"""Agent population generator."""

from __future__ import annotations

import numpy as np

from nolemming.core.types import AgentProfile, NeuralArchetype

FIRST_NAMES = [
    "James", "Maria", "Wei", "Priya", "Ahmed", "Sofia", "Kenji", "Fatima",
    "Carlos", "Olga", "Yuki", "Aisha", "David", "Lin", "Raj", "Elena",
    "Marcus", "Mei", "Omar", "Anya", "Tomás", "Hana", "Leo", "Nadia",
    "Kofi", "Ingrid", "Diego", "Suki", "Ivan", "Zara", "Noah", "Lucia",
    "Tariq", "Freya", "Jin", "Rosa", "Emeka", "Mila", "Arjun", "Chloe",
]

LAST_NAMES = [
    "Chen", "Patel", "Kim", "Garcia", "Müller", "Santos", "Nguyen", "Ali",
    "Sato", "Ivanova", "Johnson", "López", "Lee", "Singh", "Brown", "Silva",
    "Nakamura", "Hassan", "Williams", "Petrov", "Tanaka", "Park", "Wilson",
    "Martinez", "Okafor", "Johansson", "Rivera", "Zhang", "Das", "Kowalski",
    "Thompson", "Yamada", "Eriksson", "Torres", "Zhao", "Sharma", "Costa",
    "Ahmed", "Fischer", "Andersen",
]

ACTIVITY_ALPHA = 2.0
ACTIVITY_BETA = 4.5


class AgentFactory:
    """Generates agent populations from neural archetypes."""

    def __init__(
        self,
        archetypes: list[NeuralArchetype],
        seed: int = 42,
    ) -> None:
        self._archetypes = archetypes
        self._rng = np.random.default_rng(seed)

    def generate_population(self, n_agents: int) -> list[AgentProfile]:
        """Sample agents from archetype distribution."""
        counts = self._sample_archetype_counts(n_agents)
        agents: list[AgentProfile] = []
        agent_id = 0
        for arch in self._archetypes:
            for _ in range(counts[arch.archetype_id]):
                name, username = self._generate_identity(agent_id, arch)
                activity = self._sample_activity_level()
                agents.append(self._build_profile(agent_id, arch, name, username, activity))
                agent_id += 1
        return agents

    def _sample_archetype_counts(self, n_agents: int) -> dict[int, int]:
        """Allocate agents to archetypes proportionally."""
        fractions = [a.population_fraction for a in self._archetypes]
        raw = [int(f * n_agents) for f in fractions]
        remainder = n_agents - sum(raw)
        indices = self._rng.choice(len(self._archetypes), size=remainder, replace=True)
        for idx in indices:
            raw[idx] += 1
        return {a.archetype_id: raw[i] for i, a in enumerate(self._archetypes)}

    def _generate_identity(
        self,
        agent_id: int,
        archetype: NeuralArchetype,
    ) -> tuple[str, str]:
        """Return (name, username) for an agent."""
        first = self._rng.choice(FIRST_NAMES)
        last = self._rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        digits = self._rng.integers(10, 9999)
        username = f"{first.lower()}{last.lower()}{digits}"
        return name, username

    def _sample_activity_level(self) -> float:
        """Sample activity level from beta distribution centered ~0.3."""
        return float(self._rng.beta(ACTIVITY_ALPHA, ACTIVITY_BETA))

    def _build_profile(
        self,
        agent_id: int,
        archetype: NeuralArchetype,
        name: str,
        username: str,
        activity: float,
    ) -> AgentProfile:
        return AgentProfile(
            agent_id=agent_id,
            archetype=archetype,
            name=name,
            username=username,
            bio="",
            persona="",
            activity_level=activity,
        )

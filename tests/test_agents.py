"""Tests for AgentFactory and AgentMemory."""

import numpy as np
import pytest

from nolemming.agents.factory import AgentFactory
from nolemming.agents.memory import AgentMemory
from nolemming.core.types import AgentProfile, NeuralArchetype


def _make_archetypes(n: int = 3) -> list[NeuralArchetype]:
    """Create n test archetypes with equal population fractions."""
    fraction = 1.0 / n
    return [
        NeuralArchetype(
            archetype_id=i,
            centroid=np.zeros(8, dtype=np.float32),
            label=f"archetype-{i}",
            description=f"Test archetype {i}",
            population_fraction=fraction,
            dominant_regions=["visual_cortex", "language_areas"],
        )
        for i in range(n)
    ]


def _make_uneven_archetypes() -> list[NeuralArchetype]:
    """Archetypes with skewed population fractions."""
    return [
        NeuralArchetype(
            archetype_id=0,
            centroid=np.zeros(8, dtype=np.float32),
            label="dominant",
            description="Dominant archetype",
            population_fraction=0.7,
            dominant_regions=["prefrontal_cortex"],
        ),
        NeuralArchetype(
            archetype_id=1,
            centroid=np.zeros(8, dtype=np.float32),
            label="minority",
            description="Minority archetype",
            population_fraction=0.3,
            dominant_regions=["amygdala_proxy"],
        ),
    ]


# --- AgentFactory ---


class TestAgentFactory:
    def test_generate_correct_count(self) -> None:
        archetypes = _make_archetypes(3)
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(30)
        assert len(agents) == 30

    def test_agents_have_required_fields(self) -> None:
        archetypes = _make_archetypes(2)
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(10)
        for agent in agents:
            assert isinstance(agent, AgentProfile)
            assert isinstance(agent.agent_id, int)
            assert isinstance(agent.name, str)
            assert len(agent.name) > 0
            assert isinstance(agent.username, str)
            assert len(agent.username) > 0
            assert agent.archetype in archetypes
            assert 0.0 <= agent.activity_level <= 1.0

    def test_unique_agent_ids(self) -> None:
        archetypes = _make_archetypes(4)
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(50)
        ids = [a.agent_id for a in agents]
        assert len(ids) == len(set(ids))

    def test_population_distribution_matches_fractions(self) -> None:
        archetypes = _make_uneven_archetypes()
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(100)
        counts = {}
        for agent in agents:
            aid = agent.archetype.archetype_id
            counts[aid] = counts.get(aid, 0) + 1
        # 70/30 split should be roughly maintained
        assert counts[0] >= 50  # dominant should have majority
        assert counts[1] >= 15  # minority should still be present
        assert counts[0] + counts[1] == 100

    def test_reproducibility_same_seed(self) -> None:
        archetypes = _make_archetypes(3)
        agents1 = AgentFactory(archetypes, seed=42).generate_population(20)
        agents2 = AgentFactory(archetypes, seed=42).generate_population(20)
        names1 = [a.name for a in agents1]
        names2 = [a.name for a in agents2]
        assert names1 == names2

    def test_different_seed_gives_different_names(self) -> None:
        archetypes = _make_archetypes(3)
        agents1 = AgentFactory(archetypes, seed=42).generate_population(20)
        agents2 = AgentFactory(archetypes, seed=99).generate_population(20)
        names1 = [a.name for a in agents1]
        names2 = [a.name for a in agents2]
        assert names1 != names2

    def test_single_agent(self) -> None:
        archetypes = _make_archetypes(1)
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(1)
        assert len(agents) == 1
        assert agents[0].archetype == archetypes[0]

    def test_activity_levels_from_beta_distribution(self) -> None:
        """Activity levels should cluster below 0.5 (beta(2, 4.5) mean ~ 0.31)."""
        archetypes = _make_archetypes(2)
        factory = AgentFactory(archetypes, seed=42)
        agents = factory.generate_population(200)
        activities = [a.activity_level for a in agents]
        mean_activity = sum(activities) / len(activities)
        assert 0.15 < mean_activity < 0.50


# --- AgentMemory ---


class TestAgentMemory:
    def test_add_and_get_recent(self) -> None:
        memory = AgentMemory()
        memory.add(agent_id=1, content="saw bullish signal", timestamp=1.0)
        memory.add(agent_id=1, content="market dipped", timestamp=2.0)
        recent = memory.get_recent(agent_id=1, n=10)
        assert len(recent) == 2
        assert recent[0] == "market dipped"  # most recent first
        assert recent[1] == "saw bullish signal"

    def test_get_recent_limits_count(self) -> None:
        memory = AgentMemory()
        for i in range(20):
            memory.add(agent_id=1, content=f"event-{i}", timestamp=float(i))
        recent = memory.get_recent(agent_id=1, n=5)
        assert len(recent) == 5

    def test_get_recent_empty(self) -> None:
        memory = AgentMemory()
        recent = memory.get_recent(agent_id=999, n=10)
        assert recent == []

    def test_clear(self) -> None:
        memory = AgentMemory()
        memory.add(agent_id=1, content="test", timestamp=1.0)
        memory.clear(agent_id=1)
        assert memory.get_recent(agent_id=1) == []

    def test_clear_nonexistent_agent(self) -> None:
        memory = AgentMemory()
        memory.clear(agent_id=999)  # should not raise

    def test_agent_count(self) -> None:
        memory = AgentMemory()
        memory.add(agent_id=1, content="a", timestamp=1.0)
        memory.add(agent_id=2, content="b", timestamp=1.0)
        memory.add(agent_id=1, content="c", timestamp=2.0)
        assert memory.agent_count() == 2

    def test_memories_isolated_between_agents(self) -> None:
        memory = AgentMemory()
        memory.add(agent_id=1, content="agent1-only", timestamp=1.0)
        memory.add(agent_id=2, content="agent2-only", timestamp=1.0)
        assert memory.get_recent(agent_id=1) == ["agent1-only"]
        assert memory.get_recent(agent_id=2) == ["agent2-only"]

    def test_ordering_by_timestamp(self) -> None:
        memory = AgentMemory()
        memory.add(agent_id=1, content="old", timestamp=1.0)
        memory.add(agent_id=1, content="newest", timestamp=100.0)
        memory.add(agent_id=1, content="middle", timestamp=50.0)
        recent = memory.get_recent(agent_id=1, n=3)
        assert recent == ["newest", "middle", "old"]

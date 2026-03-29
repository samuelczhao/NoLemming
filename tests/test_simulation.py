"""Tests for SimulationEngine (fallback mode) and TwitterPlatform."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from nolemming.core.types import (
    AgentProfile,
    EngagementTemplate,
    NeuralArchetype,
    Platform,
    SimulationConfig,
    SimulationResult,
    Stimulus,
    StimulusType,
)
from nolemming.simulation.engine import SimulationEngine
from nolemming.simulation.platforms.twitter import TwitterPlatform


def _make_archetype(arch_id: int = 0) -> NeuralArchetype:
    return NeuralArchetype(
        archetype_id=arch_id,
        centroid=np.zeros(8, dtype=np.float32),
        label=f"test-archetype-{arch_id}",
        description="Test archetype for simulation",
        population_fraction=0.5,
        dominant_regions=["prefrontal_cortex", "language_areas"],
    )


def _make_agents(n: int = 5) -> list[AgentProfile]:
    arch = _make_archetype(0)
    return [
        AgentProfile(
            agent_id=i,
            archetype=arch,
            name=f"Agent {i}",
            username=f"agent{i}",
            bio="test bio",
            persona="test persona",
            activity_level=0.8,
        )
        for i in range(n)
    ]


def _make_engagement_template() -> EngagementTemplate:
    archetypes = [_make_archetype(0), _make_archetype(1)]
    return EngagementTemplate(
        archetype_engagement=np.array([0.6, 0.4], dtype=np.float32),
        similarity_decay=1.5,
        archetypes=archetypes,
    )


def _make_stimulus(tmp_path: Path) -> Stimulus:
    p = tmp_path / "test.txt"
    p.write_text("test stimulus")
    return Stimulus(path=p, stimulus_type=StimulusType.TEXT)


def _make_config(tmp_path: Path) -> SimulationConfig:
    return SimulationConfig(
        stimulus=_make_stimulus(tmp_path),
        n_agents=5,
        n_rounds=3,
        minutes_per_round=60,
        platform=Platform.TWITTER,
        encoder_name="mock",
        seed=42,
    )


# --- TwitterPlatform (fallback mode) ---


class TestTwitterPlatform:
    async def test_setup_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(3)
        await platform.setup(agents)
        assert db_path.exists()

    async def test_setup_inserts_users(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(3)
        await platform.setup(agents)
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        assert count == 3

    async def test_step_generates_posts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(3)
        await platform.setup(agents)
        actions = await platform.step(agents)
        assert len(actions) == 3
        for action in actions:
            assert action["action"] == "create_post"
            assert "content" in action

    async def test_step_persists_to_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(2)
        await platform.setup(agents)
        await platform.step(agents)
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        conn.close()
        assert count == 2

    async def test_get_trending_content_empty_initially(self, tmp_path: Path) -> None:
        platform = TwitterPlatform(db_path=tmp_path / "test.db")
        await platform.setup(_make_agents(2))
        assert platform.get_trending_content() == []

    async def test_get_trending_content_after_step(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(3)
        await platform.setup(agents)
        await platform.step(agents)
        trending = platform.get_trending_content()
        assert len(trending) == 3

    async def test_fallback_post_contains_username(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        agents = _make_agents(1)
        await platform.setup(agents)
        actions = await platform.step(agents)
        content = str(actions[0]["content"])
        assert agents[0].username in content

    async def test_db_schema_has_required_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        platform = TwitterPlatform(db_path=db_path)
        await platform.setup(_make_agents(1))
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "users" in tables
        assert "posts" in tables
        assert "follows" in tables


# --- SimulationEngine (fallback mode) ---


class TestSimulationEngine:
    async def test_setup_and_run(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        agents = _make_agents(5)
        template = _make_engagement_template()
        await engine.setup(agents, template)
        result = await engine.run()
        assert isinstance(result, SimulationResult)

    async def test_result_has_simulation_id(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        await engine.setup(_make_agents(3), _make_engagement_template())
        result = await engine.run()
        assert len(result.simulation_id) == 12

    async def test_result_has_agents(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        agents = _make_agents(5)
        await engine.setup(agents, _make_engagement_template())
        result = await engine.run()
        assert result.agents == agents

    async def test_result_db_path_exists(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        await engine.setup(_make_agents(3), _make_engagement_template())
        result = await engine.run()
        assert result.db_path.exists()

    async def test_actions_log_created(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        await engine.setup(_make_agents(3), _make_engagement_template())
        result = await engine.run()
        assert result.actions_path.exists()

    async def test_result_config_preserved(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        await engine.setup(_make_agents(3), _make_engagement_template())
        result = await engine.run()
        assert result.config.n_rounds == 3
        assert result.config.n_agents == 5

    async def test_result_has_archetypes_and_template(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        engine = SimulationEngine(config, output_base=tmp_path / "output")
        template = _make_engagement_template()
        await engine.setup(_make_agents(3), template)
        result = await engine.run()
        assert result.engagement_template is template
        assert len(result.archetypes) == 2

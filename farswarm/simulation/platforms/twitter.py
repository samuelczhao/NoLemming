"""Twitter-like platform wrapping OASIS."""

from __future__ import annotations

import csv
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Protocol

from farswarm.core.types import AgentProfile
from farswarm.simulation.platforms.base import SimulationPlatform

logger = logging.getLogger(__name__)

OASIS_CSV_HEADERS = ["user_id", "name", "username", "user_char", "description"]

FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    name TEXT,
    archetype_id INTEGER
);
CREATE TABLE IF NOT EXISTS posts (
    post_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    round INTEGER,
    content TEXT,
    likes INTEGER DEFAULT 0,
    reposts INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS follows (
    follower_id INTEGER,
    followee_id INTEGER,
    round INTEGER
);
"""

try:
    import oasis  # type: ignore[import-untyped]
    OASIS_AVAILABLE = True
except ImportError:
    OASIS_AVAILABLE = False


class OasisEnv(Protocol):
    """Protocol for OASIS environment."""

    async def step(self, actions: dict[object, object]) -> object: ...


class TwitterPlatform(SimulationPlatform):
    """Twitter-like platform backed by OASIS or LLM fallback."""

    def __init__(
        self,
        db_path: Path | None = None,
        llm: object | None = None,
        stimulus_context: str = "",
    ) -> None:
        self._env: OasisEnv | None = None
        self._agents: list[AgentProfile] = []
        self._oasis_agents: list[object] = []
        self._posts: list[str] = []
        self._db_path = db_path
        self._round = 0
        self._conn: sqlite3.Connection | None = None
        self._llm = llm
        self._stimulus_context = stimulus_context

    async def setup(self, agents: list[AgentProfile]) -> None:
        """Initialize platform and write agent CSV for OASIS."""
        self._agents = agents
        if OASIS_AVAILABLE:
            await self._setup_oasis(agents)
        else:
            logger.warning("OASIS not installed; using fallback simulation")
            self._setup_fallback_db(agents)

    async def step(self, active_agents: list[AgentProfile]) -> list[dict[str, object]]:
        """Execute one step with active agents."""
        if OASIS_AVAILABLE and self._env is not None:
            return await self._step_oasis(active_agents)
        if self._llm is not None:
            result = await self._step_llm_fallback(active_agents)
        else:
            result = self._step_fallback(active_agents)
        self._round += 1
        return result

    def get_trending_content(self) -> list[str]:
        """Return recent posts as trending content."""
        return self._posts[-50:] if self._posts else []

    def _setup_fallback_db(self, agents: list[AgentProfile]) -> None:
        """Create SQLite DB with schema for fallback mode."""
        if self._db_path is None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(FALLBACK_SCHEMA)
        for agent in agents:
            self._conn.execute(
                "INSERT INTO users (user_id, username, name, archetype_id) VALUES (?, ?, ?, ?)",
                (agent.agent_id, agent.username, agent.name, agent.archetype.archetype_id),
            )
        self._conn.commit()

    async def _setup_oasis(self, agents: list[AgentProfile]) -> None:
        """Configure OASIS environment."""
        csv_path = self._write_agent_csv(agents)
        agent_graph = oasis.generate_twitter_agent_graph(csv_path)  # type: ignore[attr-defined]
        self._env = oasis.make("twitter", agent_graph=agent_graph)  # type: ignore[attr-defined]
        self._oasis_agents = list(agent_graph.agents)

    async def _step_oasis(
        self, active_agents: list[AgentProfile],
    ) -> list[dict[str, object]]:
        """Run one OASIS step."""
        llm_action = oasis.LLMAction  # type: ignore[attr-defined]
        active_ids = {a.agent_id for a in active_agents}
        actions = {}
        for i, oasis_agent in enumerate(self._oasis_agents):
            if i in active_ids:
                actions[oasis_agent] = llm_action()
        result = await self._env.step(actions)  # type: ignore[union-attr]
        return self._parse_oasis_result(result)

    async def _step_llm_fallback(
        self, active_agents: list[AgentProfile],
    ) -> list[dict[str, object]]:
        """Generate posts using LLM for each active agent."""
        import asyncio

        actions: list[dict[str, object]] = []
        sem = asyncio.Semaphore(3)  # limit concurrency for low-compute

        async def gen_post(agent: AgentProfile) -> dict[str, object]:
            async with sem:
                post = await self._generate_llm_post(agent)
                self._posts.append(post)
                self._persist_post(agent.agent_id, post)
                return {
                    "agent_id": agent.agent_id,
                    "action": "create_post",
                    "content": post,
                }

        tasks = [gen_post(a) for a in active_agents]
        actions = await asyncio.gather(*tasks)
        return list(actions)

    async def _generate_llm_post(self, agent: AgentProfile) -> str:
        """Use LLM to generate a realistic social media post."""
        recent = self._posts[-5:] if self._posts else ["(no posts yet)"]
        recent_str = "\n".join(f"- {p[:100]}" for p in recent)

        response = await self._llm.generate(  # type: ignore[union-attr]
            system_prompt=(
                f"You are {agent.name} (@{agent.username}). "
                f"Cognitive profile: {agent.archetype.label} — {agent.archetype.description} "
                f"Your stance: {agent.stance}. "
                "Write a single short social media post (1-2 sentences, max 200 chars). "
                "Be opinionated. React to the topic and recent discussion."
            ),
            user_prompt=(
                f"Topic: {self._stimulus_context}\n\n"
                f"Recent posts in your feed:\n{recent_str}\n\n"
                "Write your post:"
            ),
            temperature=0.9,
            max_tokens=100,
        )
        return response.content.strip()[:280]

    def _step_fallback(
        self, active_agents: list[AgentProfile],
    ) -> list[dict[str, object]]:
        """Generate synthetic actions without LLM (fastest, for testing)."""
        actions: list[dict[str, object]] = []
        for agent in active_agents:
            post = self._generate_static_post(agent)
            self._posts.append(post)
            self._persist_post(agent.agent_id, post)
            actions.append({
                "agent_id": agent.agent_id,
                "action": "create_post",
                "content": post,
            })
        return actions

    def _generate_static_post(self, agent: AgentProfile) -> str:
        """Create a synthetic post without LLM."""
        label = agent.archetype.label
        stance = agent.stance
        return f"[{agent.username}] As a {label} thinker ({stance}): discussing the topic."

    def _persist_post(self, user_id: int, content: str) -> None:
        """Write post to SQLite if connection exists."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO posts (user_id, round, content) VALUES (?, ?, ?)",
            (user_id, self._round, content),
        )
        self._conn.commit()

    def _write_agent_csv(self, agents: list[AgentProfile]) -> Path:
        """Write OASIS-format agent CSV."""
        csv_path = Path(tempfile.mktemp(suffix=".csv"))
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(OASIS_CSV_HEADERS)
            for agent in agents:
                user_char = f"{agent.archetype.description}\n{agent.persona}"
                writer.writerow([
                    agent.agent_id, agent.name, agent.username,
                    user_char, agent.bio,
                ])
        return csv_path

    def _parse_oasis_result(
        self, result: object,
    ) -> list[dict[str, object]]:
        """Convert OASIS step result to action dicts."""
        if isinstance(result, list):
            return [{"raw": r} for r in result]
        return [{"raw": result}]

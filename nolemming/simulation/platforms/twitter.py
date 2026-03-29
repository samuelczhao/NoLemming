"""Twitter-like platform wrapping OASIS."""

from __future__ import annotations

import csv
import hashlib
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Protocol

from nolemming.core.types import AgentProfile
from nolemming.simulation.platforms.base import SimulationPlatform

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
        sem = asyncio.Semaphore(1)  # sequential to respect rate limits

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

        persona_hint = agent.persona[:200] if agent.persona else ""

        response = await self._llm.generate(  # type: ignore[union-attr]
            system_prompt=(
                f"You are {agent.name} (@{agent.username}). "
                f"Cognitive profile: {agent.archetype.label} — "
                f"{agent.archetype.description} "
                f"{persona_hint} "
                "Write a single short social media post (1-2 sentences, max 200 chars). "
                "Be opinionated and specific. React to the topic."
            ),
            user_prompt=(
                f"Topic:\n{self._stimulus_context[:300]}\n\n"
                f"Recent posts:\n{recent_str}\n\n"
                "Your post:"
            ),
            temperature=0.9,
            max_tokens=100,
        )
        content = response.content.strip()[:280]
        return content if content else f"[{agent.username}] ..."

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
        """Generate a content-aware post without LLM using templates."""
        return _generate_template_post(
            agent, self._stimulus_context, self._posts, self._round,
        )

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


# --- Template-based post generation (no LLM required) ---

_ARCHETYPE_REACTIONS: dict[str, list[str]] = {
    "fear-dominant": [
        "This is concerning. {topic} raises serious red flags.",
        "Don't get complacent. The risks here are being underestimated.",
        "Everyone celebrating but I see warning signs in {topic}.",
        "This won't end well. {topic} is masking deeper problems.",
        "Sell the news. {topic} has 'top signal' written all over it.",
    ],
    "reward-seeking": [
        "Incredible results! {topic} confirms the growth thesis.",
        "This is just the beginning. Bullish on {topic}.",
        "Beating estimates again. The momentum is unstoppable.",
        "Loading up. {topic} shows exactly why this is a buy.",
        "Record numbers! {topic} proves the bears wrong again.",
    ],
    "analytical": [
        "Looking at the numbers: {topic} shows mixed signals on margins.",
        "Key metric to watch: the guidance implies deceleration next Q.",
        "Revenue beat but cost structure tells a different story.",
        "Interesting that {topic} mentions capex — run the FCF math.",
        "The headline number is good but dig into the segments.",
    ],
    "social-attuned": [
        "Everyone's talking about {topic} — the consensus is forming fast.",
        "The reaction to {topic} is telling. Watch the follow-through.",
        "Interesting to see the split between retail and institutional takes.",
        "Social sentiment on {topic} is shifting. Pay attention.",
        "The crowd is pricing in {topic} already. What's next?",
    ],
    "risk-averse": [
        "The spending numbers in {topic} are alarming. Where's the ROI?",
        "I'd wait for more clarity. {topic} raises more questions than answers.",
        "Valuation doesn't make sense given {topic}. Staying on sidelines.",
        "Capital preservation first. {topic} is not worth the risk.",
        "Too much uncertainty around {topic}. Need to see follow-through.",
    ],
    "contrarian": [
        "Everyone's bullish on {topic}? That's usually when you sell.",
        "The consensus is wrong about {topic}. Here's why...",
        "Contrarian take: {topic} is actually bearish when you look deeper.",
        "If {topic} is so good, why is smart money selling?",
        "The market is misreading {topic}. The real story is elsewhere.",
    ],
    "verbal-analytical": [
        "Deep dive on {topic}: management's language on guidance is notable.",
        "Three things to note about {topic}: margins, capex, and outlook.",
        "The transcript reveals more than the numbers. {topic} analysis thread.",
        "Nuanced take: {topic} is positive short-term but has structural concerns.",
        "Breaking down {topic} segment by segment...",
    ],
}

_DEFAULT_REACTIONS = [
    "Thoughts on {topic}: need to see more data before taking a position.",
    "Following {topic} closely. The market will sort this out.",
    "Interesting developments with {topic}. Waiting for dust to settle.",
]


def _generate_template_post(
    agent: AgentProfile,
    stimulus_context: str,
    existing_posts: list[str],
    round_num: int,
) -> str:
    """Generate a diverse, content-aware post without an LLM."""
    topic = _extract_topic(stimulus_context)
    label = agent.archetype.label
    reactions = _ARCHETYPE_REACTIONS.get(label, _DEFAULT_REACTIONS)

    seed_str = f"{agent.agent_id}-{round_num}-{label}"
    idx = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    template = reactions[idx % len(reactions)]

    post = template.format(topic=topic)
    return f"@{agent.username}: {post}"


def _extract_topic(stimulus_context: str) -> str:
    """Extract a short topic phrase from stimulus context."""
    first_line = stimulus_context.split("\n")[0][:80]
    if "." in first_line:
        first_line = first_line.split(".")[0]
    return first_line.strip()

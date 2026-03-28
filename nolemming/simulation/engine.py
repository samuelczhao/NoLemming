"""Main simulation orchestrator."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nolemming.core.types import (
    AgentProfile,
    EngagementTemplate,
    Platform,
    SimulationConfig,
    SimulationResult,
)
from nolemming.simulation.dynamics import EngagementDynamics
from nolemming.simulation.platforms.base import SimulationPlatform
from nolemming.simulation.platforms.twitter import TwitterPlatform

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_BASE = Path("outputs")


class SimulationEngine:
    """Core simulation orchestrator integrating neural archetypes with OASIS."""

    def __init__(
        self,
        config: SimulationConfig,
        output_base: Path = DEFAULT_OUTPUT_BASE,
        llm: object | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._simulation_id = uuid.uuid4().hex[:12]
        self._output_dir = output_base / self._simulation_id
        self._agents: list[AgentProfile] = []
        self._platform: SimulationPlatform | None = None
        self._dynamics: EngagementDynamics | None = None
        self._engagement_template: EngagementTemplate | None = None
        self._stimulus_embedding: NDArray[np.float32] | None = None
        self._rng = np.random.default_rng(config.seed)

    async def setup(
        self,
        agents: list[AgentProfile],
        engagement_template: EngagementTemplate,
    ) -> None:
        """Prepare simulation environment with neurally-grounded agents."""
        self._agents = agents
        self._engagement_template = engagement_template
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._platform = self._create_platform()
        self._dynamics = self._create_dynamics(engagement_template)
        self._stimulus_embedding = self._compute_stimulus_embedding()
        await self._platform.setup(agents)
        logger.info("Simulation %s setup complete with %d agents", self._simulation_id, len(agents))

    async def run(self) -> SimulationResult:
        """Execute the full simulation loop."""
        actions_path = self._output_dir / "actions.jsonl"
        db_path = self._output_dir / "simulation.db"
        for round_num in range(self._config.n_rounds):
            await self._execute_round(round_num, actions_path)
        return self._build_result(db_path, actions_path)

    async def _execute_round(
        self,
        round_num: int,
        actions_path: Path,
    ) -> None:
        """Run a single simulation round."""
        active = self._get_active_agents(round_num)
        step_results = await self._platform.step(active)  # type: ignore[union-attr]
        self._log_actions(actions_path, round_num, step_results)
        if round_num % 10 == 0:
            total = self._config.n_rounds
            logger.info("Round %d/%d: %d active agents", round_num, total, len(active))

    def _get_active_agents(self, round_num: int) -> list[AgentProfile]:
        """Select active agents using engagement template modulation."""
        content_sim = self._estimate_content_similarity()
        active: list[AgentProfile] = []
        for agent in self._agents:
            prob = self._dynamics.modulate_activation(agent, content_sim)  # type: ignore[union-attr]
            if self._rng.random() < prob:
                active.append(agent)
        return active

    def _estimate_content_similarity(self) -> float:
        """Estimate current content similarity to stimulus."""
        trending = self._platform.get_trending_content() if self._platform else []  # type: ignore[union-attr]
        if not trending or self._stimulus_embedding is None:
            return 0.5
        return self._dynamics.compute_content_similarity(  # type: ignore[union-attr]
            trending, self._stimulus_embedding,
        )

    def _create_platform(self) -> SimulationPlatform:
        """Instantiate the correct platform."""
        stimulus_context = self._read_stimulus_context()
        if self._config.platform == Platform.TWITTER:
            return TwitterPlatform(
                db_path=self._output_dir / "simulation.db",
                llm=self._llm,
                stimulus_context=stimulus_context,
            )
        msg = f"Unsupported platform: {self._config.platform}"
        raise ValueError(msg)

    def _read_stimulus_context(self) -> str:
        """Read stimulus file text for platform context."""
        path = self._config.stimulus.path
        if path.suffix in (".txt", ".md") and path.exists():
            return path.read_text(encoding="utf-8")[:500]
        return path.stem

    def _create_dynamics(
        self,
        template: EngagementTemplate,
    ) -> EngagementDynamics:
        """Create engagement dynamics with optional embedding model."""
        encoder = self._load_embedding_model()
        return EngagementDynamics(template=template, embedding_model=encoder)

    def _load_embedding_model(self) -> object | None:
        """Load sentence-transformers model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self._config.encoder_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; using fixed similarity")
            return None

    def _compute_stimulus_embedding(self) -> NDArray[np.float32] | None:
        """Embed the stimulus text for similarity comparison."""
        if self._dynamics is None or self._dynamics._encoder is None:
            return None
        stimulus_text = self._config.stimulus.metadata.get("text", "")
        if not stimulus_text:
            return None
        embeddings = self._dynamics._encoder.encode([stimulus_text])
        return embeddings[0]

    def _log_actions(
        self,
        path: Path,
        round_num: int,
        actions: list[dict[str, object]],
    ) -> None:
        """Append actions to JSONL log."""
        with path.open("a") as f:
            for action in actions:
                record = {"round": round_num, **action}
                f.write(json.dumps(record, default=str) + "\n")

    def _build_result(
        self,
        db_path: Path,
        actions_path: Path,
    ) -> SimulationResult:
        """Construct final simulation result."""
        return SimulationResult(
            simulation_id=self._simulation_id,
            config=self._config,
            agents=self._agents,
            db_path=db_path,
            actions_path=actions_path,
            archetypes=self._engagement_template.archetypes,  # type: ignore[union-attr]
            engagement_template=self._engagement_template,  # type: ignore[arg-type]
        )

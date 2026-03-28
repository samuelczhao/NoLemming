"""Abstract simulation platform interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nolemming.core.types import AgentProfile


class SimulationPlatform(ABC):
    """Base class for all simulation platforms."""

    @abstractmethod
    async def setup(self, agents: list[AgentProfile]) -> None:
        """Initialize the platform with agent profiles."""

    @abstractmethod
    async def step(self, active_agents: list[AgentProfile]) -> list[dict[str, object]]:
        """Execute one simulation step for active agents."""

    @abstractmethod
    def get_trending_content(self) -> list[str]:
        """Return current trending/top content on the platform."""

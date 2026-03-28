"""Agent memory interface."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Single memory record."""

    content: str
    timestamp: float


class AgentMemory:
    """Simple in-memory store for agent observations."""

    def __init__(self) -> None:
        self._store: dict[int, list[MemoryEntry]] = defaultdict(list)

    def add(self, agent_id: int, content: str, timestamp: float) -> None:
        """Record a new memory for an agent."""
        self._store[agent_id].append(MemoryEntry(content=content, timestamp=timestamp))

    def get_recent(self, agent_id: int, n: int = 10) -> list[str]:
        """Return the n most recent memories for an agent."""
        entries = self._store.get(agent_id, [])
        sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
        return [e.content for e in sorted_entries[:n]]

    def clear(self, agent_id: int) -> None:
        """Remove all memories for an agent."""
        self._store.pop(agent_id, None)

    def agent_count(self) -> int:
        """Number of agents with stored memories."""
        return len(self._store)

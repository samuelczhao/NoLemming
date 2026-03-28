"""Social network and coalition analysis for simulation results."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from nolemming.core.types import SimulationResult


@dataclass
class CoalitionReport:
    """Result of coalition analysis across archetypes."""

    groups: list[list[int]] = field(default_factory=list)
    archetype_affinity: dict[str, list[str]] = field(default_factory=dict)
    polarization_index: float = 0.0


def _load_follows(db_path: str) -> list[tuple[int, int]]:
    """Load follow relationships from the simulation DB."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT follower_id, followee_id FROM follows"
            )
            return list(cursor.fetchall())
    except sqlite3.OperationalError:
        return []


def _load_interactions(db_path: str) -> list[tuple[int, int, str]]:
    """Load interactions (likes, reposts, replies) from the DB."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT source_id, target_id, action_type "
                "FROM interactions"
            )
            return list(cursor.fetchall())
    except sqlite3.OperationalError:
        return []


def _build_interaction_graph(
    follows: list[tuple[int, int]],
    interactions: list[tuple[int, int, str]],
) -> dict[int, dict[int, int]]:
    """Build weighted adjacency: agent_a -> agent_b -> weight."""
    graph: dict[int, dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for follower, followed in follows:
        graph[follower][followed] += 1
    for source, target, _ in interactions:
        graph[source][target] += 1
    return dict(graph)


def _find_coalitions(
    graph: dict[int, dict[int, int]],
    agents_by_archetype: dict[int, list[int]],
) -> list[list[int]]:
    """Identify coalitions as groups of heavily-interacting agents."""
    archetype_pairs: dict[tuple[int, int], int] = defaultdict(int)
    agent_to_arch: dict[int, int] = {}
    for arch_id, agents in agents_by_archetype.items():
        for a in agents:
            agent_to_arch[a] = arch_id

    for src, targets in graph.items():
        src_arch = agent_to_arch.get(src, -1)
        for tgt, weight in targets.items():
            tgt_arch = agent_to_arch.get(tgt, -1)
            pair = (min(src_arch, tgt_arch), max(src_arch, tgt_arch))
            archetype_pairs[pair] += weight

    return _cluster_archetypes(archetype_pairs, agents_by_archetype)


def _cluster_archetypes(
    pair_weights: dict[tuple[int, int], int],
    agents_by_archetype: dict[int, list[int]],
) -> list[list[int]]:
    """Simple greedy clustering of archetypes by interaction weight."""
    if not pair_weights:
        return [
            agents
            for agents in agents_by_archetype.values()
            if agents
        ]

    threshold = _compute_threshold(pair_weights)
    merged: dict[int, int] = {}

    for (a, b), weight in pair_weights.items():
        if weight >= threshold:
            root_a = _find_root(merged, a)
            root_b = _find_root(merged, b)
            if root_a != root_b:
                merged[root_b] = root_a

    groups_by_root: dict[int, list[int]] = defaultdict(list)
    for arch_id, agents in agents_by_archetype.items():
        root = _find_root(merged, arch_id)
        groups_by_root[root].extend(agents)

    return list(groups_by_root.values())


def _compute_threshold(
    pair_weights: dict[tuple[int, int], int],
) -> float:
    """Compute median weight as clustering threshold."""
    weights = sorted(pair_weights.values())
    mid = len(weights) // 2
    return float(weights[mid]) if weights else 0.0


def _find_root(merged: dict[int, int], node: int) -> int:
    """Union-find root lookup with path compression."""
    while node in merged:
        parent = merged[node]
        if parent in merged:
            merged[node] = merged[parent]
        node = parent
    return node


def _compute_polarization(
    agents_by_archetype: dict[int, list[int]],
    graph: dict[int, dict[int, int]],
) -> float:
    """Compute polarization index (0-1) based on cross-archetype interaction."""
    agent_to_arch: dict[int, int] = {}
    for arch_id, agents in agents_by_archetype.items():
        for a in agents:
            agent_to_arch[a] = arch_id

    cross = 0
    total = 0
    for src, targets in graph.items():
        src_arch = agent_to_arch.get(src, -1)
        for tgt, weight in targets.items():
            tgt_arch = agent_to_arch.get(tgt, -1)
            total += weight
            if src_arch != tgt_arch:
                cross += weight

    if total == 0:
        return 0.0
    return 1.0 - (cross / total)


def _build_archetype_affinity(
    graph: dict[int, dict[int, int]],
    agent_to_arch: dict[int, int],
    arch_labels: dict[int, str],
) -> dict[str, list[str]]:
    """Map each archetype label to its most-interacted archetype labels."""
    pair_weights: dict[tuple[int, int], int] = defaultdict(int)
    for src, targets in graph.items():
        src_arch = agent_to_arch.get(src, -1)
        for tgt, weight in targets.items():
            tgt_arch = agent_to_arch.get(tgt, -1)
            if src_arch != tgt_arch:
                pair_weights[(src_arch, tgt_arch)] += weight

    affinity: dict[str, list[str]] = {}
    for (a, b), _ in sorted(
        pair_weights.items(), key=lambda x: -x[1],
    ):
        label_a = arch_labels.get(a, str(a))
        label_b = arch_labels.get(b, str(b))
        affinity.setdefault(label_a, [])
        if label_b not in affinity[label_a]:
            affinity[label_a].append(label_b)

    return affinity


class NetworkAnalyzer:
    """Analyzes social network structure and coalitions."""

    def analyze_coalitions(
        self, result: SimulationResult,
    ) -> CoalitionReport:
        """Identify archetype-based coalitions from interaction data."""
        follows = _load_follows(str(result.db_path))
        interactions = _load_interactions(str(result.db_path))
        graph = _build_interaction_graph(follows, interactions)

        agents_by_arch = _group_agents_by_archetype(result)
        arch_labels = _archetype_labels(result)
        agent_to_arch = _agent_to_archetype_map(result)

        groups = _find_coalitions(graph, agents_by_arch)
        affinity = _build_archetype_affinity(
            graph, agent_to_arch, arch_labels,
        )
        polarization = _compute_polarization(agents_by_arch, graph)

        return CoalitionReport(
            groups=groups,
            archetype_affinity=affinity,
            polarization_index=polarization,
        )

    def compute_influence_scores(
        self, result: SimulationResult,
    ) -> dict[int, float]:
        """Per-agent influence based on reposts, likes, followers."""
        interactions = _load_interactions(str(result.db_path))
        follows = _load_follows(str(result.db_path))

        scores: dict[int, float] = defaultdict(float)
        for _, target, action in interactions:
            weight = _action_weight(action)
            scores[target] += weight
        for _, followed in follows:
            scores[followed] += 1.0

        return dict(scores)

    def compute_archetype_influence(
        self, result: SimulationResult,
    ) -> dict[int, float]:
        """Aggregate influence scores by archetype."""
        agent_scores = self.compute_influence_scores(result)
        arch_scores: dict[int, float] = defaultdict(float)

        for agent in result.agents:
            arch_id = agent.archetype.archetype_id
            arch_scores[arch_id] += agent_scores.get(
                agent.agent_id, 0.0,
            )

        return dict(arch_scores)


def _action_weight(action_type: str) -> float:
    """Weight different interaction types for influence scoring."""
    weights: dict[str, float] = {
        "repost": 3.0,
        "like": 1.0,
        "reply": 2.0,
    }
    return weights.get(action_type, 1.0)


def _group_agents_by_archetype(
    result: SimulationResult,
) -> dict[int, list[int]]:
    """Group agent IDs by their archetype ID."""
    groups: dict[int, list[int]] = defaultdict(list)
    for agent in result.agents:
        groups[agent.archetype.archetype_id].append(agent.agent_id)
    return dict(groups)


def _archetype_labels(
    result: SimulationResult,
) -> dict[int, str]:
    """Map archetype ID to label."""
    return {a.archetype_id: a.label for a in result.archetypes}


def _agent_to_archetype_map(
    result: SimulationResult,
) -> dict[int, int]:
    """Map agent ID to archetype ID."""
    return {
        a.agent_id: a.archetype.archetype_id
        for a in result.agents
    }

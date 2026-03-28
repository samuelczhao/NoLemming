"""Tests for SentimentAnalyzer, SignalExtractor, NetworkAnalyzer."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from nolemming.analysis.networks import (
    CoalitionReport,
    NetworkAnalyzer,
)
from nolemming.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentTrajectory,
    _score_text,
)
from nolemming.analysis.signals import (
    PredictionSignals,
    SignalExtractor,
    _compute_momentum,
    _compute_volatility,
)
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


def _make_archetype(arch_id: int = 0, label: str = "test") -> NeuralArchetype:
    return NeuralArchetype(
        archetype_id=arch_id,
        centroid=np.zeros(8, dtype=np.float32),
        label=label,
        description="Test archetype",
        population_fraction=0.5,
        dominant_regions=["prefrontal_cortex"],
    )


def _make_simulation_result(
    tmp_path: Path,
    posts: list[tuple[int, int, str]] | None = None,
) -> SimulationResult:
    """Build a SimulationResult backed by a real SQLite DB with test data."""
    db_path = tmp_path / "sim.db"
    actions_path = tmp_path / "actions.jsonl"
    actions_path.write_text("")

    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
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
    """)

    arch0 = _make_archetype(0, "analytical")
    arch1 = _make_archetype(1, "fear-dominant")

    agents = [
        AgentProfile(
            agent_id=0, archetype=arch0, name="Agent 0",
            username="agent0", bio="", persona="", activity_level=0.5,
        ),
        AgentProfile(
            agent_id=1, archetype=arch1, name="Agent 1",
            username="agent1", bio="", persona="", activity_level=0.5,
        ),
    ]

    conn.execute("INSERT INTO users VALUES (0, 'agent0', 'Agent 0', 0)")
    conn.execute("INSERT INTO users VALUES (1, 'agent1', 'Agent 1', 1)")

    if posts is None:
        posts = [
            (0, 0, "bullish great growth strong"),
            (0, 1, "bearish crash decline weak"),
            (1, 0, "amazing outperform buy strong"),
            (1, 1, "sell disappointing terrible miss"),
            (2, 0, "great beat upgrade excellent"),
            (2, 1, "discussing the topic neutrally"),
        ]
    for round_num, user_id, content in posts:
        conn.execute(
            "INSERT INTO posts (user_id, round, content) VALUES (?, ?, ?)",
            (user_id, round_num, content),
        )
    conn.commit()
    conn.close()

    stimulus_path = tmp_path / "stim.txt"
    stimulus_path.write_text("test")
    stimulus = Stimulus(path=stimulus_path, stimulus_type=StimulusType.TEXT)

    config = SimulationConfig(
        stimulus=stimulus, n_agents=2, n_rounds=3,
        minutes_per_round=60, platform=Platform.TWITTER,
    )

    template = EngagementTemplate(
        archetype_engagement=np.array([0.5, 0.5], dtype=np.float32),
        similarity_decay=1.5,
        archetypes=[arch0, arch1],
    )

    return SimulationResult(
        simulation_id="test123",
        config=config,
        agents=agents,
        db_path=db_path,
        actions_path=actions_path,
        archetypes=[arch0, arch1],
        engagement_template=template,
    )


# --- Sentiment internals ---


class TestScoreText:
    def test_positive_text(self) -> None:
        score = _score_text("bullish great growth")
        assert score > 0

    def test_negative_text(self) -> None:
        score = _score_text("bearish crash decline")
        assert score < 0

    def test_neutral_text(self) -> None:
        score = _score_text("discussing the topic")
        assert score == 0.0

    def test_mixed_text(self) -> None:
        score = _score_text("bullish but crash")
        assert -1.0 <= score <= 1.0

    def test_empty_text(self) -> None:
        score = _score_text("")
        assert score == 0.0


# --- SentimentAnalyzer ---


class TestSentimentAnalyzer:
    def test_extract_trajectory(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        analyzer = SentimentAnalyzer()
        trajectory = analyzer.extract_trajectory(result)
        assert isinstance(trajectory, SentimentTrajectory)
        assert len(trajectory.timestamps) > 0
        assert len(trajectory.scores) == len(trajectory.timestamps)
        assert len(trajectory.volumes) == len(trajectory.timestamps)

    def test_trajectory_timestamps_ordered(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        trajectory = SentimentAnalyzer().extract_trajectory(result)
        assert trajectory.timestamps == sorted(trajectory.timestamps)

    def test_trajectory_volumes_positive(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        trajectory = SentimentAnalyzer().extract_trajectory(result)
        assert all(v > 0 for v in trajectory.volumes)

    def test_trajectory_scores_bounded(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        trajectory = SentimentAnalyzer().extract_trajectory(result)
        for score in trajectory.scores:
            assert -1.0 <= score <= 1.0

    def test_extract_per_archetype(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        per_arch = SentimentAnalyzer().extract_per_archetype(result)
        assert isinstance(per_arch, dict)
        for arch_id, trajectory in per_arch.items():
            assert isinstance(arch_id, int)
            assert isinstance(trajectory, SentimentTrajectory)

    def test_trajectory_to_dict(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        trajectory = SentimentAnalyzer().extract_trajectory(result)
        d = trajectory.to_dict()
        assert "timestamps" in d
        assert "scores" in d
        assert "volumes" in d


# --- Signal helper functions ---


class TestSignalHelpers:
    def test_compute_momentum_flat(self) -> None:
        assert _compute_momentum([0.5, 0.5, 0.5]) == 0.0

    def test_compute_momentum_rising(self) -> None:
        assert _compute_momentum([0.0, 0.5, 1.0]) > 0

    def test_compute_momentum_falling(self) -> None:
        assert _compute_momentum([1.0, 0.5, 0.0]) < 0

    def test_compute_momentum_single_point(self) -> None:
        assert _compute_momentum([0.5]) == 0.0

    def test_compute_momentum_empty(self) -> None:
        assert _compute_momentum([]) == 0.0

    def test_compute_volatility_constant(self) -> None:
        trajectory = SentimentTrajectory(
            timestamps=[0.0, 1.0, 2.0],
            scores=[0.5, 0.5, 0.5],
            volumes=[10, 10, 10],
        )
        assert _compute_volatility(trajectory) == 0.0

    def test_compute_volatility_varying(self) -> None:
        trajectory = SentimentTrajectory(
            timestamps=[0.0, 1.0, 2.0],
            scores=[0.0, 1.0, 0.0],
            volumes=[10, 10, 10],
        )
        assert _compute_volatility(trajectory) > 0


# --- SignalExtractor ---


class TestSignalExtractor:
    def test_extract_returns_signals(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        signals = SignalExtractor().extract(result)
        assert isinstance(signals, PredictionSignals)

    def test_signal_fields_present(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        signals = SignalExtractor().extract(result)
        assert isinstance(signals.sentiment_score, float)
        assert isinstance(signals.sentiment_momentum, float)
        assert isinstance(signals.consensus_strength, float)
        assert isinstance(signals.volatility_estimate, float)
        assert isinstance(signals.dominant_archetype, str)

    def test_consensus_strength_bounded(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        signals = SignalExtractor().extract(result)
        assert 0.0 <= signals.consensus_strength <= 1.0

    def test_narrative_keywords_are_strings(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        signals = SignalExtractor().extract(result)
        assert isinstance(signals.narrative_keywords, list)
        for kw in signals.narrative_keywords:
            assert isinstance(kw, str)

    def test_to_dict(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        signals = SignalExtractor().extract(result)
        d = signals.to_dict()
        assert "sentiment_score" in d
        assert "dominant_archetype" in d
        assert "narrative_keywords" in d


# --- NetworkAnalyzer ---


class TestNetworkAnalyzer:
    def test_analyze_coalitions_empty_interactions(self, tmp_path: Path) -> None:
        """With no follows/interactions, should still return a valid report."""
        result = _make_simulation_result(tmp_path)
        analyzer = NetworkAnalyzer()
        report = analyzer.analyze_coalitions(result)
        assert isinstance(report, CoalitionReport)
        assert isinstance(report.groups, list)
        assert isinstance(report.polarization_index, float)

    def test_compute_influence_scores_empty(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        scores = NetworkAnalyzer().compute_influence_scores(result)
        assert isinstance(scores, dict)

    def test_compute_archetype_influence_empty(self, tmp_path: Path) -> None:
        result = _make_simulation_result(tmp_path)
        scores = NetworkAnalyzer().compute_archetype_influence(result)
        assert isinstance(scores, dict)

    def test_coalitions_with_follows(self, tmp_path: Path) -> None:
        """Add follow relationships and verify coalitions still work."""
        result = _make_simulation_result(tmp_path)
        conn = sqlite3.connect(str(result.db_path))
        conn.execute("INSERT INTO follows VALUES (0, 1, 0)")
        conn.execute("INSERT INTO follows VALUES (1, 0, 0)")
        conn.commit()
        conn.close()

        analyzer = NetworkAnalyzer()
        report = analyzer.analyze_coalitions(result)
        assert isinstance(report, CoalitionReport)
        assert 0.0 <= report.polarization_index <= 1.0

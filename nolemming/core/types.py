"""Core type definitions for NoLemming."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class StimulusType(Enum):
    """Supported stimulus input types."""

    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(frozen=True)
class Stimulus:
    """A stimulus to be processed by the brain encoder.

    Represents any content that triggers neural responses:
    earnings calls, news articles, product announcements, etc.
    """

    path: Path
    stimulus_type: StimulusType
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> Stimulus:
        """Infer stimulus type from file extension."""
        path = Path(path)
        ext = path.suffix.lower()
        type_map: dict[str, StimulusType] = {
            ".txt": StimulusType.TEXT,
            ".md": StimulusType.TEXT,
            ".wav": StimulusType.AUDIO,
            ".mp3": StimulusType.AUDIO,
            ".flac": StimulusType.AUDIO,
            ".ogg": StimulusType.AUDIO,
            ".mp4": StimulusType.VIDEO,
            ".avi": StimulusType.VIDEO,
            ".mkv": StimulusType.VIDEO,
            ".mov": StimulusType.VIDEO,
            ".webm": StimulusType.VIDEO,
        }
        if ext not in type_map:
            msg = f"Unsupported file extension: {ext}"
            raise ValueError(msg)
        return cls(path=path, stimulus_type=type_map[ext])


FSAVERAGE5_VERTICES = 20_484


@dataclass(frozen=True)
class NeuralResponse:
    """Brain encoder output: predicted neural activations.

    Attributes:
        activations: shape (n_timesteps, n_vertices) predicted cortical activations
                     on the fsaverage5 mesh (20,484 vertices per timestep).
        vertex_count: number of cortical vertices (always 20,484 for fsaverage5).
        timestep_seconds: duration of each timestep in seconds.
    """

    activations: NDArray[np.float32]
    vertex_count: int = FSAVERAGE5_VERTICES
    timestep_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.activations.ndim != 2:
            msg = f"Expected 2D array (timesteps, vertices), got shape {self.activations.shape}"
            raise ValueError(msg)
        if self.activations.shape[1] != self.vertex_count:
            msg = (
                f"Expected {self.vertex_count} vertices, "
                f"got {self.activations.shape[1]}"
            )
            raise ValueError(msg)

    @property
    def n_timesteps(self) -> int:
        return self.activations.shape[0]

    def mean_activation(self) -> NDArray[np.float32]:
        """Time-averaged activation across all timesteps."""
        return np.mean(self.activations, axis=0).astype(np.float32)


@dataclass(frozen=True)
class CompressedResponse:
    """PCA-compressed neural response.

    Attributes:
        latent: shape (n_dims,) compressed representation of mean activation.
        n_dims: number of PCA dimensions.
    """

    latent: NDArray[np.float32]

    @property
    def n_dims(self) -> int:
        return self.latent.shape[0]


@dataclass(frozen=True)
class NeuralArchetype:
    """A cluster of similar neural response patterns.

    Represents a "type of brain" — how a subpopulation processes the stimulus.

    Attributes:
        archetype_id: unique identifier within this simulation.
        centroid: PCA-compressed centroid vector of this cluster.
        label: human-readable label (auto-generated from dominant brain regions).
        description: detailed behavioral profile for LLM persona generation.
        population_fraction: fraction of the population in this archetype (0-1).
        dominant_regions: brain regions with highest activation in this archetype.
    """

    archetype_id: int
    centroid: NDArray[np.float32]
    label: str
    description: str
    population_fraction: float
    dominant_regions: list[str] = field(default_factory=list)


@dataclass
class EngagementTemplate:
    """Pre-computed mapping from (archetype, content_similarity) to engagement probability.

    This is the key bridge between neural responses and simulation behavior.
    Built once per stimulus, used every simulation step.

    Attributes:
        archetype_engagement: shape (n_archetypes,) base engagement for each archetype.
        similarity_decay: how quickly engagement drops as content drifts from stimulus.
        archetypes: the neural archetypes this template was built for.
    """

    archetype_engagement: NDArray[np.float32]
    similarity_decay: float
    archetypes: list[NeuralArchetype]

    def get_engagement(
        self,
        archetype_id: int,
        content_similarity: float,
    ) -> float:
        """Compute engagement probability for an archetype given content similarity.

        Args:
            archetype_id: index into archetype list.
            content_similarity: cosine similarity between content and stimulus (0-1).

        Returns:
            Engagement probability (0-1).
        """
        base = float(self.archetype_engagement[archetype_id])
        decay = np.exp(-self.similarity_decay * (1.0 - content_similarity))
        return float(np.clip(base * decay, 0.0, 1.0))


class Platform(Enum):
    """Simulation platform types."""

    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class AgentProfile:
    """A neurally-grounded agent in the simulation.

    Attributes:
        agent_id: unique agent identifier.
        archetype: the neural archetype this agent belongs to.
        name: display name.
        username: platform username.
        bio: short profile bio.
        persona: detailed persona for LLM system prompt.
        activity_level: base activation probability per round (0-1).
        stance: initial stance toward the stimulus topic.
    """

    agent_id: int
    archetype: NeuralArchetype
    name: str
    username: str
    bio: str
    persona: str
    activity_level: float
    stance: str = "neutral"


@dataclass
class SimulationConfig:
    """Configuration for a NoLemming simulation run."""

    stimulus: Stimulus
    n_agents: int = 1000
    n_rounds: int = 168
    minutes_per_round: int = 60
    platform: Platform = Platform.TWITTER
    encoder_name: str = "mock"
    llm_model: str = "gpt-4o-mini"
    n_archetypes: int = 8
    pca_dims: int = 64
    seed: int = 42


@dataclass
class SimulationResult:
    """Output of a completed simulation.

    Attributes:
        simulation_id: unique run identifier.
        config: the config used for this run.
        agents: list of agent profiles used.
        db_path: path to the SQLite database with simulation data.
        actions_path: path to the JSONL action log.
        archetypes: neural archetypes used.
        engagement_template: engagement template used.
    """

    simulation_id: str
    config: SimulationConfig
    agents: list[AgentProfile]
    db_path: Path
    actions_path: Path
    archetypes: list[NeuralArchetype]
    engagement_template: EngagementTemplate

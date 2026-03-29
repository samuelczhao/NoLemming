"""NoLemming — Don't follow the herd. Predict it."""

__version__ = "0.1.0"

from nolemming.core.llm import LLMBackend, OpenAICompatibleBackend
from nolemming.core.pipeline import NoLemmingPipeline
from nolemming.core.types import (
    AgentProfile,
    EngagementTemplate,
    NeuralArchetype,
    NeuralResponse,
    SimulationConfig,
    SimulationResult,
    Stimulus,
)
from nolemming.encoders.base import BrainEncoder
from nolemming.encoders.registry import encoder_registry

__all__ = [
    "AgentProfile",
    "BrainEncoder",
    "EngagementTemplate",
    "LLMBackend",
    "NeuralArchetype",
    "NeuralResponse",
    "NoLemmingPipeline",
    "OpenAICompatibleBackend",
    "SimulationConfig",
    "SimulationResult",
    "Stimulus",
    "encoder_registry",
]

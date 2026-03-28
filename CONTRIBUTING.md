# Contributing to NoLemming

## Quick Start

```bash
git clone https://github.com/samuelczhao/nolemming.git
cd nolemming
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
nolemming/
├── core/           # Types, pipeline orchestrator, LLM abstraction
├── encoders/       # Brain encoder plugins (start here to add a new encoder)
├── mapping/        # Neural response → archetypes → engagement templates
├── agents/         # Agent population generation from archetypes
├── simulation/     # Social simulation engine (OASIS or fallback)
├── analysis/       # Post-simulation analysis (sentiment, networks, signals)
├── benchmark/      # 3-way comparison framework
├── web/            # FastAPI server
└── cli.py          # CLI entry point
```

**Data flow:** `Stimulus → Encoder → NeuralResponse → Compressor → Archetypes → EngagementTemplate → AgentFactory → SimulationEngine → Analysis → Report`

## Adding a Brain Encoder

1. Create `nolemming/encoders/my_encoder.py`
2. Subclass `BrainEncoder` from `nolemming.encoders.base`
3. Implement `encode(stimulus: Stimulus) -> NeuralResponse` and `name` property
4. Register in `nolemming/encoders/registry.py`

```python
from nolemming.encoders.base import BrainEncoder
from nolemming.core.types import NeuralResponse, Stimulus

class MyEncoder(BrainEncoder):
    @property
    def name(self) -> str:
        return "my_encoder"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        # Your encoding logic here
        # Must return shape (n_timesteps, 20484)
        ...
```

## Adding an LLM Backend

1. Subclass `LLMBackend` from `nolemming.core.llm`
2. Implement `generate()` and `model_name()`
3. Register with `llm_registry.register("name", MyBackend)`

## Code Standards

- `ruff check nolemming/` must pass
- All functions have type annotations
- No function longer than 20 lines
- Tests alongside code changes: `pytest tests/ -v`
- Run order: typecheck → tests → lint

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_mapping.py -v

# With coverage
pytest tests/ --cov=nolemming --cov-report=term-missing
```

## Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `Stimulus` | core/types.py | Input content (video/audio/text) |
| `NeuralResponse` | core/types.py | Brain encoder output (n_timesteps, 20484) |
| `CompressedResponse` | core/types.py | PCA-compressed neural data |
| `NeuralArchetype` | core/types.py | Cluster of similar neural profiles |
| `EngagementTemplate` | core/types.py | Maps (archetype, content_sim) → engagement |
| `AgentProfile` | core/types.py | A neurally-grounded simulation agent |
| `SimulationResult` | core/types.py | Output of a completed simulation |
| `LLMBackend` | core/llm.py | Abstract LLM interface |

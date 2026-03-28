# Farswarm

## Project Overview
Brain-encoded swarm social simulation engine combining Meta TRIBE v2 brain encoding with OASIS multi-agent social simulation.

## Tech Stack
- Python 3.11+
- OASIS/CAMEL-AI (optional, for full simulation)
- TRIBE v2 (optional, for real brain encoding)
- Any OpenAI SDK-compatible LLM
- FastAPI (web server)
- scikit-learn (PCA, clustering)

## Development Commands
```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Typecheck
.venv/bin/mypy farswarm/

# Tests
.venv/bin/pytest tests/ -v

# Lint
.venv/bin/ruff check farswarm/

# Format
.venv/bin/ruff format farswarm/
```

## Architecture Rules
- All components are pluggable: brain encoders, LLM backends, simulation platforms
- Neural responses are always (n_timesteps, 20484) on fsaverage5 mesh
- No Big Five personality mapping — use PCA + clustering into neural archetypes
- Engagement templates are pre-computed once per stimulus, not per-interaction
- OASIS is optional — fallback simulation generates synthetic data + SQLite DB
- LLM calls go through LLMBackend abstraction, never direct openai imports in modules

## Known Pitfalls
- Python 3.14: camel-ai/oasis packages don't support it yet. Use >=3.11, <=3.13 for full OASIS integration.
- TRIBE v2 outputs 20,484 vertices (fsaverage5), not 70K voxels as some docs claim.
- VoxelCompressor.fit_single() may produce fewer PCA dims than requested if n_timesteps < n_dims.
- Fallback simulation creates SQLite DB — analysis modules must handle missing tables gracefully.
- sentence-transformers is optional — EngagementDynamics uses fixed 0.5 similarity without it.

## Data Flow
```
Stimulus → BrainEncoder → NeuralResponse (20484 vertices)
    → VoxelCompressor (PCA) → compressed timesteps
    → ArchetypeClusterer → NeuralArchetype list
    → EngagementTemplateBuilder → EngagementTemplate
    → AgentFactory → AgentProfile list
    → SimulationEngine → SimulationResult (SQLite + JSONL)
    → SentimentAnalyzer + SignalExtractor + NetworkAnalyzer
    → ReportGenerator → PredictionReport
```

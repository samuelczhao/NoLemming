# NoLemming

**Don't follow the herd. Predict it.**

NoLemming is a brain-encoded swarm social simulation engine. It takes any stimulus — an earnings call, a news article, a product launch video — predicts how human brains would process it, spawns thousands of neurally-grounded agents, and simulates how they'd collectively react on social media.

The result: a prediction of public sentiment *before* the public reacts.

Built on [Meta TRIBE v2](https://github.com/facebookresearch/tribev2) (brain encoding) + [OASIS](https://github.com/camel-ai/oasis) (social simulation). First project to combine computational neuroscience with multi-agent swarm intelligence.

---

## Why This Exists

Every multi-agent simulation today generates agent personalities from LLM heuristics — generic, interchangeable, ungrounded in reality.

NoLemming is different. It generates agents from **predicted brain activation patterns**. Different stimuli activate different neural circuits, producing fundamentally different population dynamics:

| Stimulus | Dominant Neural Response | Agent Population | Predicted Outcome |
|---|---|---|---|
| Alarming earnings miss | Amygdala (fear) | Fear-dominant agents | Fast panic spread, negative cascade |
| Breakthrough product | Reward circuits (NAcc) | Reward-seeking agents | Sustained positive engagement |
| Complex policy doc | Prefrontal cortex | Analytical agents | Slow, deliberative discourse |
| Controversial CEO | Mixed amygdala + reward | Polarized population | Tribal clustering, high volatility |

This isn't metaphor — the agents' behavior is directly parameterized by predicted cortical activation patterns.

---

## How It Works

```
Stimulus (video / audio / text)
        |
        v
  Brain Encoder (TRIBE v2 / pluggable)
        |
        v
  20,484 cortical vertex activations
        |
   -----+-----
   |         |
   v         v
  PCA     Engagement
  (50d)   Template
   |         |
   v         |
  K-means    |
  clustering |
   |         |
   v         v
  Neural Archetypes + Engagement Map
  ("fear-dominant", "reward-seeking", "analytical"...)
        |
        v
  Agent Population (1000+)
  Each grounded in a neural archetype
        |
        v
  Social Simulation (Twitter/Reddit-like)
  LLM-powered agents post, share, debate
        |
        v
  Prediction Report
  Sentiment trajectory | Archetype dynamics
  Narrative keywords   | Quantitative signals
```

---

## Quick Start

```bash
pip install nolemming

# Run with mock brain encoder (no GPU needed)
nolemming run earnings_call.txt --agents 500 --encoder mock

# Use any LLM backend
nolemming run stimulus.txt --model llama3 --base-url http://localhost:11434/v1
nolemming run stimulus.txt --model gpt-4o --api-key sk-...

# A/B test content variants
nolemming compare variant_a.txt variant_b.txt --agents 1000

# 3-way benchmark (neural vs vanilla vs random)
nolemming benchmark aapl_q4_2025
```

### Python API

```python
import asyncio
from nolemming.core.pipeline import NoLemmingPipeline
from nolemming.core.llm import OpenAICompatibleBackend

# Works with any OpenAI-compatible API (Ollama, vLLM, Together, Groq...)
llm = OpenAICompatibleBackend(
    model="qwen2.5:3b",
    base_url="http://localhost:11434/v1",
)

pipeline = NoLemmingPipeline(llm=llm)
result = asyncio.run(pipeline.run("earnings_call.txt", n_agents=1000))

report = result["analysis"]["report"]
print(report.to_markdown())
```

---

## Architecture: Why Hybrid (Architecture C)

We evaluated three architectures and chose the one that maximizes signal preservation:

| Architecture | Signal Preserved | Compute Cost | Why Rejected/Selected |
|---|---|---|---|
| A: Neural personality init | ~3% | Low | Big Five mapping destroys 95% of signal |
| B: Continuous neural scoring | ~100% theoretical | 1M forward passes | Infeasible + out-of-distribution on text |
| **C: Hybrid (selected)** | **~15-25%** | **One-time** | **PCA + clustering preserves signal, engagement templates modulate behavior** |

Architecture C runs the brain encoder **once** on the original stimulus, skips the lossy Big Five mapping entirely, and uses PCA + clustering to create **neural archetypes** directly from activation data.

---

## Plug-and-Play

Every component is swappable:

### Brain Encoders
```python
from nolemming.encoders.registry import encoder_registry

encoder_registry.get("mock")       # Synthetic (testing)
encoder_registry.get("tribe_v2")   # Meta TRIBE v2

# Add your own
from nolemming.encoders.base import BrainEncoder
class MyEncoder(BrainEncoder): ...
encoder_registry.register("my_encoder", MyEncoder)
```

### LLM Backends
```python
from nolemming.core.llm import OpenAICompatibleBackend

OpenAICompatibleBackend(model="gpt-4o")                                          # OpenAI
OpenAICompatibleBackend(model="qwen2.5:3b", base_url="http://localhost:11434/v1") # Ollama
OpenAICompatibleBackend(model="meta-llama/Llama-3-70b", base_url="https://api.together.xyz/v1")  # Together
```

### Environment Variables
```bash
export NOLEMMING_LLM_MODEL=qwen2.5:3b
export NOLEMMING_LLM_BASE_URL=http://localhost:11434/v1
export NOLEMMING_ENCODER_NAME=mock
```

---

## Use Cases

**Market Sentiment Prediction** — Feed an earnings call, predict the 72-hour social sentiment trajectory.

**Content A/B Testing** — Compare how different content variants spread through a neurally-diverse population. No real users needed.

**Misinformation Modeling** — Simulate how misinformation propagates. Identify which neural archetypes are most susceptible.

**Regime Change Detection** — Track shifts in population-level neural response distributions across sequential stimuli.

---

## Benchmark Results

3-way comparison across 5 real Q4 2025 earnings calls. Sentiment correlation with actual post-earnings social media sentiment (higher = better):

| Event | Neural | Vanilla | Random | Winner |
|-------|--------|---------|--------|--------|
| AAPL | +0.034 | **+0.653** | +0.034 | vanilla |
| GOOGL | **-0.544** | -0.679 | -0.976 | neural |
| META | **+0.575** | -0.958 | +0.000 | neural |
| NVDA | -0.768 | +0.000 | **+0.137** | random |
| TSLA | **+0.078** | +0.000 | -0.533 | neural |
| **Wins** | **3/5** | 1/5 | 1/5 | |

Standout: META earnings — neural agents tracked real sentiment direction (+0.575) while vanilla agents predicted the exact opposite (-0.958).

Run with Ollama/qwen2.5:3b (3B params, runs on CPU). Better LLM = better results.

```bash
# Run the benchmark yourself
python scripts/run_phase0.py

# Or benchmark a single event
nolemming benchmark aapl_q4_2025
```

---

## Project Structure

```
nolemming/
├── core/           # Pipeline, types, LLM abstraction
├── encoders/       # Brain encoder plugins (mock, TRIBE v2, custom)
├── mapping/        # PCA compression, archetype clustering, engagement templates
├── agents/         # Agent factory, profile generation
├── simulation/     # Social simulation engine (OASIS + LLM fallback)
├── analysis/       # Sentiment, networks, signals, reports
├── benchmark/      # 3-way comparison framework + ground truth data
├── web/            # FastAPI server
└── cli.py          # CLI entry point
```

## Requirements

- Python >= 3.11
- Any OpenAI SDK-compatible LLM (Ollama, OpenAI, Together, vLLM, Groq...)
- Optional: TRIBE v2 (GPU, CC BY-NC) | OASIS/CAMEL-AI (full social sim)

## License

MIT (NoLemming code). Brain encoder adapters carry their own licenses.

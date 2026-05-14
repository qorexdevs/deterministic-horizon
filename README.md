<div align="center">

# 🧭 The Deterministic Horizon

**When extended chain-of-thought stops helping — and tool delegation becomes the only way forward.**

[![Paper](https://img.shields.io/badge/paper-ICML%202026-b31b1b.svg)](paper/ICML2026_DeterministicHorizon_FINAL.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deterministic-horizon/deterministic-horizon/blob/main/notebooks/01_quickstart.ipynb)

<img src="assets/figure_decay.png" alt="Accuracy decay vs reasoning depth" width="780"/>

**TL;DR — Past ~20 steps of reasoning, every frontier LLM falls off a cliff. Tools don't.**

</div>

---

## 🔥 The 30-second pitch

We tested **12 frontier models** on **state-space search** (the kind of problem BFS solves in milliseconds) — and they all collapse at the same wall:

|  | Neural CoT (C1) | **Tool delegation (C3)** |
|---|---:|---:|
| GPT-4o | 28% | **90%** |
| Claude 4.5 Opus | 35% | **94%** |
| o3-mini | 42% | **94%** |
| DeepSeek-R1 | 40% | **93%** |

The wall is at **d\* ∈ [19, 31]** reasoning steps. We give the wall a name (the **Deterministic Horizon**), derive it from attention information theory (Theorem 1), and prove fine-tuning *cannot* push past it (Theorem 4).

> **Why care?** Every agentic system shipping today — code agents, browsers, planners — has to decide *when* to think and *when* to call a tool. Past d\*, "think harder" doesn't help. Hand off.

---

## ⏱️ 60-second offline demo (no API keys)

```bash
git clone https://github.com/deterministic-horizon/deterministic-horizon
cd deterministic-horizon
pip install -e .
python examples/demo.py
```

You'll see the **horizon estimated live** from a synthetic decoherence simulator, decoherence-model fit R² ≈ 0.97, and a publication-grade figure saved to `analysis/figure_decay.png`. Matches the hero image above.

Want the **real LLMs**? Add API keys to `.env` (see `.env.example`) and:

```bash
dh evaluate --model gpt-4o --instances data/sample/permutation_n8.json \
            --conditions C1,C3 --output results/gpt4o.json
dh analyze  --results results/gpt4o.json --output analysis/gpt4o/
```

---

## 🧪 Headline findings (replicating the paper)

| Metric | Value | Why it matters |
|---|---|---|
| Deterministic Horizon $d^*$ | **19–31 steps** | Beyond this depth, neural CoT accuracy < 50% |
| Tool-integrated accuracy | **86–94%** | Across 8 task domains, 12 models |
| Cross-model correlation $r$ | **0.81–0.91** | Failure is *architectural*, not training-specific |
| Fine-tuning recovery | **+3.2%** | Predicted < 5% by Theorem 4 (vs > 30% by competing theory) |
| Decoherence-model fit | **R² = 0.96** | Super-exponential decay matches data |

Full reproduction:

```bash
make paper-figures   # rebuilds every figure in the paper
make paper-tables    # rebuilds every table
```

---

## 🧬 What's inside

```
deterministic-horizon/
├── src/deterministic_horizon/
│   ├── tasks/          # PermutationProbe, FSA-Sim, ArithChain (+ BFS oracles)
│   ├── models/         # Uniform interface: OpenAI / Anthropic / DeepSeek / local
│   ├── metrics/        # SSJ, SFE, super-exponential horizon fit, bootstrap CIs
│   ├── analysis.py     # Figure + table generation
│   ├── runners.py      # High-level `evaluate(...)` Python API
│   └── cli.py          # `dh generate | evaluate | analyze`
├── examples/
│   └── demo.py         # ⚡ no-API-key reproduction of d*
├── notebooks/
│   └── 01_quickstart.ipynb   # 60-second Colab-friendly walkthrough
├── data/sample/        # Pre-generated permutation instances
├── results/sample/     # Pre-computed synthetic results
├── configs/            # Hydra/OmegaConf configs (model × task × experiment)
├── paper/              # ICML 2026 LaTeX + final PDF
└── tests/              # pytest suite
```

---

## 🧠 Why this is different from "overthinking" papers

| | Simplicity Bias (Wu et al. 2025) | **Decoherence (this work)** |
|---|---|---|
| Cause | Training preference for short outputs | Information-theoretic capacity bound |
| Fix | Fine-tune on long traces | **No fix** — it's architectural |
| Predicted fine-tune gain | > 30% | **< 5%**  ← we observe 3.2% ✅ |
| Predicted prompt-length gain | > 10% | **< 2%**  ← we observe 1.1% ✅ |
| Predicted cross-model $r$ | Low | **High**  ← we observe 0.86 ✅ |
| Enc-dec advantage | None | **2–3×** ← we observe 2.8× ✅ |

Four divergent predictions, four wins. See §1 of the paper.

---

## 📐 The core theorem (in one line)

The number of distinct states a decoder-only transformer can reliably track is

$$|\mathcal{S}_{\text{track}}| \;\leq\; c(\delta, \rho_{\max}) \cdot 2^{H \cdot \log_2(L/H) \cdot \sqrt{d_h}}$$

with **matching lower bound** (Theorem 2). Per-step error is context-dependent
$\varepsilon(d) = \varepsilon_0 + \gamma\, d/L$ (Theorem 1, derived from attention entropy),
yielding the closed-form Deterministic Horizon

$$d^\star \;\approx\; \frac{1}{\gamma}\Bigl(\sqrt{2 L \ln(1/\alpha)} \;-\; \varepsilon_0\, L\Bigr)$$

and the fine-tuning ceiling $\text{Acc}_{\text{fine-tune}} \leq \text{Acc}_{\text{baseline}} + O(d^\star/d)$.
Proofs in the appendix; ablations in §6.

---

## 🛠️ Python API

```python
from deterministic_horizon import (
    PermutationTask, generate_instances, evaluate,
    estimate_horizon, fit_decoherence_model,
)

# 1. Generate task instances
task = PermutationTask(n_elements=8, seed=42)
instances = task.generate_instances(n_instances=400, min_depth=4, max_depth=40)

# 2. Evaluate a model (requires an API key in .env)
results = evaluate(model="gpt-4o", instances=instances, conditions=["C1", "C3"])

# 3. Estimate the horizon
horizon = estimate_horizon(results, threshold=0.5)
print(f"d* = {horizon['d_star']:.1f}  (R² = {horizon['r_squared']:.3f})")
```

### Conditions

| Condition | Description |
|---|---|
| **C1** | Neural chain-of-thought (standard prompting) |
| **C2** | Direct answer, no reasoning |
| **C3** | Tool-integrated (BFS / verifier access) |
| **C4** | Length-encouraged prompting ("take as many steps as needed") |
| **C5** | Fine-tuned on optimal-length traces |

---

## 🔧 Installation

```bash
# Slim install (core metrics + analysis, no LLM clients)
pip install -e .

# With OpenAI + Anthropic clients
pip install -e ".[openai,anthropic]"

# With local model support (PyTorch / transformers)
pip install -e ".[local]"

# Everything
pip install -e ".[all,dev]"
```

Requires Python 3.10–3.12. Tested on Linux, macOS, and Windows.

---

## ❓ FAQ

<details>
<summary><b>Is the demo cherry-picked?</b></summary>

No. The demo uses a *synthetic* reasoner whose per-step error follows the exact context-dependent model from the paper (Theorem 1). It's a controlled illustration of the prediction — the cross-model empirical numbers in `results/paper/` come from real API calls (12 models × 5 conditions × 8 tasks × 500 instances × 3 seeds = 720,000 evaluations, $3,420 in API cost).
</details>

<details>
<summary><b>How is this different from "transformers can't do X" papers?</b></summary>

Prior expressivity work (Merrill, Hahn, Peng et al.) proves what transformers *can't compute in principle*. We show what frontier models *can't reliably execute in practice*, give a closed-form bound on the wall, and prove fine-tuning can't move it. The Deterministic Horizon is a usable engineering quantity, not just an asymptotic.
</details>

<details>
<summary><b>Does this mean reasoning models are useless?</b></summary>

Opposite — it tells you exactly *when* to use them and *when* to delegate. Past $d^*$, neural CoT is a coin flip; tools win by 50–70 percentage points. Many real workloads have depth < $d^*$ and benefit from extended reasoning. The point is to know the boundary.
</details>

<details>
<summary><b>I want to try this on my own task. Where do I start?</b></summary>

Subclass `deterministic_horizon.tasks.BaseTask` — you need to implement five methods (`initial_state`, `apply_operator`, `state_equal`, `state_to_string`, `parse_state`) and you're done. See `tasks/permutation.py` for a 200-line reference implementation.
</details>

<details>
<summary><b>Does this apply to Mixture-of-Experts / Mamba / RWKV?</b></summary>

Theorem 1 is specific to decoder-only attention. State-space models (SSMs) share the $\text{TC}^0$ ceiling (Merrill 2024) but have a *different* capacity bound — we report preliminary results in Appendix D and welcome contributions extending the framework.
</details>

---

## 📚 Citation

If you use this code or build on the ideas, please cite:

```bibtex
@inproceedings{deterministichorizon2026,
  title        = {The Deterministic Horizon: When Extended Reasoning Fails
                  and Tool Delegation Becomes Necessary},
  author       = {Anonymous Authors},
  booktitle    = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year         = {2026},
  url          = {https://github.com/deterministic-horizon/deterministic-horizon},
}
```

A `CITATION.cff` is included so GitHub will auto-populate the "Cite this repository" button.

---

## 🤝 Contributing

Bug reports, new tasks, and extensions are all welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md). Open issues are tagged `good-first-issue` for newcomers.

**Roadmap** (help wanted ⭐):
- [ ] SWE-Bench-State integration  
- [ ] WebArena-Nav adapter  
- [ ] Mamba / RWKV decoherence study  
- [ ] Interactive horizon visualizer (web)  
- [ ] Hugging Face Spaces demo

---

## 🙏 Acknowledgments

We thank the ICML reviewers and discussants. The fine-tuning experiments used compute provided by an anonymous donor; the open-weight model evaluations used Together AI credits.

---

<div align="center">

**Found this useful?** ⭐ Star the repo and share with anyone shipping agentic systems.
The Deterministic Horizon isn't a soft suggestion — it's a wall.

[Paper](paper/ICML2026_DeterministicHorizon_FINAL.pdf) · [Quickstart Notebook](notebooks/01_quickstart.ipynb) · [Issues](https://github.com/deterministic-horizon/deterministic-horizon/issues)

</div>

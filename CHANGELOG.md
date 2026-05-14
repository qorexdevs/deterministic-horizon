# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-14

Initial public release accompanying the ICML 2026 paper *"The Deterministic
Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary"*.

### Added
- Five experimental conditions (C1–C5): neural CoT, direct, tool-integrated, length-encouraged, fine-tuned
- Three task families: `PermutationTask`, `FSATask`, `ArithmeticTask` — each with a BFS oracle
- Unified model interface across OpenAI / Anthropic / DeepSeek / local HF models
- State-Space Jaccard (SSJ) metric with precision/recall decomposition
- Step-to-First-Error (SFE) metric
- Super-exponential decay fit (Theorem 1) and bootstrap CI for the Deterministic Horizon $d^*$
- Cross-model correlation analysis
- `dh` CLI (`generate`, `evaluate`, `analyze`, `train`)
- Offline demo (`examples/demo.py`) — no API keys required, ~10 seconds runtime
- Quickstart notebook (`notebooks/01_quickstart.ipynb`)
- Pre-generated sample instances + sample results for one-click reproduction
- Continuous integration (GitHub Actions) on Linux + macOS + Windows
- Optional dependency groups: `[openai]`, `[anthropic]`, `[local]`, `[viz]`, `[dev]`, `[all]`

### Reproducibility
- All experiments seeded with `{42, 2024, 2025}`
- Wilson score CIs for binomial proportions
- Holm–Bonferroni multiple-comparison correction
- TOST equivalence testing for null-result claims

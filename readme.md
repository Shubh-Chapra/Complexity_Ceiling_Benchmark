```markdown
# Complexity Ceiling Benchmark (CCB)

**A depth-controlled diagnostic for sequential reasoning in large language models.**

> *Paper:* [The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling](https://arxiv.org/abs/XXXX.XXXXX) — ICML 2026  
> *Institution:* BITS Pilani, Pilani Campus

---

## Overview

Standard reasoning benchmarks entangle semantic difficulty with reasoning depth, making it unclear *when* along a multi-step chain a model actually fails. CCB fixes all semantic parameters and varies only the number of required sequential reasoning steps **N** ($5 \rightarrow 50$) across three structurally distinct domains ($n=40$ independent trials per depth cell, each generated from a distinct random seed).

| Domain | Type | Primary Failure Mode |
|--------|------|----------------------|
| **D1 — Alien Grid** | Grounded spatial state-tracking ($3\times3$ grid) | Per-step state retention decay compounding |
| **D2 — Symbolic Tracking** | Abstract register/pointer chasing | Task constraint violations (illegal re-assignment) |
| **D3 — Social Logic** | Transitive relational graph inference | Cascade error propagation across reachable nodes |

### Key Findings

- **Geometric Depth Decay:** Across well-behaved domains, accuracy fits a geometric decay model $P(\text{correct} \mid N) = p_d^N$ ($R^2 > 0.90$ on D1/D2), yielding a single interpretable per-step retention parameter $p_d$.
- **Lucky-Guess Inflations:** Over the entire suite, **14.5% of all correct outputs** are verified Trace First Branch Correct (TFBC) events: the final answer is correct despite the intermediate reasoning traces diverging from ground truth. Rates reach 56%–62% on transitive relational tasks.
- **Early-Chain Cascade Collapse:** All evaluated models suffer near-universal collapse to near-zero accuracy on D3 beyond depth $N=5$, independent of general capability tier (mean divergence step $k^* \in [2.88, 4.30]$).
- **State Supervision Trade-offs:** Forcing intermediate state-tracking verbosity offers a $+10.0\text{ pp}$ accuracy uptick on D1 spatial tasks at $N=25$, but shifts the error taxonomy drastically—surfacing a 35.0% grid-integrity constraint failure profile. Conversely, it provides zero structural benefit on D3 graph paths ($p=1.0000$, $n=20$).

---

## Repository Structure


```

complexity-ceiling-benchmark/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── d1_alien_grid/
│   ├── alien_grid_main.py       # Core dataset generator, evaluator, and parser
│   ├── run_ablation.py          # Dedicated script for D1 spatial verbosity tests
│   └── results/
│       ├── benchmark_main_20260425_041745.json         # Stationary evaluations (n=400)
│       ├── summary_smart_resume_grid.txt               # Unified text summary tables
│       ├── combined_smart_resume_grid.json             # Flat merged tracking database
│       └── parser_validation_grid.json                 # Human validation sample set
│
├── d2_symbolic_tracking/
│   ├── symbolic_tracking_main.py  # Register-file tracking logic and parsing anchors
│   └── results/
│       └── ... (Symmetrical footprint to D1)
│
└── d3_social_logic/
├── social_logic_main.py     # Transitive graph engine + prompt sensitivity + graph ablations
└── results/
├── ... (Symmetrical footprint to D1)
├── results_ablation_smart_resume_fixed.json    # D3 baseline intervention logs
├── results_prompt_sens_d3.json                 # System prompt Var A/B/C metrics
├── compute_kappa.py                             # Cohen's Kappa calculator
└── compute_mcnemar.py                           # Contingency significance testing

```

---

## Installation

### 1. Clone the repository
```bash
git clone [https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark.git](https://github.com/Shubh-Chapra/Complexity_Ceiling_Benchmark.git)
cd Complexity_Ceiling_Benchmark

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure credentials

```bash
export OPENROUTER_API_KEY="sk-or-..."

```

All models are accessed at strict temperature $T=0$ via the OpenRouter routing layer using the OpenAI-compatible Python SDK.

---

## Running Evaluations

Each domain script is fully self-contained. Running a master execution script automatically scans for partial data outputs, runs missing seeds, executes necessary retries, fits the decay models, and compiles text reports:

```bash
# Evaluate Domain 1 — Alien Grid (Spatial State Tracking)
python d1_alien_grid/alien_grid_main.py

# Evaluate Domain 2 — Symbolic Tracking (Abstract Register Management)
python d2_symbolic_tracking/symbolic_tracking_main.py

# Evaluate Domain 3 — Social Logic & Integrated Interventions
python d3_social_logic/social_logic_main.py

```

### Reproducing Domain-Specific Ablations

```bash
# Run the standalone Domain 1 Spatial Verbosity Ablation matrix
python d1_alien_grid/run_ablation.py

```

*(Note: Domain 3 Graph Prompt Sensitivity and Graph Verbosity Ablations are directly triggered within the execution sequence of `d3_social_logic/social_logic_main.py`)*

---

## Statistical Analysis

To calculate inter-annotator reliability metrics ($\kappa$) or exact contingency significance partitions, pass the target evaluation file paths directly as terminal arguments:

```bash
# Inter-Annotator Agreement (Cohen's Kappa)
python d3_social_logic/results/compute_kappa.py d1_alien_grid/results/parser_validation_grid.json

# Significance Testing (McNemar Contingency Table)
python d3_social_logic/results/compute_mcnemar.py d3_social_logic/results/results_prompt_sens_d3.json

```

---

## Models Evaluated

All baseline instances were evaluated under deterministic decoding at temperature $T = 0$.

| Model Instance | OpenRouter Model Endpoint Identifier |
| --- | --- |
| Claude 3.7 Sonnet | `anthropic/claude-3.7-sonnet` |
| Gemini 2.0 Flash | `google/gemini-2.0-flash-001` |
| DeepSeek Chat | `deepseek/deepseek-chat` |
| GPT-4o-mini | `openai/gpt-4o-mini` |
| LLaMA 3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct` |

*Note: Reasoning-specialized architectures utilizing internal reinforcement scratchpads or test-time compute loops (`o1`/`o3`, `DeepSeek-R1`) are out of scope for this version due to API access limits at the time of initial manuscript submission.*

---

## Metric Reference Taxonomy

* **$p_d$ (Per-Step State Retention):** Geometric performance decay parameter extracted using absolute joint maximum likelihood estimation (MLE) across all evaluated depths.
* **$H_{0.5}$ (Effective Success Horizon):** The discrete step threshold matching an exactly bounded 50% target accuracy boundary: $\ln(0.5) / \ln(p_d)$.
* **$k^*$ (First-Branch Mismatch Step):** The exact, deterministic position in the reasoning trace where model step output diverges for the first time from ground truth.
* **TFBC (Trace First Branch Correct Rate):** Structural tracking event rate capturing context contamination where the final token block is correct despite faulty intermediate states.

---

## Citation

```bibtex
@inproceedings{chapra2026complexity,
  title     = {The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling},
  author    = {Chapra, Shubh and Sinha, Yash and Kumar, Dhruv},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  series    = {PMLR 306},
  publisher = {PMLR}
}

```

---

## License

* **Source Evaluation Code Engine:** [MIT License](https://www.google.com/search?q=LICENSE)
* **Benchmark Data Inventories** (completions, static configuration seeds): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

```

```

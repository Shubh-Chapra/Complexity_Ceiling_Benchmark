# Complexity Ceiling Benchmark (CCB)

**A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling.**

 *Paper:* **The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling**
 *Status: **Accepted at ICML CTB Workshop 2026**
 *Institution:* BITS Pilani, Pilani Campus

---

# Overview

Standard reasoning benchmarks often entangle semantic difficulty with reasoning depth, making it unclear *where* along a multi-step reasoning chain a model actually fails.

The Complexity Ceiling Benchmark (CCB) isolates sequential reasoning depth by holding semantic parameters fixed while varying only the number of required reasoning steps:

\[
N \in \{5,10,\dots,50\}
\]

across three structurally distinct reasoning domains, with \(n=40\) independent trials per depth cell.

| Domain | Type | Primary Failure Mode |
|---|---|---|
| **D1 — Alien Grid** | Spatial state tracking | Per-step retention decay |
| **D2 — Symbolic Tracking** | Abstract symbolic memory | Constraint violations |
| **D3 — Social Logic** | Transitive relational inference | Cascade error propagation |

---

# Key Findings

- Accuracy follows a geometric depth-decay relationship:

\[
P(\text{correct} \mid N)=p_d^N
\]

with \(R^2 > 0.90\) on D1/D2.

- **14.5% of all correct outputs** are TFBC (“lucky-guess”) events: the final answer is correct despite divergence in intermediate reasoning traces.

- All evaluated models exhibit sharp degradation toward near-zero accuracy on D3 beyond \(N>5\).

- Preliminary verbosity ablations suggest that increased intermediate supervision may partially mitigate retention failures on D1 while simultaneously increasing structural constraint violations.

---

# Repository Structure

```text
complexity-ceiling-benchmark/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── d1_alien_grid/
│   ├── alien_grid_main.py
│   ├── parser.py 
│   ├── run_ablation.py
│   └── results/
│       ├── benchmark_main.json
│       ├── combined_results.json
│       ├── summary.json
│       ├── parser_validation.json
│       ├── results_ablation.json
│       ├── compute_kappa.py
│       └── compute_mcnemar.py
│
├── d2_symbolic_tracking/
│   ├── symbolic_tracking_main.py
│   └── results/
│       ├── benchmark_main.json
│       ├── combined_results.json
│       ├── summary.json
│       └── parser_validation.json
│
└── d3_social_logic/
    ├── social_logic_main.py
    └── results/
        ├── benchmark_main.json
        ├── combined_results.json
        ├── summary.json
        ├── parser_validation.json
        ├── results_ablation.json
        └── results_prompt_sensitivity.json
``

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<username>/complexity-ceiling-benchmark.git
cd complexity-ceiling-benchmark
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure API credentials

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

All models are accessed through the OpenRouter API using deterministic decoding at temperature (T=0).

---

# Running Evaluations

## Domain 1 — Alien Grid

```bash
python d1_alien_grid/alien_grid_main.py
```

## Domain 2 — Symbolic Tracking

```bash
python d2_symbolic_tracking/symbolic_tracking_main.py
```

## Domain 3 — Social Logic

```bash
python d3_social_logic/social_logic_main.py
```

Each script:

* generates benchmark instances,
* evaluates all configured models,
* computes summary statistics,
* exports merged outputs and evaluation reports.

---

# Ablations

## D1 Spatial Verbosity Ablation

```bash
python d1_alien_grid/run_ablation.py
```

This reproduces:

* verbose intermediate-state supervision,
* D1 constraint-failure analysis,
* paired evaluation statistics.

## D3 Prompt Sensitivity + Verbosity Ablations

D3 ablations are integrated directly into:

```bash
python d3_social_logic/social_logic_main.py
```

This includes:

* standard vs verbose prompting,
* prompt variants (Var A/B/C),
* graph-state intervention experiments,
* McNemar paired significance testing.

---

# Statistical Analysis

## Cohen's κ

```bash
python d1_alien_grid/results/compute_kappa.py
```

## McNemar Significance Test

```bash
python d1_alien_grid/results/compute_mcnemar.py
```

Repeat analogously for D2 and D3.

---

# Models Evaluated

| Model                  | OpenRouter Endpoint                 |
| ---------------------- | ----------------------------------- |
| Claude 3.7 Sonnet      | `anthropic/claude-3.7-sonnet`       |
| Gemini 2.0 Flash       | `google/gemini-2.0-flash-001`       |
| DeepSeek Chat          | `deepseek/deepseek-chat`            |
| GPT-4o-mini            | `openai/gpt-4o-mini`                |
| LLaMA 3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct` |

Reasoning-specialized architectures (e.g. `o1/o3`, `DeepSeek-R1`) are currently out of scope for this release.

---

# Metrics

| Metric    | Description                                                   |
| --------- | ------------------------------------------------------------- |
| \(p_d\)     | Per-step retention parameter                                  |
| \(H_{0.5}\) | Effective half-accuracy horizon                               |
| \(k^*\)     | First divergence step                                         |
| TFBC      | Correct final answer despite incorrect intermediate reasoning |

---

# Reproducibility

The repository includes:

* deterministic benchmark generators,
* fixed evaluation seeds,
* parser validation datasets,
* merged model outputs,
* bootstrap confidence interval utilities,
* inter-annotator agreement scripts.

This enables exact reproduction of all reported benchmark statistics.

---

# Runtime Notes

* Full benchmark execution requires API access.
* Runtime and cost depend on provider-side latency.
* Deep traces at high depths may require elevated token budgets.

---

# Known Limitations

* The geometric decay model assumes approximately independent per-step failures.
* D3 evaluations are limited to vanilla autoregressive inference.
* The regex parser prioritises precision over recall.
* Reasoning-specialized models are not currently evaluated.

---

# Citation

```bibtex
@article{ccb2026,
  title={The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling},
  author={Anonymous},
  year={2026}
}
```

---

# License

* Code: MIT License
* Benchmark data and evaluation outputs: CC BY 4.0

```
```

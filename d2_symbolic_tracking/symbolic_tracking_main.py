"""
Resume or finalize a D2 Symbolic Tracking benchmark run across multiple models.

Scans ./results/ for any existing result files, merges partial runs by seed,
re-runs seeds that previously failed with API errors, and runs missing seeds
from scratch. Writes per-model final files, a summary table, p_d fits, and a
parser validation sample.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python symbolic_tracking_main.py
"""

import json
import time
import os
import re
import random
import math
import glob
import copy
import numpy as np
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from scipy.stats import beta as beta_dist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "./results_d2"

ALL_MODELS = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-3.3-70b-instruct",
]

DEPTH_LEVELS   = list(range(5, 55, 5))
N_PER_CELL     = 40
TEMPERATURE    = 0
DOMAIN         = "D2_Symbolic_Pointer"
BOOTSTRAP_REPS = 2000

MAX_RETRIES = 5
RETRY_WAIT  = 20
MAX_TOKENS  = 4096

POST_CALL_WAIT = {
    "anthropic":  5.0,
    "google":     3.0,
    "openai":     2.0,
    "deepseek":   2.0,
    "meta-llama": 2.0,
}
DEFAULT_WAIT = 2.0

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_smart_resume_d2"

# ---------------------------------------------------------------------------
# Client Setup & Environment Validation
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY environment variable is not configured."
    )

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task Constants & Specification
# ---------------------------------------------------------------------------

VARIABLES     = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
INITIAL_STATE = {v: i + 1 for i, v in enumerate(VARIABLES)}

OPERATIONS_DESCRIPTION = """\
REGISTER CONVENTION:
- There are 7 variables: A, B, C, D, E, F, G.
- Initial state: [A=1, B=2, C=3, D=4, E=5, F=6, G=7].
- All arithmetic is modulo 10. Results are always in the range 0–9.
- "modulo 10" means: take the result, divide by 10, keep only the remainder.
  Examples: 12 mod 10 = 2.  -4 mod 10 = 6 (add 10 until non-negative).

OPERATIONS:
- SHIFT_RIGHT        : Cyclic shift right. Each variable takes the value of its
                       left neighbour; A takes G's value.
                       New order: A=old_G, B=old_A, C=old_B, D=old_C,
                                  E=old_D, F=old_E, G=old_F.
- SHIFT_LEFT         : Cyclic shift left. Each variable takes the value of its
                       right neighbour; G takes A's value.
                       New order: A=old_B, B=old_C, C=old_D, D=old_E,
                                  E=old_F, F=old_G, G=old_A.
- SWAP_A_G           : Exchange the values of A and G simultaneously.
                       A takes old_G, G takes old_A. All other variables unchanged.
- SET_D_TO_A_PLUS_B  : D becomes (A + B) mod 10. All other variables unchanged.
- SET_C_TO_G_MINUS_E : C becomes (G - E) mod 10. If G - E is negative, add 10.
                       All other variables unchanged.

EXECUTION RULES:
- Operations are applied sequentially. Each operation uses the register state
  produced by the previous step, not the original state.
- Within a single operation, all output values are computed from the state that
  existed before the operation began.

WORKED EXAMPLES (all use starting state [A=1, B=2, C=3, D=4, E=5, F=6, G=7]):

SHIFT_RIGHT:
  State before : [A=1, B=2, C=3, D=4, E=5, F=6, G=7]
  State after  : [A=7, B=1, C=2, D=3, E=4, F=5, G=6]

SHIFT_LEFT:
  State before : [A=1, B=2, C=3, D=4, E=5, F=6, G=7]
  State after  : [A=2, B=3, C=4, D=5, E=6, F=7, G=1]

SWAP_A_G:
  State before : [A=1, B=2, C=3, D=4, E=5, F=6, G=7]
  State after  : [A=7, B=2, C=3, D=4, E=5, F=6, G=1]

SET_D_TO_A_PLUS_B:
  State before : [A=1, B=2, C=3, D=4, E=5, F=6, G=7]
  D = (1 + 2) mod 10 = 3. State after: [A=1, B=2, C=3, D=3, E=5, F=6, G=7]

SET_C_TO_G_MINUS_E:
  State before : [A=1, B=2, C=3, D=4, E=5, F=6, G=7]
  C = (7 - 5) mod 10 = 2. State after: [A=1, B=2, C=2, D=4, E=5, F=6, G=7]
  Edge case    : If G=2, E=6: C = (2-6) mod 10 = 6.\
"""

OUTPUT_FORMAT = """\
Output ONLY the following format. Do NOT use markdown code blocks. Do NOT add commentary.
1. TRACE: ["Step 1: [A=..., B=..., C=..., D=..., E=..., F=..., G=...]", "Step 2: [...]", ...]
2. ANSWER: [A=..., B=..., C=..., D=..., E=..., F=..., G=...]

CRITICAL RULES:
- Step 1 in TRACE = state AFTER applying the first operation.
- Do NOT include the initial state in TRACE.
- Output ALL 7 variables at every single step. No ellipses.\
"""

SYSTEM_PROMPT = (
    "You are an abstract register-file engine tracking 7 variables.\n\n"
    + OPERATIONS_DESCRIPTION + "\n\n"
    + OUTPUT_FORMAT
)

# ---------------------------------------------------------------------------
# Benchmark Data Generation Engine
# ---------------------------------------------------------------------------

class SymbolicPointerGenerator:
    VARIABLES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    OPS       = ["SHIFT_RIGHT", "SHIFT_LEFT", "SWAP_A_G",
                 "SET_D_TO_A_PLUS_B", "SET_C_TO_G_MINUS_E"]

    def _apply_op(self, state: dict, op: str) -> dict:
        s = dict(state)
        if op == "SHIFT_RIGHT":
            vals     = [state[v] for v in self.VARIABLES]
            new_vals = [vals[-1]] + vals[:-1]
            s        = dict(zip(self.VARIABLES, new_vals))
        elif op == "SHIFT_LEFT":
            vals     = [state[v] for v in self.VARIABLES]
            new_vals = vals[1:] + [vals[0]]
            s        = dict(zip(self.VARIABLES, new_vals))
        elif op == "SWAP_A_G":
            s['A'], s['G'] = state['G'], state['A']
        elif op == "SET_D_TO_A_PLUS_B":
            s['D'] = (state['A'] + state['B']) % 10
        elif op == "SET_C_TO_G_MINUS_E":
            s['C'] = (state['G'] - state['E'] + 10) % 10
        return s

    def _state_str(self, state: dict) -> str:
        return "[" + ", ".join(f"{v}={state[v]}" for v in self.VARIABLES) + "]"

    def solve(self, depth: int, seed: int):
        random.seed(seed)
        state        = dict(INITIAL_STATE)
        trace        = []
        events       = []
        for i in range(depth):
            op    = random.choice(self.OPS)
            state = self._apply_op(state, op)
            events.append(f"Step {i+1}: {op}")
            trace.append(f"Step {i+1}: {self._state_str(state)}")
        return "\n".join(events), trace, self._state_str(state)

    def build(self, depths, n_per_depth: int, seed_offset: int = 0):
        dataset = []
        for d in depths:
            seen      = set()
            collected = 0
            attempt   = 0
            while collected < n_per_depth:
                seed_val   = d * 10000 + attempt + seed_offset
                attempt   += 1
                pt, gt, ans = self.solve(d, seed=seed_val)
                op_chain    = tuple(re.findall(r"Step \d+: ([A-Z_]+)", pt))
                if op_chain not in seen:
                    seen.add(op_chain)
                    dataset.append({
                        "depth":    d,
                        "seed":     seed_val,
                        "prompt":   pt,
                        "gt_trace": gt,
                        "gt_ans":   ans,
                    })
                    collected += 1
        return dataset

# ---------------------------------------------------------------------------
# Trace Evaluation and Parsing
# ---------------------------------------------------------------------------

def extract_symbolic_trace(text: str) -> list:
    matches = re.findall(
        r"Step\s*\d+:\s*\[[A-Za-z_]+=.*?\]",
        text, re.IGNORECASE
    )
    return [re.sub(r'\s+', ' ', m).strip() for m in matches]


def extract_symbolic_answer(text: str) -> str:
    m = re.search(r"ANSWER:.*?(\[.*?\])", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else "N/A"


def _clean(s: str) -> str:
    return re.sub(r'\s+', '', str(s)).lower()


def compute_divergence_step(gt_trace: list, predicted_trace: list):
    for i, gt_step in enumerate(gt_trace):
        pred_step = predicted_trace[i] if i < len(predicted_trace) else "MISSING"
        if _clean(pred_step) != _clean(gt_step):
            return (i + 1, gt_step, pred_step)
    return (-1, None, None)


def evaluate_single(item: dict, model_output: str) -> dict:
    raw_ans      = extract_symbolic_answer(model_output)
    is_correct   = (_clean(raw_ans) == _clean(str(item["gt_ans"]))) and _clean(str(item["gt_ans"])) != ""
    pred_trace   = extract_symbolic_trace(model_output)
    gt_trace     = [re.sub(r'\s+', ' ', str(s)).strip() for s in item["gt_trace"]]
    output_upper = model_output.strip().upper()

    is_format_failure = (
        len(pred_trace) == 0
        or "TRACE" not in output_upper
    )

    div_step, exp_at_div, pred_at_div = compute_divergence_step(gt_trace, pred_trace)
    is_tfbc         = is_correct and (div_step != -1)
    trace_length_ok = (len(pred_trace) == item["depth"])

    return {
        "depth":              item["depth"],
        "seed":               item["seed"],
        "is_correct":         is_correct,
        "div_step":           div_step,
        "is_tfbc":            is_tfbc,
        "is_format_failure":  is_format_failure,
        "trace_length_ok":    trace_length_ok,
        "pred_trace_len":     len(pred_trace),
        "expected_trace_len": item["depth"],
        "expected_at_div":    exp_at_div,
        "predicted_at_div":   pred_at_div,
        "model_output":       model_output,
    }

# ---------------------------------------------------------------------------
# Disjoint Post-Processing and Structural Fix Framework
# ---------------------------------------------------------------------------

PROSE_PATTERNS = [
    r"\bis\s+(in)?correct\b",
    r"\bshould\s+be\b",
    r"\bthe\s+correct\b",
    r"\bgoes\s+to\s+the\s+position\b",
    r"\binstead\b",
    r"->",
    r"\bso\s+the\b",
    r"\balso\s+incorrect\b",
    r"\bbecomes\b",
    r"\bafter\s+shift\b",
]
PROSE_RE = re.compile("|".join(PROSE_PATTERNS), re.IGNORECASE)


def _extract_fix_steps(model_output: str) -> list:
    steps = re.findall(
        r"Step\s+\d+\s*:\s*(.+?)(?=Step\s+\d+\s*:|ANSWER|$)",
        model_output, re.DOTALL
    )
    return [s.strip() for s in steps]


def _step_has_prose(step_body: str) -> bool:
    return bool(PROSE_RE.search(step_body))


def _output_has_corruption(model_output: str) -> bool:
    return any(_step_has_prose(s) for s in _extract_fix_steps(model_output))


def _parse_symbolic_state(step_body: str):
    if ']]' not in step_body and ']' not in step_body:
        return None
    clean = re.sub(r'["\',\s\\]+$', '', step_body.strip())
    if not clean.endswith(']'):
        return None
    pairs = re.findall(r'\b([A-G])=(\d+)', step_body)
    if len(pairs) != 7:
        return None
    try:
        return {v: int(n) for v, n in pairs}
    except ValueError:
        return None


def _is_valid_symbolic_state(state) -> bool:
    if state is None:
        return False
    if set(state.keys()) != set(VARIABLES):
        return False
    vals = list(state.values())
    if not all(0 <= v <= 9 for v in vals):
        return False
    return len(set(vals)) == len(vals)


def _missing_trace_or_answer(model_output: str) -> bool:
    return "TRACE" not in model_output or "ANSWER" not in model_output


def _compute_failure_type(rec: dict) -> str:
    if rec.get("is_correct"):
        return "none"

    model_output    = rec.get("model_output", "")
    is_format       = rec.get("is_format_failure", False)
    is_truncated    = rec.get("is_truncated", False)
    is_constraint   = rec.get("_constraint_violation", False)

    if is_format:
        return "format"
    if is_truncated and not _output_has_corruption(model_output):
        return "trunc"
    if is_constraint:
        return "constraint"

    steps = _extract_fix_steps(model_output)
    if steps:
        all_valid = all(
            _is_valid_symbolic_state(_parse_symbolic_state(s))
            for s in steps
        )
        if all_valid:
            return "logic"

    return "format"


def fix_record(record: dict) -> dict:
    rec          = copy.deepcopy(record)
    model_output = rec.get("model_output", "")
    reasons      = []

    rec["raw_pred_trace_len"]   = rec.get("pred_trace_len")
    rec["raw_predicted_at_div"] = rec.get("predicted_at_div")
    rec["raw_trace_length_ok"]  = rec.get("trace_length_ok")

    if rec.get("is_truncated", False) and _output_has_corruption(model_output):
        rec["is_format_failure"] = True
        rec["is_truncated"]      = False
        reasons.append("F3:truncated_but_corrupted->format_failure")

    if not rec["is_format_failure"] and _missing_trace_or_answer(model_output):
        rec["is_format_failure"] = True
        reasons.append("F4:missing_TRACE_or_ANSWER")

    if not rec["is_format_failure"] and _output_has_corruption(model_output):
        rec["is_format_failure"] = True
        reasons.append("F1:prose_in_steps")

    if not rec["is_format_failure"] and not rec.get("_constraint_violation"):
        for i, s in enumerate(_extract_fix_steps(model_output), start=1):
            state = _parse_symbolic_state(s)
            if state is None:
                rec["is_format_failure"] = True
                reasons.append(f"F2a:incomplete_state_at_step{i}")
                break
            if not _is_valid_symbolic_state(state):
                rec["_constraint_violation"] = True
                reasons.append(f"F2b:constraint_violation_at_step{i}")
                break

    steps = _extract_fix_steps(model_output)
    rec["pred_trace_len"]  = len(steps)
    rec["trace_length_ok"] = (len(steps) == rec.get("expected_trace_len", -1))

    if rec["is_format_failure"]:
        rec["is_correct"]       = False
        rec["predicted_at_div"] = "MISSING"

    rec["failure_type"] = _compute_failure_type(rec)

    if reasons:
        rec["_fix_reasons"] = reasons

    return rec


def fix_all_results(results: list) -> tuple:
    fixed       = []
    change_log  = []
    type_counts = {}

    for rec in results:
        f = fix_record(rec)
        fixed.append(f)
        t = f.get("failure_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        if "_fix_reasons" in f:
            change_log.append({
                "seed":           rec.get("seed"),
                "depth":          rec.get("depth"),
                "reasons":        f["_fix_reasons"],
                "failure_type":   f.get("failure_type"),
                "fmt_fail":       f"{rec.get('is_format_failure')} -> {f.get('is_format_failure')}",
                "is_truncated":   f"{rec.get('is_truncated')} -> {f.get('is_truncated')}",
                "pred_trace_len": f"{rec.get('pred_trace_len')} -> {f.get('pred_trace_len')}",
            })

    summary = {
        "total":                  len(results),
        "modified":               len(change_log),
        "failure_type_breakdown": dict(sorted(type_counts.items())),
        "change_log":             change_log,
    }
    return fixed, summary

# ---------------------------------------------------------------------------
# Statistical Infrastructure
# ---------------------------------------------------------------------------

def clopper_pearson_ci(successes: int, n: int, confidence: float = 0.95):
    if n == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    lo = beta_dist.ppf(alpha / 2,     successes,     n - successes + 1) if successes > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes)     if successes < n else 1.0
    return (float(lo), float(hi))


def _nll(pd, depths, k, n):
    pd    = max(1e-9, min(1.0 - 1e-9, pd))
    probs = np.clip(pd ** depths, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(probs) + (n - k) * np.log(1 - probs))


def fit_pd_model(depth_levels, successes_per_depth, n_per_depth,
                 bootstrap_reps: int = 2000):
    depths = np.array(depth_levels, dtype=float)
    k      = np.array(successes_per_depth, dtype=float)
    n      = np.array(n_per_depth, dtype=float)
    mask   = n > 0
    depths, k, n = depths[mask], k[mask], n[mask]
    if len(depths) == 0:
        return (float("nan"), float("nan"), float("nan"))
    grid   = np.linspace(0.50, 0.9999, 5000)
    pd_mle = float(grid[np.argmin([_nll(p, depths, k, n) for p in grid])])
    pd_boot = []
    for _ in range(bootstrap_reps):
        k_b = np.random.binomial(n.astype(int), pd_mle ** depths)
        pd_boot.append(float(grid[np.argmin([_nll(p, depths, k_b, n) for p in grid])]))
    return (
        pd_mle,
        float(np.percentile(pd_boot, 2.5)),
        float(np.percentile(pd_boot, 97.5)),
    )


def aggregate_by_depth(results: list) -> dict:
    by_depth = {}
    for r in results:
        d = r["depth"]
        if d not in by_depth:
            by_depth[d] = {
                "n": 0, "correct": 0, "div_steps": [], "tfbc": 0,
                "format_failures": 0, "constraint_violations": 0,
                "completion_tokens": [],
            }
        s = by_depth[d]
        s["n"] += 1
        if (r.get("is_correct")
                and not r.get("is_api_failure", False)
                and not r.get("is_truncated", False)):
            s["correct"] += 1
        if r.get("div_step", -1) > 0 and not r.get("is_format_failure", False):
            s["div_steps"].append(r["div_step"])
        if r.get("is_tfbc"):
            s["tfbc"] += 1
        if r.get("is_format_failure"):
            s["format_failures"] += 1
        if r.get("_constraint_violation"):
            s["constraint_violations"] += 1
        if r.get("completion_tokens", 0) > 0:
            s["completion_tokens"].append(r["completion_tokens"])

    out = {}
    for d, v in sorted(by_depth.items()):
        n, c   = v["n"], v["correct"]
        lo, hi = clopper_pearson_ci(c, n)
        divs   = v["div_steps"]
        fmt_f  = v["format_failures"]
        toks   = v["completion_tokens"]
        out[d] = {
            "n":                       n,
            "correct":                 c,
            "accuracy":                c / n if n > 0 else 0.0,
            "ci_lo":                   lo,
            "ci_hi":                   hi,
            "n_incorrect":             n - c,
            "div_steps":               divs,
            "avg_div_step":            float(np.mean(divs)) if divs else None,
            "early_failures":          sum(1 for s in divs if s <= 3),
            "mid_failures":            sum(1 for s in divs if 4 <= s <= 10),
            "late_failures":           sum(1 for s in divs if s > 10),
            "tfbc_count":              v["tfbc"],
            "tfbc_rate":               v["tfbc"] / c if c > 0 else None,
            "format_failures":         fmt_f,
            "format_failure_rate":     fmt_f / n if n > 0 else 0.0,
            "constraint_violations":   v["constraint_violations"],
            "avg_completion_tokens":   float(np.mean(toks)) if toks else None,
        }
    return out


def aggregate_overall(results: list) -> dict:
    valid  = [r for r in results
              if not r.get("is_api_failure", False)
              and not r.get("is_truncated", False)]
    n      = len(valid)
    c      = sum(1 for r in valid if r.get("is_correct"))
    tfbc   = sum(1 for r in valid if r.get("is_tfbc"))
    fmt_f  = sum(1 for r in valid if r.get("is_format_failure"))
    constr = sum(1 for r in results if r.get("_constraint_violation"))
    lo, hi = clopper_pearson_ci(c, n)
    divs   = [r["div_step"] for r in results
              if r.get("div_step", -1) > 0
              and not r.get("is_format_failure", False)]
    toks   = [r["completion_tokens"] for r in results
              if r.get("completion_tokens", 0) > 0]
    return {
        "n":                       n,
        "correct":                 c,
        "accuracy":                c / n if n > 0 else 0.0,
        "ci_lo":                   lo,
        "ci_hi":                   hi,
        "tfbc_total":              tfbc,
        "tfbc_rate":               tfbc / c if c > 0 else None,
        "format_failures":         fmt_f,
        "format_failure_rate":     fmt_f / n if n > 0 else 0.0,
        "constraint_violations":   constr,
        "constraint_violation_rate": constr / n if n > 0 else 0.0,
        "avg_div_step":            float(np.mean(divs)) if divs else None,
        "avg_completion_tokens":   float(np.mean(toks)) if toks else None,
        "total_tokens_used":       sum(toks),
    }


def failure_timing_breakdown(results: list) -> dict:
    divs = [r["div_step"] for r in results
            if r.get("div_step", -1) > 0
            and not r.get("is_format_failure", False)]
    if not divs:
        return {"early": None, "mid": None, "late": None,
                "n_logic_failures": 0, "avg_div_step": None}
    total = len(divs)
    return {
        "early":            sum(1 for s in divs if s <= 3)      / total,
        "mid":              sum(1 for s in divs if 4 <= s <= 10) / total,
        "late":             sum(1 for s in divs if s > 10)       / total,
        "n_logic_failures": total,
        "avg_div_step":     float(np.mean(divs)),
    }

# ---------------------------------------------------------------------------
# State Resumption and Provenance Tracking
# ---------------------------------------------------------------------------

def discover_files_for_model(model_id: str, results_dir: str) -> list:
    safe     = model_id_to_safe(model_id)
    pattern  = os.path.join(results_dir, f"results_*{safe}*.json")
    files    = glob.glob(pattern)
    if not files:
        provider, *rest = model_id.split("/")
        model_part      = "_".join(rest) if rest else safe
        pattern2        = os.path.join(results_dir, f"results_*{model_part}*.json")
        files           = glob.glob(pattern2)
    return sorted(set(files), key=os.path.getmtime)


def load_and_merge_files(file_paths: list, model_id: str):
    all_by_seed  = {}
    provenance   = []
    total_loaded = 0
    dupes        = 0

    for fpath in file_paths:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"    [SKIP] {os.path.basename(fpath)}: malformed list schema.")
                continue
            n             = len(data)
            total_loaded += n
            provenance.append({"file": fpath, "n": n})

            for r in data:
                seed = r.get("seed")
                if seed is None:
                    continue
                r.setdefault("model", model_id)
                r.setdefault("label", f"Main_{model_id.split('/')[-1]}")
                if "is_format_failure" not in r:
                    r["is_format_failure"] = (len(r.get("model_output", "")) == 0)

                error_msg = str(r.get("error", "")).lower()
                is_transient_error = (
                    "402"              in error_msg
                    or "insufficient"  in error_msg
                    or "credits"       in error_msg
                    or "connection"    in error_msg
                    or "timeout"       in error_msg
                    or "timed out"     in error_msg
                    or "rate limit"    in error_msg
                    or "429"           in error_msg
                    or "500"           in error_msg
                    or "503"           in error_msg
                    or "server error"  in error_msg
                )
                is_empty_record = (
                    r.get("prompt_tokens", 0) == 0
                    and r.get("completion_tokens", 0) == 0
                    and not r.get("model_output", "").strip()
                    and not r.get("is_correct", False)
                )
                if is_transient_error or is_empty_record:
                    r["is_api_failure"]    = True
                    r["is_format_failure"] = False

                if seed not in all_by_seed:
                    all_by_seed[seed] = r
                else:
                    dupes += 1
                    existing = all_by_seed[seed]
                    if existing.get("is_api_failure", False) and not r.get("is_api_failure", False):
                        all_by_seed[seed] = r

            print(f"    Loaded {n:3d} records from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"    [ERROR] {fpath}: {e}")

    merged = list(all_by_seed.values())
    return merged, provenance, total_loaded, dupes


def classify_results(merged: list, benchmark_by_seed: dict):
    result_by_seed = {r["seed"]: r for r in merged}
    good, retry, missing = [], [], []

    for seed, item in benchmark_by_seed.items():
        if seed not in result_by_seed:
            missing.append(item)
        else:
            r = result_by_seed[seed]
            is_empty = (
                r.get("prompt_tokens", 0) == 0
                and r.get("completion_tokens", 0) == 0
                and not r.get("model_output", "").strip()
                and not r.get("is_correct", False)
            )
            if r.get("is_api_failure", False) or r.get("is_truncated", False) or is_empty:
                retry.append(item)
            else:
                good.append(r)

    return good, retry, missing

# ---------------------------------------------------------------------------
# API Execution Harness
# ---------------------------------------------------------------------------

def run_on_items(model_id: str, items: list, label: str = "",
                 output_file: str = None, existing_results: list = None) -> list:
    results          = []
    existing_results = existing_results or []
    n_total          = len(items)

    if n_total == 0:
        return results

    wait = get_post_call_wait(model_id)

    print(f"\n{'='*70}")
    print(f"[RUN] {label or model_id}")
    print(f"      {n_total} instances  |  temp={TEMPERATURE}  |  max_tokens={MAX_TOKENS}")
    print(f"{'='*70}")

    for i, item in enumerate(items):
        success  = False
        retries  = 0
        last_err = None

        while not success and retries < MAX_RETRIES:
            try:
                dynamic_tokens = MAX_TOKENS
                if item.get("depth", 0) >= 40:
                    dynamic_tokens = 6000

                t0       = time.time()
                response = client.chat.completions.create(
                    model=model_id,
                    temperature=TEMPERATURE,
                    max_tokens=dynamic_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": item["prompt"]},
                    ],
                )
                elapsed    = time.time() - t0
                raw_output = response.choices[0].message.content.strip()

                pt = getattr(response.usage, "prompt_tokens",     0) or 0
                ct = getattr(response.usage, "completion_tokens", 0) or 0
                tt = getattr(response.usage, "total_tokens",      0) or 0

                result = evaluate_single(item, raw_output)
                result["is_truncated"] = (
                    not result["is_format_failure"]
                    and result["pred_trace_len"] < item["depth"]
                )
                result.update({
                    "model":             model_id,
                    "label":             label,
                    "is_api_failure":    False,
                    "time_sec":          round(elapsed, 2),
                    "prompt_tokens":     pt,
                    "completion_tokens": ct,
                    "total_tokens":      tt,
                })
                results.append(result)

                status = "PASS" if result["is_correct"] else "FAIL"
                tags   = (
                    (" [TFBC]"   if result["is_tfbc"]          else "") +
                    (" [FORMAT]" if result["is_format_failure"] else "") +
                    (" [TRUNC]"  if result["is_truncated"]      else "")
                )
                print(
                    f"  [{i+1:3d}/{n_total}] N={item['depth']:2d} | seed={item['seed']} | "
                    f"{status}{tags} | k*={result['div_step']:3d} | tok={ct} | {elapsed:.1f}s"
                )

                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(existing_results + results, f, indent=2)

                success = True
                time.sleep(wait)

            except Exception as e:
                retries  += 1
                last_err  = str(e)
                is_credit = "402" in last_err or "credits" in last_err.lower()
                print(f"  [RETRY {retries}/{MAX_RETRIES}] seed={item['seed']} "
                      f"{'[CREDIT]' if is_credit else ''} {e}")

                if retries >= MAX_RETRIES:
                    fail = {
                        "depth":              item["depth"],
                        "seed":               item["seed"],
                        "is_correct":         False,
                        "div_step":           -1,
                        "is_tfbc":            False,
                        "is_format_failure":  False,
                        "is_api_failure":     True,
                        "is_truncated":       False,
                        "trace_length_ok":    False,
                        "pred_trace_len":     0,
                        "expected_trace_len": item["depth"],
                        "model":              model_id,
                        "label":              label,
                        "error":              last_err,
                        "prompt_tokens":      0,
                        "completion_tokens":  0,
                        "total_tokens":       0,
                    }
                    results.append(fail)
                    if output_file:
                        with open(output_file, "w") as f:
                            json.dump(existing_results + results, f, indent=2)
                else:
                    sleep_time = RETRY_WAIT * 2 if is_credit else RETRY_WAIT
                    time.sleep(sleep_time)

    return results

# ---------------------------------------------------------------------------
# Output Reporting & Export Instrumentation
# ---------------------------------------------------------------------------

def print_and_save_summary(all_model_results: dict, pd_fits: dict, filepath: str):
    lines = []

    def p(line=""):
        print(line)
        lines.append(line)

    p("=" * 80)
    p(f"DOMAIN: {DOMAIN}  |  RUN: {RUN_ID}  |  temp={TEMPERATURE}  |  n={N_PER_CELL}/cell")
    p("=" * 80)

    for model_id, results in all_model_results.items():
        short = model_id.split("/")[-1]
        p(f"\n--- {short} ---")
        p(f"{'Depth':>6}  {'N':>4}  {'Corr':>4}  {'Acc%':>6}  {'95% CI':^20}  "
          f"{'AvgK*':>7}  {'Early%':>7}  {'Mid%':>7}  {'Late%':>7}  "
          f"{'TFBC%':>7}  {'FmtFail%':>9}  {'Constr%':>8}  {'AvgTok':>7}")
        p("-" * 115)
        agg = aggregate_by_depth(results)
        for d, v in sorted(agg.items()):
            ci_str = f"[{v['ci_lo']*100:5.1f}%, {v['ci_hi']*100:5.1f}%]"
            n_inc  = v["n_incorrect"]
            early  = f"{v['early_failures']/n_inc*100:5.1f}%" if n_inc else "  N/A"
            mid    = f"{v['mid_failures']/n_inc*100:5.1f}%"   if n_inc else "  N/A"
            late   = f"{v['late_failures']/n_inc*100:5.1f}%"  if n_inc else "  N/A"
            avg_k  = f"{v['avg_div_step']:6.2f}" if v["avg_div_step"] else "   N/A"
            tfbc   = f"{v['tfbc_rate']*100:5.1f}%" if v["tfbc_rate"] is not None else "  N/A"
            fmt_f  = f"{v['format_failure_rate']*100:5.1f}%"
            constr = f"{v['constraint_violations']/v['n']*100:5.1f}%" if v["n"] else "  N/A"
            avg_t  = f"{v['avg_completion_tokens']:6.0f}" if v["avg_completion_tokens"] else "   N/A"
            p(f"{d:>6}  {v['n']:>4}  {v['correct']:>4}  {v['accuracy']*100:5.1f}%  "
              f"{ci_str:^20}  {avg_k}  {early:>7}  {mid:>7}  {late:>7}  "
              f"{tfbc:>7}  {fmt_f:>9}  {constr:>8}  {avg_t:>7}")

        ov  = aggregate_overall(results)
        ftb = failure_timing_breakdown(results)
        p("-" * 115)
        tfbc_str   = f"TFBC={ov['tfbc_rate']*100:.1f}%" if ov["tfbc_rate"] is not None else "TFBC=N/A"
        fmt_str    = f"{ov['format_failure_rate']*100:.1f}%"
        constr_str = f"{ov['constraint_violation_rate']*100:.1f}%"
        tok_str    = f"{ov['avg_completion_tokens']:.0f}" if ov["avg_completion_tokens"] else "N/A"
        p(f"{'TOTAL':>6}  {ov['n']:>4}  {ov['correct']:>4}  {ov['accuracy']*100:5.1f}%  "
          f"[{ov['ci_lo']*100:5.1f}%, {ov['ci_hi']*100:5.1f}%]  "
          f"  {tfbc_str}  FmtFail={fmt_str}  Constr={constr_str}  AvgTok={tok_str}")
        if ftb["n_logic_failures"] > 0:
            p(f"  k* timing (logic failures, n={ftb['n_logic_failures']}): "
              f"Early={ftb['early']*100:.1f}%  Mid={ftb['mid']*100:.1f}%  "
              f"Late={ftb['late']*100:.1f}%  AvgK*={ftb['avg_div_step']:.2f}")

        ft_counts = {}
        for r in results:
            t = r.get("failure_type", "unknown")
            ft_counts[t] = ft_counts.get(t, 0) + 1
        p(f"  failure_type breakdown: " +
          "  ".join(f"{k}={v}" for k, v in sorted(ft_counts.items())))

    p("\n" + "=" * 60)
    p("p_d MODEL FIT  (P(correct|N) = p_d^N, bootstrapped 95% CI)")
    p("=" * 60)
    p(f"{'Model':<35}  {'p_d MLE':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
    p("-" * 60)
    for model_id, fit in pd_fits.items():
        short = model_id.split("/")[-1]
        if math.isnan(fit[0]):
            p(f"{short:<35}  {'N/A':>8}")
        else:
            p(f"{short:<35}  {fit[0]:8.4f}  {fit[1]:8.4f}  {fit[2]:8.4f}")

    p("\n" + "=" * 80)
    p("ACCURACY MATRIX (% \u00b1 half-CI, all depths)")
    p("=" * 80)
    col_names = [m.split("/")[-1][:16] for m in all_model_results]
    p("Depth  | " + " | ".join(f"{c:^16}" for c in col_names))
    p("-" * (9 + 19 * len(col_names)))
    first_agg = aggregate_by_depth(next(iter(all_model_results.values())))
    for d in sorted(first_agg.keys()):
        row   = f" {d:2d}    | "
        cells = []
        for model_id, results in all_model_results.items():
            agg = aggregate_by_depth(results)
            v   = agg.get(d)
            cells.append(
                f"{v['accuracy']*100:5.1f}% \u00b1{(v['ci_hi']-v['ci_lo'])*50:3.1f}".center(16)
                if v else "  N/A  ".center(16)
            )
        p(row + " | ".join(cells))

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] Summary -> {filepath}")


def save_parser_validation_set(all_results_flat: list, filepath: str,
                               n_per_model: int = 10):
    by_model = {}
    for r in all_results_flat:
        by_model.setdefault(r.get("model", "unknown"), []).append(r)

    val_set = []
    for model_id, results in by_model.items():
        logic_fail = [r for r in results
                      if not r.get("is_correct")
                      and not r.get("is_format_failure", False)
                      and not r.get("is_api_failure", False)]
        sample = random.sample(logic_fail, min(n_per_model, len(logic_fail)))
        if len(sample) < n_per_model:
            others = [r for r in results
                      if r not in sample and not r.get("is_api_failure", False)]
            random.shuffle(others)
            sample += others[:n_per_model - len(sample)]
        for r in sample:
            val_set.append({
                "model":              model_id,
                "depth":              r["depth"],
                "seed":               r["seed"],
                "is_correct":         r.get("is_correct"),
                "failure_type":       r.get("failure_type"),
                "auto_div_step":      r.get("div_step"),
                "is_format_failure":  r.get("is_format_failure"),
                "pred_trace_len":     r.get("pred_trace_len"),
                "expected_len":       r.get("expected_trace_len"),
                "trace_length_ok":    r.get("trace_length_ok"),
                "expected_at_div":    r.get("expected_at_div"),
                "predicted_at_div":   r.get("predicted_at_div"),
                "model_output":       r.get("model_output", "")[:2000],
                "human_div_step":     None,
                "human2_div_step":    None,
                "annotation_note":    "",
            })

    with open(filepath, "w") as f:
        json.dump(val_set, f, indent=2)
    print(f"[SAVED] Parser validation set ({len(val_set)} traces) -> {filepath}")

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print(f"SMART RESUME D2 SYMBOLIC — {RUN_ID}")
    print(f"Results dir : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Models      : {len(ALL_MODELS)}")
    print("=" * 70)

    gen           = SymbolicPointerGenerator()
    main_dataset  = gen.build(DEPTH_LEVELS, N_PER_CELL, seed_offset=0)
    bench_by_seed = {item["seed"]: item for item in main_dataset}
    expected      = N_PER_CELL * len(DEPTH_LEVELS)

    print(f"\n[DATASET] Rebuilt {len(main_dataset)} benchmark instances")
    print(f"          Depths: {DEPTH_LEVELS}")
    print(f"          N/cell: {N_PER_CELL} \u2192 Total: {expected}")

    all_model_results = {}
    all_results_flat  = []
    provenance_map    = {}
    fix_summaries     = {}

    for model_id in ALL_MODELS:
        safe  = model_id_to_safe(model_id)
        short = model_id.split("/")[-1]

        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        found_files = discover_files_for_model(model_id, OUTPUT_DIR)
        print(f"  Found {len(found_files)} file(s):")
        for fp in found_files:
            print(f"    {os.path.basename(fp)}")

        if found_files:
            merged, prov, total_loaded, dupes = load_and_merge_files(found_files, model_id)
            print(f"  Loaded: {total_loaded}  |  After dedup: {len(merged)}  |  Dupes: {dupes}")
            provenance_map[model_id] = prov
        else:
            merged = []
            print("  No existing files \u2014 running from scratch.")
            provenance_map[model_id] = []

        good, retry_items, missing_items = classify_results(merged, bench_by_seed)
        n_good     = len(good)
        n_retry    = len(retry_items)
        n_missing  = len(missing_items)
        n_need_run = n_retry + n_missing

        print(f"\n  Status:")
        print(f"    Good (keep)         : {n_good:3d}")
        print(f"    API failures (retry): {n_retry:3d}")
        print(f"    Missing (never run) : {n_missing:3d}")
        print(f"    To run now          : {n_need_run:3d}")

        if n_need_run > 0:
            run_file = os.path.join(OUTPUT_DIR, f"results_main_{safe}_{RUN_ID}.json")
            new_results = run_on_items(
                model_id         = model_id,
                items            = retry_items + missing_items,
                label            = f"Main_{short}",
                output_file      = run_file,
                existing_results = good,
            )
            raw_results = good + new_results
        else:
            raw_results = good
            print(f"\n  Model COMPLETE \u2014 no API calls needed.")

        seeds = [r["seed"] for r in raw_results]
        assert len(set(seeds)) == len(seeds), \
            f"[BUG] Duplicate seeds in {model_id}: {len(seeds)-len(set(seeds))} dupes"

        print(f"\n  Applying classification fixes (F1\u2013F7)...")
        fixed_results, fix_summary = fix_all_results(raw_results)
        fix_summaries[model_id]    = fix_summary

        print(f"    Modified {fix_summary['modified']} / {fix_summary['total']} records")
        print(f"    failure_type breakdown: " +
              "  ".join(f"{k}={v}" for k, v in sorted(fix_summary["failure_type_breakdown"].items())))

        final_file = os.path.join(OUTPUT_DIR, f"results_FINAL_{safe}_{RUN_ID}.json")
        with open(final_file, "w") as f:
            json.dump(fixed_results, f, indent=2)
        print(f"\n  [SAVED] {os.path.basename(final_file)}")

        n_final   = len(fixed_results)
        n_api_err = sum(1 for r in fixed_results if r.get("is_api_failure",   False))
        n_trunc   = sum(1 for r in fixed_results if r.get("is_truncated",     False))
        n_fmt     = sum(1 for r in fixed_results if r.get("is_format_failure",False))
        n_correct = sum(1 for r in fixed_results if r.get("is_correct",       False))
        acc       = n_correct / n_final if n_final > 0 else 0.0
        icon      = "\u2713" if n_final == expected and n_api_err == 0 else "\u26a0"
        print(f"\n  {icon} {short}: {n_correct}/{n_final} correct = {acc*100:.1f}%"
              f"  [{n_final}/{expected} instances]")

        all_model_results[model_id] = fixed_results
        all_results_flat.extend(fixed_results)

    ordered = {m: all_model_results[m] for m in ALL_MODELS if m in all_model_results}

    print(f"\n{'='*70}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<40}  {'N':>5}  {'Exp':>5}  {'ApiErr':>6}  {'OK?':>4}")
    print("-" * 70)
    for model_id, results in ordered.items():
        n         = len(results)
        n_api_err = sum(1 for r in results if r.get("is_api_failure", False))
        ok        = (n == expected) and (n_api_err == 0)
        icon      = "\u2713" if ok else "\u2717"
        print(f"  {icon} {model_id.split('/')[-1]:<38}  {n:5d}  {expected:5d}  "
              f"{n_api_err:6d}  {'OK' if ok else 'WARN'}")

    print(f"\n{'='*70}")
    print("p_d MODEL FITS")
    pd_fits = {}
    for model_id, results in ordered.items():
        agg   = aggregate_by_depth(results)
        deps  = sorted(agg.keys())
        succs = [agg[d]["correct"] for d in deps]
        ns    = [agg[d]["n"]       for d in deps]
        fit   = fit_pd_model(deps, succs, ns, BOOTSTRAP_REPS)
        pd_fits[model_id] = fit
        short = model_id.split("/")[-1]
        if not math.isnan(fit[0]):
            print(f"  {short}: p_d={fit[0]:.4f}  95% CI=[{fit[1]:.4f}, {fit[2]:.4f}]")

    print(f"\n{'='*70}")
    print("WRITING OUTPUT FILES")

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{RUN_ID}.txt")
    print_and_save_summary(ordered, pd_fits, summary_path)

    val_path = os.path.join(OUTPUT_DIR, f"parser_validation_{RUN_ID}.json")
    save_parser_validation_set(all_results_flat, val_path, n_per_model=10)

    all_type_counts: dict = {}
    per_model_breakdown: dict = {}
    for model_id, results in ordered.items():
        mc: dict = {}
        for r in results:
            t = r.get("failure_type", "unknown")
            mc[t]              = mc.get(t, 0) + 1
            all_type_counts[t] = all_type_counts.get(t, 0) + 1
        per_model_breakdown[model_id] = dict(sorted(mc.items()))

    combined = {
        "run_id":        RUN_ID,
        "domain":        DOMAIN,
        "temperature":   TEMPERATURE,
        "n_per_cell":    N_PER_CELL,
        "depth_levels":  DEPTH_LEVELS,
        "models":        ALL_MODELS,
        "pd_fits":       {m: list(v) for m, v in pd_fits.items()},
        "main_results":  {m: ordered[m] for m in ordered},
        "provenance":    provenance_map,
        "fix_summaries": fix_summaries,
        "validation": {
            model_id: {
                "n_total":        len(r),
                "n_correct":      sum(1 for x in r if x.get("is_correct")),
                "n_api_failure":  sum(1 for x in r if x.get("is_api_failure")),
                "n_fmt_failure":  sum(1 for x in r if x.get("is_format_failure")),
                "n_truncated":    sum(1 for x in r if x.get("is_truncated")),
                "n_constraint":   sum(1 for x in r if x.get("_constraint_violation")),
                "complete":       len(r) == expected,
            }
            for model_id, r in ordered.items()
        }
    }

    combined_path = os.path.join(OUTPUT_DIR, f"combined_{RUN_ID}.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[SAVED] Combined -> {combined_path}")

    summary_json = {
        "output_file":              combined_path,
        "processed":                len(all_results_flat),
        "failure_type_breakdown":   dict(sorted(all_type_counts.items())),
        "per_model_breakdown":      per_model_breakdown,
        "fix_summaries":            fix_summaries,
    }
    summary_json_path = combined_path.replace(".json", "_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[SAVED] Summary JSON -> {summary_json_path}")

    print(f"\n{'='*60}")
    print("FAILURE TYPE BREAKDOWN (all models combined)")
    print(f"{'='*60}")
    total_all = len(all_results_flat)
    for ft, cnt in sorted(all_type_counts.items()):
        pct = 100 * cnt / total_all if total_all else 0
        print(f"  {ft:<12} {cnt:>6}  ({pct:.1f}%)")
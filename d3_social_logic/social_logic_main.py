"""
Resume or finalize a D3 Social Logic benchmark run across multiple models.

Scans ./results/ for any existing result files, merges partial runs by seed,
re-runs seeds that previously failed with API errors, and runs missing seeds
from scratch. Writes per-model final files, a summary table, p_d fits, and a
parser validation sample.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python social_logic_main.py
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
from openai import OpenAI
from scipy.stats import beta as beta_dist

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed — McNemar inline test will be skipped.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "./results_d3"

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
DOMAIN         = "D3_Social_Graph"
BOOTSTRAP_REPS = 2000

PROMPT_SENS_MODEL  = "anthropic/claude-3.7-sonnet"
PROMPT_SENS_DEPTH  = 15
PROMPT_SENS_N      = 40
PROMPT_SENS_OFFSET = 0
ABLATION_MODEL     = "anthropic/claude-3.7-sonnet"
ABLATION_DEPTH     = 15
ABLATION_N         = 20
ABLATION_OFFSET    = 200000

MAX_RETRIES = 8
RETRY_WAIT  = 20
MAX_TOKENS  = 12000

POST_CALL_WAIT = {
    "anthropic":  5.0,
    "google":     3.0,
    "openai":     2.0,
    "deepseek":   2.0,
    "meta-llama": 2.0,
}
DEFAULT_WAIT = 2.0

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_smart_resume_d3"

PEOPLE = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
VALID_PAIRS = {
    f"{PEOPLE[i]}{PEOPLE[j]}"
    for i in range(len(PEOPLE))
    for j in range(i + 1, len(PEOPLE))
}

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

RULES_DESCRIPTION = """\
PEOPLE: A, B, C, D, E, F, G, H, I, J  (10 people; initially all Neutral to each other)

RELATIONSHIP TYPES:
  - FRIENDS (F): two people are in the same faction.
  - RIVALS  (R): two people are in opposing factions.
  - NEUTRAL    : two people have no known relationship. Do NOT list Neutral pairs.

RULES OF INFERENCE (apply after every new event, recursively):
  Rule 1: Friend of a Friend = Friend  (F of F = F)
  Rule 2: Rival  of a Rival  = Friend  (R of R = F)
  Rule 3: Friend of a Rival  = Rival   (F of R = R)
  Rule 4: Rival  of a Friend = Rival   (R of F = R)

EXECUTION RULES:
  - Events are applied sequentially. Each event uses the relationship state
    produced by the previous step.
  - After each event, you MUST apply the inference rules to ALL existing
    members of BOTH involved components and list every implied pair explicitly.
  - Two people who are not yet transitively connected remain Neutral and
    must NOT appear in the output.

WORKED EXAMPLE:
  Initial state: all Neutral.
  Event 1: A and B form an ALLIANCE.
    State: [AB:F]
  Event 2: B and C form a RIVALRY.
    State: [AB:F, AC:R, BC:R]
  Event 3: D and E form an ALLIANCE.
    State: [AB:F, AC:R, BC:R, DE:F]\
"""

OUTPUT_FORMAT = """\
Output ONLY the following format. Do NOT use markdown code blocks. Do NOT add commentary.
1. TRACE: ["Step 1: [AB:F, AC:R, ...]", "Step 2: [...]", ...]
2. ANSWER: [AB:F, AC:R, ...]

CRITICAL RULES:
  - Use compact pair notation sorted alphabetically: 'AB:F' (Friends) or 'AB:R' (Rivals).
  - Every step MUST list ALL currently known non-Neutral pairs, not just new ones.
  - Do NOT use ellipses. Output the full relationship list at every single step.
  - Step 1 in TRACE = state AFTER applying the first event.\
"""

SYSTEM_PROMPT = (
    "You are a relational logic engine tracking a social graph.\n\n"
    + RULES_DESCRIPTION + "\n\n"
    + OUTPUT_FORMAT
)

# ---------------------------------------------------------------------------
# Trace Parsing Infrastructure
# ---------------------------------------------------------------------------

PAIR_RE = re.compile(r"\b([A-J]{2}):([FR])\b", re.IGNORECASE)
PAIR_B_RE = re.compile(r"\b(P\d{2}P\d{2}):([FR])\b", re.IGNORECASE)
STEP_NUM_RE = re.compile(r"Step\s*(\d+)\s*[:\-]", re.IGNORECASE)


def _extract_pairs_from_text(text: str) -> list:
    return [f"{m.group(1).upper()}:{m.group(2).upper()}"
            for m in PAIR_RE.finditer(text)]


def _extract_pairs_from_text_b(text: str) -> list:
    return [f"{m.group(1).upper()}:{m.group(2).upper()}"
            for m in PAIR_B_RE.finditer(text)]


def extract_social_trace(text: str, is_variant_b: bool = False) -> list:
    pair_fn = _extract_pairs_from_text_b if is_variant_b else _extract_pairs_from_text
    step_positions = [(m.start(), int(m.group(1))) for m in STEP_NUM_RE.finditer(text)]

    if not step_positions:
        return []

    steps = []
    for idx, (pos, step_num) in enumerate(step_positions):
        start = pos
        end   = step_positions[idx + 1][0] if idx + 1 < len(step_positions) else len(text)
        segment = text[start:end]

        pairs = pair_fn(segment)
        seen  = set()
        dedup = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        dedup_sorted = sorted(dedup)

        has_empty = bool(re.search(r"Step\s*\d+\s*[:\-]\s*\[\s*\]", segment, re.IGNORECASE))

        if dedup_sorted or has_empty:
            pair_str = ", ".join(dedup_sorted)
            steps.append(f"Step {step_num}: [{pair_str}]")

    return steps


def get_canonical_state(raw_str: str) -> str:
    if not raw_str or raw_str.strip() == "MISSING":
        return "MISSING"
    pairs = _extract_pairs_from_text(raw_str)
    if not pairs:
        if re.search(r"\[\s*\]", raw_str):
            return ""
        return ""
    return ",".join(sorted(set(p.upper() for p in pairs)))


def variant_b_to_standard(raw_str: str) -> str:
    label_map = {f"P{i+1:02d}": p for i, p in enumerate(PEOPLE)}
    result = raw_str
    for plabel, letter in sorted(label_map.items(), key=lambda x: -len(x[0])):
        result = result.replace(plabel, letter)
    return result


def extract_final_answer(text: str, is_variant_b: bool = False) -> str:
    ans_match = re.search(r"ANSWER\s*[:\-\*]+\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not ans_match:
        return "N/A"
    ans_section = ans_match.group(1).strip()
    ans_section = ans_section.split("\n\n")[0].strip()

    pair_fn = _extract_pairs_from_text_b if is_variant_b else _extract_pairs_from_text
    pairs   = pair_fn(ans_section)
    if not pairs:
        return "N/A"
    return f"[{', '.join(sorted(set(p.upper() for p in pairs)))}]"


def compute_divergence_step(gt_trace: list, pred_trace: list):
    for i, gt_step in enumerate(gt_trace):
        pred_step = pred_trace[i] if i < len(pred_trace) else "MISSING"
        if get_canonical_state(pred_step) != get_canonical_state(gt_step):
            return (i + 1, gt_step, pred_step)
    return (-1, None, None)


def evaluate_single(item: dict, model_output: str,
                    is_variant_b: bool = False) -> dict:
    raw_ans = extract_final_answer(model_output, is_variant_b=is_variant_b)

    if is_variant_b:
        ans_for_cmp  = get_canonical_state(variant_b_to_standard(raw_ans))
        pred_trace   = extract_social_trace(model_output, is_variant_b=True)
        pred_trace_c = [variant_b_to_standard(s) for s in pred_trace]
    else:
        ans_for_cmp  = get_canonical_state(raw_ans)
        pred_trace   = extract_social_trace(model_output, is_variant_b=False)
        pred_trace_c = pred_trace

    gt_canonical = get_canonical_state(item["gt_ans"])
    is_correct   = (
        ans_for_cmp == gt_canonical
        and gt_canonical not in ("", "MISSING")
        and raw_ans != "N/A"
    )

    output_upper      = model_output.upper()
    has_trace_keyword = "TRACE" in output_upper
    has_answer        = (raw_ans != "N/A")

    is_format_failure = (len(pred_trace) == 0 and not has_trace_keyword) or not has_answer
    is_truncated = (not is_format_failure and len(pred_trace) < item["depth"])

    div_step, exp_at_div, pred_at_div = compute_divergence_step(
        item["gt_trace"], pred_trace_c
    )
    is_tfbc         = is_correct and (div_step != -1)
    trace_length_ok = (len(pred_trace) == item["depth"])

    return {
        "depth":              item["depth"],
        "seed":               item["seed"],
        "is_correct":         is_correct,
        "div_step":           div_step,
        "is_tfbc":            is_tfbc,
        "is_format_failure":  is_format_failure,
        "is_truncated":       is_truncated,
        "trace_length_ok":    trace_length_ok,
        "pred_trace_len":     len(pred_trace),
        "expected_trace_len": item["depth"],
        "expected_at_div":    exp_at_div,
        "predicted_at_div":   pred_at_div,
        "parsed_ans_raw":     raw_ans,
        "gt_ans":             item["gt_ans"],
        "model_output":       model_output,
    }

# ---------------------------------------------------------------------------
# Benchmark Data Generation Engine
# ---------------------------------------------------------------------------

class SocialGraphGenerator:
    def __init__(self, people=None):
        self.people = people or PEOPLE

    def _init_state(self):
        return (
            {p: i for i, p in enumerate(self.people)},
            {p: 0 for p in self.people},
        )

    def _format_state(self, comp_id, faction):
        rels = []
        for i in range(len(self.people)):
            for j in range(i + 1, len(self.people)):
                x, y = self.people[i], self.people[j]
                if comp_id[x] == comp_id[y]:
                    tag = "F" if faction[x] == faction[y] else "R"
                    rels.append(f"{x}{y}:{tag}")
        return sorted(rels)

    def _apply_event(self, comp_id, faction, u, v, rel):
        comp_id = dict(comp_id)
        faction = dict(faction)
        cu, cv  = comp_id[u], comp_id[v]
        if cu == cv:
            return comp_id, faction
        target = faction[u] if rel == "ALLIANCE" else 1 - faction[u]
        flip   = faction[v] ^ target
        for p in self.people:
            if comp_id[p] == cv:
                comp_id[p]  = cu
                faction[p] ^= flip
        return comp_id, faction

    def solve(self, depth: int, seed: int):
        random.seed(seed)
        comp_id, faction = self._init_state()
        events, gt_trace = [], []
        for step in range(depth):
            u, v   = sorted(random.sample(self.people, 2))
            cu, cv = comp_id[u], comp_id[v]
            if cu == cv:
                rel = "ALLIANCE" if faction[u] == faction[v] else "RIVALRY"
            else:
                rel = random.choice(["ALLIANCE", "RIVALRY"])
                comp_id, faction = self._apply_event(comp_id, faction, u, v, rel)
            events.append(f"Step {step+1}: {u} and {v} form an {rel}.")
            rels = self._format_state(comp_id, faction)
            gt_trace.append(f"Step {step+1}: [{', '.join(rels)}]")
        final_rels = self._format_state(comp_id, faction)
        gt_ans     = f"[{', '.join(final_rels)}]"
        return "\n".join(events), gt_trace, gt_ans

    def build(self, depths, n_per_depth: int, seed_offset: int = 0):
        dataset = []
        for d in depths:
            seen      = set()
            collected = 0
            attempt   = 0
            while collected < n_per_depth:
                seed_val = d * 10000 + attempt + seed_offset
                attempt += 1
                prompt_text, gt_trace, gt_ans = self.solve(d, seed=seed_val)
                if prompt_text not in seen:
                    seen.add(prompt_text)
                    dataset.append({
                        "depth":    d,
                        "seed":     seed_val,
                        "prompt":   prompt_text,
                        "gt_trace": gt_trace,
                        "gt_ans":   gt_ans,
                    })
                    collected += 1
        return dataset

# ---------------------------------------------------------------------------
# Systematic Prompt Scaffold Definitions
# ---------------------------------------------------------------------------

def build_system_prompt(variant: str = "A") -> str:
    if variant == "A":
        return (
            "You are a relational logic engine tracking a social graph.\n\n"
            + RULES_DESCRIPTION + "\n\n"
            + OUTPUT_FORMAT
        )
    elif variant == "B":
        label_map = {p: f"P{i+1:02d}" for i, p in enumerate(PEOPLE)}
        rules_b   = RULES_DESCRIPTION
        for letter, plabel in sorted(label_map.items(), key=lambda x: x[0]):
            rules_b = re.sub(rf'\b{letter}\b', plabel, rules_b)
        rules_b = (rules_b
                   .replace("AB:F",  "P01P02:F")
                   .replace("BC:R",  "P02P03:R")
                   .replace("AC:R",  "P01P03:R")
                   .replace("DE:F",  "P04P05:F"))
        output_b = (OUTPUT_FORMAT
                    .replace("'AB:F' (Friends) or 'AB:R' (Rivals)",
                             "'P01P02:F' (Friends) or 'P01P02:R' (Rivals)")
                    .replace('"Step 1: [AB:F, AC:R, ...]"',
                             '"Step 1: [P01P02:F, P01P03:R, ...]"')
                    .replace("ANSWER: [AB:F, AC:R, ...]",
                             "ANSWER: [P01P02:F, P01P03:R, ...]"))
        return (
            "You are a relational logic engine tracking a social graph.\n"
            "Person labels are opaque identifiers — no alphabetical ordering.\n\n"
            + rules_b + "\n\n" + output_b
        )
    elif variant == "C":
        return (
            "You are a relational logic engine tracking a social graph.\n\n"
            + OUTPUT_FORMAT + "\n\n"
            "Initial state: all 10 people (A-J) are Neutral to each other.\n\n"
            + RULES_DESCRIPTION
        )
    elif variant == "VERBOSE":
        return (
            "You are a relational logic engine tracking a social graph.\n\n"
            + RULES_DESCRIPTION + "\n\n"
            "CRITICAL ABLATION INSTRUCTION: After EVERY single event, you MUST "
            "explicitly write out the COMPLETE list of all currently known "
            "non-Neutral relationships before proceeding.\n\n"
            + OUTPUT_FORMAT
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant!r}")

# ---------------------------------------------------------------------------
# Disjoint Post-Processing and Structural Fix Framework
# ---------------------------------------------------------------------------

PROSE_PATTERNS = [
    r"\bis\s+(in)?correct\b",
    r"\bshould\s+be\b",
    r"\bthe\s+correct\b",
    r"\binstead\b",
    r"(?<!\w)->(?!\w)",
    r"\bso\s+the\b",
    r"\balso\s+incorrect\b",
    r"\bwe\s+(can\s+)?infer\b",
    r"\btherefore\b",
    r"\bthis\s+means\b",
    r"\bby\s+inference\b",
    r"\bnote\s*:",
    r"\bsorry\b",
    r"\bhowever\b",
    r"\bactually\b",
    r"\bmistake\b",
]
PROSE_RE = re.compile("|".join(PROSE_PATTERNS), re.IGNORECASE)


def _step_has_prose(step_str: str) -> bool:
    return bool(PROSE_RE.search(step_str))


def _output_has_corruption(model_output: str, steps: list) -> bool:
    return any(_step_has_prose(s) for s in steps)


def _is_valid_social_tokens(pairs: list) -> bool:
    if not pairs:
        return True
    seen = set()
    for tok in pairs:
        m = re.fullmatch(r"([A-J]{2}):([FR])", tok.upper())
        if not m:
            return False
        pair = m.group(1)
        if pair not in VALID_PAIRS:
            return False
        if pair in seen:
            return False
        seen.add(pair)
    return True


def _extract_pairs_from_step(step_str: str) -> list:
    return _extract_pairs_from_text(step_str)


def _missing_trace_or_answer(model_output: str) -> bool:
    out_upper = model_output.upper()
    return "TRACE" not in out_upper or "ANSWER" not in out_upper


def _compute_failure_type(rec: dict, steps: list) -> str:
    if rec.get("is_api_failure", False):
        return "api"
    if rec.get("is_correct", False):
        return "none"
    if rec.get("is_format_failure", False):
        return "format"
    if rec.get("is_truncated", False) and not _output_has_corruption(rec.get("model_output", ""), steps):
        return "trunc"
    if rec.get("_constraint_violation", False):
        return "constraint"
    if steps:
        all_valid = all(
            _is_valid_social_tokens(_extract_pairs_from_step(s))
            for s in steps
        )
        if all_valid:
            return "logic"
    return "format"


def fix_record(record: dict) -> dict:
    rec          = copy.deepcopy(record)
    model_output = rec.get("model_output", "")
    reasons      = []

    if rec.get("is_api_failure", False):
        rec["failure_type"] = "api"
        return rec

    rec["raw_pred_trace_len"]    = rec.get("pred_trace_len")
    rec["raw_predicted_at_div"]  = rec.get("predicted_at_div")
    rec["raw_trace_length_ok"]   = rec.get("trace_length_ok")
    rec["raw_is_format_failure"] = rec.get("is_format_failure")

    is_variant_b = bool(re.search(r"\bP\d{2}P\d{2}:", model_output))
    steps = extract_social_trace(model_output, is_variant_b=is_variant_b)

    expected_len = rec.get("expected_trace_len", rec.get("depth", 0))
    rec["pred_trace_len"]  = len(steps)
    rec["trace_length_ok"] = (len(steps) == expected_len)

    out_upper = model_output.upper()
    raw_ans   = extract_final_answer(model_output, is_variant_b=is_variant_b)
    has_trace_kw = "TRACE" in out_upper
    has_answer   = (raw_ans != "N/A")
    new_format_failure = (len(steps) == 0 and not has_trace_kw) or not has_answer

    if rec.get("is_format_failure", False) and not new_format_failure:
        rec["is_format_failure"] = False
        reasons.append("P1:parser_fix_restored_valid_steps")

    if not rec.get("is_format_failure", False) and new_format_failure:
        rec["is_format_failure"] = True
        reasons.append("P2:format_failure_missing_answer_or_steps")

    rec["is_truncated"] = (
        not rec.get("is_format_failure", False)
        and len(steps) < expected_len
    )
    if rec.get("is_truncated") and rec.get("raw_pred_trace_len", 0) != len(steps):
        reasons.append(f"P4:truncation_recomputed({rec['raw_pred_trace_len']}->{len(steps)})")

    if rec.get("is_truncated", False) and _output_has_corruption(model_output, steps):
        rec["is_format_failure"] = True
        rec["is_truncated"]      = False
        reasons.append("F3:truncated_but_corrupted->format_failure")

    if not rec.get("is_format_failure", False) and _missing_trace_or_answer(model_output):
        rec["is_format_failure"] = True
        reasons.append("F4:missing_TRACE_or_ANSWER")

    if not rec.get("is_format_failure", False) and _output_has_corruption(model_output, steps):
        rec["is_format_failure"] = True
        reasons.append("F1:prose_in_steps")

    if not rec.get("is_format_failure", False) and not rec.get("_constraint_violation", False):
        for i, s in enumerate(steps, start=1):
            pairs = _extract_pairs_from_step(s)
            if not _is_valid_social_tokens(pairs):
                rec["_constraint_violation"] = True
                reasons.append(f"F2b:invalid_pair_token_at_step{i}")
                break

    rec["pred_trace_len"]  = len(steps)
    rec["trace_length_ok"] = (len(steps) == expected_len)

    if rec.get("is_format_failure", False):
        rec["is_correct"]       = False
        rec["predicted_at_div"] = "MISSING"
        rec["is_truncated"]      = False

    rec["failure_type"] = _compute_failure_type(rec, steps)

    if reasons:
        rec["_fix_reasons"] = reasons

    return rec


def fix_all_results(results: list) -> tuple:
    fixed        = []
    change_log  = []
    type_counts = {}
    parser_fixes = 0

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
                "pred_len":       f"{rec.get('pred_trace_len')} -> {f.get('pred_trace_len')}",
            })
            if any("P1" in r or "P2" in r or "P4" in r for r in f["_fix_reasons"]):
                parser_fixes += 1

    summary = {
        "total":                  len(results),
        "modified":               len(change_log),
        "parser_fixes":           parser_fixes,
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


def fit_pd_model(depth_levels, successes_per_depth, n_per_depth, bootstrap_reps=2000):
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
    return (pd_mle,
            float(np.percentile(pd_boot, 2.5)),
            float(np.percentile(pd_boot, 97.5)))


def aggregate_by_depth(results: list) -> dict:
    by_depth = {}
    for r in results:
        d = r["depth"]
        if d not in by_depth:
            by_depth[d] = {
                "n": 0, "correct": 0, "div_steps": [], "tfbc": 0,
                "format_failures": 0, "constraint_violations": 0,
                "truncations": 0, "completion_tokens": [],
            }
        s = by_depth[d]
        s["n"] += 1
        if r.get("is_correct") and not r.get("is_api_failure"):
            s["correct"] += 1
        ft = r.get("failure_type", "")
        if r.get("div_step", -1) > 0 and ft == "logic":
            s["div_steps"].append(r["div_step"])
        if r.get("is_tfbc"):
            s["tfbc"] += 1
        if ft in ("format",):
            s["format_failures"] += 1
        if ft == "constraint":
            s["constraint_violations"] += 1
        if ft == "trunc":
            s["truncations"] += 1
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
            "n": n, "correct": c,
            "accuracy":              c / n if n > 0 else 0.0,
            "ci_lo": lo, "ci_hi":  hi,
            "n_incorrect":          n - c,
            "div_steps":            divs,
            "avg_div_step":         float(np.mean(divs)) if divs else None,
            "early_failures":       sum(1 for s in divs if s <= 3),
            "mid_failures":         sum(1 for s in divs if 4 <= s <= 10),
            "late_failures":        sum(1 for s in divs if s > 10),
            "tfbc_count":           v["tfbc"],
            "tfbc_rate":            v["tfbc"] / c if c > 0 else None,
            "format_failures":      fmt_f,
            "format_failure_rate":  fmt_f / n if n > 0 else 0.0,
            "constraint_violations": v["constraint_violations"],
            "truncations":          v["truncations"],
            "avg_completion_tokens": float(np.mean(toks)) if toks else None,
        }
    return out


def aggregate_overall(results: list) -> dict:
    n      = len(results)
    c      = sum(1 for r in results if r.get("is_correct") and not r.get("is_api_failure"))
    tfbc   = sum(1 for r in results if r.get("is_tfbc"))
    fmt_f  = sum(1 for r in results if r.get("failure_type") == "format")
    constr = sum(1 for r in results if r.get("failure_type") == "constraint")
    api_f  = sum(1 for r in results if r.get("is_api_failure"))
    lo, hi = clopper_pearson_ci(c, n)
    divs   = [r["div_step"] for r in results
              if r.get("div_step", -1) > 0 and r.get("failure_type") == "logic"]
    toks   = [r["completion_tokens"] for r in results
              if r.get("completion_tokens", 0) > 0]
    return {
        "n": n, "correct": c,
        "accuracy":               c / n if n > 0 else 0.0,
        "ci_lo": lo, "ci_hi":    hi,
        "tfbc_total":             tfbc,
        "tfbc_rate":              tfbc / c if c > 0 else None,
        "format_failures":        fmt_f,
        "format_failure_rate":    fmt_f / n if n > 0 else 0.0,
        "constraint_violations":  constr,
        "constraint_rate":        constr / n if n > 0 else 0.0,
        "api_failures":           api_f,
        "avg_div_step":           float(np.mean(divs)) if divs else None,
        "avg_completion_tokens":  float(np.mean(toks)) if toks else None,
        "total_tokens_used":      sum(toks),
    }


def failure_timing_breakdown(results: list) -> dict:
    divs = [r["div_step"] for r in results
            if r.get("div_step", -1) > 0 and r.get("failure_type") == "logic"]
    if not divs:
        return {"early": None, "mid": None, "late": None,
                "n_logic_failures": 0, "avg_div_step": None,
                "median_div_step": None}
    total = len(divs)
    return {
        "early":             sum(1 for s in divs if s <= 3)       / total,
        "mid":               sum(1 for s in divs if 4 <= s <= 10) / total,
        "late":              sum(1 for s in divs if s > 10)        / total,
        "n_logic_failures":  total,
        "avg_div_step":      float(np.mean(divs)),
        "median_div_step":   float(sorted(divs)[total // 2]),
    }

# ---------------------------------------------------------------------------
# State Resumption and Provenance Tracking
# ---------------------------------------------------------------------------

def discover_files_for_model(model_id: str, results_dir: str) -> list:
    safe    = model_id_to_safe(model_id)
    pattern = os.path.join(results_dir, f"results_*{safe}*.json")
    files   = glob.glob(pattern)
    if not files:
        _, *rest = model_id.split("/")
        if rest:
            pat2  = os.path.join(results_dir, f"results_*{'_'.join(rest)}*.json")
            files = glob.glob(pat2)
    return sorted(set(files), key=os.path.getmtime)


def load_and_merge_files(file_paths: list, model_id: str):
    by_seed      = {}
    provenance   = []
    total_loaded = 0
    dupes        = 0

    for fpath in file_paths:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"    [SKIP] {os.path.basename(fpath)}: structural schema error.")
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
                error_msg = str(r.get("error", "")).lower()
                is_transient = any(x in error_msg for x in [
                    "402", "insufficient", "credits", "connection",
                    "timeout", "timed out", "rate limit", "429", "500", "503",
                ])
                is_empty = (
                    r.get("prompt_tokens", 0) == 0
                    and r.get("completion_tokens", 0) == 0
                    and not r.get("model_output", "").strip()
                    and not r.get("is_correct", False)
                )
                is_short = len(r.get("model_output", "").strip()) < 10
                if is_transient or is_empty or is_short:
                    r["is_api_failure"]    = True
                    r["is_format_failure"] = False

                if seed not in by_seed:
                    by_seed[seed] = r
                else:
                    dupes += 1
                    if by_seed[seed].get("is_api_failure", False) and not r.get("is_api_failure", False):
                        by_seed[seed] = r

            print(f"    Loaded {n:3d} records from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"    [ERROR] {fpath}: {e}")

    return list(by_seed.values()), provenance, total_loaded, dupes


def classify_results(merged: list, benchmark_by_seed: dict):
    by_seed = {r["seed"]: r for r in merged}
    good, retry, missing = [], [], []
    for seed, item in benchmark_by_seed.items():
        if seed not in by_seed:
            missing.append(item)
        else:
            r        = by_seed[seed]
            is_empty = (
                r.get("prompt_tokens", 0) == 0
                and r.get("completion_tokens", 0) == 0
                and not r.get("model_output", "").strip()
                and not r.get("is_correct", False)
            )
            if r.get("is_api_failure", False) or is_empty:
                retry.append(item)
            else:
                good.append(r)
    return good, retry, missing

# ---------------------------------------------------------------------------
# API Execution Harness
# ---------------------------------------------------------------------------

def run_on_items(model_id: str, items: list, label: str = "",
                 system_prompt: str = None, is_variant_b: bool = False,
                 output_file: str = None, existing_results: list = None) -> list:
    results          = []
    existing_results = existing_results or []
    n_total          = len(items)
    sys_prompt       = system_prompt or SYSTEM_PROMPT

    if n_total == 0:
        return results

    wait = get_post_call_wait(model_id)
    print(f"\n{'='*70}")
    print(f"[RUN] {label or model_id}  |  {n_total} instances  |  temp={TEMPERATURE}")
    print(f"{'='*70}")

    for i, item in enumerate(items):
        success  = False
        retries  = 0
        last_err = None

        while not success and retries < MAX_RETRIES:
            try:
                dynamic_tokens = MAX_TOKENS
                if item.get("depth", 0) >= 40:
                    dynamic_tokens = 20000

                t0       = time.time()
                response = client.chat.completions.create(
                    model=model_id,
                    temperature=TEMPERATURE,
                    max_tokens=dynamic_tokens,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": item["prompt"]},
                    ],
                )
                elapsed    = time.time() - t0
                raw_output = response.choices[0].message.content.strip()
                pt  = getattr(response.usage, "prompt_tokens",     0) or 0
                ct  = getattr(response.usage, "completion_tokens", 0) or 0
                tt  = getattr(response.usage, "total_tokens",      0) or 0

                result = evaluate_single(item, raw_output, is_variant_b=is_variant_b)
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

                ft   = "PASS" if result["is_correct"] else "FAIL"
                tags = (
                    (" [TFBC]"   if result["is_tfbc"]          else "") +
                    (" [FORMAT]" if result["is_format_failure"] else "") +
                    (" [TRUNC]"  if result["is_truncated"]      else "")
                )
                print(f"  [{i+1:3d}/{n_total}] N={item['depth']:2d} | "
                      f"seed={item['seed']} | {ft}{tags} | "
                      f"k*={result['div_step']:3d} | "
                      f"steps={result['pred_trace_len']}/{item['depth']} | "
                      f"tok={ct} | {elapsed:.1f}s")

                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(existing_results + results, f, indent=2)
                success = True
                time.sleep(wait)

            except Exception as e:
                retries  += 1
                last_err  = str(e)
                is_credit = "402" in last_err or "credits" in last_err.lower()
                print(f"  [RETRY {retries}/{MAX_RETRIES}] "
                      f"{'[CREDIT]' if is_credit else ''} {e}")
                if retries >= MAX_RETRIES:
                    results.append({
                        "depth": item["depth"], "seed": item["seed"],
                        "is_correct": False, "div_step": -1,
                        "is_tfbc": False, "is_format_failure": False,
                        "is_truncated": False, "is_api_failure": True,
                        "failure_type": "api",
                        "trace_length_ok": False, "pred_trace_len": 0,
                        "expected_trace_len": item["depth"],
                        "model": model_id, "label": label,
                        "error": last_err,
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    })
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
        print(line); lines.append(line)

    p("=" * 80)
    p(f"DOMAIN: {DOMAIN}  |  RUN: {RUN_ID}  |  temp={TEMPERATURE}  |  n={N_PER_CELL}/cell")
    p("=" * 80)

    for model_id, results in all_model_results.items():
        short = model_id.split("/")[-1]
        p(f"\n--- {short} ---")
        p(f"{'Depth':>6}  {'N':>4}  {'Corr':>4}  {'Acc%':>6}  {'95% CI':^20}  "
          f"{'AvgK*':>7}  {'Early%':>7}  {'Mid%':>7}  {'Late%':>7}  "
          f"{'TFBC%':>7}  {'FmtFail%':>9}  {'Constr%':>8}  {'Trunc%':>7}  {'AvgTok':>7}")
        p("-" * 122)

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
            trunc  = f"{v['truncations']/v['n']*100:5.1f}%"           if v["n"] else "  N/A"
            avg_t  = f"{v['avg_completion_tokens']:6.0f}" if v["avg_completion_tokens"] else "   N/A"
            p(f"{d:>6}  {v['n']:>4}  {v['correct']:>4}  {v['accuracy']*100:5.1f}%  "
              f"{ci_str:^20}  {avg_k}  {early:>7}  {mid:>7}  {late:>7}  "
              f"{tfbc:>7}  {fmt_f:>9}  {constr:>8}  {trunc:>7}  {avg_t:>7}")

        ov  = aggregate_overall(results)
        ftb = failure_timing_breakdown(results)
        p("-" * 122)
        ft_counts = {}
        for r in results:
            t = r.get("failure_type", "unknown")
            ft_counts[t] = ft_counts.get(t, 0) + 1
        p("  Failure taxonomy: " +
          "  |  ".join(f"{k}={v}" for k, v in sorted(ft_counts.items())))
        tfbc_str   = f"TFBC={ov['tfbc_rate']*100:.1f}%"   if ov["tfbc_rate"] is not None else "TFBC=N/A"
        fmt_str    = f"{ov['format_failure_rate']*100:.1f}%"
        constr_str = f"{ov['constraint_rate']*100:.1f}%"
        tok_str    = f"{ov['avg_completion_tokens']:.0f}"  if ov["avg_completion_tokens"] else "N/A"
        p(f"{'TOTAL':>6}  {ov['n']:>4}  {ov['correct']:>4}  {ov['accuracy']*100:5.1f}%  "
          f"[{ov['ci_lo']*100:5.1f}%, {ov['ci_hi']*100:5.1f}%]  "
          f"  {tfbc_str}  FmtFail={fmt_str}  Constr={constr_str}  AvgTok={tok_str}")
        if ftb["n_logic_failures"] > 0:
            p(f"  k* timing (logic failures, n={ftb['n_logic_failures']}): "
              f"Early={ftb['early']*100:.1f}%  Mid={ftb['mid']*100:.1f}%  "
              f"Late={ftb['late']*100:.1f}%  AvgK*={ftb['avg_div_step']:.2f}")

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
        for mid, res in all_model_results.items():
            agg = aggregate_by_depth(res)
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
        logic_fail = [r for r in results if r.get("failure_type") == "logic"]
        sample     = random.sample(logic_fail, min(n_per_model, len(logic_fail)))
        con_fail   = [r for r in results
                      if r.get("failure_type") == "constraint" and r not in sample]
        sample    += random.sample(con_fail, min(3, len(con_fail)))
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
                "is_truncated":       r.get("is_truncated"),
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
# Auxiliary Ablation Framework (Intervention Diagnostics)
# ---------------------------------------------------------------------------

def run_prompt_sensitivity(gen: SocialGraphGenerator) -> list:
    print(f"\n{'='*70}")
    print(f"[PROMPT SENSITIVITY]  Model: {PROMPT_SENS_MODEL}")
    print(f"Depth: {PROMPT_SENS_DEPTH}  |  n={PROMPT_SENS_N} per variant")
    print(f"{'='*70}")

    out_file = os.path.join(OUTPUT_DIR, f"results_prompt_sens_{RUN_ID}.json")
    all_sens = []

    for variant in ["A", "B", "C"]:
        dataset = gen.build([PROMPT_SENS_DEPTH], PROMPT_SENS_N,
                            seed_offset=PROMPT_SENS_OFFSET)
        results = run_on_items(
            model_id      = PROMPT_SENS_MODEL,
            items         = dataset,
            label         = f"PromptSens_Variant{variant}",
            system_prompt = build_system_prompt(variant),
            is_variant_b  = (variant == "B"),
            output_file   = out_file,
            existing_results = all_sens,
        )
        results, _ = fix_all_results(results)
        all_sens.extend(results)
        ov = aggregate_overall(results)
        print(f"  Variant {variant}: {ov['accuracy']*100:.1f}%  "
              f"[{ov['ci_lo']*100:.1f}%, {ov['ci_hi']*100:.1f}%]")

    with open(out_file, "w") as f:
        json.dump(all_sens, f, indent=2)
    return all_sens


def _run_mcnemar_inline(abl_results: list):
    if not _HAS_STATSMODELS:
        return
    std  = {r["seed"]: r["is_correct"] for r in abl_results
            if r.get("label") == "Ablation_Standard"}
    verb = {r["seed"]: r["is_correct"] for r in abl_results
            if r.get("label") == "Ablation_Verbose"}
    common = sorted(set(std) & set(verb))
    if len(common) < 5:
        return
    a = sum(1 for s in common if     std[s] and     verb[s])
    b = sum(1 for s in common if     std[s] and not verb[s])
    c = sum(1 for s in common if not std[s] and     verb[s])
    d = sum(1 for s in common if not std[s] and not verb[s])
    res = sm_mcnemar([[a, b], [c, d]], exact=True)
    print(f"\n  [McNemar inline] n={len(common)}  "
          f"both={a}  std_only={b}  verb_only={c}  both_wrong={d}")
    print(f"  p={res.pvalue:.4f}  "
          f"({'significant' if res.pvalue < 0.05 else 'not significant'})")


def run_verbosity_ablation(gen: SocialGraphGenerator) -> list:
    print(f"\n{'='*70}")
    print(f"[VERBOSITY ABLATION]  Model: {ABLATION_MODEL}")
    print(f"Depth: {ABLATION_DEPTH}  |  n={ABLATION_N}")
    print(f"{'='*70}")

    dataset  = gen.build([ABLATION_DEPTH], ABLATION_N, seed_offset=ABLATION_OFFSET)
    out_file = os.path.join(OUTPUT_DIR, f"results_ablation_{RUN_ID}.json")
    all_abl  = []

    for label, variant in [("Ablation_Standard", "A"), ("Ablation_Verbose", "VERBOSE")]:
        results = run_on_items(
            model_id      = ABLATION_MODEL,
            items         = dataset,
            label         = label,
            system_prompt = build_system_prompt(variant),
            output_file   = out_file,
            existing_results = all_abl,
        )
        results, _ = fix_all_results(results)
        all_abl.extend(results)
        ov = aggregate_overall(results)
        print(f"  {label}: {ov['accuracy']*100:.1f}%  "
              f"[{ov['ci_lo']*100:.1f}%, {ov['ci_hi']*100:.1f}%]")

    _run_mcnemar_inline(all_abl)
    with open(out_file, "w") as f:
        json.dump(all_abl, f, indent=2)
    return all_abl

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print(f"SMART RESUME D3 SOCIAL LOGIC — {RUN_ID}")
    print(f"Results dir: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Models: {len(ALL_MODELS)}  |  Depths: {DEPTH_LEVELS}")
    print(f"n/cell: {N_PER_CELL}  |  Total main calls: "
          f"{N_PER_CELL * len(DEPTH_LEVELS) * len(ALL_MODELS):,}")
    print("=" * 70)

    gen           = SocialGraphGenerator()
    main_dataset  = gen.build(DEPTH_LEVELS, N_PER_CELL, seed_offset=0)
    bench_by_seed = {item["seed"]: item for item in main_dataset}
    expected      = N_PER_CELL * len(DEPTH_LEVELS)

    print(f"\n[DATASET] Rebuilt {len(main_dataset)} benchmark instances")

    all_model_results = {}
    all_results_flat  = []
    provenance_map    = {}
    fix_summaries     = {}

    for model_id in ALL_MODELS:
        safe  = model_id_to_safe(model_id)
        short = model_id.split("/")[-1]

        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")

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
            provenance_map[model_id] = []

        good, retry_items, missing_items = classify_results(merged, bench_by_seed)
        n_need_run = len(retry_items) + len(missing_items)
        print(f"  Good: {len(good)}  |  Retry: {len(retry_items)}"
              f"  |  Missing: {len(missing_items)}  |  To run: {n_need_run}")

        if n_need_run > 0:
            run_file = os.path.join(OUTPUT_DIR, f"results_main_{safe}_{RUN_ID}.json")
            new_results = run_on_items(
                model_id         = model_id,
                items            = retry_items + missing_items,
                label            = f"Main_{short}",
                system_prompt    = SYSTEM_PROMPT,
                output_file      = run_file,
                existing_results = good,
            )
            raw_results = good + new_results
        else:
            raw_results = good
            print(f"  Complete — no API calls needed.")

        seeds = [r["seed"] for r in raw_results]
        assert len(set(seeds)) == len(seeds), f"[BUG] Duplicate seeds in {model_id}"

        print(f"\n  Applying classification fixes...")
        fixed_results, fix_summary = fix_all_results(raw_results)
        fix_summaries[model_id]    = fix_summary

        n_parser_fixed = fix_summary.get("parser_fixes", 0)
        print(f"    Modified: {fix_summary['modified']}/{fix_summary['total']}  "
              f"|  Parser fixes: {n_parser_fixed}")
        print(f"    failure_type breakdown: " +
              "  ".join(f"{k}={v}" for k, v in sorted(fix_summary["failure_type_breakdown"].items())))

        final_file = os.path.join(OUTPUT_DIR, f"results_FINAL_{safe}_{RUN_ID}.json")
        with open(final_file, "w") as f:
            json.dump(fixed_results, f, indent=2)

        n_final   = len(fixed_results)
        n_api_err = sum(1 for r in fixed_results if r.get("is_api_failure", False))
        n_correct = sum(1 for r in fixed_results if r.get("is_correct",     False))
        icon      = "✓" if n_final == expected and n_api_err == 0 else "⚠"
        print(f"\n  {icon} {short}: {n_correct}/{n_final} correct "
              f"({n_correct/n_final*100:.1f}%)  [{n_final}/{expected} instances]")

        all_model_results[model_id] = fixed_results
        all_results_flat.extend(fixed_results)

    ordered = {m: all_model_results[m] for m in ALL_MODELS if m in all_model_results}

    print(f"\n{'='*70}")
    print("FINAL VALIDATION SUMMARY")
    for model_id, res in ordered.items():
        n_api = sum(1 for r in res if r.get("is_api_failure", False))
        ok    = len(res) == expected and n_api == 0
        print(f"  {'✓' if ok else '✗'} {model_id.split('/')[-1]:<35}"
              f"  {len(res):3d}/{expected}  api={n_api}  "
              f"{'OK' if ok else 'INCOMPLETE'}")

    print(f"\n{'='*70}")
    print("p_d MODEL FITS")
    pd_fits = {}
    for model_id, res in ordered.items():
        agg   = aggregate_by_depth(res)
        deps  = sorted(agg.keys())
        succs = [agg[d]["correct"] for d in deps]
        ns    = [agg[d]["n"]       for d in deps]
        fit   = fit_pd_model(deps, succs, ns, BOOTSTRAP_REPS)
        pd_fits[model_id] = fit
        short = model_id.split("/")[-1]
        if not math.isnan(fit[0]):
            print(f"  {short}: p_d={fit[0]:.4f}  "
                  f"95% CI=[{fit[1]:.4f}, {fit[2]:.4f}]")

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{RUN_ID}.txt")
    print_and_save_summary(ordered, pd_fits, summary_path)

    val_path = os.path.join(OUTPUT_DIR, f"parser_validation_{RUN_ID}.json")
    save_parser_validation_set(all_results_flat, val_path, n_per_model=10)

    sens_flat = run_prompt_sensitivity(gen)
    abl_flat  = run_verbosity_ablation(gen)

    all_type_counts = {}
    per_model_bd    = {}
    for model_id, results in ordered.items():
        mc = {}
        for r in results:
            t = r.get("failure_type", "unknown")
            mc[t] = mc.get(t, 0) + 1
            all_type_counts[t] = all_type_counts.get(t, 0) + 1
        per_model_bd[model_id] = dict(sorted(mc.items()))

    combined = {
        "run_id":        RUN_ID,
        "domain":        DOMAIN,
        "temperature":   TEMPERATURE,
        "n_per_cell":    N_PER_CELL,
        "depth_levels":  DEPTH_LEVELS,
        "models":        ALL_MODELS,
        "pd_fits":       {m: list(v) for m, v in pd_fits.items()},
        "main_results":  all_results_flat,
        "prompt_sens":   sens_flat,
        "ablation":      abl_flat,
        "provenance":    provenance_map,
        "fix_summaries": fix_summaries,
        "validation": {
            m: {
                "n_total":        len(r),
                "n_correct":      sum(1 for x in r if x.get("is_correct")),
                "n_api":          sum(1 for x in r if x.get("is_api_failure")),
                "n_format":       sum(1 for x in r if x.get("failure_type") == "format"),
                "n_logic":        sum(1 for x in r if x.get("failure_type") == "logic"),
                "n_constraint":   sum(1 for x in r if x.get("failure_type") == "constraint"),
                "n_trunc":        sum(1 for x in r if x.get("failure_type") == "trunc"),
                "complete":       len(r) == expected,
            }
            for m, r in ordered.items()
        }
    }
    combined_path = os.path.join(OUTPUT_DIR, f"combined_{RUN_ID}.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[SAVED] Combined -> {combined_path}")

    summary_json_path = combined_path.replace(".json", "_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump({
            "failure_type_breakdown": dict(sorted(all_type_counts.items())),
            "per_model_breakdown":    per_model_bd,
            "fix_summaries":          fix_summaries,
        }, f, indent=2)
    print(f"[SAVED] Summary JSON -> {summary_json_path}")

    print(f"\n{'='*60}")
    print("FAILURE TYPE BREAKDOWN (all models combined)")
    print(f"{'='*60}")
    total_all = len(all_results_flat)
    for ft, cnt in sorted(all_type_counts.items()):
        pct = 100 * cnt / total_all if total_all else 0
        print(f"  {ft:<12} {cnt:>6}  ({pct:.1f}%)")
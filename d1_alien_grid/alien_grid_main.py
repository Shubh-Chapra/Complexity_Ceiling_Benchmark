"""

Resume or finalize a D1 Alien Grid benchmark run across multiple models.

Scans ./results/ for any existing result files, merges partial runs by seed,
re-runs seeds that previously failed with API errors, and runs missing seeds
from scratch. Writes per-model final files, a summary table, p_d fits, and a
parser validation sample.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python alien_grid_main.py
"""

import json
import time
import os
import re
import random
import math
import glob
import numpy as np
from datetime import datetime
from openai import OpenAI
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "./results"

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
DOMAIN         = "D1_Alien_Grid"
BOOTSTRAP_REPS = 2000

MAX_RETRIES    = 5
RETRY_WAIT     = 20
MAX_TOKENS     = 4096

POST_CALL_WAIT = {
    "anthropic":  5.0,
    "google":     3.0,
    "openai":     2.0,
    "deepseek":   2.0,
    "meta-llama": 2.0,
}
DEFAULT_WAIT = 2.0

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_smart_resume"

RESULTS_FILE_PATTERN = re.compile(
    r"results_(?:main|FINAL|ablation)_(.+?)_\d{8}_\d{6}.*\.json$"
)


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not set. Run: export OPENROUTER_API_KEY='sk-or-...'"
    )

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def clopper_pearson_ci(successes, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    lo = beta_dist.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return (float(lo), float(hi))


def _nll(pd, depths, k, n):
    pd = max(1e-9, min(1.0 - 1e-9, pd))
    probs = np.clip(pd ** depths, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(probs) + (n - k) * np.log(1 - probs))


def fit_pd_model(depth_levels, successes_per_depth, n_per_depth, bootstrap_reps=2000):
    depths = np.array(depth_levels, dtype=float)
    k = np.array(successes_per_depth, dtype=float)
    n = np.array(n_per_depth, dtype=float)
    mask = n > 0
    depths, k, n = depths[mask], k[mask], n[mask]
    if len(depths) == 0:
        return (float("nan"), float("nan"), float("nan"))
    grid = np.linspace(0.50, 0.9999, 5000)
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


def aggregate_by_depth(results):
    by_depth = {}
    for r in results:
        d = r["depth"]
        if d not in by_depth:
            by_depth[d] = {
                "n": 0, "correct": 0, "div_steps": [], "tfbc": 0,
                "format_failures": 0, "completion_tokens": [],
            }
        by_depth[d]["n"] += 1
        if (
            r.get("is_correct")
            and not r.get("is_api_failure", False)
            and not r.get("is_truncated", False)
        ):
            by_depth[d]["correct"] += 1
        if r.get("div_step", -1) > 0 and not r.get("is_format_failure", False):
            by_depth[d]["div_steps"].append(r["div_step"])
        if r.get("is_tfbc"):
            by_depth[d]["tfbc"] += 1
        if r.get("is_format_failure"):
            by_depth[d]["format_failures"] += 1
        if r.get("completion_tokens", 0) > 0:
            by_depth[d]["completion_tokens"].append(r["completion_tokens"])

    out = {}
    for d, v in sorted(by_depth.items()):
        n, c = v["n"], v["correct"]
        lo, hi = clopper_pearson_ci(c, n)
        divs = v["div_steps"]
        fmt_f = v["format_failures"]
        toks = v["completion_tokens"]
        out[d] = {
            "n": n, "correct": c, "accuracy": c / n if n > 0 else 0.0,
            "ci_lo": lo, "ci_hi": hi, "n_incorrect": n - c,
            "div_steps": divs,
            "avg_div_step": float(np.mean(divs)) if divs else None,
            "early_failures": sum(1 for s in divs if s <= 3),
            "mid_failures":   sum(1 for s in divs if 4 <= s <= 10),
            "late_failures":  sum(1 for s in divs if s > 10),
            "tfbc_count": v["tfbc"],
            "tfbc_rate": v["tfbc"] / c if c > 0 else None,
            "format_failures": fmt_f,
            "format_failure_rate": fmt_f / n if n > 0 else 0.0,
            "avg_completion_tokens": float(np.mean(toks)) if toks else None,
        }
    return out


def aggregate_overall(results):
    valid = [
        r for r in results
        if not r.get("is_api_failure", False)
        and not r.get("is_truncated", False)
    ]
    n = len(valid)
    c = sum(1 for r in valid if r.get("is_correct"))
    clean_accuracy = c / n if n > 0 else 0.0

    tfbc = sum(
        1 for r in results
        if r.get("is_tfbc")
        and not r.get("is_api_failure", False)
        and not r.get("is_truncated", False)
    )
    fmt_f = sum(1 for r in results if r.get("is_format_failure"))
    lo, hi = clopper_pearson_ci(c, n)
    div_steps = [
        r["div_step"] for r in results
        if r.get("div_step", -1) > 0 and not r.get("is_format_failure", False)
    ]
    toks = [r["completion_tokens"] for r in results if r.get("completion_tokens", 0) > 0]

    return {
        "n": n, "correct": c, "accuracy": clean_accuracy,
        "ci_lo": lo, "ci_hi": hi,
        "tfbc_total": tfbc,
        "tfbc_rate": tfbc / c if c > 0 else None,
        "format_failures": fmt_f,
        "format_failure_rate": fmt_f / n if n > 0 else 0.0,
        "avg_div_step": float(np.mean(div_steps)) if div_steps else None,
        "avg_completion_tokens": float(np.mean(toks)) if toks else None,
        "total_tokens_used": sum(toks),
        "clean_accuracy": clean_accuracy,
    }


def failure_timing_breakdown(results):
    div_steps = [
        r["div_step"] for r in results
        if r.get("div_step", -1) > 0 and not r.get("is_format_failure", False)
    ]
    if not div_steps:
        return {"early": None, "mid": None, "late": None, "n_logic_failures": 0}
    total = len(div_steps)
    return {
        "early": sum(1 for s in div_steps if s <= 3)     / total,
        "mid":   sum(1 for s in div_steps if 4 <= s <= 10) / total,
        "late":  sum(1 for s in div_steps if s > 10)     / total,
        "n_logic_failures": total,
        "avg_div_step": float(np.mean(div_steps)),
    }


# ---------------------------------------------------------------------------
# Benchmark generator
# ---------------------------------------------------------------------------

class AlienGridGenerator:
    def __init__(self):
        self.base_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def rotate_90_cw(self, g):
        return [list(r) for r in zip(*g[::-1])]

    def shift_row_2_left(self, g):
        g = [r[:] for r in g]
        r = g[1]
        g[1] = [r[1], r[2], r[0]]
        return g

    def swap_corners(self, g):
        g = [r[:] for r in g]
        g[0][0], g[0][2], g[2][0], g[2][2] = g[2][2], g[2][0], g[0][2], g[0][0]
        return g

    def reverse_grid(self, g):
        f = [x for r in g for x in r][::-1]
        return [f[0:3], f[3:6], f[6:9]]

    def flip_horizontal(self, g):
        return [r[::-1] for r in g]

    def transpose_grid(self, g):
        return [list(r) for r in zip(*g)]

    def shift_col_1_up(self, g):
        g = [r[:] for r in g]
        c = [g[0][0], g[1][0], g[2][0]]
        g[0][0], g[1][0], g[2][0] = c[1], c[2], c[0]
        return g

    def solve(self, depth, seed):
        random.seed(seed)
        grid = [r[:] for r in self.base_grid]
        trace, events = [], []
        ops = [
            ("ROTATE_90_CW",     self.rotate_90_cw),
            ("SHIFT_ROW_2_LEFT", self.shift_row_2_left),
            ("SWAP_CORNERS",     self.swap_corners),
            ("REVERSE_GRID",     self.reverse_grid),
            ("FLIP_HORIZONTAL",  self.flip_horizontal),
            ("TRANSPOSE_GRID",   self.transpose_grid),
            ("SHIFT_COL_1_UP",   self.shift_col_1_up),
        ]
        for i in range(depth):
            name, func = random.choice(ops)
            grid = func(grid)
            events.append(f"Step {i+1}: {name}")
            trace.append(f"Step {i+1}: {[r[:] for r in grid]!s}")
        return "\n".join(events), trace, str([r[:] for r in grid])

    def build(self, depths, n_per_depth, seed_offset=0):
        dataset = []
        for d in depths:
            seen = set()
            collected = 0
            attempt = 0
            while collected < n_per_depth:
                sv = d * 10000 + attempt + seed_offset
                attempt += 1
                pt, gt, ans = self.solve(d, seed=sv)
                op = tuple(re.findall(r"Step \d+: ([A-Z0-9_]+)", pt))
                if op not in seen:
                    seen.add(op)
                    dataset.append({
                        "depth": d, "seed": sv, "prompt": pt,
                        "gt_trace": gt, "gt_ans": ans,
                    })
                    collected += 1
        return dataset


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

OPERATIONS_DESCRIPTION = """\
GRID CONVENTION:
- Rows and columns use 1-based indexing (1 to 3).
- Row 1 = top row, Row 3 = bottom row. Col 1 = leftmost column, Col 3 = rightmost column.
- Coordinate (row, col): (1,1) = top-left, (1,3) = top-right, (3,1) = bottom-left, (3,3) = bottom-right.

OPERATIONS:
- ROTATE_90_CW      : Rotate the entire 3x3 grid 90 degrees clockwise around its center.
- SHIFT_ROW_2_LEFT  : Shift the middle row (Row 2) left by 1, wrapping the leftmost element to the right end.
- SWAP_CORNERS      : Swap coordinate (1,1) with (3,3), AND swap coordinate (1,3) with (3,1). Both swaps happen simultaneously using the grid state before either swap.
- REVERSE_GRID      : Flatten the grid row-wise into a 1D list of 9 elements (reading Row 1 left-to-right, then Row 2, then Row 3), reverse that list, then reshape back into a 3x3 grid.
- FLIP_HORIZONTAL   : Reverse the elements in each row individually (left-to-right reflection). Rows are not reordered.
- TRANSPOSE_GRID    : Reflect the grid across its main diagonal (top-left to bottom-right), so that element at (i,j) moves to position (j,i).
- SHIFT_COL_1_UP    : Shift the leftmost column (Col 1) up by 1, wrapping the top element to the bottom.

EXECUTION RULES:
- Operations are applied sequentially. Each operation uses the grid produced by the previous step, not the original grid.
- Within a single operation, all output values are computed from the grid state that existed before the operation began. No cell updated during the operation is used as input for another cell in the same operation.

WORKED EXAMPLES (all use starting grid [[1,2,3],[4,5,6],[7,8,9]]):

ROTATE_90_CW:
  Input  : [[1,2,3],[4,5,6],[7,8,9]]
  Output : [[7,4,1],[8,5,2],[9,6,3]]
  (Each column becomes a row, reading bottom-to-top.)

SHIFT_ROW_2_LEFT:
  Input  : [[1,2,3],[4,5,6],[7,8,9]]
  Row 2 before : [4,5,6]
  Shift left 1 : [5,6,4]  (4 wraps to right end)
  Output : [[1,2,3],[5,6,4],[7,8,9]]

SWAP_CORNERS:
  Input  : [[1,2,3],[4,5,6],[7,8,9]]
  Swap (1,1)=1 with (3,3)=9, AND (1,3)=3 with (3,1)=7 simultaneously.
  Output : [[9,2,7],[4,5,6],[3,8,1]]

REVERSE_GRID:
  Input        : [[1,2,3],[4,5,6],[7,8,9]]
  Flatten      : [1,2,3,4,5,6,7,8,9]
  Reverse list : [9,8,7,6,5,4,3,2,1]
  Reshape 3x3  : [[9,8,7],[6,5,4],[3,2,1]]

FLIP_HORIZONTAL:
  Input  : [[1,2,3],[4,5,6],[7,8,9]]
  Output : [[3,2,1],[6,5,4],[9,8,7]]
  (Each row is independently reversed. Row order is unchanged.)

TRANSPOSE_GRID:
  Input  : [[1,2,3],[4,5,6],[7,8,9]]
  Output : [[1,4,7],[2,5,8],[3,6,9]]
  (Element at (i,j) moves to (j,i). Example: (1,2)=2 moves to (2,1).)

SHIFT_COL_1_UP:
  Input         : [[1,2,3],[4,5,6],[7,8,9]]
  Col 1 before  : [1,4,7]  (top to bottom)
  Shift up by 1 : [4,7,1]  (1 wraps to bottom)
  Output        : [[4,2,3],[7,5,6],[1,8,9]]\
"""

OUTPUT_FORMAT = """\
Output ONLY the following format. Do NOT use markdown code blocks. Do NOT add commentary.
1. TRACE: ["Step 1: [[...]]", "Step 2: [[...]]", ...]
2. ANSWER: [[...]]

CRITICAL RULES:
- Step 1 in TRACE = grid state AFTER applying the first operation.
- Do NOT include the initial grid in TRACE.
- Output the FULL 3x3 grid for every single step. No ellipses.\
"""

SYSTEM_PROMPT = (
    "You are a spatial reasoning engine. Track a 3x3 grid "
    "(Initial: cells numbered 1-9 reading top-left to bottom-right, "
    "i.e. [[1,2,3],[4,5,6],[7,8,9]]).\n\n"
    + OPERATIONS_DESCRIPTION + "\n\n"
    "SPATIAL LOGIC EXAMPLE:\n"
    "Initial Grid: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
    "Operation: SHIFT_ROW_2_LEFT\n"
    "Result: [[1, 2, 3], [5, 6, 4], [7, 8, 9]]\n\n"
    + OUTPUT_FORMAT
)


# ---------------------------------------------------------------------------
# Parsing and evaluation
# ---------------------------------------------------------------------------

def extract_grid_trace(text):
    matches = re.findall(r"Step\s*\d+:\s*\[\[.*?\]\]", text, re.IGNORECASE)
    return [re.sub(r'\s+', ' ', m).strip() for m in matches]


def extract_answer(text):
    m = re.search(r"ANSWER:.*?(\[\[.*?\]\])", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else "N/A"


def compute_divergence_step(gt_trace, pred_trace):
    for i, gt_step in enumerate(gt_trace):
        pred_step = pred_trace[i] if i < len(pred_trace) else "MISSING"
        if re.sub(r'\s+', '', pred_step).lower() != re.sub(r'\s+', '', gt_step).lower():
            return (i + 1, gt_step, pred_step)
    return (-1, None, None)


def evaluate_single(item, model_output):
    parsed_ans = extract_answer(model_output)
    clean_parsed = re.sub(r'\s+', '', parsed_ans)
    clean_gt = re.sub(r'\s+', '', str(item["gt_ans"]))
    is_correct = (clean_parsed == clean_gt) and (clean_gt != "")
    pred_trace = extract_grid_trace(model_output)
    gt_trace = [re.sub(r'\s+', ' ', str(s)).strip() for s in item["gt_trace"]]

    output_head = model_output.strip().upper()[:50]
    is_format_failure = (len(pred_trace) == 0) or ("TRACE" not in output_head)

    div_step, exp_at_div, pred_at_div = compute_divergence_step(gt_trace, pred_trace)
    is_tfbc = is_correct and (div_step != -1)

    return {
        "depth":              item["depth"],
        "seed":               item["seed"],
        "is_correct":         is_correct,
        "div_step":           div_step,
        "is_tfbc":            is_tfbc,
        "is_format_failure":  is_format_failure,
        "trace_length_ok":    (len(pred_trace) == item["depth"]),
        "pred_trace_len":     len(pred_trace),
        "expected_trace_len": item["depth"],
        "expected_at_div":    exp_at_div,
        "predicted_at_div":   pred_at_div,
        "model_output":       model_output,
    }


# ---------------------------------------------------------------------------
# File discovery and merging
# ---------------------------------------------------------------------------

def model_id_to_safe(model_id):
    return model_id.replace("/", "_")


def discover_files_for_model(model_id, results_dir):
    safe = model_id_to_safe(model_id)
    pattern = os.path.join(results_dir, f"results_*{safe}*.json")
    files = glob.glob(pattern)
    if not files:
        provider, *rest = model_id.split("/")
        model_part = "_".join(rest) if rest else safe
        pattern2 = os.path.join(results_dir, f"results_*{model_part}*.json")
        files = glob.glob(pattern2)
    return sorted(set(files), key=os.path.getmtime)


def load_and_merge_files(file_paths, model_id):
    """
    Load all files for a model, deduplicate by seed (earliest file wins),
    but prefer non-api-failure results over api-failure results.
    """
    by_seed = {}
    provenance = []
    total_loaded = 0
    dupes = 0

    for fpath in file_paths:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"    [SKIP] {os.path.basename(fpath)}: not a list")
                continue
            n = len(data)
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

                err = str(r.get("error", "")).lower()
                if "402" in err or "insufficient" in err or "more credits" in err:
                    r["is_api_failure"] = True
                    r["is_format_failure"] = False

                if seed not in by_seed:
                    by_seed[seed] = r
                else:
                    dupes += 1
                    existing = by_seed[seed]
                    if existing.get("is_api_failure", False) and not r.get("is_api_failure", False):
                        by_seed[seed] = r

            print(f"    Loaded {n:3d} results from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"    [ERROR] Could not load {fpath}: {e}")

    return list(by_seed.values()), provenance, total_loaded, dupes


def classify_results(merged_results, benchmark_by_seed):
    """
    Split merged results into (good, to_retry, missing).
    Seeds with api_failure, truncation, or format_failure are queued for retry.
    """
    result_by_seed = {r["seed"]: r for r in merged_results}
    good, to_retry, missing = [], [], []

    for seed, item in benchmark_by_seed.items():
        if seed not in result_by_seed:
            missing.append(item)
        else:
            r = result_by_seed[seed]
            if (r.get("is_api_failure", False)
                or r.get("is_truncated", False)
                or r.get("is_format_failure", False)):
                to_retry.append(item)
            else:
                good.append(r)

    return good, to_retry, missing


# ---------------------------------------------------------------------------
# API runner
# ---------------------------------------------------------------------------

def get_post_call_wait(model_id):
    for prefix, wait in POST_CALL_WAIT.items():
        if model_id.startswith(prefix):
            return wait
    return DEFAULT_WAIT


def run_on_items(model_id, items, label="", output_file=None, existing_results=None):
    results = []
    if existing_results is None:
        existing_results = []

    n_total = len(items)
    if n_total == 0:
        print(f"  [SKIP] {model_id}: no items to run")
        return results

    wait = get_post_call_wait(model_id)

    print(f"\n{'='*70}")
    print(f"[RUN] {label or model_id}")
    print(f"      {n_total} instances | temp={TEMPERATURE} | max_tokens={MAX_TOKENS}")
    print(f"{'='*70}")

    for i, item in enumerate(items):
        success = False
        retries = 0
        last_err = None

        while not success and retries < MAX_RETRIES:
            try:
                t0 = time.time()

                # Higher token budget for deep traces to avoid truncation.
                dynamic_max_tokens = MAX_TOKENS
                if item.get("depth", 0) >= 40:
                    dynamic_max_tokens = 6000

                response = client.chat.completions.create(
                    model=model_id,
                    temperature=TEMPERATURE,
                    max_tokens=dynamic_max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["prompt"]},
                    ],
                )
                elapsed = time.time() - t0
                raw_output = response.choices[0].message.content.strip()

                pt = getattr(response.usage, "prompt_tokens", 0) or 0
                ct = getattr(response.usage, "completion_tokens", 0) or 0
                tt = getattr(response.usage, "total_tokens", 0) or 0

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
                tfbc   = " [TFBC]"   if result["is_tfbc"]           else ""
                fmt_e  = " [FORMAT]" if result["is_format_failure"] else ""
                trunc  = " [TRUNC]"  if result["is_truncated"]      else ""
                print(f"  [{i+1:3d}/{n_total}] N={item['depth']:2d} | seed={item['seed']} | "
                      f"{status}{tfbc}{fmt_e}{trunc} | k*={result['div_step']:3d} | "
                      f"tok={ct} | {elapsed:.1f}s")

                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(existing_results + results, f, indent=2)

                success = True
                time.sleep(wait)

            except Exception as e:
                retries += 1
                last_err = str(e)
                is_credit_error = "402" in last_err or "credits" in last_err.lower()

                print(f"  [RETRY {retries}/{MAX_RETRIES}] seed={item['seed']} "
                      f"{'[CREDIT ERROR]' if is_credit_error else ''} Error: {e}")

                if retries >= MAX_RETRIES:
                    print(f"  [GIVE UP] seed={item['seed']} after {MAX_RETRIES} retries")
                    results.append({
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
                    })
                    if output_file:
                        with open(output_file, "w") as f:
                            json.dump(existing_results + results, f, indent=2)
                else:
                    sleep_time = RETRY_WAIT * 2 if is_credit_error else RETRY_WAIT
                    print(f"    Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_and_save_summary(all_model_results, pd_fits, filepath):
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
          f"{'TFBC%':>7}  {'FmtFail%':>9}  {'AvgTok':>7}")
        p("-" * 105)

        agg = aggregate_by_depth(results)
        for d, v in sorted(agg.items()):
            ci_str = f"[{v['ci_lo']*100:5.1f}%, {v['ci_hi']*100:5.1f}%]"
            n_inc = v["n_incorrect"]
            early = f"{v['early_failures']/n_inc*100:5.1f}%" if n_inc else "  N/A"
            mid   = f"{v['mid_failures']/n_inc*100:5.1f}%"   if n_inc else "  N/A"
            late  = f"{v['late_failures']/n_inc*100:5.1f}%"  if n_inc else "  N/A"
            avg_k = f"{v['avg_div_step']:6.2f}" if v["avg_div_step"] else "   N/A"
            tfbc  = f"{v['tfbc_rate']*100:5.1f}%" if v["tfbc_rate"] is not None else "  N/A"
            fmt_f = f"{v['format_failure_rate']*100:5.1f}%"
            avg_t = f"{v['avg_completion_tokens']:6.0f}" if v["avg_completion_tokens"] else "   N/A"
            p(f"{d:>6}  {v['n']:>4}  {v['correct']:>4}  {v['accuracy']*100:5.1f}%  "
              f"{ci_str:^20}  {avg_k}  {early:>7}  {mid:>7}  {late:>7}  "
              f"{tfbc:>7}  {fmt_f:>9}  {avg_t:>7}")

        ov = aggregate_overall(results)
        ftb = failure_timing_breakdown(results)
        p("-" * 105)
        tfbc_str = f"TFBC={ov['tfbc_rate']*100:.1f}%" if ov["tfbc_rate"] is not None else "TFBC=N/A"
        fmt_str  = f"{ov['format_failure_rate']*100:.1f}%"
        tok_str  = f"{ov['avg_completion_tokens']:.0f}" if ov["avg_completion_tokens"] else "N/A"
        p(f"{'TOTAL':>6}  {ov['n']:>4}  {ov['correct']:>4}  {ov['accuracy']*100:5.1f}%  "
          f"[{ov['ci_lo']*100:5.1f}%, {ov['ci_hi']*100:5.1f}%]  "
          f"  {tfbc_str}  FmtFail={fmt_str}  AvgTok={tok_str}")
        if ftb["n_logic_failures"] > 0:
            p(f"  k* timing (logic failures only, n={ftb['n_logic_failures']}): "
              f"Early={ftb['early']*100:.1f}%  Mid={ftb['mid']*100:.1f}%  "
              f"Late={ftb['late']*100:.1f}%  AvgK*={ftb['avg_div_step']:.2f}")
        if ov["format_failure_rate"] > 0.10:
            p(f"  WARNING: {fmt_str} format failure rate > 10%.")
            p(f"  Report logic and format failures separately.")

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
    p("ACCURACY MATRIX (% +/- half-CI, all depths)")
    p("=" * 80)
    col_names = [m.split("/")[-1][:16] for m in all_model_results]
    p("Depth  | " + " | ".join(f"{c:^16}" for c in col_names))
    p("-" * (9 + 19 * len(col_names)))
    first_agg = aggregate_by_depth(next(iter(all_model_results.values())))
    for d in sorted(first_agg.keys()):
        row = f" {d:2d}    | "
        cells = []
        for model_id, results in all_model_results.items():
            agg = aggregate_by_depth(results)
            v = agg.get(d)
            if v:
                cells.append(
                    f"{v['accuracy']*100:5.1f}% +/-{(v['ci_hi']-v['ci_lo'])*50:3.1f}".center(16)
                )
            else:
                cells.append("  N/A  ".center(16))
        p(row + " | ".join(cells))
    p("\n[NOTE] +/-X = half-width of the 95% Clopper-Pearson CI.")
    p("[NOTE] Early/Mid/Late percentages exclude format failures.")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] Summary -> {filepath}")


def save_parser_validation_set(all_results_flat, filepath, n_per_model=10):
    by_model = {}
    for r in all_results_flat:
        by_model.setdefault(r.get("model", "unknown"), []).append(r)

    val_set = []
    for model_id, results in by_model.items():
        logic_fail = [
            r for r in results
            if not r.get("is_correct")
            and not r.get("is_format_failure", False)
            and not r.get("is_api_failure", False)
        ]
        sample = random.sample(logic_fail, min(n_per_model, len(logic_fail)))
        if len(sample) < n_per_model:
            others = [
                r for r in results
                if r not in sample and not r.get("is_api_failure", False)
            ]
            random.shuffle(others)
            sample += others[:n_per_model - len(sample)]

        for r in sample:
            val_set.append({
                "model":             model_id,
                "depth":             r["depth"],
                "seed":              r["seed"],
                "is_correct":        r.get("is_correct"),
                "auto_div_step":     r.get("div_step"),
                "is_format_failure": r.get("is_format_failure"),
                "pred_trace_len":    r.get("pred_trace_len"),
                "expected_len":      r.get("expected_trace_len"),
                "trace_length_ok":   r.get("trace_length_ok"),
                "expected_at_div":   r.get("expected_at_div"),
                "predicted_at_div":  r.get("predicted_at_div"),
                "model_output":      r.get("model_output", "")[:2000],
                "human_div_step":    None,
                "human2_div_step":   None,
                "annotation_note":   "",
            })

    with open(filepath, "w") as f:
        json.dump(val_set, f, indent=2)
    print(f"[SAVED] Parser validation set ({len(val_set)} traces) -> {filepath}")


# ---------------------------------------------------------------------------
# Helper scripts (written alongside the results)
# ---------------------------------------------------------------------------

KAPPA_SCRIPT = '''\
"""Compute Cohen's kappa between two annotators.

Usage: python compute_kappa.py parser_validation_<RUN_ID>.json
"""
import json, sys

def cohens_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

with open(sys.argv[1]) as f:
    data = json.load(f)

annotated = [
    d for d in data
    if d.get("human_div_step") is not None
    and d.get("human2_div_step") is not None
    and not d.get("is_format_failure", False)
]
if len(annotated) < 10:
    print(f"Only {len(annotated)} entries. Need >= 10.")
    sys.exit(1)

a1 = [str(d["human_div_step"])  for d in annotated]
a2 = [str(d["human2_div_step"]) for d in annotated]
k = cohens_kappa(a1, a2)
print(f"n={len(annotated)}  Cohen kappa={k:.3f}")
if k >= 0.80:
    print("PASS: kappa >= 0.80")
elif k >= 0.60:
    print("BORDERLINE: revise parser before submission")
else:
    print("FAIL: kappa < 0.60")
'''

MCNEMAR_SCRIPT = '''\
"""McNemar test for paired ablation results.

Usage: python compute_mcnemar.py results_ablation_<RUN_ID>.json
"""
import json, sys
from scipy.stats import mcnemar

with open(sys.argv[1]) as f:
    data = json.load(f)

std  = [r for r in data if r.get("label") == "Ablation_Standard"]
verb = [r for r in data if r.get("label") == "Ablation_Verbose"]
ss = {r["seed"]: r["is_correct"] for r in std}
vs = {r["seed"]: r["is_correct"] for r in verb}
common = sorted(set(ss) & set(vs))
if len(common) < 10:
    print(f"Only {len(common)} pairs.")
    sys.exit(1)

a = sum(1 for s in common if     ss[s] and     vs[s])
b = sum(1 for s in common if     ss[s] and not vs[s])
c = sum(1 for s in common if not ss[s] and     vs[s])
d = sum(1 for s in common if not ss[s] and not vs[s])
print(f"n={len(common)}  both_correct={a}  std_only={b}  verb_only={c}  both_wrong={d}")
res = mcnemar([[a, b], [c, d]], exact=True)
print(f"McNemar p={res.pvalue:.4f}")
if res.pvalue < 0.05 and c > b:
    print("VERBOSE BETTER")
elif res.pvalue < 0.05:
    print("STANDARD BETTER")
else:
    print("NO SIG DIFFERENCE")
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"RUN_ID: {RUN_ID}")
    print(f"Results dir: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Models:      {len(ALL_MODELS)}")
    print("=" * 70)

    for fname, script in [("compute_kappa.py", KAPPA_SCRIPT),
                          ("compute_mcnemar.py", MCNEMAR_SCRIPT)]:
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            f.write(script)
        print(f"[SAVED] {fname} -> {path}")

    gen = AlienGridGenerator()
    main_dataset = gen.build(DEPTH_LEVELS, N_PER_CELL, seed_offset=0)
    benchmark_by_seed = {item["seed"]: item for item in main_dataset}
    expected = N_PER_CELL * len(DEPTH_LEVELS)
    print(f"\n[DATASET] {len(main_dataset)} instances "
          f"(depths={DEPTH_LEVELS}, n/cell={N_PER_CELL}, total={expected})")

    all_model_results = {}
    all_results_flat = []
    provenance_map = {}

    for model_id in ALL_MODELS:
        safe = model_id_to_safe(model_id)
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
            print(f"  Total loaded: {total_loaded} | After dedup: {len(merged)} | Dupes: {dupes}")
            provenance_map[model_id] = prov
        else:
            merged = []
            print(f"  No existing files — running from scratch")
            provenance_map[model_id] = []

        good_results, retry_items, missing_items = classify_results(merged, benchmark_by_seed)
        n_need_run = len(retry_items) + len(missing_items)

        print(f"\n  Status:")
        print(f"    Good                 : {len(good_results):3d}")
        print(f"    Needs retry          : {len(retry_items):3d}")
        print(f"    Missing              : {len(missing_items):3d}")
        print(f"    Total to run now     : {n_need_run:3d}")

        if n_need_run > 0:
            run_file = os.path.join(OUTPUT_DIR, f"results_main_{safe}_{RUN_ID}.json")
            new_results = run_on_items(
                model_id=model_id,
                items=retry_items + missing_items,
                label=f"Main_{short}",
                output_file=run_file,
                existing_results=good_results,
            )
            final_results = good_results + new_results
        else:
            final_results = good_results
            print(f"\n  Model complete — no API calls needed")

        seeds = [r["seed"] for r in final_results]
        assert len(set(seeds)) == len(seeds), (
            f"Duplicate seeds in final results for {model_id}: "
            f"{len(seeds) - len(set(seeds))} duplicates"
        )

        final_file = os.path.join(OUTPUT_DIR, f"results_FINAL_{safe}_{RUN_ID}.json")
        with open(final_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\n  [SAVED] {os.path.basename(final_file)}")

        n_final  = len(final_results)
        n_trunc  = sum(1 for r in final_results if r.get("is_truncated", False))
        n_api    = sum(1 for r in final_results if r.get("is_api_failure", False))
        n_fmt    = sum(1 for r in final_results if r.get("is_format_failure", False))
        n_corr   = sum(1 for r in final_results if r.get("is_correct", False))
        acc      = n_corr / n_final if n_final > 0 else 0.0
        print(f"  {short}: {n_corr}/{n_final} correct = {acc*100:.1f}% "
              f"[{n_final}/{expected}]")
        if n_api   > 0: print(f"    {n_api} API failures remain")
        if n_trunc > 0: print(f"    {n_trunc} truncated ({n_trunc/n_final*100:.1f}%)")
        if n_fmt   > 0: print(f"    {n_fmt} format failures ({n_fmt/n_final*100:.1f}%)")

        all_model_results[model_id] = final_results
        all_results_flat.extend(final_results)

    ordered_results = {m: all_model_results[m] for m in ALL_MODELS if m in all_model_results}

    print(f"\n{'='*70}")
    print("FINAL VALIDATION")
    print(f"{'='*70}")
    print(f"{'Model':<40}  {'N':>5}  {'Expected':>8}  {'ApiErr':>6}  {'OK':>4}")
    print("-" * 70)
    all_ok = True
    for model_id, results in ordered_results.items():
        n = len(results)
        n_api = sum(1 for r in results if r.get("is_api_failure", False))
        ok = (n == expected) and (n_api == 0)
        print(f"  {model_id.split('/')[-1]:<38}  {n:5d}  {expected:8d}  {n_api:6d}  "
              f"{'OK' if ok else 'WARN'}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n  Some models are incomplete. Re-run to resume.")
    else:
        print("\n  All models complete.")

    print(f"\n{'='*70}")
    print("p_d MODEL FITS")
    pd_fits = {}
    for model_id, results in ordered_results.items():
        agg = aggregate_by_depth(results)
        deps = sorted(agg.keys())
        succs = [agg[d]["correct"] for d in deps]
        ns = [agg[d]["n"] for d in deps]
        fit = fit_pd_model(deps, succs, ns, BOOTSTRAP_REPS)
        pd_fits[model_id] = fit
        short = model_id.split("/")[-1]
        if not math.isnan(fit[0]):
            print(f"  {short}: p_d={fit[0]:.4f}  95% CI=[{fit[1]:.4f}, {fit[2]:.4f}]")
        else:
            print(f"  {short}: N/A")

    print(f"\n{'='*70}")
    print("WRITING OUTPUT FILES")

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{RUN_ID}.txt")
    print_and_save_summary(ordered_results, pd_fits, summary_path)

    val_path = os.path.join(OUTPUT_DIR, f"parser_validation_{RUN_ID}.json")
    save_parser_validation_set(all_results_flat, val_path, n_per_model=10)

    combined = {
        "run_id":       RUN_ID,
        "domain":       DOMAIN,
        "temperature":  TEMPERATURE,
        "n_per_cell":   N_PER_CELL,
        "depth_levels": DEPTH_LEVELS,
        "models":       ALL_MODELS,
        "pd_fits":      {m: list(v) for m, v in pd_fits.items()},
        "main_results": {m: ordered_results[m] for m in ordered_results},
        "provenance":   provenance_map,
        "validation": {
            model_id: {
                "n_total":       len(r),
                "n_correct":     sum(1 for x in r if x.get("is_correct")),
                "n_api_failure": sum(1 for x in r if x.get("is_api_failure")),
                "n_fmt_failure": sum(1 for x in r if x.get("is_format_failure")),
                "n_truncated":   sum(1 for x in r if x.get("is_truncated")),
                "complete":      len(r) == expected,
            }
            for model_id, r in ordered_results.items()
        }
    }
    combined_path = os.path.join(OUTPUT_DIR, f"combined_{RUN_ID}.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[SAVED] Combined -> {combined_path}")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"  Summary    : results/summary_{RUN_ID}.txt")
    print(f"  Combined   : results/combined_{RUN_ID}.json")
    print(f"  Validation : results/parser_validation_{RUN_ID}.json")
    print(f"  Per-model  : results/results_FINAL_<model>_{RUN_ID}.json")
    print(f"{'='*70}")

    try:
        import subprocess
        subprocess.run(["python", os.path.join(OUTPUT_DIR, "compute_kappa.py"), val_path])
    except Exception as e:
        print("[INFO] Skipping kappa:", e)


if __name__ == "__main__":
    main()

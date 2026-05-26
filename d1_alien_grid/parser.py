import json
import random

def save_parser_validation_set(all_results_flat, filepath, n_per_model=10):
    random.seed(42)

    by_model = {}
    for r in all_results_flat:
        by_model.setdefault(r.get("model","unknown"), []).append(r)

    val_set = []

    for model_id, results in by_model.items():
        logic_fail = [r for r in results
                      if not r.get("is_correct")
                      and not r.get("is_format_failure", False)
                      and not r.get("is_api_failure", False)]

        sample = random.sample(logic_fail, min(n_per_model, len(logic_fail)))

        if len(sample) < n_per_model:
            others = [r for r in results
                      if r not in sample
                      and not r.get("is_api_failure", False)]
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
                "model_output":      r.get("model_output","")[:2000],
                "human_div_step":    None,
                "human2_div_step":   None,
                "annotation_note":   "",
            })

    with open(filepath,"w") as f:
        json.dump(val_set, f, indent=2)

    print(f"[SAVED] Parser validation set ({len(val_set)} traces) -> {filepath}")


with open("combined_20260427_230055_smart_resume_fixed.json") as f:
    raw = json.load(f)

data = []
for model, records in raw["main_results"].items():
    data.extend(records)

save_parser_validation_set(data, "parser_validation_NEW.json")
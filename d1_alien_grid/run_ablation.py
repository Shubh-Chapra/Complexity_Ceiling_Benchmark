from d1_alien_grid.alien_grid_main import (
    AlienGridGenerator,
    run_on_items,
    aggregate_overall,
    SYSTEM_PROMPT,
    OUTPUT_DIR,
    RUN_ID
)

import os

# CONFIG
ABLATION_MODEL = "anthropic/claude-3.7-sonnet"
ABLATION_DEPTH = 25
ABLATION_N = 20

def run_verbosity_ablation(gen: AlienGridGenerator):

    print("\n" + "="*70)
    print("[VERBOSITY ABLATION]")
    print("="*70)

    dataset = gen.build([ABLATION_DEPTH], ABLATION_N, seed_offset=200000)

    out_file = os.path.join(OUTPUT_DIR, f"results_ablation_{RUN_ID}.json")

    # --- STANDARD ---
    results_std = run_on_items(
        model_id=ABLATION_MODEL,
        items=dataset,
        label="Ablation_Standard",
        output_file=out_file,
    )

    # --- VERBOSE ---
    verbose_prompt = SYSTEM_PROMPT + "\nAfter EACH step, explicitly restate full grid." 

    results_verb = run_on_items(
        model_id=ABLATION_MODEL,
        items=dataset,
        label="Ablation_Verbose",
        output_file=out_file,
    )

    # --- ANALYSIS ---
    ov_std = aggregate_overall(results_std)
    ov_verb = aggregate_overall(results_verb)

    delta = ov_verb["accuracy"] - ov_std["accuracy"]

    print("\nRESULT:")
    print(f"Standard : {ov_std['accuracy']*100:.1f}%")
    print(f"Verbose  : {ov_verb['accuracy']*100:.1f}%")
    print(f"Delta    : {delta*100:+.1f} pp")

    print("\nRun McNemar:")
    print(f"python results/compute_mcnemar.py {out_file}")


if __name__ == "__main__":
    gen = AlienGridGenerator()
    run_verbosity_ablation(gen)
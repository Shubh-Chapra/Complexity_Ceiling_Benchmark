"""compute_mcnemar.py
Usage: python compute_mcnemar.py results_ablation_<ID>.json
Requires: pip install statsmodels
"""
import json, sys
try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except ImportError:
    print("pip install statsmodels"); sys.exit(1)
if len(sys.argv)<2: print("Usage: python compute_mcnemar.py <file>"); sys.exit(1)
with open(sys.argv[1]) as f: data=json.load(f)
std  = [r for r in data if r.get("label")=="Ablation_Standard"]
verb = [r for r in data if r.get("label")=="Ablation_Verbose"]
if not std:  print("No Ablation_Standard records."); sys.exit(1)
if not verb: print("No Ablation_Verbose records.");  sys.exit(1)
ss={r["seed"]:r["is_correct"] for r in std}
vs={r["seed"]:r["is_correct"] for r in verb}
common=sorted(set(ss)&set(vs))
if len(common)<10: print(f"Only {len(common)} pairs."); sys.exit(1)
a=sum(1 for s in common if     ss[s] and     vs[s])
b=sum(1 for s in common if     ss[s] and not vs[s])
c=sum(1 for s in common if not ss[s] and     vs[s])
d=sum(1 for s in common if not ss[s] and not vs[s])
print(f"n={len(common)}  both={a}  std_only={b}  verb_only={c}  both_wrong={d}")
res=sm_mcnemar([[a,b],[c,d]],exact=True)
print(f"McNemar p={res.pvalue:.4f}")
if res.pvalue<0.05:
    print(f"Significant: {'VERBOSE BETTER' if c>b else 'STANDARD BETTER'}")
else:
    print("Not significant at alpha=0.05.")

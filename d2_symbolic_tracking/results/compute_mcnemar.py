"""compute_mcnemar.py  Usage: python compute_mcnemar.py results_ablation_<ID>.json"""
import json, sys
from statsmodels.stats.contingency_tables import mcnemar
with open(sys.argv[1]) as f: data=json.load(f)
std  = [r for r in data if r.get("label")=="Ablation_Standard"]
verb = [r for r in data if r.get("label")=="Ablation_Verbose"]
ss   = {r["seed"]: r["is_correct"] for r in std}
vs   = {r["seed"]: r["is_correct"] for r in verb}
common = sorted(set(ss)&set(vs))
if len(common)<10: print(f"Only {len(common)} pairs."); sys.exit(1)
a=sum(1 for s in common if     ss[s] and     vs[s])
b=sum(1 for s in common if     ss[s] and not vs[s])
c=sum(1 for s in common if not ss[s] and     vs[s])
d=sum(1 for s in common if not ss[s] and not vs[s])
print(f"n={len(common)}  both_correct={a}  std_only={b}  verb_only={c}  both_wrong={d}")
res=mcnemar([[a,b],[c,d]],exact=True); print(f"McNemar p={res.pvalue:.4f}")
print("VERBOSE BETTER" if res.pvalue<0.05 and c>b
      else ("STANDARD BETTER" if res.pvalue<0.05 else "NO SIGNIFICANT DIFFERENCE"))

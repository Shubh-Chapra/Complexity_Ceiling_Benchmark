"""compute_kappa.py
Usage: python compute_kappa.py parser_validation_<ID>.json
Cohen's kappa between two human annotators on div_step.
"""
import json, sys
def cohens_kappa(a, b):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    labels=set(a)|set(b); pe=sum((a.count(l)/n)*(b.count(l)/n) for l in labels)
    return (po-pe)/(1-pe) if pe<1 else 1.0
if len(sys.argv)<2: print("Usage: python compute_kappa.py <file>"); sys.exit(1)
with open(sys.argv[1]) as f: data=json.load(f)
annotated=[d for d in data
           if d.get("human_div_step") is not None
           and d.get("human2_div_step") is not None
           and not d.get("is_format_failure", False)]
if len(annotated)<10: print(f"Only {len(annotated)} entries (need >=10)."); sys.exit(1)
a1=[str(d["human_div_step"]) for d in annotated]
a2=[str(d["human2_div_step"]) for d in annotated]
k=cohens_kappa(a1,a2); print(f"n={len(annotated)}  Cohen kappa={k:.3f}")
if k>=0.80:   print("PASS: kappa>=0.80 — TFBC defensible")
elif k>=0.60: print("BORDERLINE: revise before submission")
else:         print("FAIL: kappa<0.60 — redesign parser")
for ft in ("logic","constraint"):
    sub=[d for d in annotated if d.get("failure_type")==ft]
    if len(sub)<3: continue
    km=cohens_kappa([str(d["human_div_step"]) for d in sub],
                    [str(d["human2_div_step"]) for d in sub])
    print(f"  {ft}: n={len(sub)}  kappa={km:.3f}")

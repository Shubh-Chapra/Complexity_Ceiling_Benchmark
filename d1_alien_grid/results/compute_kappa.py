import json, sys

def cohens_kappa(a,b):
    n=len(a)
    po=sum(x==y for x,y in zip(a,b))/n
    labels=set(a)|set(b)
    pe=sum((a.count(l)/n)*(b.count(l)/n) for l in labels)
    return (po-pe)/(1-pe) if pe<1 else 1.0

with open(sys.argv[1]) as f:
    data = json.load(f)  
annotated = [
    d for d in data
    if d.get("human_div_step") is not None
    and d.get("human2_div_step") is not None
    and not d.get("is_format_failure", False)
]

if len(annotated) < 10:
    print(f"Only {len(annotated)} entries. Need >=10.")
    sys.exit(1)

a1 = [str(d["human_div_step"]) for d in annotated]
a2 = [str(d["human2_div_step"]) for d in annotated]

k = cohens_kappa(a1, a2)

print(f"n={len(annotated)}  Cohen kappa={k:.3f}")

if k >= 0.80:
    print("PASS: kappa>=0.80 → TFBC metric defensible")
elif k >= 0.60:
    print("BORDERLINE: revise parser before submission")
else:
    print("FAIL: kappa<0.60 → redesign parser")
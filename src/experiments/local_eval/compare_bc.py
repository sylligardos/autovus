"""B (all false alarms) vs C (fair share FP / K), AUC-PR and VUS (L=20), all toy examples."""
import numpy as np
from examples import examples
from local import local_scores
from vus_local import local_vus

rng = np.random.default_rng(3)
for name, x, y, A, dets in examples():
    print(f"\n== {name}")
    for dn, d in dets.items():
        r, v = local_scores(y, d), local_vus(y, d, 20)
        print(f"  {dn}: AUC-PR {r['ap']:.2f} VUS {v['vus_global']:.2f}")
        for k in ["naive", "A", "B_adj", "C_adj"]:
            print(f"     {k:9s} " + " ".join(f"{u:5.2f}" for u in r[k]))
        print(f"     {'VUS B':9s} " + " ".join(f"{u:5.2f}" for u in v["adj_k"]))
        print(f"     {'VUS C':9s} " + " ".join(f"{u:5.2f}" for u in v["C_adj"]))
    if name[:2] in ("2.", "6b", "10"):
        rs = [local_vus(y, rng.random(len(y)), 20) for _ in range(30)]
        print("  random VUS B " + " ".join(f"{u:+.3f}" for u in np.mean([q["adj_k"] for q in rs], 0)))
        print("  random VUS C " + " ".join(f"{u:+.3f}" for u in np.mean([q["C_adj"] for q in rs], 0)))

"""Check: global VUS-PR of the reference == FastVUS; print per-anomaly VUS (B style)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autovus import FastVUS
from examples import examples
from local import local_scores
from vus_local import local_vus

L = 20
for name, x, y, A, dets in examples():
    print(f"\n== {name}")
    for dn, d in dets.items():
        r = local_vus(y, d, L)
        f = FastVUS(L=L, max_thresholds=None).compute(y, d)
        b = local_scores(y, d)["B_adj"]
        print(f"  {dn}: VUS-PR fast {f:.4f} ref {r['vus_global']:.4f} (diff {abs(f - r['vus_global']):.1e})")
        print("     B_adj(AUC-PR) " + " ".join(f"{v:5.2f}" for v in b))
        print("     VUS_k adj     " + " ".join(f"{v:5.2f}" for v in r["adj_k"]))

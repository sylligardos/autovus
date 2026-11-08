"""Per-anomaly scores on examples 8-9: AUC-PR B, VUS_k (L=20), AutoVUS_k (caps = b*)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autovus import FastVUS, AutoVUS
from examples import examples
from local import local_scores
from vus_local import local_vus

L = 20
rng = np.random.default_rng(2)
for name, x, y, A, dets in examples():
    if name[0] not in "89":
        continue
    print(f"\n== {name}")
    for dn, d in dets.items():
        auto = AutoVUS(FastVUS(L=L, cumulative=True))
        res = auto.compute(y, d)
        caps = np.array([a["left"]["buffer_val"] for a in res["anomalies"]])
        v = local_vus(y, d, L)
        av = local_vus(y, d, L, caps=caps)
        print(f"  {dn}: global VUS {v['vus_global']:.3f}  AutoVUS {res['adjusted']:.3f}  b* {caps.tolist()}")
        print("     B_adj(AUC-PR) " + " ".join(f"{u:5.2f}" for u in local_scores(y, d)["B_adj"]))
        print("     VUS_k adj     " + " ".join(f"{u:5.2f}" for u in v["adj_k"]))
        print("     AutoVUS_k adj " + " ".join(f"{u:5.2f}" for u in av["adj_k"]))
    # random baseline check for the VUS version
    rs = [local_vus(y, rng.random(len(y)), L)["adj_k"] for _ in range(50)]
    print("  random VUS_k adj mean " + " ".join(f"{u:+.3f}" for u in np.mean(rs, 0)))

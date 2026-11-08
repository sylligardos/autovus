"""Example 7: pick detectors by coverage of the per-anomaly map vs by global AUC-PR.

Coverage of a set S = mean over anomalies of max_{d in S} B_adj[k, d] (greedy).
Ensembles are scored on the series: max and mean of min-max normalized scores.
"""
import numpy as np
from sklearn.metrics import average_precision_score
from examples import examples
from local import local_scores

name, x, y, A, dets = [e for e in examples() if e[0].startswith("7")][0]
names = list(dets)
res = {d: local_scores(y, dets[d]) for d in names}
Bmap = np.array([res[d]["B_adj"] for d in names])          # (detectors, anomalies)
print("map B_adj (rows detectors, cols anomalies)")
for d, row in zip(names, Bmap):
    print(f"  {d}: " + " ".join(f"{v:.2f}" for v in row) + f"   AUC-PR {res[d]['ap']:.3f}")


def coverage(S):
    return Bmap[[names.index(d) for d in S]].max(0).mean()


def greedy(k):
    S = []
    for _ in range(k):
        S.append(max((d for d in names if d not in S), key=lambda d: coverage(S + [d])))
    return S


def norm(v):
    return (v - v.min()) / (v.max() - v.min())


def ens_ap(S):
    Z = np.array([norm(dets[d]) for d in S])
    return average_precision_score(y, Z.max(0)), average_precision_score(y, Z.mean(0))


by_ap = sorted(names, key=lambda d: -res[d]["ap"])[:2]
by_cov = greedy(2)
for label, S in [("top-2 by AUC-PR", by_ap), ("greedy coverage", by_cov)]:
    mx, mn = ens_ap(S)
    print(f"{label:16s} {S}  coverage {coverage(S):.3f}  ensemble AUC-PR max {mx:.3f} mean {mn:.3f}")

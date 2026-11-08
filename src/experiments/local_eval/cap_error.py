"""Error of the 2000-threshold cap on the per-anomaly scores (cap 2000 vs no cap).

    conda run -n autovus python cap_error.py
Series where the uncapped run fits (distinct scores x T <= 2e8), 60 random
(seed 1), 3 random detectors each. Per anomaly: |adjusted capped - uncapped|
for FastVUS L=128 (at L=20 and L=128) and for AutoVUS L=20 (plus b* changes).
Output: experiments/local_eval/cap_error.csv
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AutoVUS
from tsad_parser import TSADParser
from bench import load_series, load_score


def main():
    meta = json.load(open(ROOT / "app/data/tsb-ad/series.json"))
    rng = np.random.default_rng(1)
    p = TSADParser()
    dets = list(p.get_detectors())
    rows, used = [], 0
    for f in rng.permutation(sorted(meta)):
        if used == 60:
            break
        if meta[f]["n"] > 50_000:
            continue
        _, y = load_series(p, f)
        did = False
        for d in rng.choice(dets, 3, replace=False):
            s = load_score(p, f, d)
            if s is None or len(s) != len(y):
                continue
            nd = len(np.unique(s))
            if nd * len(y) > 2e8:
                continue
            out = {}
            for cap in (2000, None):
                e = FastVUS(L=128, cumulative=True, per_anomaly=True, max_thresholds=cap)
                g = e.compute(y, s)
                a = AutoVUS(FastVUS(L=20, cumulative=True, per_anomaly=True, max_thresholds=cap)).compute(y, s)
                out[cap] = (g, e.local["adjusted"], a["local"]["adjusted"][:, -1],
                            np.array([x["left"]["buffer_val"] for x in a["anomalies"]]), a["adjusted"])
            c, u = out[2000], out[None]
            lens = e.local["end"] - e.local["start"] + 1
            for k in range(len(c[1])):
                rows.append({"series": f, "detector": d, "T": len(y), "K": len(c[1]), "distinct": nd,
                             "len_k": int(lens[k]), "k": k,
                             "err_global128": abs(c[0][-1] - u[0][-1]),
                             "err_local20": abs(c[1][k, 20] - u[1][k, 20]),
                             "err_local128": abs(c[1][k, -1] - u[1][k, -1]),
                             "err_auto20": abs(c[2][k] - u[2][k]),
                             "bstar_changed": int(c[3][k] != u[3][k]),
                             "err_auto_global": abs(c[4] - u[4])})
            did = True
        if did:
            used += 1
            print(f"{used:2d} {f[:40]}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments/local_eval/cap_error.csv", index=False)


if __name__ == "__main__":
    main()

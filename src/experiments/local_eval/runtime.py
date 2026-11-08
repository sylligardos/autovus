"""Runtime of the per-anomaly scores: plain vs per_anomaly=True.

    conda run -n autovus python runtime.py
Sample: 60 random series (seed 0) + the series with the most anomalies (Stock,
K=2453) + the longest series; 2 random detectors each. Configs:
FastVUS(L=128, cumulative) and AutoVUS(FastVUS(L=20, cumulative)), each with
and without per_anomaly. Output: experiments/local_eval/runtime.csv
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AutoVUS
from tsad_parser import TSADParser
from bench import load_series, load_score


def timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    meta = json.load(open(ROOT / "app/data/tsb-ad/series.json"))
    rng = np.random.default_rng(0)
    files = list(rng.choice(sorted(meta), 60, replace=False))
    files += [max(meta, key=lambda f: meta[f]["anomalies"]), max(meta, key=lambda f: meta[f]["n"])]
    p = TSADParser()
    dets = list(p.get_detectors())
    rows = []
    for f in files:
        _, y = load_series(p, f)
        for d in rng.choice(dets, 2, replace=False):
            s = load_score(p, f, d)
            if s is None or len(s) != len(y):
                continue
            row = {"series": f, "T": len(y), "K": meta[f]["anomalies"], "detector": d,
                   "distinct": len(np.unique(s))}
            for pa in (False, True):
                tag = "local" if pa else "plain"
                row[f"fast128_{tag}"] = timed(lambda: FastVUS(L=128, cumulative=True, per_anomaly=pa).compute(y, s))
                row[f"auto20_{tag}"] = timed(lambda: AutoVUS(FastVUS(L=20, cumulative=True, per_anomaly=pa)).compute(y, s))
            rows.append(row)
            print(f"{f[:30]:30s} T={len(y):7d} K={row['K']:5d} {d:18s} "
                  f"fast128 {row['fast128_plain']:.2f}->{row['fast128_local']:.2f}s  "
                  f"auto20 {row['auto20_plain']:.2f}->{row['auto20_local']:.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments/local_eval/runtime.csv", index=False)


if __name__ == "__main__":
    main()

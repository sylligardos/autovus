"""Per-anomaly map on a few real TSB-AD series (all detectors), plus coverage ensembles.

    conda run -n autovus python real_map.py
Map cell = local VUS-PR of anomaly k, B style, chance adjusted (L=20).
Ensembles (mean of min-max normalized scores) of 3 detectors, picked
(a) top-3 by global VUS-PR, (b) greedy coverage of the map. Both picks use
the label (in-sample), so they compare the criteria, not a deployable method.
Output: experiments/local_eval/map_<series>.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS
from tsad_parser import TSADParser
from bench import load_series, load_score
from vus_local import local_vus

import os
L = 20
KEY = os.environ.get("KEY", "adj_k")   # adj_k = B, C_adj = C (fair share)
SERIES = ["282_NEK_id_6_WebService_tr_500_1st_64.csv",
          "631_YAHOO_id_81_WebService_tr_500_1st_96.csv",
          "682_YAHOO_id_132_WebService_tr_500_1st_354.csv"]


def norm(v):
    r = v.max() - v.min()
    return (v - v.min()) / r if r > 0 else v * 0


def main():
    p = TSADParser()
    dets = list(p.get_detectors())
    out = ROOT / "experiments" / "local_eval"
    for f in SERIES:
        x, y = load_series(p, f)
        S = {d: load_score(p, f, d) for d in dets}
        S = {d: s for d, s in S.items() if s is not None and len(s) == len(y)}
        glob = {d: FastVUS(L=L).compute(y, s) for d, s in S.items()}
        names = sorted(S, key=lambda d: -glob[d])
        Mp = np.array([local_vus(y, S[d], L)[KEY] for d in names])
        K = Mp.shape[1]

        def cov(sel):
            return Mp[[names.index(d) for d in sel]].max(0).mean()

        greedy = []
        for _ in range(3):
            greedy.append(max((d for d in names if d not in greedy), key=lambda d: cov(greedy + [d])))
        top = names[:3]
        print(f"\n== {f}  ({len(y)} points, {K} anomalies, {len(names)} detectors)")
        print(f"  best single: {names[0]} VUS {glob[names[0]]:.3f}  mean map {Mp[0].mean():.3f}")
        for lab, sel in [("top-3 by VUS", top), ("greedy cover", greedy)]:
            e = np.mean([norm(S[d]) for d in sel], 0)
            print(f"  {lab}: {sel}  coverage {cov(sel):.3f}  ensemble VUS {FastVUS(L=L).compute(y, e):.3f}")
        best_cell = Mp.max(0)
        print("  per-anomaly best detector: " + ", ".join(f"A{k+1} {names[int(Mp[:, k].argmax())]} {best_cell[k]:.2f}" for k in range(K)))
        print("  mean map (per-anomaly, equal weight) vs VUS rank, top 8:")
        for d in names[:8]:
            print(f"     {d:14s} VUS {glob[d]:.3f}  mean map {Mp[names.index(d)].mean():.3f}")

        fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 2 + 0.22 * len(names)),
                                     gridspec_kw={"height_ratios": [1, 0.18 * len(names)]})
        a0.plot(x, lw=0.6, color="k")
        _, (st, en) = FastVUS(L=0).anomaly_coordinates(y)
        for k, (s_, e_) in enumerate(zip(st, en)):
            a0.axvspan(s_, e_ + 1, color="red", alpha=0.3)
            a0.text((s_ + e_) / 2, a0.get_ylim()[1], f"A{k+1}", ha="center", va="bottom", fontsize=7)
        a0.set_title(f.replace(".csv", ""), fontsize=9, loc="left")
        im = a1.imshow(Mp, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        a1.set_yticks(range(len(names)), [f"{d} ({glob[d]:.2f})" for d in names], fontsize=6)
        a1.set_xticks(range(K), [f"A{k+1}" for k in range(K)], fontsize=7)
        for i in range(len(names)):
            for k in range(K):
                a1.text(k, i, f"{Mp[i, k]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if Mp[i, k] > 0.6 else "black")
        a1.set_xlabel("anomaly (cell = local VUS-PR, chance adjusted; rows sorted by global VUS-PR)", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"map_{KEY}_{f.split('_WebService')[0]}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

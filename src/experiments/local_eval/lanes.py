"""Lane view for one real series: score lanes with per-anomaly values and the
false-alarm cost strip (where the lost VUS comes from).

    conda run -n autovus python lanes.py
Output: experiments/local_eval/lanes_<series>.png
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

L = 20
F = "682_YAHOO_id_132_WebService_tr_500_1st_354.csv"
DETS = ["TimesFM", "CNN", "Lag_Llama", "KShapeAD", "OmniAnomaly"]

p = TSADParser()
x, y = load_series(p, F)
_, (st, en) = FastVUS(L=0).anomaly_coordinates(y)
fig, axes = plt.subplots(1 + 2 * len(DETS), 1, figsize=(11, 1.2 + 1.35 * len(DETS)), sharex=True,
                         gridspec_kw={"height_ratios": [1.2] + [1, 0.25] * len(DETS)})
axes[0].plot(x, lw=0.6, color="k")
for k, (s_, e_) in enumerate(zip(st, en)):
    axes[0].axvspan(s_ - 2, e_ + 3, color="red", alpha=0.4)
axes[0].set_title(F.replace(".csv", ""), fontsize=9, loc="left")
for i, d in enumerate(DETS):
    s = load_score(p, F, d)
    r = local_vus(y, s, L)
    print(d, "check_costC", r["check_costC"])
    a, c = axes[1 + 2 * i], axes[2 + 2 * i]
    a.plot(s, lw=0.6, color="C0")
    for k, (s_, e_) in enumerate(zip(st, en)):
        a.axvspan(s_ - 2, e_ + 3, color="red", alpha=0.15)
        a.text((s_ + e_) / 2, 1.02, f"{r['C_adj'][k]:.2f}", transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=6, color="green" if r["C_adj"][k] > 0.5 else "red")
    a.set_ylabel(d, fontsize=7, rotation=0, ha="right")
    a.text(0.995, 0.92, f"VUS {r['vus_global']:.2f} | mean per-anomaly {r['C_adj'].mean():.2f}",
           transform=a.transAxes, ha="right", va="top", fontsize=7)
    a.set_yticks([])
    c.imshow(r["costC"][None, :], aspect="auto", cmap="Reds", extent=(0, len(y), 0, 1),
             vmin=0, vmax=max(r["costC"].max(), 1e-9))
    c.set_yticks([])
    c.set_ylabel("false\nalarms", fontsize=5, rotation=0, ha="right", va="center")
fig.tight_layout()
out = ROOT / "experiments" / "local_eval" / f"lanes_{F.split('_WebService')[0]}.png"
fig.savefig(out, dpi=150)
print("saved", out)

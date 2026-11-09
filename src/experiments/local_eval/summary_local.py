"""Summary of the benchmark-wide per-anomaly export (export_local.py).

    conda run -n autovus python src/experiments/local_eval/summary_local.py
Per-anomaly score = chance-adjusted local VUS-PR (variant C):
  fast@L  from fast128 (running mean of the per-buffer area and random up to L)
  auto20  from auto20 (each anomaly capped at its b*)
Writes experiments/local_benchmark/anomalies.csv (one row per anomaly x detector).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIR = ROOT / "experiments" / "local_benchmark" / "series"


def adjusted(area, random, L):
    v = area[..., :L + 1].mean(-1)
    r = random[..., :L + 1].mean(-1)
    return (v - r) / np.maximum(1 - r, 1e-10)


def main():
    rows, pairs, skipped, size = [], [], 0, 0
    files = sorted(DIR.glob("*.npz"))
    for f in files:
        size += f.stat().st_size
        z = np.load(f)
        dets = z["detectors"]
        skipped += len(z["errors"])
        lens = z["end"] - z["start"] + 1
        f20 = adjusted(z["fast128_area"], z["fast128_random"][None], 20)       # (D, K)
        f128 = adjusted(z["fast128_area"], z["fast128_random"][None], 128)
        a20 = adjusted(z["auto20_area"], z["auto20_random"], 20)
        series = f.name[:-4]
        for d, det in enumerate(dets):
            pairs.append({"series": series, "detector": det, "K": len(lens),
                          "autovus20": float(z["auto20_adjusted"][d]),
                          "vus128": float(z["fast128_vus"][d, -1]),
                          "mean_local_auto20": float(a20[d].mean())})
            for k in range(len(lens)):
                rows.append((series, det, k, int(lens[k]), f20[d, k], f128[d, k], a20[d, k], int(z["auto20_bstar"][d, k])))
    df = pd.DataFrame(rows, columns=["series", "detector", "k", "len", "fast20", "fast128", "auto20", "bstar"])
    df.to_csv(ROOT / "experiments" / "local_benchmark" / "anomalies.csv", index=False)
    P = pd.DataFrame(pairs)

    print(f"series {len(files)}/870, pairs {len(P)}, anomaly x detector rows {len(df)}, "
          f"detector runs skipped {skipped}, size {size / 1e9:.2f} GB")
    for c in ["fast20", "fast128", "auto20"]:
        x = df[c]
        print(f"  {c:8s} median {x.median():.3f}  share >= 0.9 {(x >= 0.9).mean():.3f}  share < 0.1 {(x < 0.1).mean():.3f}")
    print(f"  auto20 b* = 0 on {(df.bstar == 0).mean():.3f} of anomaly x detector pairs")
    rho = P[["autovus20", "mean_local_auto20"]].corr(method="spearman").iloc[0, 1]
    print(f"  Spearman(global AutoVUS@20, mean local auto20) over pairs: {rho:.3f}")

    # complementarity: anomalies the series' best detector (global AutoVUS@20) misses but another catches
    best = P.loc[P.groupby("series")["autovus20"].idxmax(), ["series", "detector"]]
    b = df.merge(best, on=["series", "detector"])[["series", "k", "auto20"]].rename(columns={"auto20": "best"})
    top = df.groupby(["series", "k"])["auto20"].max().rename("top").reset_index()
    m = b.merge(top, on=["series", "k"])
    print(f"  anomalies where the series' best detector scores < 0.5 but some detector >= 0.9: "
          f"{((m.best < 0.5) & (m.top >= 0.9)).mean():.3f} ({((m.best < 0.5) & (m.top >= 0.9)).sum()} of {len(m)})")
    sb = (P.groupby("series")["mean_local_auto20"].idxmax() != P.groupby("series")["autovus20"].idxmax()).mean()
    print(f"  series where the best detector by mean local differs from the best by global AutoVUS@20: {sb:.3f}")


if __name__ == "__main__":
    main()

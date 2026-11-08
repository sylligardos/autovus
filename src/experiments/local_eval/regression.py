"""Bitwise regression of the engines: save (before) or compare (after).

    conda run -n autovus python regression.py save|compare
Cases: toy examples + 10 real series x 6 detectors; FastVUS pr/roc, cumulative,
side=left, caps; AutoVUS L=20 adjusted and b*.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AutoVUS
from tsad_parser import TSADParser
from bench import load_series, load_score
from examples import examples
from detectors_check import SERIES

OUT = ROOT / "experiments" / "local_eval" / "regression_before.npz"
DETS = ["TimesFM", "CNN", "LOF", "Sub_PCA", "KShapeAD", "IForest"]


def cases():
    for name, x, y, A, dets in examples():
        for dn, d in dets.items():
            yield f"{name}|{dn}", y, d
    p = TSADParser()
    for f in SERIES:
        _, y = load_series(p, f)
        for d in DETS:
            s = load_score(p, f, d)
            if s is not None and len(s) == len(y):
                yield f"{f}|{d}", y, s


def values(y, s):
    out = {}
    out["pr"] = FastVUS(L=20).compute(y, s)
    out["roc"] = FastVUS(L=20).compute(y, s, metric="roc")
    out["cum"] = FastVUS(L=128, cumulative=True).compute(y, s)
    out["left"] = FastVUS(L=20, side="left").compute(y, s)
    K = len(FastVUS(L=0).anomaly_coordinates(y)[1][0])
    caps = (np.arange(K) * 7) % 21
    out["caps"] = FastVUS(L=20, cumulative=True).compute(y, s, caps, caps[::-1].copy())
    r = AutoVUS(FastVUS(L=20, cumulative=True)).compute(y, s)
    out["auto"] = np.r_[r["adjusted"], [a["left"]["buffer_val"] for a in r["anomalies"]]]
    return out


def main(mode):
    res = {}
    for key, y, s in cases():
        for k, v in values(y, s).items():
            res[f"{key}|{k}"] = np.atleast_1d(np.asarray(v, dtype=float))
    if mode == "save":
        np.savez(OUT, **res)
        print("saved", len(res))
    else:
        old = np.load(OUT)
        bad = [k for k in res if not np.array_equal(old[k], res[k])]
        print(f"compared {len(res)}, bitwise different {len(bad)}")
        for k in bad[:10]:
            print("  ", k, np.abs(old[k] - res[k]).max())


if __name__ == "__main__":
    main(sys.argv[1])

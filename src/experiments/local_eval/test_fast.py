"""FastVUS(per_anomaly=True) vs the brute-force reference (ref_local.py).

    conda run -n autovus python test_fast.py toys|real
Compared: area and random (n_anomalies, n_buffers), cost per point, anomaly
coordinates; plus the cost identity and K=1 vs global VUS / AdjustedVUS.
max_thresholds=None (the reference uses every distinct score).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AdjustedVUS
from examples import examples
from ref_local import reference


def compare(y, s, L, caps=None, side="both"):
    eng = FastVUS(L=L, per_anomaly=True, max_thresholds=None, side=side, cumulative=True)
    g = eng.compute(y, s) if caps is None else eng.compute(y, s, caps[0], caps[1])
    f, r = eng.local, reference(y, s, L, caps, side)
    assert np.array_equal(f["start"], r["start"]) and np.array_equal(f["end"], r["end"])
    ident = abs(f["cost"].sum() - (1 - f["area"]).mean())
    return (np.abs(f["area"] - r["area"]).max(), np.abs(f["random"] - r["random"]).max(),
            np.abs(f["cost"] - r["cost"]).max(), ident)


def toys():
    worst = np.zeros(4)
    n = 0
    rng = np.random.default_rng(5)
    for name, x, y, A, dets in examples():
        K = len(A)
        for dn, d in dets.items():
            runs = [(20, None, "both"), (128, None, "both"), (20, None, "left"), (20, None, "right"),
                    (20, (rng.integers(0, 21, K), rng.integers(0, 21, K)), "both")]
            for L, caps, side in runs:
                e = compare(y, d, L, caps, side)
                worst = np.maximum(worst, e)
                n += 1
            # random detectors on the same label
            e = compare(y, rng.random(len(y)), 20)
            worst = np.maximum(worst, e)
            n += 1
    print(f"toys: {n} runs; max |diff| area {worst[0]:.1e} random {worst[1]:.1e} cost {worst[2]:.1e}; "
          f"cost identity {worst[3]:.1e}")


def k1():
    """One anomaly: local == global VUS-PR curve, adjusted == AdjustedVUS."""
    worst = 0.0
    for name, x, y, A, dets in examples():
        if len(A) != 1:
            continue
        for d in dets.values():
            eng = FastVUS(L=128, per_anomaly=True, cumulative=True)
            vus = eng.compute(y, d)
            adj = AdjustedVUS(FastVUS(L=128, cumulative=True)).compute(y, d)
            worst = max(worst, np.abs(eng.local["vus"][0] - vus).max(),
                        abs(eng.local["adjusted"][0, -1] - adj["adjusted"]))
    print(f"K=1: max |local - global| (VUS curve and adjusted) {worst:.1e}")


def real():
    from tsad_parser import TSADParser
    from bench import load_series, load_score
    from detectors_check import SERIES
    p = TSADParser()
    worst = np.zeros(4)
    rng = np.random.default_rng(6)
    for f in SERIES:
        _, y = load_series(p, f)
        K = int(np.sum(np.diff(np.r_[0, y]) == 1))
        for det in ["TimesFM", "LOF", "KShapeAD"]:
            s = load_score(p, f, det)
            if s is None:
                continue
            caps = (rng.integers(0, 21, K), rng.integers(0, 21, K))
            for c in (None, caps):
                e = compare(y, s, 20, c)
                worst = np.maximum(worst, e)
        print(f"  {f[:20]} done, worst so far area {worst[0]:.1e} random {worst[1]:.1e} cost {worst[2]:.1e} ident {worst[3]:.1e}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "toys":
        toys()
        k1()
    else:
        real()

"""Global AdjustedVUS at every L = 0..128 for every (series, detector) of the export.

    conda run -n autovus python src/experiments/local_eval/global_adjusted.py
Labels only: the random curve of AdjustedVUS is the running mean over buffers
of label_mass(b) / T (FastVUS buffered label, same masking as compute()).
VUS comes from the export (fast128_vus). Output:
experiments/local_benchmark/global_adjusted.npz with series, detector (P,),
vus, random, adjusted (P, 129) float32.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series  # noqa: E402

DIR = ROOT / "experiments" / "local_benchmark"


def random_curve(y, L=128):
    """AdjustedVUS random curve, as FastVUS.compute + AdjustedVUS.curves build it."""
    e = FastVUS(L=L, cumulative=True)
    _, (start, end) = e.anomaly_coordinates(y)
    mask = e.safe_mask(y, start, end, extra_safe=True)
    lab = y[mask]
    _, (start, end) = e.anomaly_coordinates(lab)
    dl, dr = e.distance_from_anomaly(lab, start, end)
    mass = e.buffered_labels(lab, dl, dr).sum(axis=1) / len(y)
    return np.cumsum(mass) / np.arange(1, len(mass) + 1)


def main():
    p = TSADParser()
    S, D, V, R = [], [], [], []
    for f in sorted((DIR / "series").glob("*.npz")):
        series = f.name[:-4]
        z = np.load(f)
        _, y = load_series(p, series)
        r = random_curve(y)
        for d, v in zip(z["detectors"], z["fast128_vus"]):
            S.append(series); D.append(d); V.append(v); R.append(r)
    V, R = np.array(V, dtype=np.float64), np.array(R)
    A = (V - R) / np.maximum(1 - R, 1e-10)
    np.savez(DIR / "global_adjusted.npz", series=np.array(S), detector=np.array(D),
             vus=V.astype(np.float32), random=R.astype(np.float32), adjusted=A.astype(np.float32))
    print(f"pairs {len(S)}, series {len(set(S))}")


if __name__ == "__main__":
    main()

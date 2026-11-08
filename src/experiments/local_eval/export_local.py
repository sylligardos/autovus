"""Benchmark-wide per-anomaly scores (variant C) for every (series, detector) of TSB-AD.

    conda run -n autovus python src/experiments/local_eval/export_local.py --workers 4
    (run under nice 19; resumable: series with an output file are skipped)

Two runs per pair, both with FastVUS(per_anomaly=True), CPU, threshold cap 2000:
  fast128  FastVUS(L=128, cumulative=True): per-anomaly area per buffer 0..128;
           the per-anomaly VUS / adjusted at any L <= 128 is a running mean of it.
  auto20   AutoVUS(FastVUS(L=20, cumulative=True)): per-anomaly area and random
           baseline per buffer with every anomaly capped at its b*.

One file per series: experiments/local_benchmark/series/<series>.npz with
  detectors (D,), start, end (K,) inclusive, original coordinates
  fast128_area (D, K, 129) float32, fast128_random (K, 129) float32 (label only),
  fast128_vus (D, 129) float32 (global cumulative VUS-PR), fast128_cost (D, T) float32
  auto20_area (D, K, 21), auto20_random (D, K, 21) float32, auto20_bstar (D, K) int16
  (symmetric), auto20_adjusted (D,) float32 (global AutoVUS), auto20_cost (D, T) float32
  errors: detectors skipped (missing / NaN / wrong length / exception)
cost = per-point false-alarm cost at the run's full L (sum = mean over anomalies
and buffers of 1 - area). Log: experiments/local_benchmark/run.log.
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AutoVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402

OUT = ROOT / "experiments" / "local_benchmark"
SERIES_DIR = OUT / "series"


def one_series(file):
    t0 = time.time()
    p = TSADParser()
    _, y = load_series(p, file)
    T = len(y)
    dets, errors = [], []
    f_area, f_vus, f_cost, a_area, a_rand, a_b, a_adj, a_cost = [], [], [], [], [], [], [], []
    f_random, start, end = None, None, None
    for d in p.get_detectors():
        try:
            s = load_score(p, file, d)
            if s is None or len(s) != T:
                errors.append(f"{d}: missing, NaN or wrong length")
                continue
            e = FastVUS(L=128, cumulative=True, per_anomaly=True)
            vus = e.compute(y, s)
            loc = e.local
            r = AutoVUS(FastVUS(L=20, cumulative=True, per_anomaly=True)).compute(y, s)
            la = r["local"]
        except Exception:
            errors.append(f"{d}: {traceback.format_exc(limit=2)}")
            continue
        dets.append(d)
        start, end, f_random = loc["start"], loc["end"], loc["random"]
        f_area.append(loc["area"]); f_vus.append(vus); f_cost.append(loc["cost"])
        a_area.append(la["area"]); a_rand.append(la["random"]); a_cost.append(la["cost"])
        a_b.append([x["left"]["buffer_val"] for x in r["anomalies"]]); a_adj.append(r["adjusted"])
    if dets:
        f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
        tmp = SERIES_DIR / (file + ".tmp.npz")
        np.savez(tmp, detectors=np.array(dets), start=start, end=end, errors=np.array(errors),
                 fast128_area=f32(f_area), fast128_random=f32(f_random), fast128_vus=f32(f_vus),
                 fast128_cost=f32(f_cost), auto20_area=f32(a_area), auto20_random=f32(a_rand),
                 auto20_bstar=np.asarray(a_b, dtype=np.int16), auto20_adjusted=f32(a_adj),
                 auto20_cost=f32(a_cost))
        os.replace(tmp, SERIES_DIR / (file + ".npz"))
    return file, T, len(dets), errors, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending series (smoke test)")
    ap.add_argument("--only", nargs="*", default=None, help="only these series files")
    a = ap.parse_args()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(ROOT / "app/data/tsb-ad/series.json"))
    files = a.only or sorted(meta, key=lambda f: -(meta[f]["n"] + 50 * meta[f]["anomalies"]))  # heaviest first
    todo = [f for f in files if not (SERIES_DIR / (f + ".npz")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    log = open(OUT / "run.log", "a")
    t0 = time.time()
    print(f"{time.strftime('%F %T')} start: {len(todo)} series to do, {len(files) - len(todo)} done, "
          f"{a.workers} workers", file=log, flush=True)
    n_err = 0
    with Pool(a.workers) as pool:
        for i, (f, T, nd, errors, sec) in enumerate(pool.imap_unordered(one_series, todo, chunksize=1), 1):
            n_err += len(errors)
            print(f"{time.strftime('%F %T')} [{i}/{len(todo)}] {f} T={T} detectors={nd} "
                  f"skipped={len(errors)} {sec:.1f}s", file=log, flush=True)
            for e in errors:
                print("    " + e.replace("\n", " | "), file=log, flush=True)
    print(f"{time.strftime('%F %T')} done: {len(todo)} series in {(time.time() - t0) / 60:.1f} min, "
          f"{n_err} detector runs skipped", file=log, flush=True)


if __name__ == "__main__":
    main()

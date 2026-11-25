"""Benchmark-wide per-anomaly AutoVUS with per-side buffers, for every (series, detector) of TSB-AD.

    nice -n 19 conda run -n autovus python src/experiments/local_eval/export_local_ps.py --L 20 --workers 4
    (resumable: series with an output file are skipped; --mem_gb caps each worker)

Same as the auto20 run of export_local.py, with sides='per_side' and the maximum buffer L as an option:
  AutoVUS(FastVUS(L=L, cumulative=True, per_anomaly=True), sides='per_side'), CPU.

One file per series: experiments/local_benchmark/series_ps<L>/<series>.npz with
  detectors (D,), start, end (K,) inclusive, original coordinates
  ps<L>_area (D, K, L+1), ps<L>_random (D, K, L+1) float32: per-anomaly area and random
  baseline per buffer, every anomaly capped at its b* on each side
  ps<L>_bleft, ps<L>_bright (D, K) int16: the selected buffers
  ps<L>_adjusted (D,) float32 (global AutoVUS), ps<L>_cost (D, T) float32
  errors: detectors skipped (missing / NaN / wrong length / exception)
Log: experiments/local_benchmark/run_ps<L>.log.
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
L = 20                      # set by --L in main and in every worker
SERIES_DIR = OUT / "series_ps20"


def setup(L_, mem_gb):
    global L, SERIES_DIR
    L, SERIES_DIR = L_, OUT / f"series_ps{L_}"
    if mem_gb:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_gb << 30, mem_gb << 30))


def one_series(file):
    t0 = time.time()
    p = TSADParser()
    _, y = load_series(p, file)
    T = len(y)
    dets, errors = [], []
    area, rand, cost, bl, br, adj = [], [], [], [], [], []
    start, end = None, None
    for d in p.get_detectors():
        try:
            s = load_score(p, file, d)
            if s is None or len(s) != T:
                errors.append(f"{d}: missing, NaN or wrong length")
                continue
            r = AutoVUS(FastVUS(L=L, cumulative=True, per_anomaly=True), sides="per_side").compute(y, s)
            la = r["local"]
        except Exception:
            errors.append(f"{d}: {traceback.format_exc(limit=2)}")
            continue
        dets.append(d)
        start, end = la["start"], la["end"]
        area.append(la["area"]); rand.append(la["random"]); cost.append(la["cost"])
        bl.append([x["left"]["buffer_val"] for x in r["anomalies"]])
        br.append([x["right"]["buffer_val"] for x in r["anomalies"]])
        adj.append(r["adjusted"])
    if dets:
        f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
        tmp = SERIES_DIR / (file + ".tmp.npz")
        np.savez(tmp, detectors=np.array(dets), start=start, end=end, errors=np.array(errors),
                 **{f"ps{L}_area": f32(area), f"ps{L}_random": f32(rand), f"ps{L}_cost": f32(cost),
                    f"ps{L}_bleft": np.asarray(bl, dtype=np.int16), f"ps{L}_bright": np.asarray(br, dtype=np.int16),
                    f"ps{L}_adjusted": f32(adj)})
        os.replace(tmp, SERIES_DIR / (file + ".npz"))
    return file, T, len(dets), errors, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=20, help="maximum buffer")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem_gb", type=int, default=10, help="address-space cap per worker")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending series (smoke test)")
    ap.add_argument("--only", nargs="*", default=None, help="only these series files")
    a = ap.parse_args()
    setup(a.L, 0)
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(ROOT / "app/data/tsb-ad/series.json"))
    files = a.only or sorted(meta, key=lambda f: -(meta[f]["n"] + 50 * meta[f]["anomalies"]))  # heaviest first
    todo = [f for f in files if not (SERIES_DIR / (f + ".npz")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    log = open(OUT / f"run_ps{a.L}.log", "a")
    t0 = time.time()
    print(f"{time.strftime('%F %T')} start: {len(todo)} series to do, {len(files) - len(todo)} done, "
          f"{a.workers} workers", file=log, flush=True)
    n_err = 0
    with Pool(a.workers, setup, (a.L, a.mem_gb)) as pool:
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

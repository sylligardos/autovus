"""Per-anomaly buffers and detection types for every (series, detector) of TSB-AD.

    nice -n 19 conda run -n autovus python src/experiments/buffers/export_buffers.py --workers 4

AutoVUS(FastVUS(L=20, cumulative=True), sides='per_side'), paper rule, CPU.
One CSV per series in experiments/buffers/series/<series>.csv (resumable: done
series are skipped), columns: series, detector, k, start, end, b_left, b_right, type.
At the end all files are joined into experiments/buffers/anomalies_ps20.csv.
Log: experiments/buffers/run.log.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AutoVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402

OUT = ROOT / "experiments" / "buffers"
SERIES_DIR = OUT / "series"
COLS = ["series", "detector", "k", "start", "end", "b_left", "b_right", "type"]


def one_series(file):
    t0 = time.time()
    p = TSADParser()
    _, y = load_series(p, file)
    rows, errors = [], 0
    for d in p.get_detectors():
        try:
            s = load_score(p, file, d)
            if s is None or len(s) != len(y):
                errors += 1
                continue
            auto = AutoVUS(FastVUS(L=20, cumulative=True), sides="per_side")
            auto.compute(y, s)
            for k, a in enumerate(auto.anomalies):
                rows.append([file, d, k, int(a["start"]), int(a["end"]),
                             int(a["left"]["buffer_val"]), int(a["right"]["buffer_val"]), a["detection"]])
        except Exception:
            errors += 1
            traceback.print_exc()
    tmp = SERIES_DIR / (file + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    os.replace(tmp, SERIES_DIR / (file + ".csv"))
    return file, len(rows), errors, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N series (smoke test)")
    args = ap.parse_args()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(ROOT / "app/data/tsb-ad/series.json"))
    files = sorted(meta, key=lambda f: -(meta[f]["n"] + 50 * meta[f]["anomalies"]))  # heaviest first
    if args.limit:
        files = files[:args.limit]
    todo = [f for f in files if not (SERIES_DIR / (f + ".csv")).exists()]
    t0 = time.time()
    with open(OUT / "run.log", "a") as log:
        log.write(f"start {time.ctime()} todo {len(todo)}/{len(files)}\n")
        with Pool(args.workers) as pool:
            for i, (f, n, e, dt) in enumerate(pool.imap_unordered(one_series, todo), 1):
                log.write(f"{i}/{len(todo)} {f} rows {n} errors {e} {dt:.1f}s total {time.time() - t0:.0f}s\n")
                log.flush()
        with open(OUT / "anomalies_ps20.csv", "w", newline="") as out:
            w = csv.writer(out)
            w.writerow(COLS)
            for f in files:
                path = SERIES_DIR / (f + ".csv")
                if path.exists():
                    with open(path) as fin:
                        next(fin)
                        out.write(fin.read())
        log.write(f"done {time.ctime()} {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()

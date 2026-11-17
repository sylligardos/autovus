"""Benchmark-wide AutoVUS for every L' up to L=256 on TSB-AD-U (870 series x 32 detectors),
with AutoVUS.compute_all_L, for sides='symmetric' and sides='per_side'.

Usage: nice -n 19 conda run -n autovus python src/experiments/autovus_all_L.py [--workers 8] [--mem_gb 10] [--limit N]
Resumable: one series = one file; rerun to continue. One thread per worker.

Writes experiments/autovus_all_L/
    series/<series>.npz   one file per series:
        detectors (D,), sides (2,) = ['symmetric', 'per_side'], buffers (257,) = L' values
        start, end (K,)                 anomaly coordinates
        adjusted, vus, random (D, 2, 257) float32   the AutoVUS value and its two parts per L'; NaN = no result
        changes (N, 6) int32            b* stored only where it changes; columns:
                                        detector index, sides index, side (0 left, 1 right), anomaly k, L', b*
                                        every b* is 0 at L'=0; a row says b* of anomaly k becomes b* from L' on.
                                        symmetric rows are stored once, as side 0 (left = right).
    pairs.csv             one row per (series, detector, sides): length, n_anomalies, n_sets, seconds, error
    run.log               one line per finished series
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, csv, sys, time  # noqa: E402
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "app", "api"))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
from autovus import FastVUS, AutoVUS  # noqa: E402
from autovus_benchmark import DETECTORS  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "autovus_all_L")
L = 256
SIDES = ('symmetric', 'per_side')
KEYS = ['series', 'detector', 'sides', 'length', 'n_anomalies', 'n_sets', 'seconds', 'error']


def changes(bstar):
    """(k, column, b*) for every place a row of the (K, n_buffers) b* matrix changes; all rows start at 0."""
    k, i = np.where(bstar[:, 1:] != bstar[:, :-1])
    return k, i + 1, bstar[k, i + 1]


def one_series(file):
    p = TSADParser()
    _, label = load_series(p, file)
    rows = []
    if label.sum() == 0:
        return file, rows
    engine = FastVUS(L=L, cumulative=True)
    autos = [AutoVUS(engine, sides=s) for s in SIDES]
    n = len(engine.buffers)
    D = len(DETECTORS)
    curves = {key: np.full((D, len(SIDES), n), np.nan, dtype=np.float32) for key in ('adjusted', 'vus', 'random')}
    start, end = AutoVUS._coordinates(label)
    change_rows = []
    for d, det in enumerate(DETECTORS):
        score = load_score(p, file, det)
        if score is None:
            continue
        m = min(len(label), len(score))
        lab, sco = label[:m], np.asarray(score[:m], dtype=float)
        if lab.sum() == 0:
            continue
        for si, auto in enumerate(autos):
            row = {'series': file, 'detector': det, 'sides': SIDES[si], 'length': len(label)}
            t = time.perf_counter()
            try:
                out = auto.compute_all_L(lab, sco)
                for key in curves:
                    curves[key][d, si] = out[key]
                for side, name in enumerate(('bstar_left', 'bstar_right')[:1 if SIDES[si] == 'symmetric' else 2]):
                    k, i, b = changes(out[name])
                    change_rows.append(np.column_stack([np.full(len(k), d), np.full(len(k), si), np.full(len(k), side),
                                                        k, engine.buffers[i], b]))
                row.update(n_anomalies=len(out['anomalies']), n_sets=len(out['set_start']))
            except Exception as e:  # noqa: BLE001 (MemoryError included)
                row['error'] = f'{type(e).__name__}: {str(e)[:100]}'
            row['seconds'] = round(time.perf_counter() - t, 3)
            rows.append(row)
    table = np.concatenate(change_rows).astype(np.int32) if change_rows else np.zeros((0, 6), dtype=np.int32)
    path = os.path.join(OUT, 'series', file.replace('.csv', '.npz'))
    np.savez_compressed(path + '.tmp.npz', detectors=np.array(DETECTORS), sides=np.array(SIDES), buffers=engine.buffers,
                        start=start, end=end, changes=table, **curves)
    os.replace(path + '.tmp.npz', path)
    return file, rows


def cap_memory(gb):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (gb << 30, gb << 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--mem_gb', type=int, default=10, help='address-space cap per worker')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'series'), exist_ok=True)
    p = TSADParser()
    files = p.meta['filename'].tolist()
    files.sort(key=lambda f: os.path.getsize(p.raw_path / f))
    if args.limit:
        files = files[:args.limit]
    todo = [f for f in files if not os.path.exists(os.path.join(OUT, 'series', f.replace('.csv', '.npz')))]   # resume
    n_done = len(files) - len(todo)
    pairs_path = os.path.join(OUT, 'pairs.csv')
    new = not os.path.exists(pairs_path)
    t0 = time.time()
    with open(pairs_path, 'a', newline='') as fp, open(os.path.join(OUT, 'run.log'), 'a') as log, \
            Pool(args.workers, cap_memory, (args.mem_gb,)) as pool:
        w = csv.DictWriter(fp, fieldnames=KEYS)
        if new:
            w.writeheader()
        for i, (file, rows) in enumerate(pool.imap_unordered(one_series, todo), 1):
            w.writerows(rows); fp.flush()
            log.write(f'{n_done + i}/{len(files)} {file} rows={len(rows)} {time.time() - t0:.0f}s\n'); log.flush()
    print(f'{len(todo)} series done, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

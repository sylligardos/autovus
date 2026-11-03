"""Benchmark-wide AutoVUS vs VUS on TSB-AD-U (870 series x 32 detectors), for the Sec V paragraph.

Per (series, detector) pair, one FastVUS pass at L=128 (cumulative) gives VUS(b) for every b,
so VUS at maximum buffer 0, 2, 4, ..., 128 comes from the same curve. AutoVUS is computed at
L=128 and at L=20. AUC-PR is the no-buffer reference. Random detectors (uniform scores,
SEEDS seeds per series) are the chance control.

Usage: nice -n 19 conda run -n autovus python src/experiments/autovus_benchmark.py [--workers 2] [--mem_gb 10] [--limit N]
Resumable: one series per checkpoint; rerun to continue.
Writes experiments/autovus_benchmark/pairs.csv, anomalies.csv (b* per anomaly at L=128), run.log.
"""
import argparse, csv, os, sys, time
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "app", "api"))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
from autovus import FastVUS, AutoVUS  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "autovus_benchmark")
DETECTORS = ['AnomalyTransformer', 'AutoEncoder', 'CNN', 'Chronos', 'Donut', 'FITS', 'IForest', 'KMeansAD_U',
             'KShapeAD', 'LOF', 'LSTMAD', 'Lag_Llama', 'MOMENT_FT', 'MOMENT_ZS', 'MatrixProfile', 'OFA',
             'OmniAnomaly', 'POLY', 'SAND', 'SR', 'Series2Graph', 'Sub_HBOS', 'Sub_IForest', 'Sub_KNN', 'Sub_LOF',
             'Sub_MCD', 'Sub_OCSVM', 'Sub_PCA', 'TimesFM', 'TimesNet', 'TranAD', 'USAD']
LMAX = 128
LS = [0, 2, 4, 8, 16, 20, 32, 64, 128]
SEEDS = 3


def one_pair(eng128, auto128, auto20, label, score):
    n = min(len(label), len(score))
    lab, sco = label[:n], np.asarray(score[:n], dtype=float)
    if np.any(np.isnan(sco)) or lab.sum() == 0:
        return None, None
    curve = np.asarray(eng128.compute(lab, sco))
    row = {f'vus_{b}': float(curve[b]) for b in LS}
    row['auc_pr'] = float(average_precision_score(lab, sco))
    r = auto128.compute(lab, sco)
    bs = [int(a['left']['buffer_val']) for a in r['anomalies']]
    row.update(autovus_128=r['adjusted'], vus_adapt_128=r['vus'], random_128=r['random'],
               b_mean_128=float(np.mean(bs)), b_max_128=int(max(bs)), n_anomalies=len(bs))
    r20 = auto20.compute(lab, sco)
    bs20 = [int(a['left']['buffer_val']) for a in r20['anomalies']]
    row.update(autovus_20=r20['adjusted'], vus_adapt_20=r20['vus'], random_20=r20['random'],
               b_mean_20=float(np.mean(bs20)), b_max_20=int(max(bs20)))
    anoms = [(k, a['start'], a['end'], bs[k], a['detection']) for k, a in enumerate(r['anomalies'])]
    return row, anoms


def one_series(file):
    p = TSADParser()
    _, label = load_series(p, file)
    rows, anoms = [], []
    if label.sum() == 0:
        return file, rows, anoms
    eng128 = FastVUS(L=LMAX, cumulative=True)
    eng20 = FastVUS(L=20, cumulative=True)
    auto128, auto20 = AutoVUS(eng128), AutoVUS(eng20)
    scores = [(d, load_score(p, file, d)) for d in DETECTORS]
    rng = np.random.default_rng(0)
    scores += [(f'Random_{i}', rng.random(len(label))) for i in range(SEEDS)]
    for det, s in scores:
        if s is None:
            continue
        try:
            row, an = one_pair(eng128, auto128, auto20, label, s)
        except Exception as e:  # noqa: BLE001
            row, an = {'error': str(e)[:100]}, None
        if row is None:
            continue
        row.update(series=file, detector=det, length=len(label))
        rows.append(row)
        for a in an or []:
            anoms.append((file, det) + a)
    return file, rows, anoms


KEYS = ['series', 'detector', 'length', 'n_anomalies', 'auc_pr'] + [f'vus_{b}' for b in LS] + \
       ['autovus_128', 'vus_adapt_128', 'random_128', 'b_mean_128', 'b_max_128',
        'autovus_20', 'vus_adapt_20', 'random_20', 'b_mean_20', 'b_max_20', 'error']
ANOM_KEYS = ['series', 'detector', 'k', 'start', 'end', 'b_star_128', 'detection']


def cap_memory(gb):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (gb << 30, gb << 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--mem_gb', type=int, default=10, help='address-space cap per worker')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    p = TSADParser()
    files = p.meta['filename'].tolist()
    if args.limit:
        files = files[:args.limit]
    files.sort(key=lambda f: os.path.getsize(p.raw_path / f))
    pairs_path, anoms_path = os.path.join(OUT, 'pairs.csv'), os.path.join(OUT, 'anomalies.csv')
    done = set()
    if os.path.exists(pairs_path):                                    # resume: one series = one checkpoint
        with open(pairs_path, newline='') as f:
            done = {r['series'] for r in csv.DictReader(f)}
    todo = [f for f in files if f not in done]
    t0 = time.time()
    with open(pairs_path, 'a', newline='') as fp, open(anoms_path, 'a', newline='') as fa, \
            open(os.path.join(OUT, 'run.log'), 'a') as log, Pool(args.workers, cap_memory, (args.mem_gb,)) as pool:
        wp, wa = csv.DictWriter(fp, fieldnames=KEYS, extrasaction='ignore'), csv.writer(fa)
        if not done:
            wp.writeheader(); wa.writerow(ANOM_KEYS)
        for i, (file, rows, anoms) in enumerate(pool.imap_unordered(one_series, todo), 1):
            wp.writerows(rows); wa.writerows(anoms); fp.flush(); fa.flush()
            log.write(f'{len(done) + i}/{len(files)} {file} rows={len(rows)} {time.time() - t0:.0f}s\n'); log.flush()
    print(f'{len(todo)} series done, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

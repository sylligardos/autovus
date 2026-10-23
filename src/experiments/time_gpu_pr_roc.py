"""FastVUSGPU time of metric='pr', 'roc', 'both' on every TSB-AD (series, detector) pair, L=256.

Run from the repo root: python src/experiments/time_gpu_pr_roc.py experiments/gpu_pr_roc_timing/times.csv
Output columns: file, detector, T, pr, roc, both (seconds of one compute() each, after a warm-up per series).
"""
import csv, sys, time
import numpy as np, torch
sys.path.insert(0, 'app'); sys.path.insert(0, 'src')
import export as E
from autovus import FastVUSGPU
from tsad_parser import TSADParser

p = TSADParser()
dets = list(p.get_detectors())
files = sorted(p.meta['filename'].tolist(), key=lambda f: (p.raw_path / f).stat().st_size)
eng = FastVUSGPU(L=256)
out = open(sys.argv[1], 'w', newline=''); w = csv.writer(out); w.writerow(['file', 'detector', 'T', 'pr', 'roc', 'both'])
tot = {'pr': 0.0, 'roc': 0.0, 'both': 0.0}; t_all = time.time(); pairs = 0; skipped = 0
for i, f in enumerate(files, 1):
    _, label = E.load_series(p, f)
    if label.sum() == 0 or label.all():
        skipped += 1; continue
    first = True
    for det in dets:
        s = E.load_score(p, f, det)
        if s is None:
            continue
        n = min(len(label), len(s)); lab, sco = label[:n], s[:n]
        if np.any(np.isnan(sco)):
            continue
        if first:
            eng.compute(lab, sco); first = False      # warm-up per series
        row = [f, det, n]
        for m in ('pr', 'roc', 'both'):
            torch.cuda.synchronize(); t = time.perf_counter()
            eng.compute(lab, sco, metric=m)
            torch.cuda.synchronize(); dt = time.perf_counter() - t
            tot[m] += dt; row.append(dt)
        w.writerow(row); pairs += 1
    if i % 50 == 0:
        out.flush()
        print(f'{i}/{len(files)} series, {pairs} pairs, {time.time() - t_all:.0f} s, totals ' + ', '.join(f'{k} {v:.1f}' for k, v in tot.items()), flush=True)
out.close()
print(f'DONE {pairs} pairs ({skipped} series skipped), wall {time.time() - t_all:.0f} s, GPU totals: ' + ', '.join(f'{k} {v:.1f} s' for k, v in tot.items()), flush=True)

"""VUS-PR of every TSB-AD (series, detector) pair for every buffer length L = 0..LMAX, in one FastVUSGPU pass (cumulative=True).

Run from the repo root: python src/experiments/vus_vs_L_gpu.py experiments/vus_vs_L
Writes pairs.csv (file, detector, T), curves.npy (pairs x (LMAX+1), float64: VUS-PR at L = 0..LMAX), run.log.
Uses: C80 (mean VUS-PR as L grows) and C39 (GPU error: column L=128 vs the CPU vus_pr in app/data/tsb-ad/scores.json).
"""
import csv, os, sys, time
import numpy as np, torch
sys.path.insert(0, 'app'); sys.path.insert(0, 'src')
import export as E
from autovus import FastVUSGPU
from tsad_parser import TSADParser

LMAX = 256
out_dir = sys.argv[1]
p = TSADParser()
dets = list(p.get_detectors())
files = sorted(p.meta['filename'].tolist(), key=lambda f: (p.raw_path / f).stat().st_size)
eng = FastVUSGPU(L=LMAX, cumulative=True)
pairs, curves, skipped, t_all, t_gpu = [], [], 0, time.time(), 0.0
for i, f in enumerate(files, 1):
    _, label = E.load_series(p, f)
    if label.sum() == 0 or label.all():
        skipped += 1; continue
    for det in dets:
        s = E.load_score(p, f, det)
        if s is None:
            continue
        n = min(len(label), len(s)); lab, sco = label[:n], s[:n]
        if np.any(np.isnan(sco)):
            continue
        torch.cuda.synchronize(); t = time.perf_counter()
        c = eng.compute(lab, sco, metric='pr')
        torch.cuda.synchronize(); t_gpu += time.perf_counter() - t
        pairs.append((f, det, n)); curves.append(np.asarray(c, dtype=np.float64))
    if i % 50 == 0:
        print(f'{i}/{len(files)} series, {len(pairs)} pairs, wall {time.time() - t_all:.0f} s, gpu {t_gpu:.0f} s', flush=True)
np.save(os.path.join(out_dir, 'curves.npy'), np.vstack(curves))
with open(os.path.join(out_dir, 'pairs.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['file', 'detector', 'T']); w.writerows(pairs)
print(f'DONE {len(pairs)} pairs ({skipped} series skipped), LMAX={LMAX}, wall {time.time() - t_all:.0f} s, gpu {t_gpu:.0f} s, device {eng.device}', flush=True)

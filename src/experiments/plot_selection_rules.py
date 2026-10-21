"""G6: visual evidence for the buffer-selection rules, synthetic and real.

For every (label, score) case and every anomaly, the window used by AutoVUS
(halfway to the neighbours) is scored once per side. Three rules are drawn
on the same axes with their b*:
  paper      (dA+1)/(dg+1) on the marginal curves          (Section 3.3)
  cumulative (dC+1)/(dG+1) on the cumulative curves        (Mingyi's code)
  ratio      A(b) / g(b)                                   (June design)
Smallest argmax, no guards. Figures: experiments/figures/selection_rules/{synthetic,real}/.

    conda run -n autovus python src/experiments/plot_selection_rules.py              # synthetic only
    conda run -n autovus python src/experiments/plot_selection_rules.py --real       # + the 8-series TSB-AD sample x 4 detectors
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autovus import FastVUS, AutoVUS

OUT = 'experiments/figures/selection_rules'
RULES = ('paper', 'cumulative', 'ratio')
COLORS = {'paper': 'C0', 'cumulative': 'C1', 'ratio': 'C2'}


# ---------------------------------------------------------------- synthetic cases
T = 1000
LABELS = {
    'baseline (len 20)': [(400, 419)],
    'short (len 5)': [(400, 404)],
    'long (len 60)': [(300, 359)],
    'two close (gap 25)': [(300, 319), (345, 364)],
}


def make_label(anomalies):
    label = np.zeros(T, int)
    for s, e in anomalies:
        label[s:e + 1] = 1
    return label


def bounds(label):
    diff = np.diff(np.r_[0, label, 0])
    return np.where(diff == 1)[0], np.where(diff == -1)[0] - 1


def shifted(label, offset):
    """The label shifted by offset points (negative = early)."""
    score = np.zeros(T)
    if offset > 0:
        score[offset:] = label[:-offset]
    elif offset < 0:
        score[:offset] = label[-offset:]
    else:
        score[:] = label
    return score


def exact_inner(label):
    score = np.zeros(T)
    for s, e in zip(*bounds(label)):
        q = max(1, (e - s + 1) // 4)
        score[s + q:e - q + 1] = 1.0 if s + q < e - q + 1 else score[s + q:e - q + 1]
        if s + q >= e - q + 1:
            score[s:e + 1] = 1.0
    return score


def partial(label):
    score = np.zeros(T)
    for s, e in zip(*bounds(label)):
        score[s:s + max(1, (e - s + 1) // 2)] = 1.0
    return score


def over(label, L, early):
    score = np.zeros(T)
    for s, e in zip(*bounds(label)):
        reach = L + (e - s + 1) // 2
        if early:
            score[max(0, s - reach):e + 1] = 1.0
        else:
            score[s:e + 1] = 1.0
            n = min(T, e + 1 + reach) - (e + 1)
            score[e + 1:e + 1 + n] = np.linspace(1.0, 0.0, n)
    return score


def isolated_fp(label):
    score = np.zeros(T)
    s, e = bounds(label)
    score[(e[-1] + T) // 2] = 1.0          # far from every anomaly
    return score


def cases(label, L):
    return {
        'exact (inner 50%)': exact_inner(label),
        'early by 10': shifted(label, -10),
        'late by 10': shifted(label, 10),
        'early by 30': shifted(label, -30),
        'late by 30': shifted(label, 30),
        'partial (first half)': partial(label),
        'over-prediction, early side': over(label, L, early=True),
        'over-prediction, late side (decay)': over(label, L, early=False),
        'isolated false positive, anomaly missed': isolated_fp(label),
        'missed (noise only)': np.zeros(T),
    }


# ---------------------------------------------------------------- rules
def rule_curves(engine, label, score):
    auto = AutoVUS(engine)
    C, G = auto.curves(label, score)
    A, g = AutoVUS.decumulate(C), AutoVUS.decumulate(G)
    return {
        'paper': (A - A[0] + 1.0) / (g - g[0] + 1.0),
        'cumulative': (C - C[0] + 1.0) / (G - G[0] + 1.0),
        'ratio': A / np.clip(g, 1e-10, None),
    }, A, g


def windows(label):
    start, end = bounds(label)
    out = []
    for k in range(len(start)):
        lo = int((end[k - 1] + start[k]) // 2) if k > 0 else 0
        hi = int((end[k] + start[k + 1]) // 2) if k < len(start) - 1 else len(label)
        out.append((int(start[k]), int(end[k]), lo, hi))
    return out


def figure(label, score, L, title, path, series=None, max_anomalies=4):
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(0)
    if series is None:                                        # synthetic: add a little noise so ranks are defined
        score = np.clip(score + rng.normal(0, 0.02, len(score)), 0, None)
    wins = windows(label)[:max_anomalies]
    n = len(wins)
    fig, axes = plt.subplots(n, 3, figsize=(17, 3.2 * n), squeeze=False)
    engines = {side: FastVUS(L=L, cumulative=True, side=side) for side in ('left', 'right')}
    for row, (s, e, lo, hi) in enumerate(wins):
        ax = axes[row]
        wl, ws = label[lo:hi], score[lo:hi]
        picks = {}
        for col, side in enumerate(('left', 'right'), start=1):
            rules, A, g = rule_curves(engines[side], wl, ws)
            b = np.arange(L + 1)
            ax2 = ax[col].twinx()
            ax2.plot(b, A, color='grey', lw=1, ls='--', label='A (marginal AUC-PR)'); ax2.plot(b, g, color='grey', lw=1, ls=':', label='g (random)')
            ax2.set_ylim(0, 1.05); ax2.tick_params(axis='y', colors='grey')
            for name, r in rules.items():
                bstar = int(np.argmax(r)); picks[(side, name)] = bstar
                ax[col].plot(b, r / r.max(), color=COLORS[name], lw=1.5, label=f"{name}: b*={bstar}, peak {r.max():.2f}")
                ax[col].axvline(bstar, color=COLORS[name], lw=0.8, alpha=0.6)
            ax[col].set_title(f"anomaly {row + 1}, {side} side"); ax[col].set_xlabel('buffer b'); ax[col].set_ylabel('rule curve / its max')
            ax[col].legend(loc='lower right', fontsize=8)
            if row == 0 and col == 1:
                ax2.legend(loc='upper left', fontsize=8)
        x = np.arange(lo, hi)
        if series is not None:
            ax[0].plot(x, series[lo:hi], color='k', lw=0.6, alpha=0.6, label='series')
        ax[0].plot(x, ws, color='C3', lw=0.9, label='score')
        ax[0].fill_between(x, 0, wl * ws.max() if ws.max() > 0 else wl, color='k', alpha=0.15, label='anomaly')
        for name in RULES:
            bl, br = picks[('left', name)], picks[('right', name)]
            y = {'paper': -0.05, 'cumulative': -0.1, 'ratio': -0.15}[name] * (ws.max() if ws.max() > 0 else 1)
            ax[0].plot([s - bl, e + br], [y, y], color=COLORS[name], lw=3, solid_capstyle='butt', label=f"{name} buffer ({bl}, {br})")
        ax[0].set_xlim(lo, hi); ax[0].set_title(f"window of anomaly {row + 1} [{s}, {e}]"); ax[0].legend(loc='upper right', fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def synthetic(L=64):
    os.makedirs(f'{OUT}/synthetic', exist_ok=True)
    for lname, anomalies in LABELS.items():
        label = make_label(anomalies)
        for cname, score in cases(label, L).items():
            fname = f"{lname.split(' ')[0]}__{cname.split(' (')[0].replace(' ', '_').replace(',', '')}.png"
            figure(label, score, L, f"{lname}: {cname}   (L={L})", f'{OUT}/synthetic/{fname}')
    print('synthetic figures in', f'{OUT}/synthetic')


def real(L=128):
    from tsad_parser import TSADParser
    from check_real import pick_series, DETECTORS
    os.makedirs(f'{OUT}/real', exist_ok=True)
    p = TSADParser()
    fnames, series, labels = p.load_timeseries(filename=pick_series(p))
    scores, _ = p.load_scores_and_delete(fnames, series, labels, detectors=DETECTORS)
    for fname, ts, label, det_scores in zip(fnames, series, labels, scores):
        label = np.asarray(label).astype(int)
        for det, score in zip(DETECTORS, det_scores):
            if len(score) != len(label) or not np.isfinite(score).all():
                continue
            figure(label, score, L, f"{fname} / {det}   (L={L}, first anomalies)",
                   f"{OUT}/real/{fname[:-4]}__{det}.png", series=np.asarray(ts, dtype=float))
    print('real figures in', f'{OUT}/real')


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--real', action='store_true')
    args = a.parse_args()
    synthetic()
    if args.real:
        real()

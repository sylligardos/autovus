"""G6: compare the buffer-selection rules on detections with a known lag (toy).

One anomaly (length 20) in a window of 400 points; the detector fires a bump
of the same length shifted by `lag` points (negative = early) on top of noise.
The curves come from FastVUS (cumulative, side = the lagged side); A is the
marginal AUC-PR per buffer, g the random baseline per buffer, C and G their
cumulative (running mean) versions. Rules:
  paper      (dA+1)/(dg+1) on the marginal curves, dA = A(b) - A(0)   (Section 3.3)
  cumulative (dC+1)/(dG+1) on the cumulative curves                  (Mingyi's code, ported as is)
  ratio      A(b) / g(b)                                             (June design)
All take the smallest argmax; none has a guard. Ideal b* = |lag|.

    conda run -n autovus python src/experiments/study_selection_rule.py            # table
    conda run -n autovus python src/experiments/study_selection_rule.py --lag 20   # + curves to experiments/figures/selection_rule_lag20.pdf
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np

from autovus import FastVUS, AutoVUS

L = 64
LAGS = (-40, -20, -10, -5, 0, 5, 10, 20, 40)


def window(lag, noise=0.1, width=20, seed=0, bump=True):
    rng = np.random.default_rng(seed)
    label = np.zeros(400, int); label[190:210] = 1
    score = rng.random(400) * noise
    if bump:
        s = 190 + lag
        score[s:s + width] += 1.0
    return label, np.clip(score, 0, 1)


def curves(label, score, side):
    """Marginal A, g and cumulative C, G over the buffers 0..L."""
    auto = AutoVUS(FastVUS(L=L, cumulative=True, side=side))
    C, G = auto.curves(label, score)
    return AutoVUS.decumulate(C), AutoVUS.decumulate(G), C, G


def rule_curves(A, g, C, G):
    return {
        'paper': (A - A[0] + 1.0) / (g - g[0] + 1.0),
        'cumulative': (C - C[0] + 1.0) / (G - G[0] + 1.0),
        'ratio': A / np.clip(g, 1e-10, None),
    }


def select(r):
    return int(np.argmax(r))          # smallest argmax; buffers are 0..L with step 1


def table():
    print(f"L={L}, anomaly length 20, bump width 20. b* per rule (ideal = |lag|), then peak value of the rule's curve.")
    print(f"{'rule':11s}" + ''.join(f"{'lag ' + str(l):>9s}" for l in LAGS) + "   noise only (5 seeds, left)")
    noise = [curves(*window(0, seed=s, bump=False), 'left') for s in range(5)]
    for name in ('paper', 'cumulative', 'ratio'):
        bs, peaks = [], []
        for lag in LAGS:
            r = rule_curves(*curves(*window(lag), 'left' if lag < 0 else 'right'))[name]
            bs.append(select(r)); peaks.append(r.max())
        nb = [select(rule_curves(*c)[name]) for c in noise]
        npk = [rule_curves(*c)[name].max() for c in noise]
        print(f"{name:11s}" + ''.join(f"{b:9d}" for b in bs) + f"   b* {nb}")
        print(f"{'  peak':11s}" + ''.join(f"{p:9.2f}" for p in peaks) + f"   peak {[round(p, 2) for p in npk]}")


def plot(lag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label, score = window(lag)
    A, g, C, G = curves(label, score, 'left' if lag < 0 else 'right')
    rules = rule_curves(A, g, C, G)
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    ax[0].plot(label, 'k', lw=1); ax[0].plot(score, 'C1', lw=0.8); ax[0].set_title(f'label and score, lag {lag}'); ax[0].set_xlim(150, 260)
    ax[1].plot(A, label='A (marginal AUC-PR)'); ax[1].plot(g, label='g (random)'); ax[1].legend(); ax[1].set_title('marginal curves'); ax[1].set_xlabel('buffer b')
    ax[2].plot(C, label='C (cumulative VUS)'); ax[2].plot(G, label='G (random)'); ax[2].legend(); ax[2].set_title('cumulative curves'); ax[2].set_xlabel('buffer b')
    for name, r in rules.items():
        ax[3].plot(r / r.max(), label=f"{name}: b*={select(r)}")
    ax[3].axvline(abs(lag), color='k', ls=':', label=f'|lag| = {abs(lag)}'); ax[3].legend(); ax[3].set_title('rule curves (scaled to max 1)'); ax[3].set_xlabel('buffer b')
    os.makedirs('experiments/figures', exist_ok=True)
    out = f'experiments/figures/selection_rule_lag{lag}.pdf'
    fig.tight_layout(); fig.savefig(out); print('saved', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lag', type=int, default=None, help='also save the curves for this lag')
    a = p.parse_args()
    table()
    if a.lag is not None:
        plot(a.lag)

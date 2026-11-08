"""Toy examples for local (per-anomaly) evaluation.

Each example: a series, its label, and a few detector scores.
Zones = cut at the midpoint between neighbouring anomalies (simplest split).
Printed per detector: global AUC-PR and the naive local AUC-PR per zone
(score computed on the cut alone, thresholds from the cut alone).
Output: experiments/local_eval/examples.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

N = 1000
rng = np.random.default_rng(0)


def series(anoms):
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(N)
    y = np.zeros(N, int)
    for s, e in anoms:
        x[s:e] += 1.5
        y[s:e] = 1
    return x, y


def floor(level=0.15):
    return level * rng.random(N)


def bump(sc, s, e, h):
    sc = sc.copy()
    sc[s:e] = np.maximum(sc[s:e], h)
    return sc


def zones(anoms):
    cuts = [0] + [(anoms[i][1] + anoms[i + 1][0]) // 2 for i in range(len(anoms) - 1)] + [N]
    return list(zip(cuts[:-1], cuts[1:]))


def examples():
    ex = []

    # 1. Tiny bump: D caught A1 strongly, A2 only with a bump just above its local
    #    noise floor; two medium false peaks near A1 outrank the bump globally.
    A = [(250, 270), (700, 720)]
    x, y = series(A)
    d = bump(bump(floor(), *A[0], 1.0), *A[1], 0.2)
    d = bump(bump(d, 80, 100, 0.5), 400, 420, 0.5)
    ex.append(("1. Tiny bump", x, y, A, {"D": d}))

    # 2. Partial coverage: D1 catches A1 and A2, D2 catches only A3.
    A = [(150, 170), (480, 500), (800, 820)]
    x, y = series(A)
    d1 = bump(bump(floor(), *A[0], 0.9), *A[1], 0.9)
    d2 = bump(floor(), *A[2], 0.9)
    ex.append(("2. Partial coverage", x, y, A, {"D1": d1, "D2": d2}))

    # 3. Far false alarm: D catches A1 but also fires hard far from it.
    A = [(300, 320)]
    x, y = series(A)
    d = bump(bump(floor(), *A[0], 0.8), 850, 870, 1.0)
    ex.append(("3. Far false alarm", x, y, A, {"D": d}))

    # 4. Coupling: both catch A2 at 0.5 under a false peak at 0.6; only Da catches A1.
    A = [(250, 270), (700, 720)]
    x, y = series(A)
    base = bump(bump(floor(), *A[1], 0.5), 450, 470, 0.6)
    ex.append(("4. Coupling", x, y, A, {"Da": bump(base, *A[0], 1.0), "Db": base}))

    # 5. Long vs short: A1 is 100 points, A2 is 5 points. Ds catches only the short
    #    one, Dl only the long one.
    A = [(200, 300), (700, 705)]
    x, y = series(A)
    ex.append(("5. Long vs short", x, y, A,
               {"Ds": bump(floor(), *A[1], 0.9), "Dl": bump(floor(), *A[0], 0.9)}))

    # 6. Fairness: (a) 1 anomaly, missed; (b) 10 anomalies, 9 caught, 1 missed.
    A = [(500, 520)]
    x, y = series(A)
    ex.append(("6a. One anomaly, missed", x, y, A, {"D": floor()}))
    A = [(50 + 95 * i, 70 + 95 * i) for i in range(10)]
    x, y = series(A)
    d = floor()
    for s_, e_ in A[:9]:
        d = bump(d, s_, e_, 0.9)
    ex.append(("6b. Ten anomalies, one missed", x, y, A, {"D": d}))

    # 7. Ensemble: D1 catches A1, A2; D2 catches A3, A4; D3 catches all four at a
    #    medium level, under one false peak. D3 has the best AUC-PR alone.
    A = [(100, 120), (330, 350), (560, 580), (800, 820)]
    x, y = series(A)
    d1 = bump(bump(floor(), *A[0], 0.9), *A[1], 0.9)
    d2 = bump(bump(floor(), *A[2], 0.9), *A[3], 0.9)
    d3 = floor()
    for s_, e_ in A:
        d3 = bump(d3, s_, e_, 0.6)
    d3 = bump(d3, 450, 470, 0.7)
    ex.append(("7. Ensemble", x, y, A, {"D1": d1, "D2": d2, "D3": d3}))

    # 8. Lag: one detector; A1 exact, A2 fires 8 points after its end, A3 fires
    #    40 points after its end (beyond L=20), A4 missed.
    A = [(100, 120), (330, 350), (560, 580), (800, 820)]
    x, y = series(A)
    d = bump(floor(), *A[0], 0.9)
    d = bump(d, A[1][1] + 8, A[1][1] + 18, 0.9)
    d = bump(d, A[2][1] + 40, A[2][1] + 50, 0.9)
    ex.append(("8. Lag", x, y, A, {"D": d}))

    # 9. Sharp vs broad: Ds marks both anomalies exactly; Db marks a plateau 15
    #    points wider on each side of both.
    A = [(250, 270), (700, 720)]
    x, y = series(A)
    ds = bump(bump(floor(), *A[0], 0.9), *A[1], 0.9)
    db = bump(bump(floor(), A[0][0] - 15, A[0][1] + 15, 0.9), A[1][0] - 15, A[1][1] + 15, 0.9)
    ex.append(("9. Sharp vs broad", x, y, A, {"Ds": ds, "Db": db}))

    # 10. K scaling: every anomaly is caught at 0.9 and has one false peak (10
    #     points at 0.95) 60 points after it. (a) 1 anomaly, (b) 8 anomalies.
    for K, tag in [(1, "10a"), (8, "10b")]:
        A = [(40 + 120 * i, 60 + 120 * i) for i in range(K)] if K > 1 else [(400, 420)]
        x, y = series(A)
        d = floor()
        for s_, e_ in A:
            d = bump(bump(d, s_, e_, 0.9), e_ + 60, e_ + 70, 0.95)
        ex.append((f"{tag}. K scaling, {K} anomal{'y' if K == 1 else 'ies'}", x, y, A, {"D": d}))
    return ex


def main():
    ex = examples()
    rows = sum(1 + len(dets) for *_, dets in ex)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 1.1 * rows), sharex=True)
    i = 0
    for name, x, y, A, dets in ex:
        Z = zones(A)
        ax = axes[i]; i += 1
        ax.plot(x, lw=0.6, color="k")
        for s, e in A:
            ax.axvspan(s, e, color="red", alpha=0.3)
        ax.set_ylabel("series", fontsize=7)
        ax.set_title(name, loc="left", fontsize=9, fontweight="bold")
        print(name)
        for dn, d in dets.items():
            ax = axes[i]; i += 1
            ax.plot(d, lw=0.7, color="C0")
            ax.set_ylim(0, 1.05)
            g = average_precision_score(y, d)
            loc = [average_precision_score(y[a:b], d[a:b]) for a, b in Z]
            txt = f"global {g:.2f} | naive local " + " ".join(f"A{k+1} {v:.2f}" for k, v in enumerate(loc))
            ax.text(0.01, 0.85, f"{dn}: {txt}", transform=ax.transAxes, fontsize=7, va="top")
            ax.set_ylabel(dn, fontsize=7)
            print(f"  {dn}: {txt}")
        for ax_ in axes[i - 1 - len(dets):i]:
            for a, b in Z[1:]:
                ax_.axvline(a, ls="--", lw=0.6, color="grey")
            for s, e in A:
                ax_.axvspan(s, e, color="red", alpha=0.12)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    out = str(Path(__file__).resolve().parents[3] / "experiments" / "local_eval" / "examples.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()

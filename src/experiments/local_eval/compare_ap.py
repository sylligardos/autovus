"""Run every local score on every toy example and print one table per example.

    conda run -n autovus python src/experiments/local_eval/compare_ap.py
"""
import numpy as np
from examples import examples
from local import local_scores

KEYS = ["naive", "A", "A_adj", "B", "B_adj", "roc", "roc_adj", "loss"]


def main():
    for name, x, y, A, dets in examples():
        print(f"\n== {name}")
        for dn, d in dets.items():
            r = local_scores(y, d)
            print(f"  {dn}: AUC-PR {r['ap']:.3f}  AUC-ROC {r['auc']:.3f}  "
                  f"checks A {r['check_A']:.1e} roc {r['check_roc']:.1e} loss {r['check_loss']:.1e}")
            print("     " + "".join(f"{k:>9}" for k in ["anom", "w"] + KEYS))
            for i in range(len(A)):
                row = [f"A{i+1}", f"{r['w'][i]:.2f}"] + [f"{r[k][i]:.3f}" for k in KEYS]
                print("     " + "".join(f"{v:>9}" for v in row))


if __name__ == "__main__":
    main()

import argparse

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)


def expand_labels(labels):
    """
    Build a matrix of progressively buffered label sequences.

    Row 0  : original labels (buffer = 0)
    Row k  : labels with each anomaly region expanded by k points on each side
    Row -1 : all ones (maximum expansion)
    """
    n = len(labels)
    rows = [labels.copy()]
    current = labels.copy()
    while current.sum() < n:
        expanded = current.copy()
        for i in range(n):
            if current[i] == 1:
                if i > 0:
                    expanded[i - 1] = 1
                if i < n - 1:
                    expanded[i + 1] = 1
        current = expanded
        rows.append(current.copy())
    return np.array(rows)


def compute_per_buffer_scores(scores, label_matrix):
    """
    For each buffer level, compute AP (= AUC-PR) and AUC-ROC.
    Degenerate rows (all-ones / all-zeros) get ap=roc=None.
    """
    n = label_matrix.shape[1]
    results = []
    for i, row in enumerate(label_matrix):
        n_pos = int(row.sum())
        if n_pos == 0 or n_pos == n:
            results.append({"buf": i, "n_pos": n_pos, "ap": None, "roc": None})
            continue
        ap = average_precision_score(row, scores)
        roc = roc_auc_score(row, scores)
        results.append({"buf": i, "n_pos": n_pos, "ap": ap, "roc": roc})
    return results


def aggregate(values, kind):
    """
    Aggregate a list of per-buffer metric values into a single scalar.

    - 'uniform'     : simple arithmetic mean
    - 'trapezoidal' : trapezoidal integration over buffer index, normalised
                      by the buffer range so the result is in [0, 1]
    """
    v = np.asarray(values, dtype=float)
    if kind == "uniform":
        return float(v.mean())
    if kind == "trapezoidal":
        if len(v) < 2:
            return float(v[0])
        x = np.arange(len(v))
        return float(np.trapezoid(v, x=x) / (x[-1] - x[0]))
    raise ValueError(f"unknown aggregation: {kind}")


def print_per_buffer_table(results, n, agg):
    print(f"\n{'buf':>4} {'#pos':>5} {'frac':>6} {'AP':>8} {'AUC-ROC':>9}")
    print("-" * 38)
    for r in results:
        frac = r["n_pos"] / n
        ap = "     n/a" if r["ap"] is None else f"{r['ap']:>8.4f}"
        roc = "      n/a" if r["roc"] is None else f"{r['roc']:>9.4f}"
        print(f"{r['buf']:>4} {r['n_pos']:>5} {frac:>6.2f} {ap} {roc}")

    ap_vals = [r["ap"] for r in results if r["ap"] is not None]
    roc_vals = [r["roc"] for r in results if r["roc"] is not None]
    if not ap_vals:
        return

    kinds = ["uniform", "trapezoidal"] if agg == "both" else [agg]
    print("-" * 38)
    for k in kinds:
        ap_agg = aggregate(ap_vals, k)
        roc_agg = aggregate(roc_vals, k)
        tag = f"{k[:5]}"
        print(f"{tag:>4} {'':>5} {'':>6} {ap_agg:>8.4f} {roc_agg:>9.4f}"
              f"   <- VUS-PR / VUS-ROC ({k})")


def plot_label_matrix(labels, scores, label_matrix, out_path):
    import matplotlib.pyplot as plt

    n_rows = label_matrix.shape[0]
    fig, axes = plt.subplots(
        n_rows + 1, 1,
        figsize=(12, 1.0 * (n_rows + 1)),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    ax = axes[0]
    ax.fill_between(range(len(scores)), scores, alpha=0.6, color="steelblue")
    ax.plot(scores, color="steelblue", lw=1.5)
    ax.set_ylabel("score", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=7)

    colors = {0: "#f0f0f0", 1: "#e05c5c"}
    for i, row in enumerate(label_matrix):
        ax = axes[i + 1]
        for j, val in enumerate(row):
            ax.axvspan(j - 0.5, j + 0.5, color=colors[val], lw=0)
        if i == 0:
            tag = "original"
        elif i == n_rows - 1:
            tag = "all 1s"
        else:
            tag = f"buf {i}"
        ax.set_ylabel(tag, fontsize=8, rotation=0, labelpad=40, va="center")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("time index", fontsize=9)
    axes[-1].set_xlim(-0.5, len(labels) - 0.5)
    fig.suptitle("Expanding label buffers", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_pr_roc_curves(scores, label_matrix, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_rows = label_matrix.shape[0]
    colors = cm.plasma(np.linspace(0, 0.85, n_rows))

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    for i, row in enumerate(label_matrix):
        n_pos = int(row.sum())
        if n_pos == 0 or n_pos == len(row):
            continue
        precision, recall, _ = precision_recall_curve(row, scores)
        ap = average_precision_score(row, scores)
        ax_pr.plot(recall, precision, color=colors[i], lw=1.8,
                   label=f"buf {i} (AP={ap:.3f})")

        fpr, tpr, _ = roc_curve(row, scores)
        roc = roc_auc_score(row, scores)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=1.8,
                    label=f"buf {i} (AUC={roc:.3f})")

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(fontsize=8, loc="lower left")
    ax_pr.set_title("PR curves per buffer (AP = AUC-PR)")
    ax_pr.grid(True, alpha=0.3)

    ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=1, alpha=0.6)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.05)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_title("ROC curves per buffer")
    ax_roc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_metric_vs_buffer(results, agg, out_path):
    import matplotlib.pyplot as plt

    bufs = [r["buf"] for r in results if r["ap"] is not None]
    aps = [r["ap"] for r in results if r["ap"] is not None]
    rocs = [r["roc"] for r in results if r["roc"] is not None]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bufs, aps, "o-", color="#d62728", lw=2, label="AP (AUC-PR)")
    ax.plot(bufs, rocs, "s-", color="#1f77b4", lw=2, label="AUC-ROC")

    kinds = ["uniform", "trapezoidal"] if agg == "both" else [agg]
    style = {"uniform": ":", "trapezoidal": "--"}
    for k in kinds:
        ap_agg = aggregate(aps, k)
        roc_agg = aggregate(rocs, k)
        ax.axhline(ap_agg, color="#d62728", ls=style[k], lw=1, alpha=0.7,
                   label=f"VUS-PR ({k}) = {ap_agg:.3f}")
        ax.axhline(roc_agg, color="#1f77b4", ls=style[k], lw=1, alpha=0.7,
                   label=f"VUS-ROC ({k}) = {roc_agg:.3f}")

    ax.set_xlabel("buffer level")
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(bufs)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Per-buffer metric values & aggregations")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# ---------- Dummy scenarios ----------------------------------------------

def make_triangular_bump(n, center, height=0.95, half_width=2, base=0.05):
    s = np.full(n, base)
    for offset in range(-half_width, half_width + 1):
        i = center + offset
        if 0 <= i < n:
            decay = (half_width - abs(offset)) / max(half_width, 1)
            s[i] = base + (height - base) * decay
    s[center] = height
    return s


def get_scenarios():
    n = 31
    anomaly_idx = 15

    labels = np.zeros(n, dtype=int)
    labels[anomaly_idx] = 1

    return {
        # Original aligned example with multiple anomalies
        "aligned": {
            "labels": np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            "scores": np.array([0.1, 0.2, 0.4, 0.9, 0.8, 0.3, 0.1, 0.2,
                                0.5, 0.7, 0.3, 0.1, 0.2, 0.1, 0.0]),
            "desc": "scores peak ON anomalies",
        },
        # Single anomaly at 15; detection peak just 2 indices away (17)
        "close_miss": {
            "labels": labels.copy(),
            "scores": make_triangular_bump(n, center=17),
            "desc": f"anomaly @ {anomaly_idx}, detection peak @ 17 (offset 2)",
        },
        # Same anomaly; detection peak far away (25 -> offset 10)
        "far_miss": {
            "labels": labels.copy(),
            "scores": make_triangular_bump(n, center=25),
            "desc": f"anomaly @ {anomaly_idx}, detection peak @ 25 (offset 10)",
        },
    }


def run_scenario(name, labels, scores, desc, agg, do_plots):
    print("\n" + "=" * 60)
    print(f"Scenario: {name}  —  {desc}")
    print("=" * 60)
    print(f"labels: {labels.tolist()}")
    print(f"scores: {[round(float(s), 2) for s in scores]}")

    label_matrix = expand_labels(labels)
    print(f"\nLabel matrix shape: {label_matrix.shape} "
          f"({label_matrix.shape[0]} buffers, the last being all-1s)")

    results = compute_per_buffer_scores(scores, label_matrix)
    print_per_buffer_table(results, n=label_matrix.shape[1], agg=agg)

    if do_plots:
        plot_label_matrix(labels, scores, label_matrix,
                          f"/tmp/vus_v2_{name}_labels.png")
        plot_pr_roc_curves(scores, label_matrix,
                           f"/tmp/vus_v2_{name}_pr_roc.png")
        plot_metric_vs_buffer(results, agg,
                              f"/tmp/vus_v2_{name}_metric_vs_buffer.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", choices=["uniform", "trapezoidal", "both"],
                        default="both",
                        help="how to aggregate per-buffer metric values")
    parser.add_argument("--scenario", default="all",
                        help="scenario name ('all' runs every dummy scenario)")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plotting (table only)")
    args = parser.parse_args()

    scenarios = get_scenarios()
    if args.scenario != "all":
        if args.scenario not in scenarios:
            raise SystemExit(f"unknown scenario {args.scenario!r}; "
                             f"choose from {list(scenarios)} or 'all'")
        scenarios = {args.scenario: scenarios[args.scenario]}

    for name, cfg in scenarios.items():
        run_scenario(name, cfg["labels"], cfg["scores"], cfg["desc"],
                     agg=args.agg, do_plots=not args.no_plots)

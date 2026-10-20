"""
Stage 2 of the Adjusted-VUS ranking study: analysis + plots.

Reads the CSV(s) written by compute_ranking_adapt.py and quantifies how much the
detector ranking changes when we move from community VUS to Adjusted VUS (aVUS),
both globally and sliced per-dataset / per-detector, so the cases where our metric
shines can be pinned down. Saves tables (CSV/MD) and figures (PDF+PNG).

Headline metric: skill_adapt_A = Adjusted VUS (aVUS). Skill columns are CLIPPED at
0 for display (stored unclipped, ARI/AMI style). Baseline = community VUS, falling
back to our non-adaptive vus_full if community is unavailable.

Usage:
  conda run -n autovus python src/analyze_ranking_adapt.py --scores experiments/ranking_adapt/scores_NAB.csv
  conda run -n autovus python src/analyze_ranking_adapt.py            # globs scores_*.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

DISPLAY = {
    'vus_community': 'VUS (community)',
    'vus_full':      'VUS (full, ours)',
    'vus_adapt':     'VUS_adapt (raw)',
    'skill_adapt_A': 'aVUS-A (adaptive)',
    'skill_adapt_B': 'aVUS-B (adaptive)',
    'skill_full_A':  'aVUS-A (full)',
    'skill_full_B':  'aVUS-B (full)',
    'ratio_adapt_A': 'ratio-A (adaptive)',
    'ratio_adapt_B': 'ratio-B (adaptive)',
}
SKILL_COLS = ['skill_adapt_A', 'skill_adapt_B', 'skill_full_A', 'skill_full_B']
HEADLINE = 'skill_adapt_A'


def load_scores(scores_arg):
    if scores_arg:
        paths = [scores_arg]
    else:
        paths = sorted(glob.glob(os.path.join('experiments', 'ranking_adapt', 'scores_*.csv')))
    if not paths:
        raise SystemExit("No score CSVs found. Run compute_ranking_adapt.py first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # drop no-anomaly rows (adaptive scoring undefined)
    df = df[df.get('note').isna()] if 'note' in df.columns else df
    df = df.dropna(subset=[HEADLINE])
    # clipped display copies of the skill columns
    for c in SKILL_COLS:
        if c in df.columns:
            df[c + '_disp'] = df[c].clip(lower=0.0)
    return df.reset_index(drop=True)


def pick_baseline(df):
    if 'vus_community' in df.columns and df['vus_community'].notna().any():
        return 'vus_community'
    return 'vus_full'


def per_series_agreement(df, baseline, candidates):
    rows = []
    for (ds, fn), g in df.groupby(['dataset', 'filename']):
        b = g[baseline].values.astype(float)
        if np.isfinite(b).sum() < 2:
            continue
        best_b_idx = np.nanargmax(np.where(np.isfinite(b), b, -np.inf))
        best_b = g.iloc[best_b_idx]['detector']
        for c in candidates:
            v = g[c].values.astype(float)
            m = np.isfinite(b) & np.isfinite(v)
            if m.sum() < 2:
                continue
            tau = kendalltau(b[m], v[m]).correlation
            rho = spearmanr(b[m], v[m]).correlation
            best_c_idx = np.nanargmax(np.where(np.isfinite(v), v, -np.inf))
            best_c = g.iloc[best_c_idx]['detector']
            rows.append({
                'dataset': ds, 'filename': fn, 'candidate': c,
                'kendall_tau': tau, 'spearman': rho,
                'top1_flip': bool(best_b != best_c),
                'best_baseline': best_b, 'best_candidate': best_c,
                'n_detectors': int(m.sum()),
            })
    return pd.DataFrame(rows)


def separation_table(df, baseline, avus_disp):
    """Per series: does aVUS separate detectors that VUS squashes together?"""
    rows = []
    for (ds, fn), g in df.groupby(['dataset', 'filename']):
        b = g[baseline].dropna()
        a = g[avus_disp].dropna()
        if len(b) < 2 or len(a) < 2:
            continue
        rows.append({
            'dataset': ds, 'filename': fn,
            'vus_spread':  float(b.max() - b.min()),
            'avus_spread': float(a.max() - a.min()),
            'vus_max':     float(b.max()),
            'best_vus':    g.loc[b.idxmax(), 'detector'],
            'best_avus':   g.loc[a.idxmax(), 'detector'],
        })
    d = pd.DataFrame(rows)
    if len(d):
        d['separation_gain'] = d['avus_spread'] - d['vus_spread']
        d = d.sort_values('separation_gain', ascending=False)
    return d


# ----------------------------- plots ---------------------------------------

def _to_markdown(df):
    """Minimal markdown table renderer (avoids the optional 'tabulate' dep)."""
    cols = list(df.columns)
    def fmt(x):
        return f'{x:.4g}' if isinstance(x, float) else str(x)
    head = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body = '\n'.join('| ' + ' | '.join(fmt(v) for v in row) + ' |'
                     for row in df.itertuples(index=False))
    return '\n'.join([head, sep, body])


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  saved {name}")


def plot_tau_violin(agg, outdir):
    cands = [c for c in agg['candidate'].unique()]
    data = [agg.loc[agg.candidate == c, 'kendall_tau'].dropna().values for c in cands]
    fig, ax = plt.subplots(figsize=(1.6 * len(cands) + 2, 5))
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(cands) + 1))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in cands], rotation=30, ha='right', fontsize=8)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_ylabel("Kendall's τ vs community VUS (per series)")
    ax.set_title('Ranking agreement with VUS (lower = more reranking)')
    _save(fig, outdir, 'tau_distribution')


def plot_dataset_heatmap(agg, outdir):
    piv = agg.pivot_table(index='dataset', columns='candidate', values='kendall_tau', aggfunc='mean')
    piv = piv[[c for c in DISPLAY if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in piv.columns], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label="mean Kendall's τ")
    ax.set_title('Mean ranking agreement with VUS, per dataset')
    _save(fig, outdir, 'dataset_heatmap')


def plot_flip_heatmap(agg, outdir):
    """Per-dataset top-1 flip rate — the punchy version of the reranking signal."""
    piv = agg.pivot_table(index='dataset', columns='candidate', values='top1_flip', aggfunc='mean')
    piv = piv[[c for c in DISPLAY if c in piv.columns]]
    piv = piv.sort_values(HEADLINE + '_disp', ascending=False)
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect='auto', cmap='OrRd', vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([DISPLAY.get(c, c) for c in piv.columns], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=7,
                        color='white' if v > 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='top-1 flip rate')
    ax.set_title('Best-detector flip rate vs VUS, per dataset')
    _save(fig, outdir, 'flip_rate_heatmap')


def plot_detector_rank_shift(df, baseline, avus_disp, outdir):
    # mean rank (1 = best) per detector under each metric, across all series
    df = df.copy()
    df['rank_vus'] = df.groupby(['dataset', 'filename'])[baseline].rank(ascending=False, method='average')
    df['rank_avus'] = df.groupby(['dataset', 'filename'])[avus_disp].rank(ascending=False, method='average')
    m = df.groupby('detector')[['rank_vus', 'rank_avus']].mean().sort_values('rank_vus')
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(m) + 2))
    y = np.arange(len(m))
    ax.hlines(y, m['rank_vus'], m['rank_avus'], color='lightgray', zorder=1)
    ax.scatter(m['rank_vus'], y, color='steelblue', label='VUS', zorder=2)
    ax.scatter(m['rank_avus'], y, color='crimson', label='aVUS', zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(m.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('mean rank (1 = best)')
    ax.legend()
    ax.set_title('Per-detector mean rank: VUS vs Adjusted VUS')
    _save(fig, outdir, 'detector_rank_shift')


def plot_bump_charts(df, per_series, baseline, avus_disp, outdir, n_panels=6, top_k=10):
    """Readable bump charts: the most-reranked series from each of the most-reranked
    datasets (one panel per dataset, diversified), showing only the top_k detectors
    (union of top_k under VUS and under aVUS), re-ranked within that subset."""
    sub = per_series[per_series.candidate == HEADLINE + '_disp'].dropna(subset=['kendall_tau'])
    if sub.empty:
        return
    # one representative (most-reranked) series per dataset, then most-reranked datasets
    picks = sub.loc[sub.groupby('dataset')['kendall_tau'].idxmin()].sort_values('kendall_tau').head(n_panels)
    ncol = 3
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, (_, srow) in enumerate(picks.iterrows()):
        ax = axes[k]
        g = df[(df.dataset == srow.dataset) & (df.filename == srow.filename)].copy()
        g = g.dropna(subset=[baseline, avus_disp])
        r_vus_full = g[baseline].rank(ascending=False, method='first')
        r_avus_full = g[avus_disp].rank(ascending=False, method='first')
        keep = set(g.loc[r_vus_full <= top_k, 'detector']) | set(g.loc[r_avus_full <= top_k, 'detector'])
        gg = g[g.detector.isin(keep)].copy()
        gg['r_vus'] = gg[baseline].rank(ascending=False, method='first')
        gg['r_avus'] = gg[avus_disp].rank(ascending=False, method='first')
        for _, dr in gg.iterrows():
            flip = dr['r_vus'] != dr['r_avus']
            ax.plot([0, 1], [dr['r_vus'], dr['r_avus']],
                    color='crimson' if flip else 'lightgray', lw=1.8 if flip else 1.0,
                    zorder=3 if flip else 1, alpha=0.95 if flip else 0.6)
            ax.text(-0.04, dr['r_vus'], dr['detector'], ha='right', va='center', fontsize=7.5)
            ax.text(1.04, dr['r_avus'], dr['detector'], ha='left', va='center', fontsize=7.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['VUS', 'aVUS'], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(-0.75, 1.75)
        ax.set_title(f'{srow.dataset}   (τ={srow.kendall_tau:.2f})', fontsize=10)
        ax.tick_params(labelleft=False)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
    for k in range(len(picks), len(axes)):
        axes[k].axis('off')
    fig.suptitle(f'Detector-ranking changes, VUS → Adjusted VUS (top {top_k} detectors per series)', fontsize=12)
    _save(fig, outdir, 'bump_top_disagreements')


def plot_saturation_case(df, baseline, avus_disp, sep, outdir):
    """The single clearest 'VUS fails, we don't' figure: for one saturated series,
    per-detector VUS (all ~equal) next to Adjusted VUS (spread out)."""
    if sep is None or len(sep) == 0:
        return
    cand = sep[sep.dataset == 'Stock']
    top = (cand.iloc[0] if len(cand) else sep.iloc[0])
    g = df[(df.dataset == top.dataset) & (df.filename == top.filename)].dropna(subset=[baseline, avus_disp]).copy()
    g = g.sort_values(avus_disp, ascending=False)
    y = np.arange(len(g))
    fig, (axv, axa) = plt.subplots(1, 2, figsize=(11, 0.32 * len(g) + 2), sharey=True)
    axv.barh(y, g[baseline], color='steelblue')
    axv.set_title(f'VUS  (spread = {top.vus_spread:.3f})', fontsize=10)
    axa.barh(y, g[avus_disp], color='crimson')
    axa.set_title(f'Adjusted VUS  (spread = {top.avus_spread:.3f})', fontsize=10)
    axv.set_yticks(y); axv.set_yticklabels(g['detector'], fontsize=7); axv.invert_yaxis()
    for ax in (axv, axa):
        ax.set_xlim(0, 1); ax.set_xlabel('score')
    fig.suptitle(f'{top.dataset} / {os.path.basename(str(top.filename))}\n'
                 'VUS rates all detectors near-equal; Adjusted VUS separates them', fontsize=10)
    _save(fig, outdir, 'saturation_case_stock')


def plot_scatter(df, baseline, avus_disp, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))
    for ds, g in df.groupby('dataset'):
        ax.scatter(g[baseline], g[avus_disp], s=8, alpha=0.4, label=ds)
    ax.set_xlabel(DISPLAY.get(baseline, baseline))
    ax.set_ylabel('Adjusted VUS (aVUS-A, adaptive)')
    ax.set_title('VUS vs Adjusted VUS, per (series, detector)')
    if df['dataset'].nunique() <= 12:
        ax.legend(fontsize=7, markerscale=2)
    _save(fig, outdir, 'scatter_vus_vs_avus')


def main():
    p = argparse.ArgumentParser(description='Stage 2: analyze Adjusted-VUS rankings')
    p.add_argument('--scores', type=str, default=None, help='Path to a scores_*.csv (default: glob all)')
    p.add_argument('--outdir', type=str, default=os.path.join('experiments', 'ranking_adapt'))
    p.add_argument('--top-n', type=int, default=6, help='Number of bump-chart panels (most-reranked datasets)')
    p.add_argument('--top-k', type=int, default=10, help='Detectors shown per bump panel')
    args = p.parse_args()

    df = load_scores(args.scores)
    baseline = pick_baseline(df)
    avus_disp = HEADLINE + '_disp'
    candidates = [c + '_disp' if c in SKILL_COLS else c
                  for c in ['skill_adapt_A', 'skill_adapt_B', 'skill_full_A',
                            'ratio_adapt_A', 'vus_adapt']]
    candidates = [c for c in candidates if c in df.columns]
    # map display names for the *_disp aliases
    for c in SKILL_COLS:
        DISPLAY[c + '_disp'] = DISPLAY[c]

    print(f"Loaded {len(df)} rows, {df['filename'].nunique()} series, "
          f"{df['detector'].nunique()} detectors, baseline={baseline}")

    agg = per_series_agreement(df, baseline, candidates)
    sep = separation_table(df, baseline, avus_disp)

    figdir = os.path.join(args.outdir, 'figures')
    tabdir = os.path.join(args.outdir, 'tables')
    os.makedirs(tabdir, exist_ok=True)

    # ---- summary tables ----
    overall = (agg.groupby('candidate')
               .agg(mean_tau=('kendall_tau', 'mean'),
                    median_tau=('kendall_tau', 'median'),
                    top1_flip_rate=('top1_flip', 'mean'),
                    n_series=('kendall_tau', 'size'))
               .reindex(candidates))
    overall.to_csv(os.path.join(tabdir, 'summary_overall.csv'))
    agg.groupby(['dataset', 'candidate']).agg(
        mean_tau=('kendall_tau', 'mean'),
        top1_flip_rate=('top1_flip', 'mean'),
        n_series=('kendall_tau', 'size')).to_csv(os.path.join(tabdir, 'summary_per_dataset.csv'))
    agg.to_csv(os.path.join(tabdir, 'per_series_agreement.csv'), index=False)
    sep.to_csv(os.path.join(tabdir, 'separation.csv'), index=False)

    # case studies: low VUS-spread (detectors look equal) but high aVUS-spread
    cs = sep.copy()
    if len(cs):
        cs = cs.sort_values('separation_gain', ascending=False).head(30)
        with open(os.path.join(tabdir, 'case_studies.md'), 'w') as f:
            f.write("# Case studies — where Adjusted VUS separates what VUS squashes\n\n")
            f.write("Series with the largest (aVUS spread − VUS spread): VUS rates all "
                    "detectors near-equal, aVUS pulls them apart.\n\n")
            f.write(_to_markdown(cs))
    print("\n=== Overall (per-series ranking agreement vs VUS) ===")
    print(overall.round(3).to_string())

    # ---- figures ----
    print("\nFigures:")
    plot_tau_violin(agg, figdir)
    if df['dataset'].nunique() > 1:
        plot_dataset_heatmap(agg, figdir)
        plot_flip_heatmap(agg, figdir)
    plot_detector_rank_shift(df, baseline, avus_disp, figdir)
    plot_bump_charts(df, agg, baseline, avus_disp, figdir, n_panels=args.top_n, top_k=args.top_k)
    plot_saturation_case(df, baseline, avus_disp, sep, figdir)
    plot_scatter(df, baseline, avus_disp, figdir)

    print(f"\nTables -> {os.path.abspath(tabdir)}\nFigures -> {os.path.abspath(figdir)}")


if __name__ == '__main__':
    main()

"""Fig 8 (practicality): (a) evaluation vs detector inference time on TSB-AD (notebook cell 86),
(b) deep-learning pipeline time breakdown (notebook cell 64). Shorter panels for the 12-page cut.
Data: experiments/all_metrics_tsb_def_params/results, experiments/compare_train_metrics/sylli_logs_29012026."""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from tsad_parser import TSADParser

experiments_path = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments')
save_fig_path = os.path.join(experiments_path, 'figures')
my_vus_name = 'FastVUS'
color_palette = {
    f'{my_vus_name} (L: 128)': '#c8b400', f'{my_vus_name}-GPU (L: 128)': '#40da70',
    f'{my_vus_name}': '#c8b400', f'{my_vus_name}-GPU': '#40da70', 'AUC': '#f1a73f',
    'VUS (L: 128)': '#b02a41', 'Affiliation': '#A5D1C2', 'Range-AUC (L: 128)': '#547C6E', 'RF': '#123327',
}
formal_names = {
    'FF-VUS': f'{my_vus_name} (L: 128)', 'FF-VUS-GPU': f'{my_vus_name}-GPU (L: 128)', 'AUC': 'AUC',
    'VUS': 'VUS (L: 128)', 'RF': 'RF', 'AFFILIATION': 'Affiliation', 'RANGE-AUC': 'Range-AUC (L: 128)',
    'auc': 'AUC', 'vus': 'VUS (L: 128)', 'ffvus': f'{my_vus_name} (L: 128)', 'ffvusgpu': f'{my_vus_name}-GPU (L: 128)',
}
sns.set_style("whitegrid")


def finalize_plot(save_dir, filename):
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', pad_inches=0, dpi=250)
    print("saved", save_path + ".pdf")


def load_results(path):
    df = pd.DataFrame()
    for file in [f for f in os.listdir(path) if f.endswith('.csv')]:
        curr_df = pd.read_csv(os.path.join(path, file), index_col=0).reset_index()
        df = pd.concat([df, curr_df], ignore_index=True)
    return df


# ---------------- (a) scatter: notebook cells 77-79 + 86 (figsize 2.7 x 3.1 -> 2.7 x 2.4) ----------------
df = load_results(os.path.join(experiments_path, 'all_metrics_tsb_def_params', 'results'))
df['Metric'] = df['Metric'].map(formal_names)
vus_metrics = ['VUS (L: 128)', f'{my_vus_name} (L: 128)', f'{my_vus_name}-GPU (L: 128)']
vus_times = (df[df['Metric'].isin(vus_metrics)]
             .pivot_table(index=['Time series', 'Detector'], columns='Metric', values='Metric time')
             .reset_index()[['Time series', 'Detector'] + vus_metrics])
vus_times.columns.name = None
detector_times = (TSADParser().load_runtime()[['File Name', 'Model Name', 'Time Cost']]
                  .rename(columns={'File Name': 'Time series', 'Model Name': 'Detector', 'Time Cost': 'Detector time'}))
detector_times = detector_times[detector_times['Detector time'] > 0]
merged = vus_times.merge(detector_times, on=['Time series', 'Detector'], how='inner')
print("pairs:", len(merged))

DET, VUS, CPU, GPU = 'Detector time', 'VUS (L: 128)', f'{my_vus_name} (L: 128)', f'{my_vus_name}-GPU (L: 128)'
clouds = [(VUS, 'VUS'), (CPU, my_vus_name), (GPU, f'{my_vus_name}-GPU')]

fig, ax = plt.subplots(figsize=(2.7, 2.4))
vals = np.concatenate([merged[DET].values] + [merged[c].values for c, _ in clouds])
lo, hi = vals.min() * 0.6, vals.max() * 1.6
lims = [lo, hi]
ax.fill_between(lims, lims, hi, color=color_palette[VUS], alpha=0.05, lw=0, zorder=0)
ax.plot(lims, lims, ls='--', lw=1.0, color='#4d4d4d', zorder=1)
handles = []
for col, label in clouds:
    c = color_palette[col]
    ax.scatter(merged[DET], merged[col], s=5, alpha=0.16, color=c, edgecolors='none', rasterized=True, zorder=2)
    frac = (merged[col] >= merged[DET] / 10).mean() * 100
    print(f"{label}: >= detector/10 in {frac:.0f}% of runs")
    handles.append(plt.Line2D([], [], marker='o', ls='', ms=6, mec='white', mew=0.8, color=c, label=f'{label}  {frac:.0f}%'))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Detector time (secs; log)', fontsize=8)
ax.set_ylabel('Evaluation time (secs; log)', fontsize=8)
ax.tick_params(labelsize=7)
fig.canvas.draw()
p0 = ax.transData.transform((lo, lo)); p1 = ax.transData.transform((hi, hi))
diag_angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
gm = 10 ** ((np.log10(lo) + np.log10(hi)) / 2)
ax.text(gm * 150, gm * 100, 'evaluation = inference', rotation=diag_angle + 2.5, rotation_mode='anchor',
        ha='center', va='top', fontsize=7, color='#000000', fontweight='bold',
        path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9, fontsize=6.5,
          title='$\\gtrsim$ detector', title_fontsize=7, handletextpad=0.4, labelspacing=0.3, borderpad=0.3)
finalize_plot(save_fig_path, 'runtime_vs_detector_tsb_scatter_small')

# ---------------- (b) deep-learning pipeline: notebook cells 60, 71, 64 (figsize 2.62 x 4.1 -> 2.62 x 3.2) ----------------
curr_experiment_path = os.path.join(experiments_path, 'compare_train_metrics', 'sylli_logs_29012026')
tdf = pd.DataFrame()
for file in [f for f in os.listdir(curr_experiment_path) if f.endswith('.csv')]:
    curr_df = pd.read_csv(os.path.join(curr_experiment_path, file), index_col=0).reset_index()
    curr_df = curr_df.rename(columns={'epoch': 'Epoch', 'train_time': 'Train time', 'val_time': 'Val time',
                                      'total_time': 'Total time', 'train_pct': 'Train %', 'val_pct': 'Val %'})
    curr_df['Metric'] = file.split('_')[3]
    tdf = pd.concat([tdf, curr_df], ignore_index=True)
tdf['Metric'] = tdf['Metric'].map(formal_names)
tdf = tdf[tdf['Epoch'] != 'TOTAL']
grouped = tdf.groupby('Metric')[['Train time', 'Val time', 'Total time']].sum().reset_index()
metric_order = ['VUS (L: 128)', f'{my_vus_name} (L: 128)', f'{my_vus_name}-GPU (L: 128)', 'AUC']
grouped_sorted = grouped.set_index('Metric').loc[metric_order].reset_index()
print(grouped_sorted)

fig, ax = plt.subplots(1, 1, figsize=(5.5 / 2.1, 3.2))
train_times = grouped_sorted['Train time'].values
val_times = grouped_sorted['Val time'].values
x_pos = np.arange(len(grouped_sorted))
ax.bar(x_pos, train_times, color=[color_palette[m] for m in grouped_sorted['Metric']], edgecolor='black', label='Training (bottom part)')
ax.bar(x_pos, val_times, bottom=train_times, color='none', edgecolor=[color_palette[m] for m in grouped_sorted['Metric']],
       hatch='///', linewidth=1, label='Validation (top part)')
ax.set_xticks(x_pos)
ax.set_xticklabels([x.replace(' (L: 128)', '') for x in metric_order], rotation=30)
ax.set_yscale('log')
ax.set_ylabel('Total time (secs)')
ax.set_ylim(3500, max(grouped_sorted['Total time']) * 1.3)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
legend_handles = [Patch(facecolor='none', edgecolor='black', hatch='///', linewidth=1, label='Validation'),
                  Patch(facecolor='gray', edgecolor='black', label='Training')]
ax.legend(handles=legend_handles, loc='upper right', framealpha=0.95, fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for i, (val, total) in enumerate(zip(val_times, grouped_sorted['Total time'])):
    if val >= 3600:
        ax.text(i, total + 3, f'{int(val // 3600)}h{int((val % 3600) // 60)}m', ha='center', va='bottom')
    elif val >= 60:
        ax.text(i, total + 3, f'{int(val // 60)}m{int(val % 60)}s', ha='center', va='bottom')
    else:
        ax.text(i, total + 3, f'{val:.1f}s', ha='center', va='bottom')
for i, (train, val, total) in enumerate(zip(train_times, val_times, grouped_sorted['Total time'])):
    val_pct = (val / total) * 100
    ax.text(i, 4000, '1.3h', ha='center', va='center', weight='bold', color='white')
    if val_pct > 8:
        ax.text(i, train + val / 2, f'{val_pct:.0f}%', ha='center', va='center', weight='bold')
finalize_plot(save_fig_path, 'toy_example_time_breakdown_small')

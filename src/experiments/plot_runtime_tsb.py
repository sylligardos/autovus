"""Fig 1(a): the notebook's Fig 6 (exec_time_comparison_tsb, cell 13 of experimental_results.ipynb) with the
"Nx faster" arrows of the old Fig 1(a) (exec_time_comparison_tsb_small, cell 14). Same code, same style.
Max-error annotations dropped (now in Fig 7a).

Input: experiments/all_metrics_tsb_def_params/results/*.csv. Output: experiments/figures/exec_time_comparison_tsb_arrows.pdf/.png
Usage: conda run -n autovus python src/experiments/plot_runtime_tsb.py
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

experiments_path = 'experiments'
save_fig_path = os.path.join(experiments_path, 'figures')
os.makedirs(save_fig_path, exist_ok=True)

# --- notebook cells 3, 4, 6, 8 (verbatim) ---------------------------------------------------
my_vus_name = 'FastVUS'
color_palette = {
    f'{my_vus_name} (L: 128)': '#c8b400', f'{my_vus_name}-GPU (L: 128)': '#40da70',
    f'{my_vus_name}': '#c8b400', f'{my_vus_name}-GPU': '#40da70',
    'AUC': '#f1a73f', 'VUS (L: 128)': '#b02a41', 'Affiliation': '#A5D1C2',
    'Range-AUC (L: 128)': '#547C6E', 'RF': '#123327',
}
formal_names = {
    'FF-VUS': f'{my_vus_name} (L: 128)', 'FF-VUS-GPU': f'{my_vus_name}-GPU (L: 128)', 'AUC': 'AUC',
    'VUS': 'VUS (L: 128)', 'RF': 'RF', 'AFFILIATION': 'Affiliation', 'RANGE-AUC': 'Range-AUC (L: 128)',
}
sns.set_style("whitegrid")


def finalize_plot(save_dir, filename):
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', pad_inches=0, dpi=250)


def load_results(path):
    df = pd.DataFrame()
    for file in [f for f in os.listdir(path) if f.endswith('.csv')]:
        curr_df = pd.read_csv(os.path.join(path, file), index_col=0).reset_index()
        df = pd.concat([df, curr_df], ignore_index=True)
    return df


df = load_results(os.path.join(experiments_path, 'all_metrics_tsb_def_params', 'results'))
df['Metric'] = df['Metric'].map(formal_names)

metric_to_compare = 'VUS (L: 128)'
speed_up_dict = {}
df_vus = df[df['Metric'] == metric_to_compare].set_index('Time series')
for metric in df['Metric'].unique():
    curr_df = df[df['Metric'] == metric].set_index('Time series')
    common_index = df_vus.index.intersection(curr_df.index)
    speed_up_dict[metric] = (df_vus.loc[common_index, 'Metric time'] / curr_df.loc[common_index, 'Metric time']).mean()
    print(f"{metric}: {speed_up_dict[metric]:.2f}x speed up")

# --- notebook cell 13 (Fig 6), without the max-error annotations --------------------------
median_times = df.groupby('Metric')['Metric time'].median()
order = median_times.sort_values().index.to_list()

fig = plt.figure(figsize=(5.1, 3.3))
medianprops = dict(linestyle='-', linewidth=2, color='white')
axis = sns.boxplot(
    y='Metric time', x='Metric', hue='Metric', data=df, showfliers=False, log_scale=True,
    showmeans=False, medianprops=medianprops, palette=color_palette, order=order, saturation=1,
)
axis.set_ylabel('Runtime per series (secs; log)')
axis.set_xlabel('')
axis.tick_params(axis='x', rotation=25)
axis.grid(visible=False, axis='x')
axis.hlines(y=10.5, xmin=-1, xmax=7, linestyle='--', lw=1, color='k')
axis.set_xlim(-0.5, 6.5)
axis.set_ylim(0, 80)
axis.text(2.5, 40, 'Total time:')

for i, metric in enumerate(order):
    total = df.loc[df['Metric'] == metric].groupby(['Time series', 'Detector'])['Metric time'].mean().sum()
    if total > 3600:
        h, m = divmod(int(total) // 60, 60); label = f'{h}h {m}m'
    elif total > 100:
        m, s = divmod(int(total), 60); label = f'{m}m {s}s'
    else:
        label = f'{total:.0f}s'
    axis.text(i, 25, label, ha='center', va='top', fontsize='small')
    print(f"{metric}: total time {label}")

# --- notebook cell 14 (old Fig 1a): the "Nx faster" arrows, VUS box now at position 6 -----
i_vus = order.index(metric_to_compare)
for i, metric in enumerate(order):
    if my_vus_name in metric:
        axis.annotate(
            f'{speed_up_dict[metric]:.0f}$\\times$\nfaster',
            xy=(i_vus, 1.5),
            xytext=(i - 0.35, 0.1) if 'GPU' in metric else (i - 0.25, .8),
            arrowprops=dict(arrowstyle="<-", connectionstyle=f"arc3,rad={-.42 if 'GPU' in metric else -.2}", color='k', alpha=0.8),
            va='top', ha='left',
        )

xticklabels = axis.get_xticklabels()
for idx in range(len(xticklabels)):
    if my_vus_name in xticklabels[idx].get_text():
        xticklabels[idx].set_fontweight('bold')
    if '(L:' in xticklabels[idx].get_text():
        xticklabels[idx].set_text(xticklabels[idx].get_text().replace(' (L: 128)', ''))
axis.set_xticks(axis.get_xticks())
axis.set_xticklabels([label.get_text() for label in xticklabels])

axis.set_title('L1: VUS is computationally expensive', fontweight='bold', fontsize=10)
axis.text(0.985, 0.03, '(a)', transform=axis.transAxes, ha='right', va='bottom', fontsize='medium', fontweight='bold')

finalize_plot(save_fig_path, 'exec_time_comparison_tsb_arrows')

# --- layout of the saved (tight-cropped) panel, so Fig 1(b) can align its frame to it -----
import json
_fig = plt.gcf(); _r = _fig.canvas.get_renderer()
_tb = _fig.get_tightbbox(_r)                                  # inches
_ab = axis.get_window_extent(_r).transformed(_fig.dpi_scale_trans.inverted())
_tt = axis.title.get_window_extent(_r).transformed(_fig.dpi_scale_trans.inverted())
layout = dict(width=_tb.width, ax_left=_ab.x0 - _tb.x0, ax_right=_tb.x1 - _ab.x1,
              title_gap=_tt.y0 - _ab.y1, font_size=plt.rcParams['font.size'],
              edge_color=list(axis.spines['left'].get_edgecolor()), edge_width=axis.spines['left'].get_linewidth())
json.dump(layout, open(os.path.join(save_fig_path, 'fig1a_layout.json'), 'w'), indent=1)
print('layout (inches):', layout)
# re-save with the exact box measured above, so the width matches the layout Fig 1(b) is drawn to
_stem = os.path.join(save_fig_path, 'exec_time_comparison_tsb_arrows')
_fig.savefig(f"{_stem}.pdf", bbox_inches=_tb); _fig.savefig(f"{_stem}.png", bbox_inches=_tb, dpi=250)

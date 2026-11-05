"""Fig 7: runtime versus series length, number of anomalies and average anomaly length on synthetic data.
Notebook cells 19, 21 and 23 of experimental_results.ipynb, verbatim, with a shorter figure (3.2 in instead of 3.8 in).

Input: experiments/allmetrics_defparams_syn/results/*.csv (moved here from the archive, 2026-11-05).
Output: experiments/figures/exec_time_per_ts_char_syn.pdf/.png
Usage: conda run -n autovus python src/experiments/plot_exec_time_syn.py
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

experiments_path = 'experiments'
save_fig_path = os.path.join(experiments_path, 'figures')
os.makedirs(save_fig_path, exist_ok=True)

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
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave room for the figure legend above the axes
    save_path = os.path.join(save_dir, filename)
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', pad_inches=0, dpi=250)


def load_results(path):
    df = pd.DataFrame()
    for file in [f for f in os.listdir(path) if f.endswith('.csv')]:
        curr_df = pd.read_csv(os.path.join(path, file), index_col=0).reset_index()
        df = pd.concat([df, curr_df], ignore_index=True)
    return df


# cell 19
df = load_results(os.path.join(experiments_path, 'allmetrics_defparams_syn', 'results'))
df['Metric'] = df['Metric'].map(formal_names)

# cell 21
length_analysis_df = df[df['Length'] != 100_000]
n_anomalies_analysis_df = df[np.logical_and(df['Time series'].str.contains('len_100000_'), df['Time series'].str.contains('_avglen_10.npz'))].copy()
avg_anomaly_length_analysis_df = df[df['Time series'].str.contains('len_100000_n_10_')].copy()
avg_anomaly_length_analysis_df.loc[:, 'Anomalies average length'] = avg_anomaly_length_analysis_df.apply(lambda row: int(row['Time series'].split('_')[-1].replace('.npz', '')), axis=1)
n_anomalies_analysis_df.loc[:, 'Number of anomalies'] = n_anomalies_analysis_df.apply(lambda row: int(row['Time series'].split('_')[-3]), axis=1)
analysis_dfs = [length_analysis_df, n_anomalies_analysis_df, avg_anomaly_length_analysis_df]
attributes = ["Length", "Number of anomalies", "Anomalies average length"]

# cell 23 (figsize 5 x 3.8 -> 5 x 3.2)
fig = plt.figure(figsize=(5, 2.7))
gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 2], wspace=0.25, hspace=0.6)
attr_axis = {"Length": gs[0, :], "Number of anomalies": gs[1, 0], "Anomalies average length": gs[1, 1]}
abc_numbered = {0: 'a', 1: 'b', 2: 'c'}

legend_axis = None
for i, attribute in enumerate(attributes):
    axis = fig.add_subplot(attr_axis[attribute])
    curr_df = analysis_dfs[i]
    sns.lineplot(x=attribute, y='Metric time', hue='Metric', data=curr_df, ax=axis, palette=color_palette,
                 markers=True, style='Metric', dashes=False, linewidth=2, zorder=1, estimator=np.median)
    axis.set_yscale('log')
    axis.grid(axis='both', alpha=0.5)
    axis.set_xlabel(f"({abc_numbered[i]}) {attribute if attribute != 'Length' else 'Time series length'}")
    axis.set_ylabel('')
    if attribute == "Length":
        axis.set_xscale('log', base=10)
    if i != 0:
        axis.get_legend().remove()
        if i == 1:
            axis.set_yticks([0.001, 0.01, 0.1, 1, 10])
    else:
        legend_axis = axis
        axis.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
        curr_color = "#383838"
        for x_val, y_val, txt in [(1e6, 1000, "1 hour of ECG"), (1e8, 0.1, "5 days of ECG"), (2**31, 0.004, "3 months of ECG\n(2 billion points)")]:
            axis.annotate(txt, xy=(x_val, y_val), xytext=(x_val / 2, y_val), ha='right', va='center', fontsize='small',
                          color=curr_color, arrowprops=dict(arrowstyle="->", color=curr_color, alpha=0.5))
            axis.axvline(x=x_val, ymin=0, ymax=20000, color=curr_color, linestyle='--', alpha=0.3)

fig.text(0, 0.5, "Runtime (secs; log scale)", va='center', rotation='vertical')
if legend_axis:
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_axis.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize='x-small')
finalize_plot(save_fig_path, 'exec_time_per_ts_char_syn')

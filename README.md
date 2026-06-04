## Ideas from Mingyi to review

Mingyi Huang  [4:21 PM]
I don’t think there are any remaining unsolved issues, at least from what I remember.
There may still be some other issues that are hard to avoid. Since you are trying to make the computation more efficient, one thing you might want to consider is the thresholding strategy. In the current version, VUS uses 300 thresholds by default to compute the R-AUC ROC/PR values. This kind of sampling can make the curve uneven. This is especially problematic when the anomaly contains only one time point, the highest anomaly score is slightly shifted away from it, and the time series is very long. In such cases, threshold sampling may sometimes severely underestimate the accuracy, for example from 1.0 to 0.01.
Using all possible thresholds would be more accurate, but for computational efficiency, the current strategy is also understandable and somewhat unavoidable. Maybe you could think about whether there is a better way to handle this trade-off.Mingyi Huang  [4:26 PM]
Also, algorithms like this may lead us to overestimate the capability of many detectors. For example, if you look at some STOCK time series in TSB-AD, the anomalies are extremely dense. After adding a buffer area around each individual anomaly, these buffered regions can end up covering almost the entire time series. As a result, the whole time series effectively receives nonzero labels, and anomaly scores of almost any shape can still obtain a reasonably high score.

# FF-VUS

FF-VUS provides efficient implementations of the Volume Under the Precision-Recall Surface (VUS-PR) for time series anomaly detection. This repository accompanies the paper "FF-VUS: A Freaking-Fast Evaluation Measure for Time Series Anomaly Detection" and contains the exact CPU algorithm (FF-VUS) and a fully vectorized GPU approximation (FF-VUS-GPU).

## Summary

VUS-PR is a threshold-independent metric that evaluates detection quality across slope tolerances, producing a 3D precision–recall surface whose volume summarizes performance. FF-VUS removes computational bottlenecks through algorithmic optimizations and a GPU-parallel approximation, enabling practical evaluation at large scale.

## Key contributions

- FF-VUS: an exact, CPU-based algorithm that avoids redundant computations.
- FF-VUS-GPU: a vectorized approximation exploiting GPU parallelism for massive speedups.
- Adaptive buffer selection: analytical denominator (E[AUC-PR] = g_b) used to build a ratio curve; best buffer selected via argmax, elbow, or first-peak on the ratio.
- Extensive experiments on the TSB-AD benchmark and synthetic series demonstrating orders-of-magnitude speed improvements.

Representative results (paper)

- Average speed-up of FF-VUS over baseline VUS: ~111×
- Average speed-up of FF-VUS-GPU over baseline VUS: ~467×
- Extreme scenario: benchmark evaluation reduced from ~61 hours to ~2 minutes (>1800× improvement)

## Repository contents

- src/: metric implementations, experiment scripts, visualisation tools
  - src/vus/vus_numpy.py — FF-VUS (CPU) + adaptive buffer selection
  - src/vus/vus_torch.py — FF-VUS-GPU (vectorized, optional)
  - src/generate_synthetic.py — synthetic data & score generation
  - src/test_adaptive_buffer.py — adaptive buffer tests on synthetic label configs
  - src/test_adaptive_buffer_real.py — adaptive buffer visualisation on TSB-AD series
  - src/notebooks/: analysis and visualisation notebooks used in the paper
- experiments/: generated results and figures

## Requirements & installation

Python 3.12+, with (minimum):

- numpy, scipy, tqdm
- matplotlib, seaborn (optional, for plotting)
- torch (optional, for GPU acceleration)

Install:

```bash
pip install -r requirements.txt
```

## Minimal usage

Compute VUS (CPU):

```python
from src.vus.vus_numpy import VUSNumpy
vus = VUSNumpy(
  slope_size=128                    # The buffer length that we add on the left and right of the label
  step=1,                           # The step for which we repeat the buffers, e.g. slope_size = 100 and step = 1 results to 100 buffers but if step = 5 it results to 20 buffers
  zita=(1/math.sqrt(2)),            # The height on the y-axis at which the buffer reaches, the higher this value, the higher the reward
  global_mask=True,                 # Optimization option which is very helpful in very long time series
  slopes='precomputed',             # The way we compute the buffers, please leave this to the default option for every version, rest of the modes were used for cross evaluation while designing
  existence='optimized',            # The way we compute the existence reward, same as for 'slopes'
  conf_matrix='dynamic_plus',       # The optimization level for the matrix, same as for 'slopes'
  interpolation='stepwise',         # We have 2 different modes for interpolating the final surface, although the default is the one we should mainly use
  metric='vus_pr',                  # This option is not implemented yet and it won't affect anything, in the feature we may compute more measures
  side='both',                      # The side to which we would like to add the buffers, e.g. for streaming applications, it only makes sense to add the buffers after the event not before
)
value, timing = vus.compute(label, score)
```

Compute VUS (GPU, optional):

```python
from src.vus.vus_torch import VUSTorch
vus = VUSTorch(slope_size=128, device='cuda')
value, timing = vus.compute(label, score)
```

## Experiments

Notebooks under src/notebooks reproduce figures and runtime analyses reported in the paper. Results and plotting scripts save outputs to experiments/figures/.

## Interactive demo

`src/demo_vus.py` is a Streamlit app that lets you explore every aspect of FF-VUS interactively. It is the companion to the paper and intended to sit alongside it.

### Prerequisites

The demo requires two precomputed data files, both produced by a single script:

```bash
conda run -n ffvus python src/precompute_demo.py --L 1024
```

| Output file | Contents |
|---|---|
| `experiments/precomputed/rankings_L1024.npz` | cumVUS curves, analytical denominator, b\* values (argmax/elbow/first\_peak) for every (series, real detector, buffer) triple, plus 11 synthetic detectors |
| `experiments/precomputed/existence_maps.npz` | Per-anomaly existence curves for every (series, detector) pair |

Options: `--skip rankings`, `--skip existence`, `--force` (re-run even if output exists), `--backend torch` (GPU, requires CUDA).

### Launch

```bash
conda run -n ffvus streamlit run src/demo_vus.py
```

### Tabs

**Explorer** — Single (series, detector) view. Shows the raw time series + score preview and the cumVUS curve as a function of buffer size b, with gradient and key metrics (AUC-PR at b=0, VUS at b=L, Δ, peak slope). Requires `rankings_L1024.npz`.

**Impact Analysis** — How much does the choice of buffer size affect VUS-PR? Aggregates Δ (VUS − AUC-PR) and AUC-above-baseline across series and detectors, at any scope (global / dataset / single series).

**Ranking Instability** — Bump chart of detector rankings as b varies from 0 to L. Highlights rank crossings among the top-k pool, per-dataset instability summary, and a tool to find the most unstable individual series.

**Localization Map** — Per-timestep confusion matrix heatmap stacked across detectors. Color each timestep TP/FN/FP/TN relative to the soft buffer label at the chosen b. Useful for spotting which detector covers which anomaly region.

**Existence Map** — Per-anomaly existence score: for a given (series, detector), how often is each labeled anomaly detected across all thresholds and buffers 0→b? Identifies hard-to-detect anomalies globally. Requires `existence_maps.npz`.

**Adaptive Buffer** — The main claim of the paper, visualized. For each selected detector (real or synthetic), shows the expanded binary label at b\* alongside the original ground truth (red = anomaly, blue = buffer zone). Controls: b\* selection method (argmax / elbow / first\_peak) and a detector multiselect covering all real detectors plus 11 synthetic detectors (perfect, const\_0, const\_1, rand\_bernoulli, rand\_uniform, C1, C2, E1, E3, E4, E5). Convergence panel at the bottom plots `mean_anom_len + b*` per detector — values should cluster near `mean_anom_len` regardless of detector, which is the key convergence property. Requires `rankings_L1024.npz`.

**Notes** — Persistent annotation layer. Save observations tied to a (series, detector, buffer) triple via the sidebar, filter and review them here.

### Sidebar controls

The global sidebar (Dataset → Series → Buffer size b) drives the Explorer, Impact Analysis, Ranking Instability, Localization Map, and Existence Map tabs. The Adaptive Buffer tab uses only Dataset/Series (not the buffer slider) because b is determined per-detector by b\*.

### Synthetic detectors (Adaptive Buffer tab)

The 11 synthetic detectors are designed to test specific behaviors of b\* selection:

| Name | Description | Expected b\* |
|---|---|---|
| perfect | score = label | ≈ 0 |
| const\_0 | always 0 | ≈ 0 |
| const\_1 | always 1 | ≈ 0 |
| rand\_bernoulli | Bernoulli(p = label.mean()) | ≈ 0 |
| rand\_uniform | Uniform(0, 1) | ≈ 0 |
| C1 | fires in the left buffer zone (early detection) | > 0 |
| C2 | fires in the right buffer zone (late detection) | > 0 |
| E1 | isolated false positive far from any anomaly | ≈ 0 |
| E3 | detects only the first half of each anomaly | small |
| E4 | early over-prediction (extends before left boundary) | large |
| E5 | late over-prediction (extends after right boundary) | large |

## Citation

If you use this work, please cite the paper: "FF-VUS: A Freaking-Fast Evaluation Measure for Time Series Anomaly Detection." (under review)

## Authors & License

Authors: Anonymous  
License: MIT

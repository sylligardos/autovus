# AutoVUS

Fast and adaptive VUS-PR for time series anomaly detection.

- **FastVUS**: the engine. Exact CPU implementation (`src/vus/vus_numpy.py`) and a vectorized GPU approximation (`src/vus/vus_torch.py`).
- **AutoVUS**: adaptive buffer selection per anomaly, built on top of FastVUS.

Target: ICDE 2027 (research track, Nov 11 2026; demo track, Nov 23 2026).
Work in progress; `TRACKER.md` says what is done and what is left.

## Setup

```bash
conda env create -f environment.yml -n autovus
```

Benchmarks are read from `~/Documents/tsad-benchmarks` through the `tsad-parser` package (installed editable in the env).

## Usage

```python
from src.vus.vus_numpy import VUSNumpy
(value, timing), elapsed = VUSNumpy(slope_size=128).compute(label, score)
```

GPU:

```python
from src.vus.vus_torch import VUSTorch
vus = VUSTorch(slope_size=128, device='cuda')
```

Demo:

```bash
conda run -n autovus streamlit run src/demo/demo_vus.py
```

## Layout

- `src/vus/`: the metric (FastVUS CPU and GPU, AutoVUS).
- `src/baselines/`: upstream TSB-AD metrics (vendored, untouched) and PATE, behind one `compute()` function.
- `src/experiments/`: experiment and test scripts. Run them from the repo root.
- `src/demo/`: the Streamlit demo.
- `src/notebooks/`: analysis notebooks.
- `scripts/`: CLEPS cluster job scripts (cluster env is still called `ffvus`).
- `references/`: paper plan, feedback, past submissions. The paper itself lives in Overleaf.
- `experiments/`: outputs (not in git). Old results are in `~/Documents/autovus-archive/`.

## License

MIT

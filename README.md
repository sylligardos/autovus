# AutoVUS

Fast and adaptive VUS-PR for time series anomaly detection.

- **FastVUS**: the engine. Exact CPU implementation (`src/autovus/fastvus.py`) and a vectorized GPU approximation (`src/autovus/fastvus_gpu.py`).
- **AutoVUS**: adaptive buffer selection per anomaly, built on top of FastVUS.

Code for the ICDE 2027 submission "FastVUS and AutoVUS".

## Setup

```bash
conda env create -f environment.yml -n autovus
```

Benchmarks are read from `$TSAD_BENCHMARKS` (default `~/Documents/tsad-benchmarks`) in the TSB-AD layout, through the vendored reader `src/tsad_parser/`.

## Usage

```python
import sys; sys.path.insert(0, 'src')
from autovus import FastVUS, FastVUSGPU, AutoVUS

value = FastVUS(L=128).compute(label, score)            # exact, CPU
value = FastVUSGPU(L=128).compute(label, score)         # GPU, same interface
result = AutoVUS(FastVUS(L=128, cumulative=True)).compute(label, score)   # adaptive buffers
```

Checks (toy data, no benchmark files): `src/experiments/check_fastvus.py`, `check_fastvus_gpu.py`, `check_autovus.py`, `check_speed.py`.

Demo:

```bash
# see app/README.md: FastAPI (app/api) + Next.js (app/web)
```

## Layout

- `src/autovus/`: the metric (FastVUS CPU and GPU, AutoVUS).
- `src/baselines/`: upstream TSB-AD metrics (vendored, untouched) and PATE, behind one `compute()` function.
- `src/experiments/`: experiment and check scripts. Run them from the repo root.
- `app/`: the demo (FastAPI + Next.js) and its export.
- `src/notebooks/`: analysis notebooks.
- `docs/supplementary/`: results left out of the paper for space reasons.
- `experiments/`: outputs (not in git).

## License

MIT

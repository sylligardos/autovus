# AutoVUS

Fast and adaptive VUS-PR for time series anomaly detection.

- **FastVUS**: the engine. Exact CPU implementation (`src/autovus/fastvus.py`) and a vectorized GPU approximation (`src/autovus/fastvus_gpu.py`).
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
- `src/experiments/`: experiment and check scripts. Run them from the repo root. The experiment scripts still use the old classes and are ported one by one when rerun.
- `app/`: the demo (FastAPI + Next.js) and its export.
- `src/notebooks/`: analysis notebooks.
- `scripts/`: CLEPS cluster job scripts (cluster env is still called `ffvus`).
- `references/`: paper plan, feedback, past submissions. The paper itself lives in Overleaf.
- `docs/supplementary/`: results left out of the paper for space reasons.
- `experiments/`: outputs (not in git). Old results are in `~/Documents/autovus-archive/`.

## License

MIT

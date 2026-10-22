"""Direct, single-process reads of a benchmark through the parser's file helpers.

The parser's load_timeseries / load_scores open a process pool on every call,
which is slow for one file and forbidden inside a worker process. These read
one file at a time with the parser's own per-file loaders.
"""
import numpy as np
from tsad_parser import TSADParser


def load_series(p: TSADParser, file: str):
    """values (first channel, float), label (int) of one series."""
    values, label = p._load_single_csv(str(p.raw_path / file))
    values = np.asarray(values, dtype=float)
    values = values[:, 0] if values.ndim > 1 else values.ravel()
    return values, np.asarray(label).astype(int).ravel()


def load_score(p: TSADParser, file: str, detector: str):
    """One detector's score on one series, or None when the file is missing or holds NaN."""
    s = p._load_single_npy(str(p.scores_path / detector / file.replace('.csv', '.npy')))
    if s is None:
        return None
    s = np.asarray(s, dtype=float).ravel()
    return None if np.any(np.isnan(s)) else s

"""Shared setup for the fading-precision experiments: thread caps, import paths, data folder."""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent.parent
DATA = HERE / "data"          # big intermediate files (deleted by run.py at the end)
RESULTS = HERE / "results"    # small JSON / text results that the README quotes
for p in (str(PROTO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def save_json(name, obj):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def median_time(fn, repeats=3):
    """Median wall time of fn() over `repeats` runs, plus the last return value."""
    ts, out = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out

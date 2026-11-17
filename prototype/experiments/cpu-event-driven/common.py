"""Shared setup for the cpu-event-driven experiments: thread caps, import path, timing, memory."""

import os
import sys
import time

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
# OpenMP worker threads (the C kernels) must not spin after a parallel region: the machine is shared.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", ".."))
if PROTO not in sys.path:
    sys.path.insert(0, PROTO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

MAX_THREADS = 3


def set_threads(k: int):
    import numba
    numba.set_num_threads(min(k, MAX_THREADS))


def _status(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def rss() -> int:
    return _status("VmRSS")


def peak_extra_rss(fn):
    """Run fn() and return (result, bytes of resident memory above the starting RSS at the peak).

    Resets the kernel's high-water mark first (Linux /proc/self/clear_refs, value 5), so this
    catches temporaries allocated by NumPy, Numba or C alike.
    """
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
    before = rss()
    out = fn()
    return out, max(0, _status("VmHWM") - before)


def peak_tracemalloc(fn):
    """Peak bytes allocated through Python/NumPy during fn() (what the baseline numbers used)."""
    import tracemalloc
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    out = fn()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return out, peak


def timeit(fn, reps=5, warmup=1, setup=None):
    """Median wall time of fn() over reps (setup() runs untimed before each rep)."""
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    ts = []
    for _ in range(reps):
        if setup:
            setup()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


MiB = 1024 * 1024

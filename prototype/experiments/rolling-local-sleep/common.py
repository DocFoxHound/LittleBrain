"""Shared setup: thread caps, paths into prototype/, flow matrices, seeding helpers."""

import os
import sys
from pathlib import Path

# Resource caps from the brief (must be set before numba / numpy spin up their pools)
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent.parent
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402

from neuron import WORKHORSE, NeuronType  # noqa: E402,F401
from packed_neuron import (ELIG_STEPS, MAX_PENDING, TAG_MAX, USED_RATE,  # noqa: E402,F401
                           _flow_matrix)

VRAM_CAP = int(2.5 * 2**30)
MASK64 = (1 << 64) - 1


@lru_cache(maxsize=4096)
def flow3(t: NeuronType, steps: int) -> np.ndarray:
    """The prototype's L x L flow matrix, padded to 3 x 3 (identity on unused levels). Read-only."""
    m = _flow_matrix(t, int(steps))
    out = np.eye(3, dtype=np.float32)
    L = m.shape[0]
    out[:L, :L] = m
    out.flags.writeable = False
    return out


def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


class SeedStream:
    """Per-call 64-bit seeds for the counter-based RNG used inside the kernels."""

    def __init__(self, base: int):
        self.base, self.k = int(base) & MASK64, 0

    def next(self) -> int:
        self.k += 1
        return splitmix64(self.base ^ (self.k * 0xD1B54A32D192ED03 & MASK64))


def make_words(t: NeuronType, n: int, rng: np.random.Generator, chunk: int = 32768) -> np.ndarray:
    """Fresh synapse words, same distribution as PackedPopulation.__init__, built in chunks
    so a 1M-neuron population needs no multi-GB temporaries."""
    syn = t.n_branches * t.slots
    out = np.empty((n, syn), np.uint32)
    base = np.uint32((32 << 4) | (32 << 16) | (32 << 22))  # fast = c1 = c2 = 0, gen = tag = used = 0
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        v = rng.uniform(0.0, 0.1, (b - a, syn)).astype(np.float32) * np.float32(31 / t.w_max)
        q = np.clip(np.floor(v + rng.random(v.shape, dtype=np.float32)), -31, 31) + 32
        out[a:b] = base | (q.astype(np.uint32) << np.uint32(10))
    return out


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

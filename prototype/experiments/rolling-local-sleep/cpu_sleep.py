"""
CPU sleep kernels: PackedPopulation.sleep, run directly on the packed uint32 words, in place.

One pass, one synapse at a time in registers:
  catch up the lazy cascade clock -> consolidate tagged fast weights -> sleep_flow_steps of flow
  (one 3x3 matrix) -> downscale -> prune/regrow -> clear fast + used -> fade tags.
No temporaries beyond a per-neuron prune counter (4 bytes per neuron).

Two implementations with identical math and RNG (bit-identical results for the same seed):
  "c"     csleep.c via ctypes, clang -O3 -march=native -fopenmp (AVX2, 8 synapses per instruction;
          gcc 15 does not vectorize this loop, clang 21 does)
  "numba" @njit(parallel=True), same code; LLVM does not vectorize it (64-bit typing), ~10x slower
Randomness is a counter-based hash (lowbias32 of seed + 4*synapse index + k), 16-bit uniforms.
Not bit-identical to the NumPy prototype (different random numbers), only statistically equal.
"""

import ctypes
import os
import subprocess
import time

import numpy as np
from numba import njit, prange

import common  # noqa: F401  (thread caps, sys.path)
from common import HERE, flow3

# ---------------------------------------------------------------- C version (ctypes)

_SO = HERE / "build" / "_csleep.so"
_C = HERE / "csleep.c"
CFLAGS = ["-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-fno-math-errno",
          "-fno-trapping-math", "-shared", "-fPIC"]


def _build():
    _SO.parent.mkdir(exist_ok=True)
    if not _SO.exists() or _SO.stat().st_mtime < _C.stat().st_mtime:
        subprocess.check_call(["clang", *CFLAGS, str(_C), "-o", str(_SO)])


os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # idle OpenMP threads must not spin
os.environ.setdefault("KMP_BLOCKTIME", "0")
_build()
_lib = ctypes.CDLL(str(_SO))  # CDLL releases the GIL during the call
_P = ctypes.c_void_p
_lib.sleep_words.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, _P, ctypes.c_int32, _P,
                             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                             ctypes.c_float, ctypes.c_uint32, ctypes.c_int64, _P, ctypes.c_int32]
_lib.sleep_words.restype = None

# ---------------------------------------------------------------- Numba version

M32 = np.uint64(0xFFFFFFFF)


@njit(inline="always")
def _h32(x):
    x = x & M32
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7feb352d)) & M32
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846ca68b)) & M32
    x ^= x >> np.uint64(16)
    return x


@njit(inline="always")
def _ulo(r):
    return (np.float32(r & np.uint64(0xFFFF)) + np.float32(0.5)) * np.float32(1 / 65536)


@njit(inline="always")
def _uhi(r):
    return (np.float32(r >> np.uint64(16)) + np.float32(0.5)) * np.float32(1 / 65536)


@njit(inline="always")
def _clamp(v, lo, hi):
    v = lo if v < lo else v
    return hi if v > hi else v


@njit(inline="always")
def _enc(v, wmax, qinv, u):
    q = np.floor(_clamp(v, -wmax, wmax) * qinv + u)
    return np.uint64(np.int64(_clamp(q, np.float32(-31.0), np.float32(31.0))) + 32)


@njit(inline="always")
def _dec(s, wq):
    return (np.float32(s) - np.float32(32.0)) * wq


@njit(parallel=True, nogil=True, cache=True)
def _sleep_numba(words, syn, P, do_catch, S, fast_scale, wmax, keep, prune_eps, tag_decay,
                 seed, base_index, pruned):
    n_neurons = words.shape[0] // syn
    wq = wmax / np.float32(31.0)
    qinv = np.float32(31.0) / wmax
    seed = np.uint64(seed) & M32
    for n in prange(n_neurons):
        cnt = 0
        for j in range(syn):
            i = n * syn + j
            w = np.uint64(words[i])
            ctr = np.uint64(base_index + i) * np.uint64(4) + seed
            r0, r1 = _h32(ctr), _h32(ctr + np.uint64(1))
            r2, r3 = _h32(ctr + np.uint64(2)), _h32(ctr + np.uint64(3))
            gen = w & np.uint64(15)
            tagi = (w >> np.uint64(28)) & np.uint64(7)
            used = w >> np.uint64(31)
            a0 = _dec((w >> np.uint64(10)) & np.uint64(63), wq)
            a1 = _dec((w >> np.uint64(16)) & np.uint64(63), wq)
            a2 = _dec((w >> np.uint64(22)) & np.uint64(63), wq)
            if do_catch:
                b0 = P[0, 0] * a0 + P[0, 1] * a1 + P[0, 2] * a2
                b1 = P[1, 0] * a0 + P[1, 1] * a1 + P[1, 2] * a2
                b2 = P[2, 0] * a0 + P[2, 1] * a1 + P[2, 2] * a2
                a0 = _dec(_enc(b0, wmax, qinv, _ulo(r2)), wq)
                a1 = _dec(_enc(b1, wmax, qinv, _uhi(r2)), wq)
                a2 = _dec(_enc(b2, wmax, qinv, _ulo(r3)), wq)
            tag = np.float32(tagi) / np.float32(7.0)
            c0 = _clamp(a0 + _dec((w >> np.uint64(4)) & np.uint64(63), wq) * fast_scale * tag, -wmax, wmax)
            d0 = (S[0, 0] * c0 + S[0, 1] * a1 + S[0, 2] * a2) * keep
            d1 = S[1, 0] * c0 + S[1, 1] * a1 + S[1, 2] * a2
            d2 = S[2, 0] * c0 + S[2, 1] * a1 + S[2, 2] * a2
            m = max(abs(d0), max(abs(d1), abs(d2)))
            if m < prune_eps and used == 0 and tagi == 0:
                d0 = _uhi(r3) * np.float32(0.1)
                d1 = np.float32(0.0)
                d2 = np.float32(0.0)
                gen = (gen + np.uint64(1)) & np.uint64(15)
                cnt += 1
            q0 = _enc(d0, wmax, qinv, _ulo(r0))
            q1 = _enc(d1, wmax, qinv, _uhi(r0))
            q2 = _enc(d2, wmax, qinv, _ulo(r1))
            tq = np.uint64(np.int64(np.floor(np.float32(tagi) * tag_decay + _uhi(r1)))) & np.uint64(7)
            words[i] = np.uint32(gen | np.uint64(32 << 4) | (q0 << np.uint64(10)) | (q1 << np.uint64(16))
                                 | (q2 << np.uint64(22)) | (tq << np.uint64(28)))
        pruned[n] = cnt


# ---------------------------------------------------------------- entry points

DEFAULT_THREADS = int(os.environ.get("OMP_NUM_THREADS", "3"))
KERNEL_SECONDS = []  # wall time of each kernel call (for the schedule benchmark's breakdown)


def cpu_sleep(words2d: np.ndarray, t, pending: int, fast_scale: float, seed: int,
              base_index: int = 0, impl: str = "c", threads: int = DEFAULT_THREADS) -> int:
    """One night for a block of packed words (in place). Returns the number of synapses rewired."""
    assert words2d.dtype == np.uint32 and words2d.flags.c_contiguous
    flat = words2d.reshape(-1)
    syn = t.n_branches * t.slots
    n = flat.shape[0] // syn
    pruned = np.zeros(n, np.int32)
    P, S = flow3(t, pending), flow3(t, t.sleep_flow_steps)
    seed32 = int(seed) & 0xFFFFFFFF
    t0 = time.perf_counter()
    if impl == "c":
        _lib.sleep_words(flat.ctypes.data, n, syn, P.ctypes.data, int(pending > 0), S.ctypes.data,
                         float(np.float32(fast_scale)), float(t.w_max), float(np.float32(1 - t.downscale)),
                         float(t.prune_eps), float(t.tag_decay), seed32, int(base_index),
                         pruned.ctypes.data, int(threads))
    else:
        import numba
        numba.set_num_threads(threads)
        _sleep_numba(flat, syn, P, bool(pending > 0), S, np.float32(fast_scale), np.float32(t.w_max),
                     np.float32(1 - t.downscale), np.float32(t.prune_eps), np.float32(t.tag_decay),
                     np.uint64(seed32), np.int64(base_index), pruned)
    KERNEL_SECONDS.append(time.perf_counter() - t0)
    return int(pruned.sum())


def sleep_neuron_state(t, plasticity: np.ndarray, age: np.ndarray, post_hist: np.ndarray):
    """The per-neuron part of sleep (tiny arrays): mature, age, clear firing history."""
    post_hist[:] = 0
    age += 1
    plasticity[:] = (t.plasticity_floor + (plasticity - t.plasticity_floor) * t.maturation
                     ).astype(np.float16)

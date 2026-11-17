"""
Python side of strategy 3 (bit-sliced branches, wiring bitmap in the free gen planes). See bitslice.c.

Behavior differences from the baseline (deliberate, checked in behavior.py):
  * Wiring is LOCAL: each branch draws its 32 synapses from a window of 128 consecutive inputs
    (window start = hash of neuron and branch), at most one synapse per input position.
  * Rewiring moves a synapse to a random FREE input inside the same window (not anywhere).
  * Inputs must be binary spikes.
"""

import common  # noqa: F401

import ctypes
import os
import subprocess

import numpy as np
from numba import njit

from packed_neuron import MAX_PENDING, Q, USED_RATE, PackedPopulation, _flow_matrix

SRC = os.path.join(common.HERE, "bitslice.c")
LIB = os.path.join(common.HERE, "libbitslice.so")


def _build():
    # clang vectorizes the per-synapse loops (8-wide AVX2); gcc 15 does not. gcc is the fallback.
    for cc in ("clang", "gcc"):
        try:
            subprocess.run([cc, "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", SRC, "-o", LIB, "-lm"],
                           check=True, capture_output=True)
            return ctypes.CDLL(LIB)
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("could not build bitslice.c")


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        lib = _build()
    else:
        try:
            lib = ctypes.CDLL(LIB)
        except OSError:
            lib = _build()
    P, I64, I, F, D, U64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_double, ctypes.c_uint64
    lib.bs_pack.argtypes = [I64, P, P, P]
    lib.bs_unpack.argtypes = [I64, P, P, P]
    lib.bs_windows.argtypes = [I64, I, I64, I64, P]
    lib.bs_forward.argtypes = [I64, I, I64, I64, P, P, F, F, F, F, P, I, P, F, F, F, F, I, F, F, F, P, P, I]
    lib.bs_modulate.argtypes = [I64, I, I64, I64, P, P, P, I, P, P, P, F, F, F, F, U64, I, I]
    lib.bs_modulate_exact.argtypes = [I64, I, I64, I64, P, P, P, I, P, P, P, F, F, F, F, I, P, P, P, I]
    lib.bs_catch_up.argtypes = [I64, P, P, F, F, U64, I]
    lib.bs_sleep.argtypes = [I64, I, P, P, F, F, F, F, F, D, U64, I]
    lib.bs_sleep.restype = I64
    return lib


lib = _load()


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


@njit(cache=True)
def _random_bitmaps(nbr, seed, out):
    """Floyd sampling: 32 distinct positions out of 128 for every branch."""
    st = np.uint64(seed)
    for i in range(nbr):
        lo = np.uint64(0)
        hi = np.uint64(0)
        for j in range(96, 128):
            st += np.uint64(0x9E3779B97F4A7C15)
            z = st
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z ^= z >> np.uint64(31)
            t = np.int64(z % np.uint64(j + 1))
            taken = (lo >> np.uint64(t)) & np.uint64(1) if t < 64 else (hi >> np.uint64(t - 64)) & np.uint64(1)
            if taken:
                t = j
            if t < 64:
                lo |= np.uint64(1) << np.uint64(t)
            else:
                hi |= np.uint64(1) << np.uint64(t - 64)
        out[i, 0] = lo
        out[i, 1] = hi


class BitSlicedPopulation:
    def __init__(self, ntype, n_neurons, n_inputs, rng, first_id=0, nthreads=3, seed=None):
        t = self.t = ntype
        assert n_inputs % 64 == 0 and ntype.slots == 32 and ntype.n_branches <= 64 and ntype.cascade_levels <= 3
        self.n, self.n_inputs, self.rng, self.first_id, self.nthreads = n_neurons, n_inputs, rng, first_id, nthreads
        B = t.n_branches
        self.nW = n_inputs // 64
        base = PackedPopulation(ntype, n_neurons, n_inputs, rng, first_id)
        nbr = n_neurons * B
        wire = np.empty((nbr, 2), np.uint64)
        _random_bitmaps(nbr, (first_id * 0x2545F4914F6CDD1D + 99) % (1 << 63), wire)
        self.rec = np.empty(nbr * 32, np.uint32)
        lib.bs_pack(nbr, ptr(base.words), ptr(wire), ptr(self.rec))
        self.gate_bits = np.ascontiguousarray(base.gate_bits)
        self.gbytes = self.gate_bits.shape[1]
        self.threshold = base.threshold.copy()
        self.plasticity = base.plasticity.copy()
        self.post_hist = base.post_hist.copy()
        self.age = base.age.copy()
        self.elig_weights = base.elig_weights.astype(np.float32)
        del base
        self.Xh = np.zeros((4, self.nW), np.uint64)  # last 4 input spike patterns, as bitsets
        self.pending, self.fast_scale = 0, 1.0
        self.wscale = np.float32(t.w_max / Q)
        self.key = seed if seed is not None else (first_id * 0x9E3779B97F4A7C15 + 4242) % (1 << 64)
        self.counter = 0
        self.rate_buf = np.empty(n_neurons, np.float32)

    def _next_key(self):
        self.counter += 1
        return (self.key + self.counter * 0x632BE59BD9B4E019) % (1 << 64)

    def bitset(self, x):
        x = np.asarray(x)
        assert ((x == 0) | (x == 1)).all(), "bit-sliced branches need binary spikes"
        return np.packbits(x > 0, bitorder="little").view(np.uint64)

    def forward(self, x, context=0.0, learn_trace=True, X=None):
        t = self.t
        X = self.bitset(x) if X is None else X
        M0 = _flow_matrix(t, self.pending)[0]
        b = [float(self.wscale * M0[k]) if k < t.cascade_levels else 0.0 for k in range(3)]
        lib.bs_forward(self.n, t.n_branches, self.nW, self.first_id, ptr(X), ptr(self.rec),
                       float(self.wscale * np.float32(self.fast_scale)), b[0], b[1], b[2], ptr(self.gate_bits),
                       self.gbytes, ptr(self.threshold), t.branch_threshold, t.spike_bonus,
                       1.0 + t.context_gain * context, t.soma_slope, int(learn_trace), USED_RATE,
                       t.threshold_lr, t.target_rate, ptr(self.rate_buf), ptr(self.post_hist), self.nthreads)
        if learn_trace:
            self.Xh[1:] = self.Xh[:-1].copy()
            self.Xh[0] = X
        return self.rate_buf * t.sign

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0, uniforms=None):
        if dopamine == 0.0:
            return
        t = self.t
        coef = (t.lr * self.plasticity * ach * dopamine).astype(np.float32)   # float16 math, as baseline
        oneshot = abs(dopamine) * ach > t.oneshot_level
        if oneshot:
            self._catch_up()
        common_args = (self.n, t.n_branches, self.nW, self.first_id, ptr(self.Xh), ptr(self.rec),
                       ptr(self.gate_bits), self.gbytes, ptr(self.post_hist), ptr(self.elig_weights), ptr(coef),
                       float(np.float32(self.fast_scale)), t.w_max, float(self.wscale),
                       float(np.float32(arousal * abs(dopamine))))
        if uniforms is None:
            lib.bs_modulate(*common_args, self._next_key(), int(oneshot), self.nthreads)
        else:   # validation: NumPy's uniforms, baseline arithmetic order
            uf, ut, uo = (np.ascontiguousarray(u, np.float32) for u in uniforms)
            lib.bs_modulate_exact(*common_args, int(oneshot), ptr(uf), ptr(ut), ptr(uo), self.nthreads)
        self.fast_scale = 1.0

    def tick(self):
        self.pending += 1
        self.fast_scale *= self.t.fast_decay
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def rest(self, steps):
        self.pending += steps
        self.fast_scale *= self.t.fast_decay ** steps
        self._catch_up()

    def _M3(self, steps):
        M3 = np.zeros((3, 3), np.float32)
        L = self.t.cascade_levels
        M3[:L, :L] = _flow_matrix(self.t, steps)
        return M3

    def _catch_up(self):
        if self.pending:
            M3 = self._M3(self.pending)
            lib.bs_catch_up(self.n * self.t.n_branches, ptr(self.rec), ptr(M3), self.t.w_max, float(self.wscale),
                            self._next_key(), self.nthreads)
            self.pending = 0

    def sleep(self):
        t = self.t
        self._catch_up()
        Ms = self._M3(t.sleep_flow_steps)
        n = lib.bs_sleep(self.n, t.n_branches, ptr(self.rec), ptr(Ms), float(np.float32(self.fast_scale)),
                         t.w_max, float(self.wscale), t.downscale, t.prune_eps, t.tag_decay, self._next_key(),
                         self.nthreads)
        self.fast_scale = 1.0
        self.Xh[:] = 0
        self.post_hist[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16)
        return int(n)

    def bytes_per_neuron(self):
        per = [self.rec, self.gate_bits, self.threshold, self.plasticity, self.post_hist, self.age]
        return sum(a.nbytes for a in per) / self.n

    # ---------- inspection / validation ----------

    def unpack(self):
        """(words (N,B,32) in baseline format with gen = 0, src (N,B,32) input index of each slot)."""
        N, B = self.n, self.t.n_branches
        words = np.empty(N * B * 32, np.uint32)
        wire = np.empty((N * B, 2), np.uint64)
        lib.bs_unpack(N * B, ptr(self.rec), ptr(words), ptr(wire))
        ws = np.empty(N * B, np.int64)
        lib.bs_windows(N, B, self.first_id, self.nW, ptr(ws))
        bits = np.unpackbits(wire.view(np.uint8), axis=1, bitorder="little").astype(bool)  # (NB, 128)
        pos = np.nonzero(bits)[1].reshape(N * B, 32)
        src = (ws[:, None] * 64 + pos) % self.n_inputs
        return words.reshape(N, B, 32), src.reshape(N, B, 32)


class ExplicitWiringReference(PackedPopulation):
    """The baseline NumPy code, but reading the input index of each synapse from an explicit array."""

    def __init__(self, bs: BitSlicedPopulation):
        super().__init__(bs.t, 1, bs.n_inputs, np.random.default_rng(0))
        words, self.src = bs.unpack()
        self.n, self.words = bs.n, words.copy()
        for name in ("gate_bits", "threshold", "plasticity", "post_hist", "age"):
            setattr(self, name, getattr(bs, name).copy())
        self.pending, self.fast_scale = bs.pending, bs.fast_scale

    def _src(self, w):
        return self.src


class BitSlicedNeuron:
    def __init__(self, ntype, n_inputs, rng):
        self.pop = BitSlicedPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)), nthreads=1)

    def forward(self, x, context=0.0, learn_trace=True):
        return float(self.pop.forward(x, context, learn_trace)[0])

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        self.pop.modulate(dopamine, ach, arousal)

    def tick(self):
        self.pop.tick()

    def rest(self, steps):
        self.pop.rest(steps)

    def sleep(self):
        return self.pop.sleep()

    def memory_bytes(self):
        return int(self.pop.bytes_per_neuron())

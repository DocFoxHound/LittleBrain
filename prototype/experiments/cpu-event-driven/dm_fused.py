"""
Strategy 1: fused CPU kernels for the CURRENT ("dendrite-major") packed layout.

Same storage as packed_neuron.PackedPopulation (one uint32 word per synapse, words[n, branch, slot],
input index = hash(neuron id, slot, gen)). The difference is HOW a step runs: one Numba kernel
walks the words, unpacks each into registers, computes and writes it back. No (N, B, S) temporaries.

Semantics are the baseline's, line by line (including the global fast-weight clock that is folded
into every synapse at each modulate). The only intended difference is where the random numbers for
stochastic rounding come from: a counter-based hash (splitmix64) instead of NumPy's generator,
so no random arrays are allocated. In `exact=True` mode the kernel instead consumes NumPy uniforms
drawn in the same order as the baseline, which lets us check modulate bit-for-bit.
"""

import common  # noqa: F401  (sets thread caps and sys.path)

import numpy as np
from numba import njit, prange

from packed_neuron import ELIG_STEPS, MAX_PENDING, Q, TAG_MAX, USED_RATE, PackedPopulation, _flow_matrix

C1 = np.uint64(0x9E3779B97F4A7C15)
C2 = np.uint64(0xBF58476D1CE4E5B9)
C3 = np.uint64(0x94D049BB133111EB)
C4 = np.uint64(0xD6E8FEB86659FD93)
U27, U30, U31, U32, U40 = (np.uint64(v) for v in (27, 30, 31, 32, 40))
INV24 = np.float32(1.0 / 16777216.0)
CHUNK = 64


@njit(inline="always")
def mix64(h):
    """splitmix64 finalizer: a good 64-bit hash, used as a counter-based RNG."""
    h ^= h >> U30
    h *= C2
    h ^= h >> U27
    h *= C3
    h ^= h >> U31
    return h


@njit(inline="always")
def unif(key, i):
    """Uniform float32 in [0, 1) for counter i under key (stateless, thread-safe)."""
    return np.float32(mix64(key + np.uint64(i) * C1) >> U40) * INV24


@njit(inline="always")
def src_of(idp, slotc, genc, w, nin, mask):
    """Baseline wiring hash: _targets(ids, slot, gen, n_inputs) for one synapse."""
    h = idp ^ slotc ^ genc[w & np.uint32(15)]
    h ^= h >> U31
    h *= C4
    h ^= h >> U32
    if mask != np.uint64(0):
        return h & mask
    return h % nin


def _forward(words, first_id, nin, mask, slotc, genc, x, a_f, b0, b1, b2, off, gate_bits,
             threshold, bthr, bonus, ctx, slope, learn, used_rate, thr_lr, target, rate):
    N, B, S = words.shape
    nchunk = (N + CHUNK - 1) // CHUNK
    for c in prange(nchunk):
        act = np.zeros(B, np.uint64)
        for n in range(c * CHUNK, min(N, (c + 1) * CHUNK)):
            idp = (np.uint64(first_id + n) + np.uint64(1)) * C1
            soma = np.float32(0.0)
            for b in range(B):
                drive = np.float32(0.0)
                m = np.uint64(0)
                for s in range(S):
                    w = words[n, b, s]
                    xv = x[src_of(idp, slotc[b * S + s], genc, w, nin, mask)]
                    wt = (np.float32((w >> np.uint32(4)) & np.uint32(63)) * a_f
                          + np.float32((w >> np.uint32(10)) & np.uint32(63)) * b0
                          + np.float32((w >> np.uint32(16)) & np.uint32(63)) * b1
                          + np.float32((w >> np.uint32(22)) & np.uint32(63)) * b2 - off)
                    drive += wt * xv
                    if xv > np.float32(0.0):
                        m |= np.uint64(1) << np.uint64(s)
                act[b] = m
                g = (gate_bits[n, b >> 3] >> (7 - (b & 7))) & 1
                if g:
                    sp = np.float32(1.0) if drive > bthr else np.float32(0.0)
                    soma += max(drive, np.float32(0.0)) + bonus * sp
            soma *= ctx
            r = np.float32(1.0) / (np.float32(1.0) + np.exp(-slope * (soma - threshold[n])))
            rate[n] = r
            if learn:
                if r >= used_rate:
                    for b in range(B):
                        if (gate_bits[n, b >> 3] >> (7 - (b & 7))) & 1:
                            m = act[b]
                            for s in range(S):
                                if (m >> np.uint64(s)) & np.uint64(1):
                                    words[n, b, s] |= np.uint32(0x80000000)
                threshold[n] += thr_lr * (r - target)


def _modulate(words, first_id, nin, mask, slotc, genc, gate_bits, pre32, post32, ew, coef,
              fs, wmax, wscale, ar_da, use_u, u_fast, u_tag, key):
    N, B, S = words.shape
    nchunk = (N + CHUNK - 1) // CHUNK
    q = np.float32(Q)
    for c in prange(nchunk):
        for n in range(c * CHUNK, min(N, (c + 1) * CHUNK)):
            idp = (np.uint64(first_id + n) + np.uint64(1)) * C1
            p0, p1, p2, p3 = post32[n, 0], post32[n, 1], post32[n, 2], post32[n, 3]
            cf = coef[n]
            for b in range(B):
                g = np.float32((gate_bits[n, b >> 3] >> (7 - (b & 7))) & 1)
                for s in range(S):
                    idx = (n * B + b) * S + s
                    w = words[n, b, s]
                    src = src_of(idp, slotc[b * S + s], genc, w, nin, mask)
                    e = (ew[0] * pre32[0, src]) * p0
                    e += (ew[1] * pre32[1, src]) * p1
                    e += (ew[2] * pre32[2, src]) * p2
                    e += (ew[3] * pre32[3, src]) * p3
                    e = e * g
                    d = (cf * e) * g
                    fv = (np.float32((w >> np.uint32(4)) & np.uint32(63)) - np.float32(32.0)) * wscale
                    fv = fv * fs + d
                    fv = min(max(fv, -wmax), wmax)
                    h = mix64(key + np.uint64(idx) * C1)   # one hash, two 24-bit uniforms
                    u = u_fast[idx] if use_u else np.float32(h >> U40) * INV24
                    fq = np.floor(fv * q + u)
                    fq = min(max(fq, -q), q) + np.float32(32.0)
                    if use_u:   # baseline arithmetic (float64 tag), for the bit-exact check
                        tg = np.float64((w >> np.uint32(28)) & np.uint32(7)) / 7.0 + np.float64((ar_da * e) * g)
                        tg = min(max(tg, 0.0), 1.0)
                        tq = np.uint32(np.floor(tg * 7.0 + np.float64(u_tag[idx])))
                    elif e == np.float32(0.0):   # no eligibility: the tag cannot change
                        tq = (w >> np.uint32(28)) & np.uint32(7)
                    else:
                        tg = np.float32((w >> np.uint32(28)) & np.uint32(7)) * np.float32(1.0 / 7.0) + (ar_da * e) * g
                        tg = min(max(tg, np.float32(0.0)), np.float32(1.0))
                        u2 = np.float32((h >> np.uint64(16)) & np.uint64(0xFFFFFF)) * INV24
                        tq = np.uint32(np.int32(np.floor(tg * np.float32(7.0) + u2)))
                    w = (w & np.uint32(0x8FFFFC0F)) | (np.uint32(fq) << np.uint32(4)) \
                        | ((tq & np.uint32(7)) << np.uint32(28))
                    words[n, b, s] = w


@njit(inline="always")
def enc6(v, wmax, u):
    """_enc for one value: clip to +-w_max, stochastic round to 1/31 steps, store as 1..63."""
    q = np.float32(Q)
    v = min(max(v, -wmax), wmax)
    f = np.floor(v * (q / wmax) + u)
    return np.uint32(min(max(f, -q), q) + np.float32(32.0))


@njit(inline="always")
def dec6(w, off, wscale):
    return (np.float32((w >> np.uint32(off)) & np.uint32(63)) - np.float32(32.0)) * wscale


def _catch_up(words, M, wmax, wscale, key):
    N, B, S = words.shape
    for n in prange(N):
        for b in range(B):
            for s in range(S):
                idx = (n * B + b) * S + s
                w = words[n, b, s]
                l0, l1, l2 = dec6(w, 10, wscale), dec6(w, 16, wscale), dec6(w, 22, wscale)
                c0 = M[0, 0] * l0 + M[0, 1] * l1 + M[0, 2] * l2
                c1 = M[1, 0] * l0 + M[1, 1] * l1 + M[1, 2] * l2
                c2 = M[2, 0] * l0 + M[2, 1] * l1 + M[2, 2] * l2
                w = (w & np.uint32(0xF00003FF)) | (enc6(c0, wmax, unif(key, 3 * idx)) << np.uint32(10)) \
                    | (enc6(c1, wmax, unif(key, 3 * idx + 1)) << np.uint32(16)) \
                    | (enc6(c2, wmax, unif(key, 3 * idx + 2)) << np.uint32(22))
                words[n, b, s] = w


def _oneshot(words, first_id, nin, mask, slotc, genc, gate_bits, pre32, post32, ew, coef,
             wmax, wscale, key):
    """BTSP-like one-shot write: c0 += 0.5 * delta, all levels re-encoded (as the baseline)."""
    N, B, S = words.shape
    for n in prange(N):
        idp = (np.uint64(first_id + n) + np.uint64(1)) * C1
        for b in range(B):
            g = np.float32((gate_bits[n, b >> 3] >> (7 - (b & 7))) & 1)
            for s in range(S):
                idx = (n * B + b) * S + s
                w = words[n, b, s]
                src = src_of(idp, slotc[b * S + s], genc, w, nin, mask)
                e = np.float32(0.0)
                for k in range(4):
                    e += (ew[k] * pre32[k, src]) * post32[n, k]
                d = (coef[n] * (e * g)) * g
                c0 = dec6(w, 10, wscale) + np.float32(0.5) * d
                w = (w & np.uint32(0xF00003FF)) | (enc6(c0, wmax, unif(key, 3 * idx)) << np.uint32(10)) \
                    | (enc6(dec6(w, 16, wscale), wmax, unif(key, 3 * idx + 1)) << np.uint32(16)) \
                    | (enc6(dec6(w, 22, wscale), wmax, unif(key, 3 * idx + 2)) << np.uint32(22))
                words[n, b, s] = w


def _sleep(words, Ms, fs, wmax, wscale, downscale, prune_eps, tag_decay, key):
    N, B, S = words.shape
    pruned = 0
    for n in prange(N):
        for b in range(B):
            for s in range(S):
                idx = (n * B + b) * S + s
                w = words[n, b, s]
                tag = np.float64((w >> np.uint32(28)) & np.uint32(7)) / 7.0
                l0 = np.float32(min(max(np.float64(dec6(w, 10, wscale))
                                        + np.float64(dec6(w, 4, wscale) * fs) * tag, -wmax), wmax))
                l1, l2 = dec6(w, 16, wscale), dec6(w, 22, wscale)
                c0 = Ms[0, 0] * l0 + Ms[0, 1] * l1 + Ms[0, 2] * l2
                c1 = Ms[1, 0] * l0 + Ms[1, 1] * l1 + Ms[1, 2] * l2
                c2 = Ms[2, 0] * l0 + Ms[2, 1] * l1 + Ms[2, 2] * l2
                c0 *= np.float32(1.0) - downscale
                gen = w & np.uint32(15)
                weak = (max(abs(c0), abs(c1), abs(c2)) < prune_eps) and (w >> np.uint32(31)) == 0 \
                    and tag < 0.05
                if weak:
                    pruned += 1
                    gen = (gen + np.uint32(1)) & np.uint32(15)
                    c0 = np.float32(0.1) * unif(key, 5 * idx + 4)
                    c1 = np.float32(0.0)
                    c2 = np.float32(0.0)
                tq = np.uint32(np.floor(tag * tag_decay * 7.0 + np.float64(unif(key, 5 * idx + 3))))
                w = gen | (np.uint32(32) << np.uint32(4)) \
                    | (enc6(c0, wmax, unif(key, 5 * idx)) << np.uint32(10)) \
                    | (enc6(c1, wmax, unif(key, 5 * idx + 1)) << np.uint32(16)) \
                    | (enc6(c2, wmax, unif(key, 5 * idx + 2)) << np.uint32(22)) \
                    | ((tq & np.uint32(7)) << np.uint32(28))
                words[n, b, s] = w
    return pruned


def _compile(fastmath):
    o = dict(parallel=True, fastmath=fastmath, nogil=True, cache=fastmath, error_model="numpy")
    return {name: njit(**o)(f) for name, f in
            [("forward", _forward), ("modulate", _modulate), ("catch_up", _catch_up),
             ("oneshot", _oneshot), ("sleep", _sleep)]}


K_FAST = _compile(True)    # production
K_EXACT = _compile(False)  # strict IEEE order, for bit-exact checks against NumPy


class FusedPopulation(PackedPopulation):
    """PackedPopulation with every step replaced by a fused Numba kernel. Same state, same layout."""

    def __init__(self, ntype, n_neurons, n_inputs, rng, first_id=0, exact=False, seed=None):
        super().__init__(ntype, n_neurons, n_inputs, rng, first_id)
        assert ntype.n_branches * ntype.slots <= 1 << 20 and ntype.slots <= 64
        self.exact = exact
        self.K = K_EXACT if exact else K_FAST
        self.slotc = (np.arange(ntype.n_branches * ntype.slots, dtype=np.uint64) * C2)
        self.genc = np.arange(16, dtype=np.uint64) * C3
        self.nin_u = np.uint64(n_inputs)
        self.mask_u = np.uint64(n_inputs - 1) if n_inputs & (n_inputs - 1) == 0 else np.uint64(0)
        self.key = np.uint64(seed if seed is not None else (first_id * 0x2545F4914F6CDD1D + 12345) % (1 << 63))
        self.counter = 0
        self.rate_buf = np.empty(n_neurons, np.float32)
        self.wscale = np.float32(ntype.w_max / Q)

    def _next_key(self):
        self.counter += 1
        return np.uint64((int(self.key) + self.counter * 0x632BE59BD9B4E019) % (1 << 64))

    # ---------- acting ----------

    def forward(self, x, context=0.0, learn_trace=True):
        t = self.t
        x32 = np.ascontiguousarray(x, np.float32)
        M0 = _flow_matrix(t, self.pending)[0]
        ws = self.wscale
        a_f = np.float32(ws * np.float32(self.fast_scale))
        b0, b1, b2 = (np.float32(ws * M0[k]) if k < t.cascade_levels else np.float32(0) for k in range(3))
        off = np.float32(32.0) * (a_f + b0 + b1 + b2)
        self.K["forward"](self.words, self.first_id, self.nin_u, self.mask_u, self.slotc, self.genc, x32,
                          a_f, b0, b1, b2, off, self.gate_bits, self.threshold,
                          np.float32(t.branch_threshold), np.float32(t.spike_bonus),
                          np.float32(1.0 + t.context_gain * context), np.float32(t.soma_slope),
                          bool(learn_trace), np.float32(USED_RATE), np.float32(t.threshold_lr),
                          np.float32(t.target_rate), self.rate_buf)
        rate = self.rate_buf
        if learn_trace:
            self.pre_hist[1:] = self.pre_hist[:-1].copy()
            self.pre_hist[0] = x
            self.post_hist[:, 1:] = self.post_hist[:, :-1].copy()
            self.post_hist[:, 0] = 0.5 + rate
        return rate * t.sign

    # ---------- learning ----------

    def _mod_args(self, dopamine, ach):
        t = self.t
        coef16 = t.lr * self.plasticity * ach * dopamine          # float16, exactly as the baseline
        return (self.pre_hist.astype(np.float32), self.post_hist.astype(np.float32),
                self.elig_weights.astype(np.float32), coef16.astype(np.float32))

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0, uniforms=None):
        if dopamine == 0.0:
            return
        t = self.t
        pre32, post32, ew, coef = self._mod_args(dopamine, ach)
        if uniforms is None:
            use_u, uf, ut = False, np.zeros(1, np.float32), np.zeros(1, np.float32)
        else:
            use_u, (uf, ut) = True, uniforms
        self.K["modulate"](self.words, self.first_id, self.nin_u, self.mask_u, self.slotc, self.genc,
                           self.gate_bits, pre32, post32, ew, coef, np.float32(self.fast_scale),
                           np.float32(t.w_max), self.wscale, np.float32(arousal * abs(dopamine)),
                           use_u, uf.ravel(), ut.ravel(), self._next_key())
        self.fast_scale = 1.0
        if abs(dopamine) * ach > t.oneshot_level:
            self._catch_up()
            self.K["oneshot"](self.words, self.first_id, self.nin_u, self.mask_u, self.slotc, self.genc,
                              self.gate_bits, pre32, post32, ew, coef, np.float32(t.w_max), self.wscale,
                              self._next_key())

    def _catch_up(self):
        if self.pending:
            M = _flow_matrix(self.t, self.pending)
            M3 = np.zeros((3, 3), np.float32)
            L = self.t.cascade_levels
            M3[:L, :L] = M
            self.K["catch_up"](self.words, M3, np.float32(self.t.w_max), self.wscale, self._next_key())
            self.pending = 0

    # ---------- sleeping ----------

    def sleep(self):
        t = self.t
        self._catch_up()
        M = _flow_matrix(t, t.sleep_flow_steps)
        M3 = np.zeros((3, 3), np.float32)
        M3[:t.cascade_levels, :t.cascade_levels] = M
        n = self.K["sleep"](self.words, M3, np.float32(self.fast_scale), np.float32(t.w_max), self.wscale,
                            np.float32(t.downscale), np.float32(t.prune_eps), float(t.tag_decay),
                            self._next_key())
        self.fast_scale = 1.0
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16)
        return int(n)


class FusedNeuron:
    """Single-neuron adapter (same interface as PackedNeuron) for the behavior test."""

    def __init__(self, ntype, n_inputs, rng):
        self.pop = FusedPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)))

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

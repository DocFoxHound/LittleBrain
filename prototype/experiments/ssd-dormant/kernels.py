"""
Numba kernels that work directly on packed 32-bit synapse words (same bit layout and the same
maths as prototype/packed_neuron.py), one neuron = one row of 256 words = 1 KiB.

Kernels take `rows` (where each neuron sits in the buffer) and `ids` (its global id), so the same
code runs on: an in-RAM array, a RAM block cache filled from the SSD, an np.memmap of the brain
file, or a chunk streaming through the sleep sweep.

Randomness (stochastic rounding, regrowth) comes from a counter-based hash of
(seed, neuron id, synapse, field), so results do not depend on thread scheduling.

Simplification vs packed_neuron.py (documented in README): the shared clocks (fast_scale and
pending cascade flow) are passed in per call. In a real sparse brain they become per-neuron
"last touched" stamps; the maths is the same.
"""

import numpy as np
from numba import njit, prange

U = np.uint64
K1, K2, K3, K4 = U(0x9E3779B97F4A7C15), U(0xBF58476D1CE4E5B9), U(0x94D049BB133111EB), U(0xD6E8FEB86659FD93)
Q = 31
TAG_MAX = 7
USED_RATE = 0.05
NB, NS = 8, 32


@njit(inline="always")
def _mod(h, n):
    """h % n, with a fast path when n is a power of two (64-bit division is ~30 cycles)."""
    if (n & (n - U(1))) == U(0):
        return h & (n - U(1))
    return h % n


@njit(inline="always")
def _target(nid, slot, gen, n_inputs):
    """Same hash as packed_neuron._targets: input = hash(neuron id, slot, generation)."""
    h = ((nid + U(1)) * K1) ^ (slot * K2) ^ (gen * K3)
    h ^= h >> U(31)
    h *= K4
    h ^= h >> U(32)
    return _mod(h, n_inputs)


@njit(inline="always")
def _urand(seed, nid, j, field):
    """Uniform [0, 1) from a counter-based hash (splitmix64 finalizer)."""
    x = seed ^ (nid * K1) ^ (U(j) * K2) ^ (U(field) * K4)
    x ^= x >> U(30)
    x *= K2
    x ^= x >> U(27)
    x *= K3
    x ^= x >> U(31)
    return np.float32(x >> U(40)) * np.float32(1.0 / 16777216.0)


@njit(inline="always")
def _dec(s, scale):          # 6-bit field (1..63, 32 = zero) -> float
    return (np.float32(s) - np.float32(32.0)) * scale


@njit(inline="always")
def _enc(v, inv_scale, u):   # float -> 6-bit field, stochastic rounding, clipped to +-31
    q = np.floor(v * inv_scale + u)
    if q > Q:
        q = Q
    if q < -Q:
        q = -Q
    return np.uint32(np.int32(q) + 32)


# ----------------------------------------------------------------------------- acting

@njit(parallel=True, nogil=True, cache=True)
def k_forward(words, rows, ids, id0, gate, thresh, post_hist, X, n_in, col_shift, F0, fast_scale,
              wmax, bthr, bonus, ctx_mult, slope, thr_lr, target_rate, learn, rate_out):
    """Forward pass for the neurons at `rows`. Input to neuron i, synapse s:
    X[(hash(i, s, gen) + column(i) * col_shift) % n_in]. If learn: set 'used' bits,
    adapt the threshold, push the firing history (for eligibility).
    Per-neuron state (gate, thresh, post_hist) is indexed by ids - id0."""
    scale = np.float32(wmax / Q)
    for a in prange(rows.shape[0]):
        r = rows[a]
        nid = U(ids[a])
        i = ids[a] - id0
        shift = (nid // U(1024)) * U(col_shift)
        g = gate[i]
        soma = np.float32(0.0)
        for b in range(NB):
            if not ((g >> b) & 1):
                continue                  # a silenced branch contributes nothing
            drive = np.float32(0.0)
            for s in range(NS):
                j = b * NS + s
                w = words[r, j]
                t = _mod(_target(nid, U(j), U(w & np.uint32(15)), U(n_in)) + shift, U(n_in))
                xv = X[t]
                if xv != 0.0:
                    wt = _dec((w >> np.uint32(4)) & np.uint32(63), scale) * fast_scale \
                        + F0[0] * _dec((w >> np.uint32(10)) & np.uint32(63), scale) \
                        + F0[1] * _dec((w >> np.uint32(16)) & np.uint32(63), scale) \
                        + F0[2] * _dec((w >> np.uint32(22)) & np.uint32(63), scale)
                    drive += wt * xv
            out = drive if drive > 0.0 else np.float32(0.0)
            if drive > bthr:
                out += bonus
            soma += out
        soma *= ctx_mult
        rate = np.float32(1.0) / (np.float32(1.0) + np.exp(-slope * (soma - thresh[i])))
        rate_out[a] = rate
        if learn:
            if rate >= USED_RATE:
                for b in range(NB):
                    if (g >> b) & 1:
                        for s in range(NS):
                            j = b * NS + s
                            w = words[r, j]
                            t = _mod(_target(nid, U(j), U(w & np.uint32(15)), U(n_in)) + shift, U(n_in))
                            if X[t] > 0.0:
                                words[r, j] = w | np.uint32(0x80000000)
            thresh[i] += thr_lr * (rate - target_rate)
            for k in range(post_hist.shape[1] - 1, 0, -1):
                post_hist[i, k] = post_hist[i, k - 1]
            post_hist[i, 0] = np.float32(0.5) + rate


@njit(parallel=True, nogil=True, cache=True)
def k_modulate(words, rows, ids, id0, gate, plast, post_hist, pre_hist, elig_w, n_in, col_shift,
               fast_scale, wmax, lr, dopamine, ach, arousal, oneshot, seed):
    """Three-factor update of fast weights and tags (and the BTSP one-shot write into c0;
    the caller must have caught the cascade up first, as packed_neuron does)."""
    scale = np.float32(wmax / Q)
    inv = np.float32(Q / wmax)
    nh = pre_hist.shape[0]
    for a in prange(rows.shape[0]):
        r = rows[a]
        i = ids[a] - id0
        nid = U(ids[a])
        shift = (nid // U(1024)) * U(col_shift)
        g = gate[i]
        p = np.float32(plast[i])
        for b in range(NB):
            open_ = (g >> b) & 1          # gated branches don't learn, but their fast weights still fade
            for s in range(NS):
                j = b * NS + s
                w = words[r, j]
                e = np.float32(0.0)
                if open_:
                    t = _mod(_target(nid, U(j), U(w & np.uint32(15)), U(n_in)) + shift, U(n_in))
                    for k in range(nh):
                        e += elig_w[k] * pre_hist[k, t] * post_hist[i, k]
                if e == 0.0 and fast_scale == 1.0:
                    continue          # exact no-op: no change, no decay to apply (saves the RNG draws)
                delta = lr * p * ach * dopamine * e
                f = _dec((w >> np.uint32(4)) & np.uint32(63), scale) * fast_scale + delta
                f = min(max(f, -wmax), wmax)
                fq = _enc(f, inv, _urand(seed, nid, j, 1))
                tg = np.float32((w >> np.uint32(28)) & np.uint32(7)) / TAG_MAX + arousal * abs(dopamine) * e
                tg = min(max(tg, np.float32(0.0)), np.float32(1.0))
                tq = np.uint32(np.floor(tg * TAG_MAX + _urand(seed, nid, j, 2)))
                if tq > TAG_MAX:
                    tq = np.uint32(TAG_MAX)
                w = (w & ~np.uint32((63 << 4) | (7 << 28))) | (fq << np.uint32(4)) | (tq << np.uint32(28))
                if oneshot:
                    c0 = _dec((w >> np.uint32(10)) & np.uint32(63), scale) + np.float32(0.5) * delta
                    c0 = min(max(c0, -wmax), wmax)
                    w = (w & ~np.uint32(63 << 10)) | (_enc(c0, inv, _urand(seed, nid, j, 3)) << np.uint32(10))
                words[r, j] = w


# ----------------------------------------------------------------------------- sleeping

@njit(parallel=True, nogil=True, cache=True)
def k_sleep(words, rows, ids, Mpre, fast_scale, Msleep, keep0, prune_eps, tag_mult, wmax, seed,
            full, pruned_out):
    """Per-synapse night. For every synapse:
        c = Mpre @ c                         (cascade catch-up; for a dormant neuron this is ALL
                                              the nights and rests it skipped, composed into one 3x3)
        c0 += fast * fast_scale * tag        (consolidate tagged fast weights)
        c = Msleep @ c ; c0 *= keep0         (night's cascade flow + downscaling)
        prune & regrow if weak, unused, untagged (gen + 1 -> new hashed input, fresh small c0)
        fast = 0, used = 0, tag *= tag_mult  (stochastic rounding everywhere)
    If not `full`, only the catch-up is applied (fast/tag/used untouched, no pruning)."""
    scale = np.float32(wmax / Q)
    inv = np.float32(Q / wmax)
    m = np.empty((3, 3), np.float32)
    n_ = np.empty((3, 3), np.float32)
    for x in range(3):
        for y in range(3):
            m[x, y] = Mpre[x, y]
            n_[x, y] = Msleep[x, y]
    for a in prange(rows.shape[0]):
        r = rows[a]
        nid = U(ids[a])
        npr = 0
        for j in range(NB * NS):
            w = words[r, j]
            gen = w & np.uint32(15)
            f = _dec((w >> np.uint32(4)) & np.uint32(63), scale)
            a0 = _dec((w >> np.uint32(10)) & np.uint32(63), scale)
            a1 = _dec((w >> np.uint32(16)) & np.uint32(63), scale)
            a2 = _dec((w >> np.uint32(22)) & np.uint32(63), scale)
            tq = (w >> np.uint32(28)) & np.uint32(7)
            used = w >> np.uint32(31)
            c0 = m[0, 0] * a0 + m[0, 1] * a1 + m[0, 2] * a2
            c1 = m[1, 0] * a0 + m[1, 1] * a1 + m[1, 2] * a2
            c2 = m[2, 0] * a0 + m[2, 1] * a1 + m[2, 2] * a2
            if not full:
                w = (w & np.uint32(0xF00003FF)) \
                    | (_enc(min(max(c0, -wmax), wmax), inv, _urand(seed, nid, j, 10)) << np.uint32(10)) \
                    | (_enc(min(max(c1, -wmax), wmax), inv, _urand(seed, nid, j, 11)) << np.uint32(16)) \
                    | (_enc(min(max(c2, -wmax), wmax), inv, _urand(seed, nid, j, 12)) << np.uint32(22))
                words[r, j] = w
                continue
            tag = np.float32(tq) / TAG_MAX
            c0 = min(max(c0 + f * fast_scale * tag, -wmax), wmax)
            d0 = (n_[0, 0] * c0 + n_[0, 1] * c1 + n_[0, 2] * c2) * keep0
            d1 = n_[1, 0] * c0 + n_[1, 1] * c1 + n_[1, 2] * c2
            d2 = n_[2, 0] * c0 + n_[2, 1] * c1 + n_[2, 2] * c2
            big = max(abs(d0), max(abs(d1), abs(d2)))
            if big < prune_eps and used == 0 and tq == 0:
                gen = (gen + np.uint32(1)) & np.uint32(15)
                d0 = np.float32(0.1) * _urand(seed, nid, j, 13)
                d1 = np.float32(0.0)
                d2 = np.float32(0.0)
                npr += 1
            q0 = _enc(min(max(d0, -wmax), wmax), inv, _urand(seed, nid, j, 10))
            q1 = _enc(min(max(d1, -wmax), wmax), inv, _urand(seed, nid, j, 11))
            q2 = _enc(min(max(d2, -wmax), wmax), inv, _urand(seed, nid, j, 12))
            nt = np.uint32(np.floor(tag * tag_mult * TAG_MAX + _urand(seed, nid, j, 14)))
            if nt > TAG_MAX:
                nt = np.uint32(TAG_MAX)
            words[r, j] = gen | (np.uint32(32) << np.uint32(4)) | (q0 << np.uint32(10)) \
                | (q1 << np.uint32(16)) | (q2 << np.uint32(22)) | (nt << np.uint32(28))
        pruned_out[a] = npr


@njit(parallel=True, nogil=True, cache=True)
def k_init(words, first_id, seed):
    """Fresh neurons exactly as PackedPopulation.__init__: fast = c1 = c2 = 0,
    c0 = U(0, 0.1) with stochastic rounding, gen = tag = used = 0."""
    inv = np.float32(Q / 1.0)
    for a in prange(words.shape[0]):
        nid = U(first_id + a)
        for j in range(NB * NS):
            c0 = np.float32(0.1) * _urand(seed, nid, j, 20)
            q0 = _enc(c0, inv, _urand(seed, nid, j, 21))
            words[a, j] = (np.uint32(32) << np.uint32(4)) | (q0 << np.uint32(10)) \
                | (np.uint32(32) << np.uint32(16)) | (np.uint32(32) << np.uint32(22))


@njit(parallel=True, nogil=True, cache=True)
def k_checksum(words):
    """Cheap order-independent checksum (sum of mixed words), to verify round trips."""
    acc = np.zeros(words.shape[0], np.uint64)
    for a in prange(words.shape[0]):
        h = U(0)
        for j in range(words.shape[1]):
            x = U(words[a, j]) * K1 + U(j)
            x ^= x >> U(29)
            h += x
        acc[a] = h
    return acc.sum()


# ----------------------------------------------------------------------------- helpers (numpy)

def flow_matrix_3(t, steps):
    """packed_neuron._flow_matrix padded to 3x3 (unused levels stay zero)."""
    from packed_neuron import _flow_matrix
    L = t.cascade_levels
    M = np.zeros((3, 3), np.float64)
    M[:L, :L] = _flow_matrix(t, int(steps)) if steps else np.eye(L)
    return M


def night_matrices(t, pending):
    """(Mpre, Msleep, keep0) for one ordinary night after `pending` awake steps."""
    return (flow_matrix_3(t, pending).astype(np.float32),
            flow_matrix_3(t, t.sleep_flow_steps).astype(np.float32),
            np.float32(1.0 - t.downscale))


def deferred_night(t, M, pending):
    """Compose one skipped night (after `pending` awake/rest steps) onto the deferred map M.
    The night is linear for a neuron with no fast weights: c -> D F^100 F^pending c."""
    D = np.diag([1.0 - t.downscale, 1.0, 1.0])
    L = t.cascade_levels
    D[L:, L:] = 0.0
    return D @ flow_matrix_3(t, t.sleep_flow_steps) @ flow_matrix_3(t, pending) @ M

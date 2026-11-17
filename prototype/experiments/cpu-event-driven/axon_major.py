"""
Strategy 2: event-driven, AXON-MAJOR layout.

Synapses are stored grouped by their PRESYNAPTIC input ("axon") instead of by their postsynaptic
neuron and branch. A step then visits only the synapses of inputs that actually spiked.

Layout (neurons are cut into blocks of BLK neurons; BLK * branches <= 65536):

    words[j]   uint32  the same 32-bit synapse word as the baseline, but bits 0-3 (gen) are FREE:
                       the input is no longer computed from a hash, it is WHERE the word is stored
    post[j]    uint16  which (neuron-in-block, branch) the synapse sits on: nl * B + b
    off[k, i]  int64   bucket of input i in block k is words[off[k, i] : off[k, i + 1]]
    stamp[k,i] int32   fast-weight clock of that bucket (see below)

So "the storage position is the wiring" holds for the PRE side only. The POST side has to be
stored (16 bits), which costs more than the 4 gen bits it frees: 6 bytes per synapse instead of 4.

Changes to timing semantics (deliberate, behavior checked in behavior.py):
  * Fast-weight decay is lazy PER BUCKET. The baseline folds the global decay clock into every
    synapse at every modulate (and re-rounds every fast weight stochastically). Here each bucket
    remembers when it was last brought up to date (stamp) and is rescaled only when touched.
    Same expected value, less rounding noise, and untouched synapses cost nothing.
  * Rewiring moves a synapse from bucket i to bucket i' (its postsynaptic side stays). It only
    happens in sleep, so sleep rebuilds each block's buckets with a counting sort (no slack, no
    per-synapse gen counter, no wrap-around after 16 rewirings).
Everything else (forward math, eligibility from 4 steps of history, tags, one-shot writes,
cascade flow with a shared clock, consolidation, downscaling, pruning rule) is the baseline's.
"""

import common  # noqa: F401

import math

import numpy as np
from numba import njit, prange

from dm_fused import C1, C2, C3, INV24, dec6, enc6, mix64, src_of, unif
from packed_neuron import ELIG_STEPS, MAX_PENDING, Q, USED_RATE, PackedPopulation, _flow_matrix

USED = np.uint32(0x80000000)


@njit(inline="always")
def f2h(v):
    """float32 in {0} U [0.5, 2) -> float16 bits, round to nearest even (checked against NumPy)."""
    if v == np.float32(0.0):
        return np.uint16(0)
    if v < np.float32(1.0):
        m = np.rint(v * np.float32(2048.0))
        if m >= np.float32(2048.0):
            return np.uint16(15 << 10)
        return np.uint16((14 << 10) | (np.int64(m) - 1024))
    m = np.rint(v * np.float32(1024.0))
    if m >= np.float32(2048.0):
        return np.uint16(16 << 10)
    return np.uint16((15 << 10) | (np.int64(m) - 1024))


@njit(inline="always")
def gate_of(gate_bits, n, b):
    return (gate_bits[n, b >> 3] >> (7 - (b & 7))) & 1


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _am_forward(words, post, off, stamp, now, act_idx, act_x, ln_fd, wscale, b0, b1, b2, bsh, BLK, N,
                gate_bits, threshold, bthr, bonus, ctx, slope, learn, used_rate, thr_lr, target,
                rate, ph64, nthreads):
    nblk = off.shape[0]
    B = 1 << bsh
    na = act_idx.shape[0]
    for t in prange(nthreads):
        d = np.zeros(BLK * B, np.float32)
        qual = np.zeros(BLK * B, np.uint8)   # 1 = this (neuron, branch) marks active synapses "used"
        for k in range(t, nblk, nthreads):
            base = k * BLK
            nb = min(BLK, N - base)
            # 1. events: only the buckets of active inputs
            for a in range(na):
                i = act_idx[a]
                lo, hi = off[k, i], off[k, i + 1]
                if lo == hi:
                    continue
                xv = act_x[a]
                af = wscale * np.float32(math.exp(ln_fd * (now - stamp[k, i])))
                o = np.float32(32.0) * (af + b0 + b1 + b2)
                ws, ps = words[lo:hi], post[lo:hi]   # slices: no index wrap-around checks
                for jj in range(hi - lo):
                    w = ws[jj]
                    wt = (np.float32((w >> np.uint32(4)) & np.uint32(63)) * af
                          + np.float32((w >> np.uint32(10)) & np.uint32(63)) * b0
                          + np.float32((w >> np.uint32(16)) & np.uint32(63)) * b1
                          + np.float32((w >> np.uint32(22)) & np.uint32(63)) * b2 - o)
                    d[ps[jj]] += wt * xv
            # 2. somas: every neuron (homeostasis runs every step)
            for nl in range(nb):
                n = base + nl
                soma = np.float32(0.0)
                for b in range(B):
                    dv = d[nl * B + b]
                    d[nl * B + b] = np.float32(0.0)
                    if gate_of(gate_bits, n, b):
                        sp = np.float32(1.0) if dv > bthr else np.float32(0.0)
                        soma += max(dv, np.float32(0.0)) + bonus * sp
                soma *= ctx
                r = np.float32(1.0) / (np.float32(1.0) + np.exp(-slope * (soma - threshold[n])))
                rate[n] = r
                if learn:
                    threshold[n] += thr_lr * (r - target)
                    ph64[n] = (ph64[n] << np.uint64(16)) | np.uint64(f2h(np.float32(0.5) + r))
                    ok = r >= used_rate
                    for b in range(B):
                        qual[nl * B + b] = np.uint8(1) if (ok and gate_of(gate_bits, n, b)) else np.uint8(0)
            # 3. "used" flags, again only on the active buckets (branch-free)
            if learn:
                for a in range(na):
                    i = act_idx[a]
                    lo, hi = off[k, i], off[k, i + 1]
                    ws, ps = words[lo:hi], post[lo:hi]
                    for jj in range(hi - lo):
                        ws[jj] |= np.uint32(qual[ps[jj]]) << np.uint32(31)


@njit(inline="always")
def h32(x):
    """32-bit integer hash (lowbias32); vectorizes with AVX2, used as a counter-based RNG.
    Every step is cast back to uint32: Numba promotes uint32 arithmetic to 64 bits, and without the
    casts the hash silently stops wrapping and returns 'uniforms' far above 1."""
    x = np.uint32(x)
    x = np.uint32(x ^ (x >> np.uint32(16)))
    x = np.uint32(x * np.uint32(0x7FEB352D))
    x = np.uint32(x ^ (x >> np.uint32(15)))
    x = np.uint32(x * np.uint32(0x846CA68B))
    x = np.uint32(x ^ (x >> np.uint32(16)))
    return x


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _am_modulate(words, post, off, stamp, now, elig_idx, pre32, ew, coef, post32, gate_bits, ln_fd,
                 wmax, wscale, ar_da, k1, k2, bsh, BLK, oneshot, maxb, nthreads):
    """Production modulate. Two phases per bucket so the arithmetic phase vectorizes (AVX2):
    (a) gather the per-neuron factors of each event, (b) update the contiguous words.
    Differences from the exact kernel: 32-bit hash RNG and float32 tag arithmetic."""
    nblk = off.shape[0]
    bm = (1 << bsh) - 1
    ne = elig_idx.shape[0]
    q = np.float32(Q)
    inv7 = np.float32(1.0 / 7.0)
    for t in prange(nthreads):
        E = np.empty(maxb, np.float32)
        D = np.empty(maxb, np.float32)
        for k in range(t, nblk, nthreads):
            base = k * BLK
            for a in range(ne):
                i = elig_idx[a]
                lo, hi = off[k, i], off[k, i + 1]
                m = hi - lo
                if m == 0:
                    continue
                sc = np.float32(math.exp(ln_fd * (now - stamp[k, i])))
                stamp[k, i] = now
                e0, e1, e2, e3 = ew[0] * pre32[0, i], ew[1] * pre32[1, i], ew[2] * pre32[2, i], ew[3] * pre32[3, i]
                ws, ps = words[lo:hi], post[lo:hi]
                for jj in range(m):                       # (a) gathers
                    p = np.int64(ps[jj])
                    n = base + (p >> bsh)
                    g = np.float32(gate_of(gate_bits, n, p & bm))
                    e = (e0 * post32[n, 0] + e1 * post32[n, 1] + e2 * post32[n, 2] + e3 * post32[n, 3]) * g
                    E[jj] = e
                    D[jj] = (coef[n] * e) * g
                c1 = k1 + np.uint32(lo) * np.uint32(0x9E3779B9)
                c2 = k2 + np.uint32(lo) * np.uint32(0x85EBCA6B)
                for jj in range(m):                       # (b) contiguous, vectorizable
                    w = ws[jj]
                    fv = (np.float32((w >> np.uint32(4)) & np.uint32(63)) - np.float32(32.0)) * wscale
                    fv = fv * sc + D[jj]
                    fv = fv if fv > -wmax else -wmax
                    fv = fv if fv < wmax else wmax
                    u = np.float32(h32(c1 + np.uint32(jj) * np.uint32(0x9E3779B9)) >> np.uint32(8)) * INV24
                    u2 = np.float32(h32(c2 + np.uint32(jj) * np.uint32(0x85EBCA6B)) >> np.uint32(8)) * INV24
                    fq = np.floor(fv * q + u)
                    fq = fq if fq > -q else -q
                    fq = (fq if fq < q else q) + np.float32(32.0)
                    tg = np.float32((w >> np.uint32(28)) & np.uint32(7)) * inv7 + ar_da * E[jj]
                    tg = tg if tg > np.float32(0.0) else np.float32(0.0)
                    tg = tg if tg < np.float32(1.0) else np.float32(1.0)
                    tq = np.uint32(np.int32(np.floor(tg * np.float32(7.0) + u2)))
                    w = (w & np.uint32(0x8FFFFC0F)) | (np.uint32(np.int32(fq)) << np.uint32(4)) \
                        | ((tq & np.uint32(7)) << np.uint32(28))
                    ws[jj] = w
                if oneshot:                               # rare: BTSP-like write into c0
                    for jj in range(m):
                        w = ws[jj]
                        u3 = np.float32(h32((c1 ^ np.uint32(0x3C6EF372)) + np.uint32(jj) * np.uint32(0x9E3779B9))
                                        >> np.uint32(8)) * INV24
                        c0 = dec6(w, 10, wscale) + np.float32(0.5) * D[jj]
                        ws[jj] = (w & np.uint32(0xFFFF03FF)) | (enc6(c0, wmax, u3) << np.uint32(10))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _am_modulate_exact(words, post, off, stamp, now, elig_idx, pre32, ew, coef, post32, gate_bits, ln_fd,
                 wmax, wscale, ar_da, key, bsh, BLK, oneshot, use_u, u_fast, u_tag, u_one, orig, nthreads):
    nblk = off.shape[0]
    bm = (1 << bsh) - 1
    ne = elig_idx.shape[0]
    q = np.float32(Q)
    for t in prange(nthreads):
        for k in range(t, nblk, nthreads):
            base = k * BLK
            for a in range(ne):
                i = elig_idx[a]
                lo, hi = off[k, i], off[k, i + 1]
                if lo == hi:
                    continue
                sc = np.float32(math.exp(ln_fd * (now - stamp[k, i])))
                stamp[k, i] = now
                e0, e1, e2, e3 = ew[0] * pre32[0, i], ew[1] * pre32[1, i], ew[2] * pre32[2, i], ew[3] * pre32[3, i]
                for j in range(lo, hi):
                    p = np.int64(post[j])
                    n = base + (p >> bsh)
                    g = np.float32(gate_of(gate_bits, n, p & bm))
                    e = e0 * post32[n, 0]
                    e += e1 * post32[n, 1]
                    e += e2 * post32[n, 2]
                    e += e3 * post32[n, 3]
                    e = e * g
                    dl = (coef[n] * e) * g
                    w = words[j]
                    fv = (np.float32((w >> np.uint32(4)) & np.uint32(63)) - np.float32(32.0)) * wscale
                    fv = min(max(fv * sc + dl, -wmax), wmax)
                    h = mix64(key + np.uint64(j) * C1)
                    u = u_fast[orig[j]] if use_u else np.float32(h >> np.uint64(40)) * INV24
                    fq = np.floor(fv * q + u)
                    fq = min(max(fq, -q), q) + np.float32(32.0)
                    tg = np.float64((w >> np.uint32(28)) & np.uint32(7)) / 7.0 + np.float64((ar_da * e) * g)
                    tg = min(max(tg, 0.0), 1.0)
                    u2 = u_tag[orig[j]] if use_u else np.float32((h >> np.uint64(16)) & np.uint64(0xFFFFFF)) * INV24
                    tq = np.uint32(np.floor(tg * 7.0 + np.float64(u2)))
                    w = (w & np.uint32(0x8FFFFC0F)) | (np.uint32(fq) << np.uint32(4)) \
                        | ((tq & np.uint32(7)) << np.uint32(28))
                    if oneshot:
                        u3 = u_one[orig[j]] if use_u else unif(key ^ np.uint64(0x3C6EF372), j)
                        c0 = dec6(w, 10, wscale) + np.float32(0.5) * dl
                        w = (w & np.uint32(0xFFFF03FF)) | (enc6(c0, wmax, u3) << np.uint32(10))
                    words[j] = w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _am_catch_up(words, M, wmax, wscale, key, nthreads):
    n = words.shape[0]
    step = (n + nthreads * 64 - 1) // (nthreads * 64)
    for c in prange(nthreads * 64):
        lo, hi = min(n, c * step), min(n, (c + 1) * step)
        ws = words[lo:hi]          # slice: lets LLVM drop the negative-index checks and vectorize
        for jj in range(hi - lo):
            j = lo + jj
            w = ws[jj]
            l0, l1, l2 = dec6(w, 10, wscale), dec6(w, 16, wscale), dec6(w, 22, wscale)
            c0 = M[0, 0] * l0 + M[0, 1] * l1 + M[0, 2] * l2
            c1 = M[1, 0] * l0 + M[1, 1] * l1 + M[1, 2] * l2
            c2 = M[2, 0] * l0 + M[2, 1] * l1 + M[2, 2] * l2
            ws[jj] = (w & np.uint32(0xF00003FF)) | (enc6(c0, wmax, unif(key, 3 * j)) << np.uint32(10)) \
                | (enc6(c1, wmax, unif(key, 3 * j + 1)) << np.uint32(16)) \
                | (enc6(c2, wmax, unif(key, 3 * j + 2)) << np.uint32(22))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _am_sleep(words, post, off, stamp, now, reg_lo, reg_n, Ms, ln_fd, wmax, wscale, downscale, prune_eps,
              tag_decay, key, nthreads, orig, track):
    """Consolidate, flow, downscale, prune; pruned synapses move to a random new input's bucket."""
    nblk, nin = stamp.shape
    pruned = 0
    for t in prange(nthreads):
        for k in range(t, nblk, nthreads):
            blo, bhi = off[k, 0], off[k, nin]
            newpre = np.empty(bhi - blo, np.int32)
            moved = 0
            for i in range(nin):
                sc = np.float32(math.exp(ln_fd * (now - stamp[k, i])))
                stamp[k, i] = 0
                qlo, qhi = off[k, i], off[k, i + 1]
                ws = words[qlo:qhi]
                for jj in range(qhi - qlo):
                    j = qlo + jj
                    w = ws[jj]
                    tag = np.float64((w >> np.uint32(28)) & np.uint32(7)) / 7.0
                    l0 = np.float32(min(max(np.float64(dec6(w, 10, wscale))
                                            + np.float64(dec6(w, 4, wscale) * sc) * tag, -wmax), wmax))
                    l1, l2 = dec6(w, 16, wscale), dec6(w, 22, wscale)
                    c0 = Ms[0, 0] * l0 + Ms[0, 1] * l1 + Ms[0, 2] * l2
                    c1 = Ms[1, 0] * l0 + Ms[1, 1] * l1 + Ms[1, 2] * l2
                    c2 = Ms[2, 0] * l0 + Ms[2, 1] * l1 + Ms[2, 2] * l2
                    c0 *= np.float32(1.0) - downscale
                    np_ = i
                    if (max(abs(c0), abs(c1), abs(c2)) < prune_eps) and (w >> np.uint32(31)) == 0 and tag < 0.05:
                        moved += 1
                        np_ = reg_lo[k] + np.int64(mix64((key ^ np.uint64(0x5DEECE66D)) + np.uint64(j) * C1) % np.uint64(reg_n[k]))
                        c0 = np.float32(0.1) * unif(key, 5 * j + 3)
                        c1 = np.float32(0.0)
                        c2 = np.float32(0.0)
                    newpre[j - blo] = np_
                    tq = np.uint32(np.floor(tag * tag_decay * 7.0 + np.float64(unif(key, 5 * j + 4))))
                    ws[jj] = (np.uint32(32) << np.uint32(4)) \
                        | (enc6(c0, wmax, unif(key, 5 * j)) << np.uint32(10)) \
                        | (enc6(c1, wmax, unif(key, 5 * j + 1)) << np.uint32(16)) \
                        | (enc6(c2, wmax, unif(key, 5 * j + 2)) << np.uint32(22)) \
                        | ((tq & np.uint32(7)) << np.uint32(28))
            if moved:
                # Rebuild this block's buckets: stable counting sort by (new) presynaptic input.
                cnt = np.zeros(nin + 1, np.int64)
                for jj in range(bhi - blo):
                    cnt[newpre[jj] + 1] += 1
                for i in range(nin):
                    cnt[i + 1] += cnt[i]
                tw = np.empty(bhi - blo, np.uint32)
                tp = np.empty(bhi - blo, np.uint16)
                to = np.empty(bhi - blo if track else 1, np.int64)
                pos = cnt[:nin].copy()
                for jj in range(bhi - blo):
                    dst = pos[newpre[jj]]
                    pos[newpre[jj]] += 1
                    tw[dst] = words[blo + jj]
                    tp[dst] = post[blo + jj]
                    if track:
                        to[dst] = orig[blo + jj]
                for jj in range(bhi - blo):
                    words[blo + jj] = tw[jj]
                    post[blo + jj] = tp[jj]
                    if track:
                        orig[blo + jj] = to[jj]
                for i in range(nin + 1):
                    off[k, i] = blo + cnt[i]
            pruned += moved
    return pruned


@njit(parallel=True, nogil=True, cache=True)
def _hash_src(N, B, S, first_id, nin, mask, slotc, genc, words, out):
    for n in prange(N):
        idp = (np.uint64(first_id + n) + np.uint64(1)) * C1
        for b in range(B):
            for s in range(S):
                out[n, b, s] = np.int32(src_of(idp, slotc[b * S + s], genc, words[n, b, s], nin, mask))


@njit(nogil=True, cache=True)
def _build(src, words_dm, BLK, nin, words, post, off, orig):
    """Counting sort of the dendrite-major synapses into (block, input) buckets, ordered by post."""
    N, B, S = src.shape
    nblk = off.shape[0]
    pos = 0
    for k in range(nblk):
        n0, n1 = k * BLK, min(N, (k + 1) * BLK)
        cnt = np.zeros(nin + 1, np.int64)
        for n in range(n0, n1):
            for b in range(B):
                for s in range(S):
                    cnt[src[n, b, s] + 1] += 1
        for i in range(nin):
            cnt[i + 1] += cnt[i]
        for i in range(nin + 1):
            off[k, i] = pos + cnt[i]
        fill = cnt[:nin].copy()
        for n in range(n0, n1):
            for b in range(B):
                for s in range(S):
                    i = src[n, b, s]
                    j = pos + fill[i]
                    fill[i] += 1
                    words[j] = words_dm[n, b, s] & np.uint32(0xFFFFFFF0)
                    post[j] = np.uint16((n - n0) * B + b)
                    orig[j] = (n * B + b) * S + s
        pos += cnt[nin]


class AxonPopulation:
    """Event-driven population, axon-major storage. Same neuron model as PackedPopulation."""

    def __init__(self, ntype, n_neurons, n_inputs, rng, first_id=0, block=4096, wiring="hash",
                 region=None, seed=None, track_origin=False, nthreads=3, from_packed=None):
        t = self.t = ntype
        B, S = t.n_branches, t.slots
        assert B & (B - 1) == 0 and block * B <= 65536 and t.cascade_levels <= 3
        self.n, self.n_inputs, self.rng, self.first_id = n_neurons, n_inputs, rng, first_id
        self.BLK = min(block, n_neurons)
        self.bsh = B.bit_length() - 1
        self.nblk = (n_neurons + self.BLK - 1) // self.BLK
        self.nthreads = nthreads
        base = from_packed if from_packed is not None else PackedPopulation(ntype, n_neurons, n_inputs, rng, first_id)
        # Wiring: the baseline's hash (same targets as PackedPopulation) or local receptive fields
        self.reg_lo = np.zeros(self.nblk, np.int64)
        self.reg_n = np.full(self.nblk, n_inputs, np.int64)
        src = np.empty((n_neurons, B, S), np.int32)
        if wiring == "hash":
            slotc = np.arange(B * S, dtype=np.uint64) * C2
            genc = np.arange(16, dtype=np.uint64) * C3
            mask = np.uint64(n_inputs - 1) if n_inputs & (n_inputs - 1) == 0 else np.uint64(0)
            _hash_src(n_neurons, B, S, first_id, np.uint64(n_inputs), mask, slotc, genc, base.words, src)
        elif wiring == "local":  # block k draws from its own window of `region` inputs
            r = np.random.default_rng(int(rng.integers(1 << 31)))
            nwin = n_inputs // region
            self.reg_lo = (np.arange(self.nblk) % nwin) * region
            self.reg_n[:] = region
            for k in range(self.nblk):
                n0, n1 = k * self.BLK, min(n_neurons, (k + 1) * self.BLK)
                src[n0:n1] = self.reg_lo[k] + r.integers(0, region, (n1 - n0, B, S))
        else:
            raise ValueError(wiring)
        total = n_neurons * B * S
        self.words = np.empty(total, np.uint32)
        self.post = np.empty(total, np.uint16)
        self.off = np.empty((self.nblk, n_inputs + 1), np.int64)
        orig = np.empty(total, np.int64)
        _build(src, base.words, self.BLK, n_inputs, self.words, self.post, self.off, orig)
        del src
        self.orig = orig if track_origin else np.zeros(1, np.int64)
        self.track = track_origin
        self.stamp = np.zeros((self.nblk, n_inputs), np.int32)
        # Per neuron (as the baseline)
        self.gate_bits = base.gate_bits.copy()
        self.threshold = base.threshold.copy()
        self.plasticity = base.plasticity.copy()
        self.post_hist = base.post_hist.copy()
        self.age = base.age.copy()
        self.pre_hist = base.pre_hist.copy()
        self.elig_weights = base.elig_weights.astype(np.float32)
        self.pending, self.now = 0, 0
        self.fast_scale_global = 1.0  # only for reporting; real decay is per bucket
        self.ln_fd = math.log(t.fast_decay)
        self.wscale = np.float32(t.w_max / Q)
        self.key = np.uint64(seed if seed is not None else (first_id * 0x2545F4914F6CDD1D + 777) % (1 << 63))
        self.counter = 0
        self.rate_buf = np.empty(n_neurons, np.float32)
        self.events_last = 0
        del base

    def _next_key(self):
        self.counter += 1
        return np.uint64((int(self.key) + self.counter * 0x632BE59BD9B4E019) % (1 << 64))

    # ---------- acting ----------

    def forward(self, x, context=0.0, learn_trace=True, active=None):
        t = self.t
        x = np.asarray(x, np.float32)
        act = np.flatnonzero(x).astype(np.int64) if active is None else active
        M0 = _flow_matrix(t, self.pending)[0]
        b = [np.float32(self.wscale * M0[k]) if k < t.cascade_levels else np.float32(0) for k in range(3)]
        _am_forward(self.words, self.post, self.off, self.stamp, self.now, act, x[act], self.ln_fd,
                    self.wscale, b[0], b[1], b[2], self.bsh, self.BLK, self.n, self.gate_bits, self.threshold,
                    np.float32(t.branch_threshold), np.float32(t.spike_bonus),
                    np.float32(1.0 + t.context_gain * context), np.float32(t.soma_slope), bool(learn_trace),
                    np.float32(USED_RATE), np.float32(t.threshold_lr), np.float32(t.target_rate),
                    self.rate_buf, self.post_hist.view(np.uint64).reshape(-1), self.nthreads)
        if learn_trace:
            self.pre_hist[1:] = self.pre_hist[:-1].copy()
            self.pre_hist[0] = x
        return self.rate_buf * t.sign

    # ---------- learning ----------

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0, uniforms=None):
        if dopamine == 0.0:
            return
        t = self.t
        oneshot = abs(dopamine) * ach > t.oneshot_level
        if oneshot:
            self._catch_up()
        elig_idx = np.flatnonzero(self.pre_hist.any(axis=0)).astype(np.int64)  # inputs active in last 4 steps
        coef = (t.lr * self.plasticity * ach * dopamine).astype(np.float32)   # float16 math, as baseline
        pre32 = self.pre_hist.astype(np.float32)
        post32 = self.post_hist.astype(np.float32)
        if uniforms is None:
            key = int(self._next_key())
            maxb = int(np.diff(self.off, axis=1).max())
            _am_modulate(self.words, self.post, self.off, self.stamp, self.now, elig_idx, pre32, self.elig_weights,
                         coef, post32, self.gate_bits, self.ln_fd, np.float32(t.w_max), self.wscale,
                         np.float32(arousal * abs(dopamine)), np.uint32(key & 0xFFFFFFFF), np.uint32(key >> 32),
                         self.bsh, self.BLK, bool(oneshot), max(maxb, 1), self.nthreads)
            return
        # validation path: scalar kernel, baseline arithmetic order, NumPy's uniforms (indexed by origin)
        uf, ut, uo = (u.ravel() for u in uniforms)
        _am_modulate_exact(self.words, self.post, self.off, self.stamp, self.now, elig_idx, pre32,
                           self.elig_weights, coef, post32, self.gate_bits, self.ln_fd, np.float32(t.w_max),
                           self.wscale, np.float32(arousal * abs(dopamine)), self._next_key(), self.bsh, self.BLK,
                           bool(oneshot), True, uf, ut, uo, self.orig, self.nthreads)

    def tick(self):
        self.now += 1
        self.pending += 1
        self.fast_scale_global *= self.t.fast_decay
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def rest(self, steps):
        self.now += steps
        self.pending += steps
        self._catch_up()

    def _catch_up(self):
        if self.pending:
            M3 = np.zeros((3, 3), np.float32)
            L = self.t.cascade_levels
            M3[:L, :L] = _flow_matrix(self.t, self.pending)
            _am_catch_up(self.words, M3, np.float32(self.t.w_max), self.wscale, self._next_key(), self.nthreads)
            self.pending = 0

    # ---------- sleeping ----------

    def sleep(self):
        t = self.t
        self._catch_up()
        M3 = np.zeros((3, 3), np.float32)
        M3[:t.cascade_levels, :t.cascade_levels] = _flow_matrix(t, t.sleep_flow_steps)
        n = _am_sleep(self.words, self.post, self.off, self.stamp, self.now, self.reg_lo, self.reg_n, M3,
                      self.ln_fd, np.float32(t.w_max), self.wscale, np.float32(t.downscale),
                      np.float32(t.prune_eps), float(t.tag_decay), self._next_key(), self.nthreads,
                      self.orig, self.track)
        self.now = 0
        self.fast_scale_global = 1.0
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16)
        return int(n)

    # ---------- inspection ----------

    def bytes_per_neuron(self):
        per = [self.words, self.post, self.off, self.stamp, self.gate_bits, self.threshold, self.plasticity,
               self.post_hist, self.age]
        return sum(a.nbytes for a in per) / self.n

    def to_dendrite_major(self):
        """(words in baseline (N,B,S) order with gen zeroed, src, real fast weights). Needs track_origin."""
        assert self.track
        N, B, S = self.n, self.t.n_branches, self.t.slots
        w = np.empty(N * B * S, np.uint32)
        w[self.orig] = self.words
        src = np.empty(N * B * S, np.int64)
        fscale = np.empty(N * B * S, np.float32)
        for k in range(self.nblk):
            for i in range(self.n_inputs):
                lo, hi = self.off[k, i], self.off[k, i + 1]
                src[self.orig[lo:hi]] = i
                fscale[self.orig[lo:hi]] = math.exp(self.ln_fd * (self.now - self.stamp[k, i]))
        return w.reshape(N, B, S), src.reshape(N, B, S), fscale.reshape(N, B, S)


class AxonNeuron:
    """Single-neuron adapter (PackedNeuron interface) for the behavior test."""

    def __init__(self, ntype, n_inputs, rng):
        self.pop = AxonPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)), nthreads=1)

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

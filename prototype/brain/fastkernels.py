"""
Compiled inner loops (note 79, plan phase 0: speed). Same arithmetic as the numpy code they replace, done in one pass
over each cell's synapses on all CPU cores, without building temporary arrays. Sums run in a different order, so
results differ from numpy in about the 7th significant digit. LITTLEBRAIN_FAST=0 uses the original numpy code.
"""

import os

import numpy as np

ENABLED = os.environ.get("LITTLEBRAIN_FAST", "1") == "1"

try:
    from numba import njit, prange
except ImportError:   # numba missing: everything falls back to numpy
    ENABLED = False

if ENABLED:
    @njit(parallel=True, cache=True)
    def lean_drive(r, src, fast, def0, d, gain, transmit, has_transmit, floor, has_floor, a0, F, qstep, wmax, norm,
                   out, xs):
        """Branch drive of a lean synapse bank, and the inputs seen (for learning traces).
        w = clip(fast) + a0 * def0 + qstep * sum_l F[l] * d[l]   (LeanBank._signed_weights)
        w = max(w, floor) ; w *= gain ; w *= transmit             (SynapseBank.weights / drive)
        out[i, b] = norm * sum_s w * r[src]"""
        n, nb, ns = src.shape
        L = d.shape[0]
        for i in prange(n):
            for b in range(nb):
                acc = np.float32(0.0)
                for s in range(ns):
                    x = r[src[i, b, s]]
                    xs[i, b, s] = x
                    f = fast[i, b, s]
                    if f > wmax:
                        f = wmax
                    elif f < -wmax:
                        f = -wmax
                    c = np.float32(0.0)
                    for l in range(L):
                        c += np.float32(F[l]) * np.float32(d[l, i, b, s])
                    w = f + (np.float32(a0) * def0[i, b, s] + c * np.float32(qstep))
                    if has_floor and w < floor[i, b, s]:
                        w = floor[i, b, s]
                    w = w * gain[i, b, s]
                    if has_transmit:
                        w = w * transmit[i, b, s]
                    acc += w * x
                out[i, b] = acc * np.float32(norm)

    @njit(parallel=True, cache=True)
    def trace(elig, pre, xs, post, decay, offset):
        """In place: elig = elig*decay + xs*(offset+post) ; pre = pre*decay + xs   (SynapseBank.trace)"""
        n, nb, ns = xs.shape
        for i in prange(n):
            p = np.float32(offset) + post[i]
            for b in range(nb):
                for s in range(ns):
                    x = xs[i, b, s]
                    elig[i, b, s] = elig[i, b, s] * np.float32(decay) + x * p
                    pre[i, b, s] = pre[i, b, s] * np.float32(decay) + x

    @njit(parallel=True, cache=True)
    def modulate_lean(fast, elig, plasticity, factor, coop, has_open, branch_open, hetero, has_thal, thal,
                      fast_range, tag_scale, tag):
        """In place, one cell at a time (SynapseBank.modulate for lean banks with branch tags):
        delta = factor * plasticity * elig ; branch cooperativity ; branch gating ; heterosynaptic balance (separately
        for thalamic afferents) ; fast = clip(fast + delta) ; tag[i,b] = clip(tag + tag_scale * max_s elig)"""
        n, nb, ns = elig.shape
        for i in prange(n):
            delta = np.empty((nb, ns), np.float32)
            for b in range(nb):
                for s in range(ns):
                    delta[b, s] = np.float32(factor) * plasticity[i] * elig[i, b, s]
            if coop != 0.0:
                for b in range(nb):
                    tot = np.float32(0.0)
                    for s in range(ns):
                        if elig[i, b, s] > 0.05:
                            tot += delta[b, s]
                    for s in range(ns):
                        if elig[i, b, s] > 0.05:
                            delta[b, s] += np.float32(coop) * tot
            if has_open:
                for b in range(nb):
                    for s in range(ns):
                        delta[b, s] *= branch_open[i, b]
            if hetero:
                if has_thal:
                    st = np.float32(0.0)
                    sc = np.float32(0.0)
                    nt = 0
                    for b in range(nb):
                        for s in range(ns):
                            if thal[i, b, s]:
                                st += delta[b, s]
                                nt += 1
                            else:
                                sc += delta[b, s]
                    mt = st / max(nt, 1)
                    mc = sc / max(nb * ns - nt, 1)
                    for b in range(nb):
                        for s in range(ns):
                            delta[b, s] -= mt if thal[i, b, s] else mc
                else:
                    m = np.float32(0.0)
                    for b in range(nb):
                        for s in range(ns):
                            m += delta[b, s]
                    m = m / (nb * ns)
                    for b in range(nb):
                        for s in range(ns):
                            delta[b, s] -= m
            for b in range(nb):
                mx = np.float32(0.0)
                for s in range(ns):
                    f = fast[i, b, s] + delta[b, s]
                    if f > fast_range:
                        f = fast_range
                    elif f < -fast_range:
                        f = -fast_range
                    fast[i, b, s] = f
                    e = elig[i, b, s]
                    if s == 0 or e > mx:
                        mx = e
                t = tag[i, b] + np.float32(tag_scale) * mx
                tag[i, b] = min(max(t, np.float32(0.0)), np.float32(1.0))

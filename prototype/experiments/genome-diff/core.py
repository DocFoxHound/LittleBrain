"""
Genome + diff population: the packed workhorse neuron, with a procedural "genome".

Every synapse's birth state is regenerated on demand from (neuron id, slot, generation):
  - wiring (which input it listens to)  : low bits of the same 64-bit hash the packed design already uses
  - innate weight (birth c0, 0..4 steps): spare high bits of that SAME hash (no extra hashing)
  - stochastic-rounding noise           : a counter-based hash (no RNG state)

Two dynamics modes share one 32-bit word layout (same as prototype/packed_neuron.py):

  mode BASE (0)     : exactly the baseline rules. The word stores c0 = innate + learned, so the
                      genome is just the birth state. Sleep's flow / downscale / re-rounding /
                      pruning move EVERY synapse away from it, touched or not.
  mode ANCHORED (1) : the innate weight is never stored; the cascade levels hold only what
                      experience added (learned deviation, starts at 0). Effective c0 = innate + l0.
                      Zero is a fixed point of flow, downscale and stochastic rounding, so a synapse
                      that experience never touched stays bit-identical to the genome forever:
                      word == PRISTINE (a constant). Optional sleep "diff GC" rules on top.

Event-driven: forward / modulate only visit neurons that have input (exact, see notes in README).
All kernels are numba, at most 3 threads.
"""

import os

for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

import sys
from collections import deque
from pathlib import Path

import numba as nb
import numpy as np

PROTO = Path(__file__).resolve().parents[2]
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))

from neuron import WORKHORSE, NeuronType  # noqa: E402
from packed_neuron import ELIG_STEPS, MAX_PENDING, Q, TAG_MAX, USED_RATE, _flow_matrix  # noqa: E402

BASE, ANCHORED = 0, 1
B, S = 8, 32
NSYN = B * S

# 64-bit mixing constants (the first four are the baseline's wiring hash)
M1 = np.uint64(0x9E3779B97F4A7C15)
M2 = np.uint64(0xBF58476D1CE4E5B9)
M3 = np.uint64(0x94D049BB133111EB)
M4 = np.uint64(0xD6E8FEB86659FD93)
ONE = np.uint64(1)
MASK20 = np.uint64(0xFFFFF)
INV20 = 1.0 / 1048576.0

# A pristine ANCHORED word: gen 0, fast 0, l0 = l1 = l2 = 0, tag 0, used 0 (signed fields stored +32)
PRISTINE = np.uint32((32 << 4) | (32 << 10) | (32 << 16) | (32 << 22))
PERSIST_MASK = np.uint32(0x7FFFFFFF & ~(63 << 4))  # gen, c0, c1, c2, tag (fast and used are daytime-only)


# ---------------------------------------------------------------- procedural genome

@nb.njit(inline="always", cache=True)
def syn_hash(nid, j, gen):
    """Exactly the baseline's wiring hash (packed_neuron._targets) before the modulo."""
    h = ((nid + ONE) * M1) ^ (j * M2) ^ (gen * M3)
    h ^= h >> np.uint64(31)
    h *= M4
    h ^= h >> np.uint64(32)
    return h


@nb.njit(inline="always", cache=True)
def target(h, i, lay):
    """lay = [n_inputs, local, n_regions, region_size, neurons_per_col, p_neighbor_256].
    local=0: baseline wiring h % n_inputs.  local=1: own region (+ neighbours with prob p/256)."""
    if lay[1] == 0:
        return np.int64(h % np.uint64(lay[0]))
    reg = (i // lay[4]) % lay[2]
    r = np.int64((h >> np.uint64(12)) & np.uint64(0xFF))
    if r < lay[5]:
        if (h >> np.uint64(20)) & ONE:
            reg += 1
        else:
            reg -= 1
        reg = reg % lay[2]
    return reg * lay[3] + np.int64(h % np.uint64(lay[3]))


@nb.njit(inline="always", cache=True)
def innate_q(h, birth_q):
    """Birth c0 in quantum steps: stochastic round of U(0, 0.1)*Q/w_max, from spare hash bits."""
    u1 = np.float64((h >> np.uint64(24)) & MASK20) * INV20
    u2 = np.float64((h >> np.uint64(44)) & MASK20) * INV20
    return np.floor(u1 * birth_q + u2)


@nb.njit(inline="always", cache=True)
def urand(key, ctr):
    """Counter-based uniform [0,1): no RNG state, reproducible, thread-safe."""
    z = key * M1 + ctr * M3
    z = (z ^ (z >> np.uint64(30))) * M2
    z = (z ^ (z >> np.uint64(27))) * M4
    z ^= z >> np.uint64(31)
    return np.float64(z >> np.uint64(40)) * (1.0 / 16777216.0)


@nb.njit(inline="always", cache=True)
def sround(v, key, ctr):
    return np.floor(v + urand(key, ctr))


@nb.njit(inline="always", cache=True)
def clipf(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------- kernels

CH = 64  # neurons per parallel chunk


@nb.njit(parallel=True, cache=True)
def k_forward(words, idx, first_id, lay, x, frow, now, fast_t, log_fd, mode, birth_q, gcoef,
              gate, thr, rate_out, post_hist, hpos, learn, ctx, P, active_today):
    # P: [branch_threshold, spike_bonus, context_gain, soma_slope, threshold_lr, target_rate,
    #     used_rate, w_max, post_floor, post_gate]
    q2w = P[7] / 31.0
    n = idx.shape[0]
    nch = (n + CH - 1) // CH
    for c in nb.prange(nch):
        act = np.empty(NSYN, np.int16)
        for ii in range(c * CH, min(n, (c + 1) * CH)):
            i = idx[ii]
            nid = np.uint64(first_id + i)
            scale = np.exp(log_fd * (now - fast_t[i]))
            g8 = gate[i]
            soma = 0.0
            na = 0
            for b in range(B):
                drive = 0.0
                for s in range(S):
                    j = b * S + s
                    w = words[i, j]
                    h = syn_hash(nid, np.uint64(j), np.uint64(w & 15))
                    xi = x[target(h, i, lay)]
                    if xi != 0.0:
                        act[na] = j
                        na += 1
                        ce = (frow[0] * (((w >> 10) & 63) - 32.0) + frow[1] * (((w >> 16) & 63) - 32.0)
                              + frow[2] * (((w >> 22) & 63) - 32.0))
                        if mode == 1 and (w & 15) == 0:
                            ce += innate_q(h, birth_q) * gcoef
                        drive += ((((w >> 4) & 63) - 32.0) * scale + ce) * q2w * xi
                if (g8 >> b) & 1:
                    spike = P[1] if drive > P[0] else 0.0
                    soma += max(drive, 0.0) + spike
            soma *= 1.0 + P[2] * ctx
            rate = 1.0 / (1.0 + np.exp(-P[3] * (soma - thr[i])))
            rate_out[ii] = rate
            if learn:
                # eligibility's postsynaptic factor; post_gate > 0 = Hebbian gate (silent cells don't learn)
                post_hist[i, hpos] = P[8] + rate if rate >= P[9] else 0.0
                active_today[i] = 1
                if rate >= P[6]:
                    for a in range(na):
                        j = act[a]
                        if (g8 >> (j // S)) & 1:
                            words[i, j] |= np.uint32(0x80000000)
                thr[i] += np.float32(P[4] * (rate - P[5]))


@nb.njit(parallel=True, cache=True)
def k_idle_threshold(thr, evaluated, P):
    """Neurons with no input at this step: drive 0, so rate = sigmoid(-slope*thr)."""
    for i in nb.prange(thr.shape[0]):
        if not evaluated[i]:
            rate = 1.0 / (1.0 + np.exp(P[3] * thr[i]))
            thr[i] += np.float32(P[4] * (rate - P[5]))


@nb.njit(parallel=True, cache=True)
def k_modulate(words, idx, first_id, lay, pre_hist, post_hist, slots, ew, now, fast_t, log_fd,
               gate, plast, da, ach, arousal, P, lr, ctr, oneshot, mode, birth_q, m0):
    w_max = P[7]
    q2w = w_max / 31.0
    w2q = 31.0 / w_max
    n = idx.shape[0]
    nk = slots.shape[0]
    for ii in nb.prange(n):
        i = idx[ii]
        nid = np.uint64(first_id + i)
        scale = np.exp(log_fd * (now - fast_t[i]))
        g8 = gate[i]
        for j in range(NSYN):
            w = words[i, j]
            fq = ((w >> 4) & 63) - 32
            gb = (g8 >> (j // S)) & 1
            h = syn_hash(nid, np.uint64(j), np.uint64(w & 15))
            e = 0.0
            if gb:
                src = target(h, i, lay)
                for k in range(nk):
                    e += ew[k] * pre_hist[slots[k], src] * post_hist[i, slots[k]]
            if e == 0.0 and fq == 0:
                continue
            key = (nid << np.uint64(8)) | np.uint64(j)
            delta = lr * plast[i] * ach * da * e
            fast = clipf(fq * q2w * scale + delta, -w_max, w_max)
            fnew = np.int64(clipf(sround(fast * w2q, key, ctr), -31.0, 31.0)) + 32
            w = (w & ~np.uint32(63 << 4)) | np.uint32(fnew << 4)
            if e != 0.0:
                t = (w >> 28) & 7
                tv = clipf(t / 7.0 + arousal * abs(da) * e, 0.0, 1.0)
                tnew = np.int64(sround(tv * 7.0, key, ctr + np.uint64(7777)))
                w = (w & ~np.uint32(7 << 28)) | np.uint32(min(tnew, 7) << 28)
                if oneshot:  # BTSP-like write straight into the surface level
                    c0 = (((w >> 10) & 63) - 32.0) * q2w
                    g = innate_q(h, birth_q) * q2w * m0 if (mode == 1 and (w & 15) == 0) else 0.0
                    eff = clipf(g + c0 + 0.5 * delta, -w_max, w_max)
                    c0q = np.int64(clipf(sround((eff - g) * w2q, key, ctr + np.uint64(99)), -31.0, 31.0)) + 32
                    w = (w & ~np.uint32(63 << 10)) | np.uint32(c0q << 10)
            words[i, j] = w
        fast_t[i] = now


@nb.njit(parallel=True, cache=True)
def k_flow(words, idx, F, w_max, ctr, first_id, mode):
    """Apply a cascade flow matrix to the levels (catch-up). In ANCHORED mode an all-zero
    synapse is skipped: zero is a fixed point, so untouched synapses cost nothing."""
    q2w = w_max / 31.0
    w2q = 31.0 / w_max
    for ii in nb.prange(idx.shape[0]):
        i = idx[ii]
        nid = np.uint64(first_id + i)
        for j in range(NSYN):
            w = words[i, j]
            a0 = ((w >> 10) & 63) - 32
            a1 = ((w >> 16) & 63) - 32
            a2 = ((w >> 22) & 63) - 32
            if mode == 1 and a0 == 0 and a1 == 0 and a2 == 0:
                continue
            key = (nid << np.uint64(8)) | np.uint64(j)
            c0 = a0 * q2w
            c1 = a1 * q2w
            c2 = a2 * q2w
            n0 = F[0, 0] * c0 + F[0, 1] * c1 + F[0, 2] * c2
            n1 = F[1, 0] * c0 + F[1, 1] * c1 + F[1, 2] * c2
            n2 = F[2, 0] * c0 + F[2, 1] * c1 + F[2, 2] * c2
            q0 = np.int64(clipf(sround(clipf(n0, -w_max, w_max) * w2q, key, ctr), -31.0, 31.0)) + 32
            q1 = np.int64(clipf(sround(clipf(n1, -w_max, w_max) * w2q, key, ctr + np.uint64(1)), -31.0, 31.0)) + 32
            q2 = np.int64(clipf(sround(clipf(n2, -w_max, w_max) * w2q, key, ctr + np.uint64(2)), -31.0, 31.0)) + 32
            words[i, j] = (w & ~np.uint32(0x3FFFF << 10)) | np.uint32((q0 << 10) | (q1 << 16) | (q2 << 22))


@nb.njit(parallel=True, cache=True)
def k_sleep(words, idx, first_id, F, now, fast_t, log_fd, active_today, dirty, mode, birth_q, ctr,
            w_max, downscale, prune_eps, tag_decay, need_activity, min_tag, snap_q, untagged_decay, m, mn,
            prune_prob):
    """One night. ANCHORED: levels hold the learned deviation l; a gen-0 synapse's innate part is
    g*m (m = the cohort's genome-clock vector before tonight, mn after). Flow is linear, so l
    evolves on its own and the innate part follows the shared clock: untouched l = 0 stays 0."""
    q2w = w_max / 31.0
    w2q = 31.0 / w_max
    n_pruned = np.zeros(idx.shape[0], np.int64)
    for ii in nb.prange(idx.shape[0]):
        i = idx[ii]
        nid = np.uint64(first_id + i)
        scale = np.exp(log_fd * (now - fast_t[i]))
        can_prune = (need_activity == 0) or (active_today[i] != 0)
        any_diff = False
        for j in range(NSYN):
            w = words[i, j]
            gen = np.int64(w & 15)
            fq = ((w >> 4) & 63) - 32
            t = np.int64((w >> 28) & 7)
            used = (w >> 31) & 1
            a0 = ((w >> 10) & 63) - 32
            a1 = ((w >> 16) & 63) - 32
            a2 = ((w >> 22) & 63) - 32
            if mode == 1 and a0 == 0 and a1 == 0 and a2 == 0 and fq == 0 and t == 0 and gen == 0 \
                    and ((used != 0) or (not can_prune)):
                words[i, j] = PRISTINE  # untouched and cannot be pruned: nothing to do
                continue
            key = (nid << np.uint64(8)) | np.uint64(j)
            tag01 = t / 7.0
            gi = 0.0
            if mode == 1 and gen == 0:
                gi = innate_q(syn_hash(nid, np.uint64(j), np.uint64(0)), birth_q) * q2w
            c0 = a0 * q2w
            c1 = a1 * q2w
            c2 = a2 * q2w
            if t >= min_tag:  # 1. consolidate tagged fast weight into the surface (clip the total)
                c0 = clipf(gi * m[0] + c0 + fq * q2w * scale * tag01, -w_max, w_max) - gi * m[0]
            n0 = F[0, 0] * c0 + F[0, 1] * c1 + F[0, 2] * c2  # 2. let it sink
            n1 = F[1, 0] * c0 + F[1, 1] * c1 + F[1, 2] * c2
            n2 = F[2, 0] * c0 + F[2, 1] * c1 + F[2, 2] * c2
            n0 *= 1.0 - downscale  # 3. downscale
            weak = (max(abs(gi * mn[0] + n0), abs(gi * mn[1] + n1), abs(gi * mn[2] + n2)) < prune_eps) \
                and used == 0 and tag01 < 0.05 and can_prune
            if weak and prune_prob < 1.0:  # optional: turnover-limited pruning (a few % per night)
                weak = urand(key, ctr + np.uint64(11)) < prune_prob
            if weak:  # 4. prune and regrow: next generation, new target, new birth weight
                gen = (gen + 1) & 15
                n0 = 0.0
                n1 = 0.0
                n2 = 0.0
                gi = 0.0
                if mode == 0 or gen != 0:  # BASE, or a regrown synapse: its birth weight is stored
                    n0 = innate_q(syn_hash(nid, np.uint64(j), np.uint64(gen)), birth_q) * q2w
                else:
                    gi = innate_q(syn_hash(nid, np.uint64(j), np.uint64(0)), birth_q) * q2w
                n_pruned[ii] += 1
            tnew = min(np.int64(sround(tag01 * tag_decay * 7.0, key, ctr + np.uint64(5))), 7)
            if untagged_decay > 0.0 and tnew == 0:  # fringe: untagged learning fades toward the genome
                n0 *= 1.0 - untagged_decay
                n1 *= 1.0 - untagged_decay
                n2 *= 1.0 - untagged_decay
            g0 = gi * mn[0]
            e0 = clipf(g0 + n0, -w_max, w_max)  # keep effective c0 inside the baseline's range
            q0 = np.int64(clipf(sround((e0 - g0) * w2q, key, ctr), -31.0, 31.0))
            q1 = np.int64(clipf(sround(clipf(n1, -w_max, w_max) * w2q, key, ctr + np.uint64(1)), -31.0, 31.0))
            q2 = np.int64(clipf(sround(clipf(n2, -w_max, w_max) * w2q, key, ctr + np.uint64(2)), -31.0, 31.0))
            if snap_q > 0 and tnew == 0 and gen == 0 and abs(q0) <= snap_q and abs(q1) <= snap_q \
                    and abs(q2) <= snap_q:
                q0 = 0  # diff GC: a tiny untagged residue snaps back to the genome
                q1 = 0
                q2 = 0
            nw = np.uint32(gen | (32 << 4) | ((q0 + 32) << 10) | ((q1 + 32) << 16) | ((q2 + 32) << 22)
                           | (tnew << 28))
            words[i, j] = nw
            if nw != PRISTINE:
                any_diff = True
        dirty[i] = 1 if any_diff else 0
    return n_pruned.sum()


@nb.njit(parallel=True, cache=True)
def k_genome_q(n, first_id, lay, birth_q, out):
    """Birth c0 (gen 0) for every synapse, in quanta. Used for BASE birth and for diff measurement."""
    for i in nb.prange(n):
        nid = np.uint64(first_id + i)
        for j in range(NSYN):
            out[i, j] = np.uint8(innate_q(syn_hash(nid, np.uint64(j), np.uint64(0)), birth_q))


@nb.njit(parallel=True, cache=True)
def k_genome_q_gen(words, first_id, birth_q, out):
    """Innate c0 for every synapse at its CURRENT generation."""
    for i in nb.prange(words.shape[0]):
        nid = np.uint64(first_id + i)
        for j in range(NSYN):
            out[i, j] = np.uint8(innate_q(syn_hash(nid, np.uint64(j), np.uint64(words[i, j] & 15)), birth_q))


# ---------------------------------------------------------------- population

def layout_global(n_inputs):
    return np.array([n_inputs, 0, 1, n_inputs, 1 << 30, 0], np.int64)


def layout_local(n_regions, region_size, neurons_per_col, p_neighbor=0.1):
    return np.array([n_regions * region_size, 1, n_regions, region_size, neurons_per_col,
                     int(round(p_neighbor * 256))], np.int64)


class GenomePop:
    """Population with a procedural genome. Interface mirrors packed_neuron.PackedPopulation,
    plus `active` (which neurons can possibly have input) for event-driven stepping."""

    def __init__(self, ntype: NeuronType, n: int, lay, seed=0, first_id=0, mode=ANCHORED,
                 need_activity=False, min_tag=0, snap_q=0, untagged_decay=0.0, post_floor=0.5, post_gate=0.0,
                 genome_clock=True, prune_prob=1.0):
        assert ntype.cascade_levels <= 3 and ntype.n_branches == B and ntype.slots == S
        t = self.t = ntype
        self.n, self.lay, self.first_id, self.mode = n, np.asarray(lay, np.int64), first_id, mode
        self.n_inputs = int(self.lay[0])
        self.need_activity, self.min_tag, self.snap_q = int(need_activity), int(min_tag), int(snap_q)
        self.untagged_decay = float(untagged_decay)
        # Genome clock: an untouched synapse's innate weight evolves by the same linear flow /
        # downscale as everything else, identically for the whole cohort, so it is one shared 3x3
        # matrix (M) instead of per-synapse state. genome_clock=False: innate weight is constant.
        self.genome_clock = bool(genome_clock)
        self.prune_prob = float(prune_prob)
        self.M = np.eye(3)
        self.birth_q = 0.1 * Q / t.w_max
        self.P = np.array([t.branch_threshold, t.spike_bonus, t.context_gain, t.soma_slope, t.threshold_lr,
                           t.target_rate, USED_RATE, t.w_max, post_floor, post_gate], np.float64)
        self.log_fd = float(np.log(t.fast_decay))
        self.ew = (t.elig_decay ** np.arange(ELIG_STEPS)).astype(np.float64)
        self.ctr = np.uint64(seed) * np.uint64(1_000_003) + np.uint64(12345)

        if mode == BASE:
            gq = np.empty((n, NSYN), np.uint8)
            k_genome_q(n, first_id, self.lay, self.birth_q, gq)
            self.words = (PRISTINE & ~np.uint32(63 << 10)) | ((gq.astype(np.uint32) + 32) << 10)
            del gq
        else:
            self.words = np.full((n, NSYN), PRISTINE, np.uint32)
        self.gate = np.full(n, 0xFF, np.uint8)
        self.threshold = np.full(n, t.threshold_init, np.float32)
        self.plasticity = np.full(n, np.float16(t.plasticity_start), np.float32)
        self.post_hist = np.zeros((n, ELIG_STEPS), np.float32)
        self.fast_t = np.zeros(n, np.float64)
        self.active_today = np.zeros(n, np.uint8)
        self.dirty = np.zeros(n, np.uint8)
        self.age = np.zeros(n, np.uint16)
        self.pre_hist = np.zeros((ELIG_STEPS, self.n_inputs), np.float32)
        self.hist_count = 0
        self.recent = deque(maxlen=ELIG_STEPS)  # neuron sets evaluated at the last few trace steps
        self.pending = 0
        self.now = 0.0
        self.all_idx = np.arange(n, dtype=np.int64)
        self.last_pruned = 0
        self._F3cache = {}

    # ---------- helpers
    def _next_ctr(self):
        self.ctr = self.ctr + np.uint64(104729)
        return self.ctr

    def _F3(self, steps):
        if steps not in self._F3cache:
            L = self.t.cascade_levels
            F = np.zeros((3, 3))
            F[:L, :L] = _flow_matrix(self.t, steps)
            if len(self._F3cache) > 64:
                self._F3cache.clear()
            self._F3cache[steps] = F
        return self._F3cache[steps]

    def _flow_idx(self):
        if self.mode == BASE:
            return self.all_idx
        return np.flatnonzero(self.dirty | self.active_today).astype(np.int64)

    def _catch_up(self):
        if self.pending:
            F = self._F3(self.pending)
            k_flow(self.words, self._flow_idx(), F, self.t.w_max, self._next_ctr(), self.first_id, self.mode)
            if self.genome_clock:
                self.M = F @ self.M
            self.pending = 0

    def _gcoef(self, frow):
        return float(frow @ self.M[:, 0]) if self.genome_clock else 1.0

    # ---------- acting
    def forward(self, x, context=0.0, learn_trace=True, active=None):
        """active: neuron indices that can receive input (None = all). Exact as long as every
        neuron with an active input is included."""
        idx = self.all_idx if active is None else np.asarray(active, np.int64)
        x = np.asarray(x, np.float32)
        rate = np.empty(idx.shape[0], np.float64)
        frow = self._F3(self.pending)[0].copy() if self.pending else np.array([1.0, 0.0, 0.0])
        hpos = self.hist_count % ELIG_STEPS
        k_forward(self.words, idx, self.first_id, self.lay, x, frow, self.now, self.fast_t, self.log_fd,
                  self.mode, self.birth_q, self._gcoef(frow), self.gate, self.threshold, rate, self.post_hist, hpos,
                  bool(learn_trace), float(context), self.P, self.active_today)
        if learn_trace:
            if active is not None and idx.shape[0] < self.n:
                ev = np.zeros(self.n, np.bool_)
                ev[idx] = True
                k_idle_threshold(self.threshold, ev, self.P)
            self.pre_hist[hpos] = x
            self.hist_count += 1
            self.recent.append(idx)
        return rate * self.t.sign

    # ---------- learning
    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0, subset=None):
        """subset: neurons the modulator reaches (None = broadcast to all, as in the baseline)."""
        if dopamine == 0.0 or self.hist_count == 0:
            return
        t = self.t
        oneshot = abs(dopamine) * ach > t.oneshot_level
        if oneshot:
            self._catch_up()
        if len(self.recent) == 1 or all(r is self.all_idx for r in self.recent):
            idx = self.recent[-1]
        else:
            idx = np.unique(np.concatenate(list(self.recent)))
        if subset is not None:
            idx = np.intersect1d(idx, subset, assume_unique=False).astype(np.int64)
        k = min(self.hist_count, ELIG_STEPS)
        slots = np.array([(self.hist_count - 1 - q) % ELIG_STEPS for q in range(k)], np.int64)
        k_modulate(self.words, idx, self.first_id, self.lay, self.pre_hist, self.post_hist, slots, self.ew[:k],
                   self.now, self.fast_t, self.log_fd, self.gate, self.plasticity, float(dopamine), float(ach),
                   float(arousal), self.P, t.lr, self._next_ctr(), bool(oneshot), self.mode, self.birth_q,
                   float(self.M[0, 0]) if self.genome_clock else 1.0)

    def tick(self):
        self.now += 1.0
        self.pending += 1
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def rest(self, steps):
        self.now += steps
        self.pending += steps
        self._catch_up()

    # ---------- sleeping
    def sleep(self):
        t = self.t
        self._catch_up()
        # ANCHORED: only neurons with a diff or with activity today need any work -- unless silent
        # neurons may prune (need_activity off), in which case every neuron has to be checked.
        idx = self._flow_idx() if (self.mode == ANCHORED and self.need_activity) else self.all_idx
        F = self._F3(t.sleep_flow_steps)
        if self.genome_clock:
            Mn = np.diag([1.0 - t.downscale, 1.0, 1.0]) @ F @ self.M
        else:
            Mn = np.eye(3)
        n = k_sleep(self.words, idx, self.first_id, F, self.now, self.fast_t,
                    self.log_fd, self.active_today, self.dirty, self.mode, self.birth_q, self._next_ctr(),
                    t.w_max, t.downscale, t.prune_eps, t.tag_decay, self.need_activity, self.min_tag,
                    self.snap_q, self.untagged_decay, self.M[:, 0].copy(), Mn[:, 0].copy(), self.prune_prob)
        self.M = Mn
        self.fast_t[:] = self.now
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.hist_count = 0
        self.recent.clear()
        self.active_today[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16).astype(np.float32)
        self.last_pruned = int(n)
        return int(n)

    # ---------- genome comparison
    def genome_words(self):
        """The genome as a full word array (BASE: birth state; ANCHORED: the constant PRISTINE)."""
        if self.mode == ANCHORED:
            return np.full_like(self.words, PRISTINE)
        gq = np.empty((self.n, NSYN), np.uint8)
        k_genome_q(self.n, self.first_id, self.lay, self.birth_q, gq)
        return (PRISTINE & ~np.uint32(63 << 10)) | ((gq.astype(np.uint32) + 32) << 10)


class GenomeNeuron:
    """One-neuron adapter with the PackedNeuron interface, for demo_single_neuron.run(...)."""

    def __init__(self, ntype, n_inputs, rng, **cfg):
        first_id = int(rng.integers(1 << 31))  # same draw as PackedNeuron -> same wiring
        seed = int(rng.integers(1 << 40))
        self.pop = GenomePop(ntype, 1, layout_global(n_inputs), seed=seed, first_id=first_id, **cfg)

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
        return int(np.count_nonzero(self.pop.words != self.pop.genome_words()) * 8 + 16)


def make_adapter(**cfg):
    def make(ntype, n_inputs, rng):
        return GenomeNeuron(ntype, n_inputs, rng, **cfg)
    return make

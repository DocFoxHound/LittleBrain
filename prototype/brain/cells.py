"""
Cell dynamics for the brain prototype (float reference, vectorized per group of same-type cells).

- SynapseBank:      learnable excitatory synapses (learning rules v2, cascade memory).
- LeanBank:         the same synapses stored lean (notes/24): seed + differences, per-type precision,
                    day buffer, per-branch tags. Used when LEAN=1 or a group is built with lean=True.
- ExcitatoryGroup:  pyramidal / stellate / thalamic relay / dentate granule cells.
- InhibitoryGroup:  PV, SST, VIP, TRN. Driven by pooled local activity; synapses fixed.
- Neuromodulators:  dopamine (VTA), acetylcholine (basal forebrain), noradrenaline (locus coeruleus),
                    serotonin (raphe). Functional models of what these cell groups compute.
"""

import os

import numpy as np

from celltypes import CellType
from lean_neuron import _gamma_bits, _hash_uniform
from backend import like, sround, to_host, xpof
from packed_neuron import MAX_PENDING, _flow_matrix  # cascade flow as one small matrix (linear dynamics)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SynapseBank:
    """N cells x B branches x S slots of learnable excitatory synapses."""

    def __init__(self, n, branches, slots, rules, rng, heterosynaptic=True):
        self.t = rules
        self.heterosynaptic = heterosynaptic
        shape = (n, branches, slots)
        self.src = np.zeros(shape, np.int32)          # index into the network's rate vector (set by wiring)
        self.long = np.zeros(shape, bool)             # from another area / structure
        self.gain = np.ones(shape, np.float32)        # fixed per-synapse strength (e.g. strong driver synapses)
        self.fast = np.zeros(shape, np.float32)
        self.cascade = np.zeros((rules.cascade_levels,) + shape, np.float32)
        self.cascade[0] = rng.uniform(0.0, 0.1, shape)
        self.elig = np.zeros(shape, np.float32)
        self.pre = np.zeros(shape, np.float32)         # input-only trace (for plateau / one-shot learning)
        self.tag = np.zeros(shape, np.float32)
        self.wave_gate = None                          # optional (N,B,S) long-range gate set by waves
        self.ach_suppress = self.transmit = None       # per-synapse ACh muting (federation, Hasselmo)
        self.learn_mask = None                         # optional (N,B,S): which synapses may learn (lesion tests)
        self.branch_open = None                        # optional (N,B): closed branches don't learn (context gating)

    def weights(self):
        return np.clip(self.fast, -self.t.w_max, self.t.w_max) + self.cascade[0]

    def drive(self, r):
        xs = r[self.src]
        if self.wave_gate is not None:
            xs = xs * self.wave_gate
        w = self.weights() * self.gain
        if self.transmit is not None:
            w = w * self.transmit   # transmission only: the synapse still learns from its input
        return (w * xs).sum(axis=2), xs

    def trace(self, xs, post):
        self.elig = self.elig * self.t.elig_decay + xs * (self.t.post_offset + post)[:, None, None]
        self.pre = self.pre * self.t.elig_decay + xs

    def modulate(self, plasticity, dopamine, ach, arousal, plateau=None):
        delta = self.t.lr * plasticity[:, None, None] * ach * dopamine * self.elig
        if self.t.arousal_gated:  # amygdala: noradrenaline gates whether learning happens at all
            delta = delta * arousal
        if self.t.cooperativity:
            # Clustered plasticity: plasticity proteins are shared along a branch, so the more synapses
            # learn together on one branch, the more each of them gains (branches compete; Govindarajan 2006)
            active = (self.elig > 0.05).astype(np.float32)
            branch_total = (delta * active).sum(2, keepdims=True)
            delta = delta + self.t.cooperativity * branch_total * active
        if self.learn_mask is not None:
            delta = delta * self.learn_mask
        if self.branch_open is not None:
            delta = delta * self.branch_open[:, :, None]
        if self.heterosynaptic:
            # Heterosynaptic plasticity: when some of a cell's synapses strengthen, the rest weaken a
            # little, so its total input stays balanced (prevents runaway excitation in recurrent circuits)
            delta = delta - delta.mean(axis=(1, 2), keepdims=True)
        self.fast = np.clip(self.fast + delta, -self.t.fast_range, self.t.fast_range)
        self.mark(arousal * abs(dopamine) * self.elig)
        if abs(dopamine) * ach > self.t.oneshot_level and plateau is not None:
            # BTSP: a dendritic plateau potentiates every recently active input in one shot, whether or
            # not the cell fired before (Bittner et al. 2017). Only the few cells that have a plateau
            # (plateau, 0..1 per cell) do this; writing it into every cell erased all selectivity (notes/26).
            self.oneshot(0.3 * np.sign(dopamine) * self.pre * (plasticity * plateau)[:, None, None])

    def set_surface(self, value):
        """Set the surface level (tests and hand-built circuits)."""
        self.cascade[0][:] = value

    def mark(self, x):
        """Tag synapses that mattered (consolidated at sleep)."""
        self.tag = np.clip(self.tag + x, 0.0, 1.0)

    def oneshot(self, step):
        self.cascade[0] = np.clip(self.cascade[0] + step, -self.t.w_max, self.t.w_max)

    def tick(self, steps=1):
        self.fast *= self.t.fast_decay ** steps
        self.cascade = np.tensordot(like(self.cascade, _flow_matrix(self.t, steps)), self.cascade, axes=1)

    def sleep(self):
        t = self.t
        self.cascade[0] = np.clip(self.cascade[0] + t.consolidation_rate * self.fast * self.tag,
                                  -t.w_max, t.w_max)
        self.fast[:] = 0.0
        self.elig[:] = 0.0
        self.cascade = np.tensordot(like(self.cascade, _flow_matrix(t, t.sleep_flow_steps)), self.cascade, axes=1)
        self.cascade[0] *= 1.0 - t.downscale
        self.tag[:] = 0.0


class LeanBank(SynapseBank):
    """The same synapses, stored lean (notes/24, prototype/lean_neuron.py):
      - long-term: cascade = M @ seed default + d * step. d is a small integer per level at the cell type's
        precision (q); M is one shared matrix of all global linear operations (flow, downscaling), applied
        lazily. Untouched synapses keep d = 0 and cost nothing to store.
      - day buffer: fast weight, eligibility, input trace (as before, cleared or decayed at sleep).
      - chemical tags: one per branch, not per synapse.
      - gentle sleep reset: surface-only differences of 1 step with empty deep levels go back to the seed.
      - norm: with more (or fewer) synapses than the cell type's reference layout, each one counts
        proportionally less (more), so thresholds mean the same at every size.
    Wiring (src, long, gain) is regenerable from the brain's seed and pathway table, so not stored."""

    def __init__(self, n, branches, slots, rules, rng, heterosynaptic=True, q=7, norm=1.0):
        self.t, self.heterosynaptic = rules, heterosynaptic
        shape = (n, branches, slots)
        self.src = np.zeros(shape, np.int32)
        self.long = np.zeros(shape, bool)
        self.gain = np.ones(shape, np.float32)
        self.fast = np.zeros(shape, np.float32)
        self.elig = np.zeros(shape, np.float32)
        self.pre = np.zeros(shape, np.float32)
        self.wave_gate = self.learn_mask = self.branch_open = None
        self.ach_suppress = self.transmit = None
        L = rules.cascade_levels
        self.ids = (np.arange(n, dtype=np.uint64) + np.uint64(rng.integers(1 << 40)))[:, None, None]
        self.slot = np.arange(branches * slots, dtype=np.uint64).reshape(1, branches, slots)
        self.def0 = 0.1 * _hash_uniform(self.ids, self.slot, np.zeros(shape, np.uint8))
        # Structural turnover with seeded choice (notes/24): rewired synapses keep only (generation, choice)
        self.gen = np.zeros(shape, np.uint8)
        self.born = np.zeros(shape, np.uint16)
        self.path = np.zeros(shape, np.uint8)   # which pathway a synapse belongs to (set by the wiring)
        self.paths = []
        self.turnover, self.grace, self.night = 0.0, 3, 0
        self.to_rewire = None
        self.M = np.eye(L)
        self.d = np.zeros((L,) + shape, np.int8)
        # Diagnostic switches (env): LEAN_Q (precision override), LEAN_TAGS=synapse, LEAN_RESET=0
        q = int(os.environ.get("LEAN_Q", q))
        self.branch_tags = os.environ.get("LEAN_TAGS", "branch") == "branch"
        self.reset = os.environ.get("LEAN_RESET", "1") == "1"
        self.q, self.qstep, self.norm, self.rng = q, rules.w_max / q, norm, rng
        self.pending = 0
        self.tag = np.zeros((n, branches) if self.branch_tags else shape, np.float32)

    def _base(self):
        """The evolved seed default, all levels (M is a small host matrix)."""
        return like(self.def0, self.M[:, 0])[:, None, None, None] * self.def0

    @property
    def cascade(self):
        self._catch_up()
        return self._base() + self.d * self.qstep

    def set_surface(self, value):
        """Set the surface level (tests and hand-built circuits)."""
        c = self.cascade
        c[0][:] = value
        self._set(c)

    def _set(self, c):
        """Nonlinear write of new levels: untouched synapses keep their difference exactly."""
        c = np.clip(c, -self.t.w_max, self.t.w_max)
        df = (c - self._base()) / self.qstep
        new = np.clip(sround(df, self.rng), -self.q, self.q).astype(np.int8)
        self.d = np.where(np.abs(df - self.d) < 1e-3, self.d, new)

    def _linear(self, A):
        self.M = A @ self.M
        df = np.tensordot(like(self.d, A), self.d.astype(np.float32), axes=1)
        self.d = np.clip(sround(df, self.rng), -self.q, self.q).astype(np.int8)

    def _catch_up(self):
        if self.pending:
            p, self.pending = self.pending, 0
            self._linear(_flow_matrix(self.t, p).astype(np.float64))

    def weights(self):
        F = _flow_matrix(self.t, self.pending)[0]
        c0 = float(F @ self.M[:, 0]) * self.def0 + np.tensordot(like(self.d, F), self.d.astype(np.float32), axes=1) * self.qstep
        return np.clip(self.fast, -self.t.w_max, self.t.w_max) + c0

    def drive(self, r):
        d, xs = super().drive(r)
        return d * self.norm, xs

    def mark(self, x):
        self.tag = np.clip(self.tag + (x.max(axis=2) if self.branch_tags else x), 0.0, 1.0)

    def oneshot(self, step):
        c = self.cascade
        c[0] += step
        self._set(c)

    def tick(self, steps=1):
        self.fast *= self.t.fast_decay ** steps
        self.pending += steps
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def sleep(self):
        t = self.t
        tag = self.tag[:, :, None] if self.branch_tags else self.tag
        mattered = (tag >= 0.05) & (np.abs(self.fast) >= 0.05)  # heterosynaptic plasticity nudges every synapse a little
        add = t.consolidation_rate * self.fast * tag
        if np.any(add):
            c = self.cascade
            c[0] += add
            self._set(c)
        self.fast[:] = 0.0
        self.elig[:] = 0.0
        D = np.eye(len(self.M))
        D[0, 0] = 1.0 - t.downscale
        self._catch_up()
        self._linear(D @ _flow_matrix(t, t.sleep_flow_steps).astype(np.float64))
        if self.reset:
            small = (np.abs(self.d[0]) <= 1) & ~np.any(self.d[1:], axis=0)  # gentle surface-only reset
            self.d[0][small] = 0
        self.tag[:] = 0.0
        self.night += 1
        if self.turnover:
            # Spine turnover: each cell's weakest slots are replaced, except synapses that mattered today
            # and newborn ones still in their grace period. The brain picks the new inputs (seeded choice).
            young = (self.gen > 0) & (self.night - self.born.astype(np.int64) < self.grace)
            strength = np.where(mattered | young | (self.gen >= 15), np.inf, np.abs(self.cascade).max(axis=0))
            n, k = strength.shape[0], int(self.turnover * strength[0].size)
            if k:
                flat = strength.reshape(n, -1)
                pick = np.argpartition(flat, k, axis=1)[:, :k]
                ok = np.isfinite(np.take_along_axis(flat, pick, axis=1))
                xp = xpof(flat)
                cells = xp.repeat(xp.arange(n), k)[ok.ravel()]
                b, sl = np.unravel_index(pick.ravel()[ok.ravel()], strength.shape[1:])
                self.to_rewire = (cells, b, sl)

    def rewire(self, where, new_src, choice):
        """New input for these slots: the seed's default strength for (new generation, choice)."""
        self.gen[where] += 1
        self.born[where] = self.night
        self.src[where] = new_src
        key = self.gen[where].astype(np.uint64) * np.uint64(256) + choice.astype(np.uint64)
        self.def0[where] = 0.1 * _hash_uniform(self.ids[where[0], 0, 0], self.slot[0][where[1], where[2]], key)
        self.d[(slice(None),) + where] = 0
        for a in (self.fast, self.elig, self.pre):
            a[where] = 0.0
        self.to_rewire = None

    def storage_bits(self):
        """Long-term bits (all of notes/24's ideas): gap-coded changed synapses + their levels."""
        L, n = self.d.shape[0], self.d.shape[1]
        b = int(np.ceil(np.log2(2 * self.q + 1)))
        d = to_host(self.d).reshape(L, n, -1)
        valued = (d != 0).any(axis=0)
        rewired = to_host(self.gen).reshape(n, -1) > 0
        changed = valued | rewired
        settled = valued & (d.max(axis=0) == d.min(axis=0))
        bits = sum(_gamma_bits(np.diff(np.concatenate([[-1], np.flatnonzero(changed[i])]))) for i in range(n))
        return (bits + int(changed.sum()) + (b + 1) * int(settled.sum()) + (L * b + 1) * int((valued & ~settled).sum())
                + 10 * int(rewired.sum()))  # rewired: generation (4 bits) + choice of 64 (6 bits)


class ExcitatoryGroup:
    """Pyramidal (IT/PT/CT), stellate, thalamic relay or dentate granule cells of one type."""

    def __init__(self, ctype: CellType, n: int, rng: np.random.Generator, lean=None, suite=None, caps=None,
                 brain_neurons=None):
        """lean: store synapses lean (default: env LEAN=1). suite: "standard", "low" or "biological" sizes
        the cell from biology (suites.py; implies lean); None keeps the cell type's reference layout.
        caps: (basal, apical) most synapses worth having in this network (federation: ~4 contacts per
        partner within reach); None = no cap. brain_neurons: size of the simulated brain (biological suite)."""
        self.c, self.n = ctype, n
        suite = suite or os.environ.get("SUITE") or None
        suite = None if suite == "none" else suite
        lean = bool(suite) or (os.environ.get("LEAN") == "1" if lean is None else lean)
        nb, sb, na, sa = ctype.n_basal, ctype.basal_slots, ctype.n_apical, ctype.apical_slots
        q = 7
        if lean:
            import suites  # here, not at the top: suites imports celltypes and lean_neuron
            lv = suites.levels(ctype)
            q = suites.Q[lv or 16]
        per_branch = sb
        if suite:
            total = suites.synapses(ctype, suite, brain_neurons)
            ref = nb * sb + na * sa
            a_total = int(round(total * na * sa / ref)) if na else 0
            b_total = max(1, total - a_total)
            per_branch = suites.BRANCH_SLOTS[suite]
        else:
            b_total, a_total = nb * sb, na * sa
        if caps is not None:  # scaled to the network: no more synapses than distinct partners can use
            # One factor for both compartments, set by the tighter one, so the cell keeps its proportions
            f = min(1.0, caps[0] / b_total, caps[1] / a_total if na else 1.0)
            b_total = max(1, int(b_total * f))
            a_total = max(1, int(a_total * f)) if na else 0
        if suite or caps is not None:
            sb = min(per_branch, b_total)
            nb = max(1, b_total // sb)
            if na:
                sa = min(per_branch, max(1, a_total))
                na = max(1, a_total // sa)
        self.nb, self.bnorm = nb, ctype.n_basal / nb   # more branches each count proportionally less
        self.anorm = (ctype.n_apical / na) if na else 1.0
        if lean:
            self.basal = LeanBank(n, nb, sb, ctype.rules, rng, q=q, norm=ctype.basal_slots / sb)
            self.apical = LeanBank(n, na, sa, ctype.rules, rng, q=q, norm=ctype.apical_slots / sa) if na else None
            if ctype.learns:  # adult spine turnover: a few % per day (Holtmaat & Svoboda 2009)
                for b in self.banks():
                    b.turnover = float(os.environ.get("TURNOVER", "0.03"))
        else:
            self.basal = SynapseBank(n, nb, sb, ctype.rules, rng)
            self.apical = SynapseBank(n, na, sa, ctype.rules, rng) if na else None
        self.rate = np.zeros(n, np.float32)
        self.adapt_state = np.zeros(n, np.float32)
        self.threshold = np.full(n, ctype.threshold, np.float32)
        self.inh_gain = np.ones(n, np.float32)         # PV inhibition strength, tuned by inhibitory plasticity
        self.scale = np.ones(n, np.float32)            # intrinsic excitability: sharpness relative to input spread
        self.avg_rate = np.full(n, ctype.target_rate, np.float32)
        self.plasticity = np.full(n, ctype.rules.plasticity_start, np.float32)
        self.h = np.zeros(n, np.float32)               # relay cells: T-type calcium de-inactivation
        self.branch_gain = np.ones((n, nb), np.float32)  # learned branch excitability
        self.branch_open = np.ones((n, nb), np.float32)  # context gating by SST-like inhibition
        self.branch_trace = np.zeros((n, nb), np.float32)
        self.prev_drive = np.zeros(n, np.float32)
        self._xs = None

    def compute(self, r, pv_in=0.0, sst_in=0.0, trn_in=0.0, gain=1.0, apical_mode=1.0, external=0.0,
                allow_burst=True, dend_in=0.0, ais_in=0.0, volume_in=0.0, cf=0.0):
        """Target rate from inputs. pv/sst/trn_in: inhibition per cell. gain: wave excitability.
        allow_burst: relay cells can burst only when brainstem ACh/NE is low (drowsy, asleep)."""
        c, rules = self.c, self.c.rules
        d, xs_b = self.basal.drive(r)
        spikes = (d > rules.branch_threshold).astype(np.float32)
        branch = (np.maximum(d, 0.0) + rules.spike_bonus * spikes) * self.branch_gain * self.branch_open
        self._spikes = spikes * self.branch_open
        # SST cells target dendrites: they gate the apical tuft strongly and basal branches mildly
        # Dendrite-targeting cells (bistratified, HIPP, LTS) gate basal branches; SST gates mostly apical
        basal = self.bnorm * branch.sum(axis=1) / (1.0 + 0.3 * sst_in + 1.5 * dend_in) + external
        apical, xs_a = 0.0, None
        if self.apical is not None:
            da, xs_a = self.apical.drive(r)
            apical = np.maximum(self.anorm * da.sum(axis=1), 0.0) * apical_mode / (1.0 + 2.0 * sst_in)
        soma = basal * (1.0 + c.apical_gain * apical) + c.apical_drive * apical
        # Layer 5 "BAC firing": basal and apical input arriving together trigger a calcium burst
        if c.bac_gain:
            soma = soma + c.bac_gain * c.threshold * sigmoid(4 * (apical - 1.0)) * sigmoid(4 * (basal / c.threshold - 0.5))
        # PV cells target the cell body: divisive (gain control) plus subtractive inhibition
        soma = soma / (1.0 + 0.5 * pv_in) - self.inh_gain * pv_in - 2.0 * trn_in
        # Neurogliaform / ivy cells: slow GABA-B volume inhibition, both subtractive and divisive
        soma = soma / (1.0 + volume_in) - volume_in
        soma = soma * gain - self.adapt_state + c.baseline
        # Thalamic relay: after a stretch of hyperpolarization, a depolarizing input causes a burst
        if c.burst_gain:
            v = soma - self.threshold
            self.h += 0.1 * ((v < -0.5).astype(np.float32) * allow_burst - self.h)
            soma = soma + c.burst_gain * self.h * np.maximum(soma - self.prev_drive, 0.0)
            self.h *= np.where(soma > self.threshold, 0.5, 1.0)  # a burst inactivates the channels
            self.prev_drive = soma.astype(np.float32)
        self._xs = (xs_b, xs_a)
        self.soma = soma
        out = sigmoid(c.slope * self.scale * (soma - self.threshold))
        if c.family == "purkinje":
            out = np.maximum(out, cf)  # climbing fiber input: a complex spike
        # Chandelier / basket "pinceau" synapses on the axon initial segment veto the output itself
        return out / (1.0 + 3.0 * ais_in)

    def update(self, target, learn=True, mask=None):
        """mask: cells to update this step (wave scheduling); others keep their state."""
        c = self.c
        step = c.alpha * (target.astype(np.float32) - self.rate)
        self.rate += step if mask is None else step * mask
        self.adapt_state = c.adapt_decay * self.adapt_state + c.adapt * self.rate
        if learn and c.learns:
            xs_b, xs_a = self._xs
            if mask is not None:
                xs_b = xs_b * mask[:, None, None]
                xs_a = None if xs_a is None else xs_a * mask[:, None, None]
            self.basal.trace(xs_b, self.rate)
            self.branch_trace = c.rules.elig_decay * self.branch_trace + self._spikes
            if self.apical is not None:
                self.apical.trace(xs_a, self.rate)
        # Homeostasis: threshold drifts to keep the cell near its target rate (sparse coding)
        self.threshold += c.rules.threshold_lr * (self.rate - c.target_rate)
        self.avg_rate += c.rules.rate_avg_tau * (self.rate - self.avg_rate)

    def inhibitory_plasticity(self, pv_in, eta=0.01):
        """Vogels-style: PV inhibition onto a cell grows when the cell is above its target rate."""
        self.inh_gain = np.maximum(self.inh_gain + eta * (self.rate - self.c.target_rate) * pv_in, 0.0)

    def banks(self):
        return [b for b in (self.basal, self.apical) if b is not None]

    def modulate(self, dopamine, ach, arousal):
        """D2 cells (dopamine_sign -1): a dopamine burst weakens and a dip strengthens their synapses."""
        if self.c.rules.encoding_learning:
            # Hippocampus: NMDA plasticity during encoding (high ACh) needs no reward signal; the pattern
            # arrives several steps after the dopamine pulse (notes/26)
            dopamine = dopamine + self.c.rules.encoding_learning * max(ach - 0.5, 0.0)
        if self.c.learns and dopamine:
            self.basal.branch_open = self.branch_open
            for b in self.banks():
                b.modulate(self.plasticity, dopamine * self.c.dopamine_sign, ach, arousal, self.plateau())
            if self.c.rules.branch_lr:
                # Branch strength plasticity: a branch that spiked before a learning signal becomes
                # more excitable as a whole (Losonczy et al. 2008)
                self.branch_gain = np.clip(self.branch_gain + self.c.rules.branch_lr * dopamine * ach
                                           * self.branch_trace * self.branch_open, 0.5, 2.0)

    def plateau(self):
        """Which cells have a dendritic plateau now (BTSP, CA1): strong entorhinal input on the tuft together
        with firing. Stand-in: the cell's own rate, which the winners-take-all rule keeps sparse."""
        return self.rate

    def climbing_fiber(self, cf):
        """Purkinje learning (supervised): parallel fibers active together with a climbing-fiber error
        signal are weakened (LTD); active without it, slowly strengthened (LTP)."""
        b, r = self.basal, self.c.rules
        cf = like(b.elig, cf)[:, None, None] * np.ones_like(b.elig)
        b.fast = np.clip(b.fast + r.lr * self.plasticity[:, None, None] * b.elig * (0.1 * (1 - cf) - cf),
                         -r.fast_range, r.fast_range)
        b.mark(b.elig * cf)

    def tick(self, steps=1):
        for b in self.banks():
            b.tick(steps)

    def sleep(self):
        for b in self.banks():
            b.sleep()
        r = self.c.rules
        self.plasticity = r.plasticity_floor + (self.plasticity - r.plasticity_floor) * r.maturation


class InhibitoryGroup:
    """Cells driven by pooled input whose synapses don't learn: interneurons (PV, SST, VIP, TRN, ...),
    inhibitory projection cells, tonic pacemaking cells (pallidum, TAN, orexin) and slow neuroendocrine
    cells. Negative drive = inhibition."""

    def __init__(self, ctype: CellType, n: int):
        self.c, self.n = ctype, n
        self.rate = np.zeros(n, np.float32)
        self.u = np.zeros(n, np.float32)      # SST: facilitation of incoming excitatory synapses
        self.h = np.zeros(n, np.float32)      # rebound after inhibition (TAN)
        self.level = np.zeros(n, np.float32)  # slow output (volume GABA-B, hormones)

    def step(self, drive, steps=1):
        """steps > 1: this group runs on a slower clock; rates catch up as if held for `steps` steps."""
        c = self.c
        drive = like(self.rate, drive) + c.baseline
        if c.burst_gain:  # inhibition de-inactivates a rebound current; release triggers a burst
            self.h += 0.2 * ((drive < 0).astype(np.float32) - self.h)
            drive = drive + c.burst_gain * self.h * (drive >= 0)
            self.h *= np.where(drive >= 0, 0.5, 1.0)
        if c.facilitation:
            # Facilitating synapses: SST cells respond to *sustained* activity, ramping up over time
            self.u += c.facilitation * (np.clip(drive, 0, None) - self.u)
            drive = drive * (0.3 + self.u)
        if c.linear:  # fast-spiking: firing keeps rising with drive, so inhibition tracks excitation
            target = np.maximum(drive - c.threshold, 0.0) * c.slope / 4.0
        else:
            target = sigmoid(c.slope * (drive - c.threshold))
        self.rate += (1 - (1 - c.alpha) ** steps) * (target.astype(np.float32) - self.rate)
        if c.slow:  # output outlasts firing: slow GABA-B volume inhibition, hormone release
            k = c.slow ** steps
            self.level = k * self.level + (1 - k) * self.rate
            return self.level
        return self.rate


class ModulatorGroup:
    """Dopamine, noradrenaline, serotonin, acetylcholine cells: a baseline rate moved by a signal
    (reward prediction error, surprise, novelty). Serotonin is slow."""

    def __init__(self, ctype: CellType, n: int):
        self.c, self.n = ctype, n
        self.rate = np.full(n, ctype.baseline, np.float32)

    def step(self, signal, steps=1):
        c = self.c
        target = np.clip(c.baseline + 0.7 * like(self.rate, signal), 0.0, 1.0)
        a = (1 - c.slow) if c.slow else c.alpha
        self.rate += (1 - (1 - a) ** steps) * (target - self.rate)
        return self.rate


class OliveGroup:
    """Inferior olive: a slow subthreshold oscillation; fires a climbing-fiber spike when an error
    arrives (most likely near the oscillation peak). Output is the cerebellum's teaching signal."""

    def __init__(self, ctype: CellType, n: int, rng=None):
        self.c, self.n = ctype, n
        rng = rng or np.random.default_rng(0)
        self.phase = rng.random(n) * 2 * np.pi
        self.rate = np.zeros(n, np.float32)

    def step(self, error, inhibition=0.0):
        c = self.c
        self.phase += 2 * np.pi / c.period
        v = like(self.rate, error) + 0.15 * np.cos(self.phase) - inhibition
        self.rate = (v > c.threshold).astype(np.float32)
        return self.rate


class PacemakerGroup:
    """Intrinsic oscillators: SCN clock (~24 h, entrained by light) and septal theta pacemakers."""

    def __init__(self, ctype: CellType, n: int):
        self.c, self.n = ctype, n
        self.phase = np.zeros(n)
        self.rate = np.full(n, 1.0, np.float32)

    def step(self, light=0.0, k=0.02):
        # Light advances or delays the clock depending on when it arrives (phase response curve)
        self.phase += 2 * np.pi / self.c.period - k * like(self.phase, light, self.phase.dtype) * np.sin(self.phase)
        self.rate = (0.5 + 0.5 * np.cos(self.phase)).astype(np.float32)
        return self.rate


def make_group(ctype: CellType, n: int, rng=None, lean=None, suite=None, caps=None, brain_neurons=None):
    """The right group class for a cell type's family."""
    rng = rng or np.random.default_rng(0)
    if ctype.family in ("pyramidal", "stellate", "relay", "granule", "msn", "purkinje", "rebound", "learning_inh"):
        return ExcitatoryGroup(ctype, n, rng, lean=lean, suite=suite, caps=caps, brain_neurons=brain_neurons)
    if ctype.family == "dopamine":
        return ModulatorGroup(ctype, n)
    if ctype.family == "olive":
        return OliveGroup(ctype, n, rng)
    if ctype.family == "pacemaker":
        return PacemakerGroup(ctype, n)
    return InhibitoryGroup(ctype, n)


class Neuromodulators:
    """
    Functional models of the neuromodulatory nuclei (their cells compute these signals):
      VTA dopamine:        reward prediction error = reward - expected (expectation learned per context)
      Basal forebrain ACh: high for novelty / mismatch (encode), low for familiar input (recall)
      Locus coeruleus NE:  arousal; phasic burst on surprise
      Raphe serotonin:     slow average of reward (patience / time horizon)
    """

    def __init__(self):
        self.expected = {}      # VTA: learned reward expectation per context
        self.serotonin = 0.0
        self.ach = 0.5
        self.ne = 0.2

    def dopamine(self, context, reward, lr=0.2):
        v = self.expected.get(context, 0.0)
        rpe = reward - v
        self.expected[context] = v + lr * rpe
        self.serotonin += 0.02 * (reward - self.serotonin)
        return float(np.clip(rpe, -1.0, 1.0))

    def acetylcholine(self, mismatch, midpoint=0.3, slope=8.0):
        self.ach += 0.5 * (float(sigmoid(slope * (mismatch - midpoint))) - self.ach)
        return self.ach

    def noradrenaline(self, surprise, midpoint=0.5, slope=6.0):
        self.ne += 0.5 * (0.1 + 0.9 * float(sigmoid(slope * (surprise - midpoint))) - self.ne)
        return self.ne

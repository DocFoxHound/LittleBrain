"""
Lean neuron (notes/24): the packed workhorse neuron, stored as seed + changes.

Long-term state per synapse is a small integer *difference* d (one per cascade level) from the
synapse's default. The default is never stored: it is regenerated from a hash of (neuron id, slot,
generation) and evolved by one shared 3 x 3 matrix M that records every global operation since
birth (cascade flow, sleep downscaling). Both are linear, so a synapse nothing ever happened to
keeps d = 0 forever and costs nothing to store.

    cascade = M @ default + d * step          step = w_max / q   (q sets the precision)

Day buffer (day-only state, never in long-term storage): fast weight, tag, used flag. Read live
while awake; folded into d and cleared at sleep.

Chemical (shared) bookkeeping, chem=True: tags are kept per branch, not per synapse (synaptic tagging
and capture works at the branch level). The used flag stays per synapse but is day-only.
Eligibility is already shared: rebuilt from per-input and per-neuron firing histories (as in
packed_neuron), never stored per synapse.

Seeded choice (rewiring="seeded"): a rewiring synapse doesn't take one random new input. The seed
offers `choices` candidates (default 64); the synapse picks the one that fired most together with its
branch's strongest inputs today, and stores only which one (6 bits for 64). Inputs that fire together end up on the
same branch (clustering), which is what lets branch spikes work. Co-firing is counted in a shared,
day-only table (a "chemical" field for the whole population, cleared at sleep).

Arrays here are a dense *working copy* (what the GPU would unpack on the fly). The stored size is
measured by `storage()`, which encodes the long-term state in the formats of notes/24.
"""

import zlib

import numpy as np

from neuron import NeuronType
from packed_neuron import ELIG_STEPS, MAX_PENDING, TAG_MAX, USED_RATE, _flow_matrix, _sround, _targets

FAST_Q = 31  # day buffer fast weight: 6 bits, as in the packed neuron
KEY_STRIDE = 256  # (generation, choice) -> one hash key; up to 256 choices
THEME = 8        # a branch's "theme": its strongest inputs
LEVEL_BITS = {31: 6, 15: 5, 7: 4, 3: 3, 2: 3, 1: 2}


def _hash_uniform(ids, slot, gen):
    """Deterministic uniform [0, 1) per synapse: the seed's default strength."""
    h = ((ids + np.uint64(7)) * np.uint64(0xD1342543DE82EF95)) \
        ^ (slot * np.uint64(0x9E3779B97F4A7C15)) \
        ^ (gen.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9))
    h ^= h >> np.uint64(29)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(32)
    return (h >> np.uint64(40)).astype(np.float32) / float(1 << 24)


def _gamma_bits(gaps):
    """Elias-gamma code length for gaps >= 1."""
    return int((2 * np.floor(np.log2(gaps)) + 1).sum()) if len(gaps) else 0


class LeanPopulation:
    def __init__(self, ntype: NeuronType, n_neurons: int, n_inputs: int, rng: np.random.Generator,
                 first_id: int = 0, q: int = 31, chem: bool = False, dendrites: str = "spikes",
                 branch_floor: float = 0.2, rewiring: str = "random", turnover: bool = False,
                 choices: int = 64, newborn: float = 0.1, guide: str = "cofire"):
        """dendrites: "linear" (no branch spikes), "spikes" (as packed_neuron), or "spikes+learning"
        (plasticity concentrates on branches that recently spiked; branch_floor elsewhere)."""
        assert ntype.cascade_levels == 3
        t = self.t = ntype
        self.n, self.n_inputs, self.rng, self.first_id = n_neurons, n_inputs, rng, first_id
        self.q, self.chem, self.step = q, chem, t.w_max / q
        self.dendrites, self.branch_floor, self.rewiring = dendrites, branch_floor, rewiring
        self.guide = guide  # seeded choice counts all co-firing ("cofire") or co-firing signed by dopamine ("value")
        self.choices = choices  # seeded choice: candidates offered per rewiring
        self.newborn = newborn  # seed default strength scale of rewired synapses (birth synapses: 0.1)
        self.turnover = turnover  # constant spine turnover: each night the weakest rewire_budget of slots are replaced
        B, S = t.n_branches, t.slots
        shape = (n_neurons, B, S)

        # Long-term: difference from the evolved default (stored only where nonzero)
        self.d = np.zeros((3,) + shape, np.int8)
        self.gen = np.zeros(shape, np.uint8)
        self.choice = np.zeros(shape, np.uint8)  # seeded choice: which candidate (0 = the seed's default)
        self.born = np.zeros(shape, np.uint16)   # night of last rewiring (kept in the rewiring list only)
        self.grace, self.anchor = 0, False       # grace: nights a newborn synapse is safe from turnover
        if rewiring == "seeded":  # shared, day-only co-firing counts
            self.coact = np.zeros((n_inputs, n_inputs), np.float32)
            self.in_count = np.zeros(n_inputs, np.float32)
        self.M = np.eye(3)  # shared: every global linear operation since birth

        # Day buffer
        self.fast = np.zeros(shape, np.float32)
        self.tag = np.zeros((n_neurons, B) if chem else shape, np.float32)
        self.used = np.zeros(shape, bool)
        self.branch_trace = np.zeros((n_neurons, B), np.float32)

        # Per neuron, as packed_neuron
        self.gate_bits = np.packbits(np.ones((n_neurons, B), bool), axis=1)
        self.threshold = np.full(n_neurons, t.threshold_init, np.float32)
        self.plasticity = np.full(n_neurons, t.plasticity_start, np.float16)
        self.post_hist = np.zeros((n_neurons, ELIG_STEPS), np.float16)
        self.age = np.zeros(n_neurons, np.uint16)
        self.avg_rate = np.full(n_neurons, t.target_rate, np.float16)

        # Shared
        self.pre_hist = np.zeros((ELIG_STEPS, n_inputs), np.float16)
        self.elig_weights = (t.elig_decay ** np.arange(ELIG_STEPS)).astype(np.float32)
        self.slot = np.arange(B * S, dtype=np.uint64).reshape(1, B, S)
        self.ids = np.arange(first_id, first_id + n_neurons, dtype=np.uint64)[:, None, None]
        self.pending = 0
        self.fast_scale = 1.0
        self._regen()

    # ---------- the seed ----------

    def _regen(self):
        """Working-copy cache of what the seed says (recomputed, never stored)."""
        key = self._key(self.gen, self.choice)
        self.src = _targets(self.ids, self.slot, key, self.n_inputs)
        self.def0 = np.where(self.gen > 0, self.newborn, 0.1) * _hash_uniform(self.ids, self.slot, key)

    @staticmethod
    def _key(gen, choice):
        return gen.astype(np.uint64) * np.uint64(KEY_STRIDE) + choice

    def _levels(self):
        return self.M[:, 0, None, None, None] * self.def0 + self.d * self.step

    def _set_levels(self, c):
        """Nonlinear writes: store the new levels as a difference from the evolved default."""
        c = np.clip(c, -self.t.w_max, self.t.w_max)
        base = self.M[:, 0, None, None, None] * self.def0
        df = (c - base) / self.step
        new = np.clip(_sround(df, self.rng), -self.q, self.q).astype(np.int8)
        self.d = np.where(np.abs(df - self.d) < 1e-3, self.d, new)  # untouched synapses keep d exactly

    def _linear(self, A):
        """A global linear operation: the default absorbs it via M; d follows exactly (0 stays 0)."""
        self.M = A @ self.M
        df = np.tensordot(A.astype(np.float32), self.d.astype(np.float32), axes=1)
        self.d = np.clip(_sround(df, self.rng), -self.q, self.q).astype(np.int8)

    def _gate(self):
        return np.unpackbits(self.gate_bits, axis=1, count=self.t.n_branches).astype(np.float32)

    def _own(self, a):
        return a[:, :, None] if self.chem else a

    def _elig(self):
        e = np.zeros(self.src.shape, np.float32)
        for k in range(ELIG_STEPS):
            e += self.elig_weights[k] * self.pre_hist[k][self.src] * self.post_hist[:, k, None, None]
        return e

    def _weights(self):
        c0 = np.tensordot(_flow_matrix(self.t, self.pending)[0], self._levels(), axes=1)
        return np.clip(self.fast * self.fast_scale, -self.t.w_max, self.t.w_max) + c0

    def _catch_up(self):
        if self.pending:
            self._linear(_flow_matrix(self.t, self.pending).astype(np.float64))
            self.pending = 0

    # ---------- acting ----------

    def forward(self, x, context=0.0, learn_trace=True, post=None):
        """post: firing rate imposed by population competition (None = the neuron's own rate)."""
        t = self.t
        xs = x[self.src]
        drive = (self._weights() * xs).sum(axis=2)
        gate = self._gate()
        spike = (drive > t.branch_threshold).astype(np.float32)
        if self.dendrites == "linear":
            spike[:] = 0.0
        soma = ((np.maximum(drive, 0.0) + t.spike_bonus * spike) * gate).sum(axis=1) * (1.0 + t.context_gain * context)
        rate = 1.0 / (1.0 + np.exp(-t.soma_slope * (soma - self.threshold)))
        if learn_trace:
            fire = rate if post is None else post
            self.pre_hist = np.roll(self.pre_hist, 1, axis=0)
            self.pre_hist[0] = x
            self.post_hist = np.roll(self.post_hist, 1, axis=1)
            self.post_hist[:, 0] = t.post_offset + fire
            self.avg_rate = (self.avg_rate + t.rate_avg_tau * (fire - self.avg_rate)).astype(np.float16)
            used = (xs > 0) & (gate[:, :, None] > 0) & (np.asarray(fire)[:, None, None] >= USED_RATE)
            self.used |= used
            if self.rewiring == "seeded" and self.guide == "cofire":
                on = np.flatnonzero(x > 0)
                self.coact[np.ix_(on, on)] += 1.0
                self.in_count[on] += 1.0
            self.branch_trace = t.elig_decay * self.branch_trace + spike * gate
            self.threshold += t.threshold_lr * (rate - t.target_rate)
        return rate * t.sign

    # ---------- learning ----------

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        if dopamine == 0.0:
            return
        t = self.t
        gate = self._gate()[:, :, None]
        elig = self._elig() * gate
        if self.dendrites == "spikes+learning":
            f = self.branch_floor
            elig *= (f + (1.0 - f) * np.minimum(self.branch_trace, 1.0))[:, :, None]
        delta = t.lr * self.plasticity[:, None, None] * ach * dopamine * elig
        fast = np.clip(self.fast * self.fast_scale + delta, -t.fast_range, t.fast_range)
        self.fast = _sround(fast * (FAST_Q / t.fast_range), self.rng) * (t.fast_range / FAST_Q)
        self.fast_scale = 1.0
        mark = arousal * abs(dopamine) * (elig.max(axis=2) if self.chem else elig)
        self.tag = _sround(np.clip(self.tag + mark, 0.0, 1.0) * TAG_MAX, self.rng) / TAG_MAX
        if self.rewiring == "seeded" and self.guide == "value":
            # Value-guided: inputs that fired together *when it mattered*, signed by the outcome
            on = np.flatnonzero(self.pre_hist[0] > 0)
            self.coact[np.ix_(on, on)] += dopamine
            self.in_count[on] += 1.0
        if abs(dopamine) * ach > t.oneshot_level:
            self._catch_up()
            c = self._levels()
            c[0] += 0.5 * delta
            self._set_levels(c)

    def tick(self):
        self.pending += 1
        self.fast_scale *= self.t.fast_decay
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def rest(self, steps):
        self.pending += steps
        self.fast_scale *= self.t.fast_decay ** steps
        self._catch_up()

    # ---------- sleeping: fold the day buffer into long-term storage ----------

    def sleep(self, reset_below=0):
        """reset_below: idea 6, differences of at most this many steps go back to the seed."""
        t = self.t
        self._catch_up()
        tag = self._own(self.tag)
        # 1. Consolidate tagged fast weights (only these synapses' differences change)
        add = t.consolidation_rate * self.fast * self.fast_scale * tag
        touched = add != 0
        if touched.any():
            c0 = self._levels()[0]
            new = np.clip(c0 + add, -t.w_max, t.w_max)
            d0 = self.d[0].astype(np.float32) + _sround((new - c0) / self.step, self.rng)
            self.d[0] = np.where(touched, np.clip(d0, -self.q, self.q), self.d[0]).astype(np.int8)
        # 2-3. Let them sink, then downscale: both linear, absorbed by M
        D = np.diag([1.0 - t.downscale, 1.0, 1.0])
        self._linear(D @ _flow_matrix(t, t.sleep_flow_steps).astype(np.float64))
        if reset_below:
            # Gentle version (reset_below < 0): only surface-level noise, where the deep levels hold nothing
            small = (np.abs(self.d).max(axis=0) <= abs(reset_below)) if reset_below > 0 else \
                (np.abs(self.d[0]) <= -reset_below) & (self.d[1] == 0) & (self.d[2] == 0)
            self.d[:, small] = 0
        # 4. Prune and regrow, as packed_neuron
        c = self._levels()
        mattered = (tag >= 0.05) & (self.fast != 0) if self.chem else tag >= 0.05
        weak = (np.abs(c).max(axis=0) < t.prune_eps) & ~self.used & ~mattered
        if t.need_gated_rewiring:
            need = np.clip(1.0 - self.avg_rate.astype(np.float32) / t.target_rate, 0.0, 1.0)
            budget = (t.rewire_budget * t.n_branches * t.slots * need * self.plasticity).astype(np.int64)
            priority = np.where(weak, self.rng.random(weak.shape), np.inf).reshape(self.n, -1)
            rank = np.argsort(np.argsort(priority, axis=1), axis=1).reshape(weak.shape)
            weak &= rank < budget[:, None, None]
        if self.turnover:
            young = (self.gen > 0) & (self.age[:, None, None] - self.born < self.grace)
            mattered = mattered | young
            strength = np.where(mattered, np.inf, np.abs(c).max(axis=0)).reshape(self.n, -1)
            rank = np.argsort(np.argsort(strength, axis=1), axis=1).reshape(c.shape[1:])
            weak = (rank < int(t.rewire_budget * t.n_branches * t.slots)) & ~mattered
        weak &= self.gen < 15
        n = int(weak.sum())
        if n:
            # Newborn synapse: whatever the seed says for its new generation (evolved default), so
            # rewiring costs only a gen-list entry, not a stored difference
            self.gen[weak] += 1
            self.born[weak] = self.age[np.nonzero(weak)[0]]
            if self.rewiring == "seeded":
                self._choose(weak)
            self._regen()
            self.d[:, weak] = 0
        if self.rewiring == "seeded":
            self.coact[:] = 0.0
            self.in_count[:] = 0.0
        # 5. Empty the day buffer; mature
        self.fast[:] = 0.0
        self.tag *= t.tag_decay
        self.used[:] = False
        self.branch_trace[:] = 0.0
        self.fast_scale = 1.0
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16)
        return n

    def _choose(self, weak):
        """Seeded choice: each rewiring synapse takes the candidate that fired most with its branch's theme."""
        ni, bi, si = np.nonzero(weak)
        c0 = self._levels()[0]
        top = np.argpartition(-c0, THEME, axis=2)[:, :, :THEME]
        theme = np.take_along_axis(self.src, top, axis=2)[ni, bi]                 # (w, THEME)
        k = np.arange(self.choices, dtype=np.uint8)[None, :]
        key = self._key(self.gen[ni, bi, si][:, None], k)
        slot = self.slot[0][bi, si].astype(np.uint64)[:, None]
        ids = self.ids[ni, 0, 0][:, None]
        cand = _targets(ids, slot, key, self.n_inputs)                           # (w, CHOICES)
        if self.anchor:  # one memory per branch: follow the branch's single strongest input
            theme = theme[np.arange(len(ni)), np.argmax(c0[ni, bi][np.arange(len(ni))[:, None], top[ni, bi]], axis=1)][:, None]
        score = self.coact[cand[:, :, None], theme[:, None, :]].sum(axis=2) / (self.in_count[cand] + 1.0)
        self.choice[ni, bi, si] = np.argmax(score, axis=1).astype(np.uint8)

    # ---------- size ----------

    def day_buffer_bits(self):
        """Today's buffer, per neuron: touched synapses (index + fast weight [+ tag + used])."""
        S = self.t.n_branches * self.t.slots
        idx = int(np.ceil(np.log2(S)))
        entries = ((self.fast != 0) | self.used).reshape(self.n, -1).sum(axis=1)
        per = idx + 6 + 1 + (0 if self.chem else 3)
        shared = self.t.n_branches * 4 if self.chem else 0
        return entries * per + shared, entries

    def storage(self):
        """Long-term bytes per neuron in each format of notes/24 (mean over the population)."""
        n, S = self.n, self.t.n_branches * self.t.slots
        b = LEVEL_BITS[self.q]
        gb = 4 + (int(np.ceil(np.log2(self.choices))) if self.rewiring == "seeded" else 0)  # rewiring entry: generation (+ choice)
        d = self.d.reshape(3, n, S)
        gen = self.gen.reshape(n, S)
        changed = (d != 0).any(axis=0) | (gen != 0)
        rewired = gen != 0
        valued = (d != 0).any(axis=0)  # rewired synapses at their new default store no values
        settled = valued & (d.max(axis=0) == d.min(axis=0))
        settled1 = valued & (d.max(axis=0) - d.min(axis=0) <= 1)
        fixed = 8 * (2 + 2 + 2 + 2 + self.gate_bits.shape[1])  # threshold, plasticity, avg_rate, age (f16), gates
        nc, nr, ns, nv = changed.sum(1), rewired.sum(1), settled.sum(1), valued.sum(1)

        gaps = []
        for i in range(n):
            ix = np.flatnonzero(changed[i])
            gaps.append(_gamma_bits(np.diff(np.concatenate([[-1], ix]))))
        gaps = np.array(gaps)

        out = {
            "changed %": 100 * changed.mean(),
            "settled % of changed": 100 * ns.sum() / max(nv.sum(), 1),
            "settled±1 % of changed": 100 * settled1.sum() / max(nv.sum(), 1),
            "rewired %": 100 * rewired.mean(),
            "packed today (32 bits each)": (32 * S + fixed) / 8,
            "seed+changes, 32 bits each": (S + 32 * nc + fixed) / 8,
            "+1 day-only out": (S + (3 * b + 4) * nc + fixed) / 8,
            "+2 settled": (S + nc + (b + 1) * ns + (3 * b + 1) * (nv - ns) + 4 * nc + fixed) / 8,
            "+4 gen list": (S + nc + (b + 1) * ns + (3 * b + 1) * (nv - ns) + gb * nr + fixed) / 8,
            "+5 gap-coded": (gaps + nc + (b + 1) * ns + (3 * b + 1) * (nv - ns) + gb * nr + fixed) / 8,
        }
        # +5 with a real compressor: serialize the changed entries and zlib them
        blob = []
        for i in range(n):
            ix = np.flatnonzero(changed[i])
            blob.append(np.diff(np.concatenate([[-1], ix])).astype(np.uint16).tobytes())
            blob.append(d[:, i, np.flatnonzero(valued[i])].astype(np.int8).tobytes())
            blob.append(gen[i, rewired[i]].tobytes())
        out["+5 zlib"] = (len(zlib.compress(b"".join(blob), 9)) * 8 + fixed * n) / 8 / n
        return {k: float(np.mean(v)) for k, v in out.items()}


class LeanNeuron:
    """One-neuron population with the interface of neuron.Neuron, for the story tests."""

    def __init__(self, ntype, n_inputs, rng, q=31, chem=False, reset_below=0, dendrites="spikes"):
        self.pop = LeanPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)), q=q, chem=chem,
                                  dendrites=dendrites)
        self.reset_below = reset_below

    def forward(self, x, context=0.0, learn_trace=True):
        return float(self.pop.forward(x, context, learn_trace)[0])

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        self.pop.modulate(dopamine, ach, arousal)

    def tick(self):
        self.pop.tick()

    def rest(self, steps):
        self.pop.rest(steps)

    def sleep(self):
        return self.pop.sleep(reset_below=self.reset_below)

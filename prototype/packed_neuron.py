"""
Packed version of the workhorse neuron: the same design, in the smallest memory we can manage.

Every synapse is ONE 32-bit word:

    bits  0-3   gen   (4)  rewiring generation; the target input is recomputed from a hash
    bits  4-9   fast  (6)  signed, -31..31  (x fast_range/31; the effective weight is capped at w_max)
    bits 10-15  c0    (6)  cascade level 0 (surface)
    bits 16-21  c1    (6)  cascade level 1
    bits 22-27  c2    (6)  cascade level 2 (deepest)
    bits 28-30  tag   (3)  0..7  ("this mattered")
    bit  31     used  (1)  used since last sleep

What is NOT stored per synapse any more:
  - target index: hash(neuron id, slot, gen). Rewiring = gen + 1.
  - eligibility:  rebuilt on demand from the last ELIG_STEPS steps of input activity (shared,
                  one row per presynaptic neuron) and of this neuron's firing (a few bytes per
                  neuron). Older history (weight 0.5^8 and below) is dropped.
  - usage count:  replaced by the 1-bit "used" flag.
  - per-step fast decay and cascade flow: one shared clock for the whole population; the
    flow is linear, so any number of steps is applied at once with a precomputed matrix.

Low-precision values are written with stochastic rounding, so small changes are kept
on average instead of being rounded away.

Neurons live in a population (one big array), not as separate Python objects:
a neuron is just an index. That removes per-object overhead.
"""

from functools import lru_cache

import numpy as np

from neuron import NeuronType

FIELDS = {"gen": (0, 4), "fast": (4, 6), "c0": (10, 6), "c1": (16, 6), "c2": (22, 6),
          "tag": (28, 3), "used": (31, 1)}
LEVELS = ("c0", "c1", "c2")
Q = 31            # signed 6-bit fields hold -31..31, stored as 1..63
TAG_MAX = 7
USED_RATE = 0.05  # firing rate above which an active synapse counts as "used"
MAX_PENDING = 512
ELIG_STEPS = 8    # recent steps the eligibility trace remembers exactly (4 was too short for v2 rules)


def _get(w, name):
    off, bits = FIELDS[name]
    return (w >> np.uint32(off)) & np.uint32((1 << bits) - 1)


def _set(w, name, v):
    off, bits = FIELDS[name]
    m = np.uint32((1 << bits) - 1)
    return (w & ~np.uint32(int(m) << off)) | ((v.astype(np.uint32) & m) << np.uint32(off))


def _sround(x, rng):
    """Stochastic rounding: 0.3 becomes 1 with probability 0.3, else 0."""
    return np.floor(x + rng.random(x.shape, dtype=np.float32))


def _enc(v, w_max, rng):
    return (np.clip(_sround(v * (Q / w_max), rng), -Q, Q) + 32).astype(np.uint32)


def _dec(s, w_max):
    return (s.astype(np.float32) - 32.0) * (w_max / Q)


def _targets(ids, slot, gen, n_inputs):
    """Which input each synapse listens to, recomputed from (neuron id, slot, generation)."""
    h = ((ids + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)) \
        ^ (slot * np.uint64(0xBF58476D1CE4E5B9)) \
        ^ (gen.astype(np.uint64) * np.uint64(0x94D049BB133111EB))
    h ^= h >> np.uint64(31)
    h *= np.uint64(0xD6E8FEB86659FD93)
    h ^= h >> np.uint64(32)
    return (h % np.uint64(n_inputs)).astype(np.intp)


@lru_cache(maxsize=1024)
def _flow_matrix(t: NeuronType, steps: int) -> np.ndarray:
    """The cascade flow is linear, so `steps` steps of it are one small L x L matrix."""
    L = t.cascade_levels
    cap = 2.0 ** np.arange(L)
    g = t.cascade_g * 2.0 ** -np.arange(max(L - 1, 0))
    step = np.zeros((L, L))
    for col in range(L):  # apply one step to each basis vector, same order as Neuron._flow
        c = np.eye(L)[:, col].copy()
        for k in range(L - 1):
            f = g[k] * (c[k] - c[k + 1])
            c[k] -= f / cap[k]
            c[k + 1] += f / cap[k + 1]
        c[-1] *= 1.0 - t.deep_leak
        step[:, col] = c
    return np.linalg.matrix_power(step, steps).astype(np.float32)


class PackedPopulation:
    def __init__(self, ntype: NeuronType, n_neurons: int, n_inputs: int,
                 rng: np.random.Generator, first_id: int = 0):
        assert ntype.cascade_levels <= 3, "the 32-bit layout has room for 3 cascade levels"
        t = self.t = ntype
        self.n, self.n_inputs, self.rng, self.first_id = n_neurons, n_inputs, rng, first_id
        B, S = t.n_branches, t.slots
        shape = (n_neurons, B, S)

        w = np.zeros(shape, np.uint32)
        for name in ("fast", "c1", "c2"):
            w = _set(w, name, np.full(shape, 32, np.uint32))
        w = _set(w, "c0", _enc(rng.uniform(0.0, 0.1, shape).astype(np.float32), t.w_max, rng))
        self.words = w

        # Per neuron (about 27 bytes)
        self.gate_bits = np.packbits(np.ones((n_neurons, B), bool), axis=1)
        self.threshold = np.full(n_neurons, t.threshold_init, np.float32)
        self.plasticity = np.full(n_neurons, t.plasticity_start, np.float16)
        self.post_hist = np.zeros((n_neurons, ELIG_STEPS), np.float16)  # recent firing, newest first
        self.age = np.zeros(n_neurons, np.uint16)
        self.avg_rate = np.full(n_neurons, t.target_rate, np.float16)

        # Shared by the whole population
        self.pre_hist = np.zeros((ELIG_STEPS, n_inputs), np.float16)  # recent input, newest first
        self.elig_weights = (ntype.elig_decay ** np.arange(ELIG_STEPS)).astype(np.float32)
        self.slot = np.arange(B * S, dtype=np.uint64).reshape(1, B, S)
        self.pending = 0        # awake steps not yet applied to the cascade
        self.fast_scale = 1.0   # fast-weight decay not yet applied

    # ---------- reading the packed words ----------

    def _ids(self):
        return np.arange(self.first_id, self.first_id + self.n, dtype=np.uint64)[:, None, None]

    def _src(self, w):
        return _targets(self._ids(), self.slot, _get(w, "gen"), self.n_inputs)

    def _gate(self):
        return np.unpackbits(self.gate_bits, axis=1, count=self.t.n_branches).astype(np.float32)

    def _levels(self, w):
        return np.stack([_dec(_get(w, LEVELS[k]), self.t.w_max) for k in range(self.t.cascade_levels)])

    def _elig(self, src):
        """Sum over recent steps of decay^k x input(t-k) x firing(t-k), per synapse."""
        e = np.zeros(src.shape, np.float32)
        for k in range(ELIG_STEPS):
            e += self.elig_weights[k] * self.pre_hist[k][src] * self.post_hist[:, k, None, None]
        return e

    def _weights(self, w):
        c0 = np.tensordot(_flow_matrix(self.t, self.pending)[0], self._levels(w), axes=1)
        fast = np.clip(_dec(_get(w, "fast"), self.t.fast_range) * self.fast_scale, -self.t.w_max, self.t.w_max)
        return fast + c0

    def _write_levels(self, w, c):
        for k in range(self.t.cascade_levels):
            w = _set(w, LEVELS[k], _enc(np.clip(c[k], -self.t.w_max, self.t.w_max), self.t.w_max, self.rng))
        return w

    def _catch_up(self):
        if self.pending:
            c = np.tensordot(_flow_matrix(self.t, self.pending), self._levels(self.words), axes=1)
            self.words = self._write_levels(self.words, c)
            self.pending = 0

    # ---------- acting ----------

    def forward(self, x, context=0.0, learn_trace=True):
        t, w = self.t, self.words
        xs = x[self._src(w)]                                     # (N, B, S)
        drive = (self._weights(w) * xs).sum(axis=2)              # (N, B)
        gate = self._gate()
        spike = (drive > t.branch_threshold).astype(np.float32)
        branch_out = (np.maximum(drive, 0.0) + t.spike_bonus * spike) * gate
        soma = branch_out.sum(axis=1) * (1.0 + t.context_gain * context)
        rate = 1.0 / (1.0 + np.exp(-t.soma_slope * (soma - self.threshold)))

        if learn_trace:
            self.pre_hist = np.roll(self.pre_hist, 1, axis=0)
            self.pre_hist[0] = x
            self.post_hist = np.roll(self.post_hist, 1, axis=1)
            self.post_hist[:, 0] = t.post_offset + rate
            self.avg_rate = (self.avg_rate + t.rate_avg_tau * (rate - self.avg_rate)).astype(np.float16)
            used = (xs > 0) & (gate[:, :, None] > 0) & (rate[:, None, None] >= USED_RATE)
            self.words = _set(w, "used", _get(w, "used") | used)
            self.threshold += t.threshold_lr * (rate - t.target_rate)
        return rate * t.sign

    # ---------- learning ----------

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        if dopamine == 0.0:
            return  # no learning and no tagging without a dopamine signal
        t, w = self.t, self.words
        gate = self._gate()[:, :, None]
        elig = self._elig(self._src(w)) * gate
        delta = t.lr * self.plasticity[:, None, None] * ach * dopamine * elig * gate

        fast = np.clip(_dec(_get(w, "fast"), t.fast_range) * self.fast_scale + delta, -t.fast_range, t.fast_range)
        w = _set(w, "fast", _enc(fast, t.fast_range, self.rng))
        self.fast_scale = 1.0

        tag = np.clip(_get(w, "tag") / TAG_MAX + arousal * abs(dopamine) * elig * gate, 0.0, 1.0)
        w = _set(w, "tag", _sround(tag * TAG_MAX, self.rng))
        self.words = w

        if abs(dopamine) * ach > t.oneshot_level:  # BTSP-like one-shot write
            self._catch_up()
            c = self._levels(self.words)
            c[0] += 0.5 * delta
            self.words = self._write_levels(self.words, c)

    def tick(self):
        self.pending += 1
        self.fast_scale *= self.t.fast_decay
        if self.pending >= MAX_PENDING:
            self._catch_up()

    def rest(self, steps):
        self.pending += steps
        self.fast_scale *= self.t.fast_decay ** steps
        self._catch_up()

    # ---------- sleeping ----------

    def sleep(self):
        t = self.t
        self._catch_up()
        w = self.words
        tag = _get(w, "tag") / TAG_MAX
        c = self._levels(w)
        # 1. Consolidate tagged fast weights into the surface; 2. let them sink; 3. downscale
        fast = _dec(_get(w, "fast"), t.fast_range) * self.fast_scale
        c[0] = np.clip(c[0] + t.consolidation_rate * fast * tag, -t.w_max, t.w_max)
        c = np.tensordot(_flow_matrix(t, t.sleep_flow_steps), c, axes=1)
        c[0] *= 1.0 - t.downscale
        # 4. Prune and regrow: weak at every level, not used, not tagged -> new generation, new target
        weak = (np.abs(c).max(axis=0) < t.prune_eps) & (_get(w, "used") == 0) & (tag < 0.05)
        if t.need_gated_rewiring:
            # Only under-active neurons look for new inputs, a few slots per night (see notes/16)
            need = np.clip(1.0 - self.avg_rate.astype(np.float32) / t.target_rate, 0.0, 1.0)
            budget = (t.rewire_budget * t.n_branches * t.slots * need * self.plasticity).astype(np.int64)
            priority = np.where(weak, self.rng.random(weak.shape), np.inf).reshape(self.n, -1)
            rank = np.argsort(np.argsort(priority, axis=1), axis=1).reshape(weak.shape)
            weak &= rank < budget[:, None, None]
        n = int(weak.sum())
        if n:
            w = _set(w, "gen", np.where(weak, _get(w, "gen") + 1, _get(w, "gen")))
            c[:, weak] = 0.0
            c[0][weak] = self.rng.uniform(0.0, 0.1, n)
        w = self._write_levels(w, c)
        # 5. Clear fast weights and use flags; fade tags; mature
        w = _set(w, "fast", np.full(w.shape, 32, np.uint32))
        w = _set(w, "tag", _sround(tag * t.tag_decay * TAG_MAX, self.rng))
        w = _set(w, "used", np.zeros(w.shape, np.uint32))
        self.words = w
        self.fast_scale = 1.0
        self.pre_hist[:] = 0
        self.post_hist[:] = 0
        self.age += 1
        self.plasticity = (t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
                           ).astype(np.float16)
        return n

    # ---------- size ----------

    def bytes_per_neuron(self) -> float:
        per = [self.words, self.gate_bits, self.threshold, self.plasticity, self.post_hist, self.age,
               self.avg_rate]
        return sum(a.nbytes for a in per) / self.n

    def shared_bytes(self) -> int:
        return self.pre_hist.nbytes + self.slot.nbytes


class PackedNeuron:
    """One-neuron population with the same interface as neuron.Neuron, for the demo."""

    def __init__(self, ntype: NeuronType, n_inputs: int, rng: np.random.Generator):
        self.pop = PackedPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)))

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

    def memory_bytes(self) -> int:
        return int(self.pop.bytes_per_neuron())

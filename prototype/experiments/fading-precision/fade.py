"""
Step 3: fading precision ("memories lose resolution, not existence").

A neuron that has not been used for N nights keeps its cascade levels (c0 surface, c1 middle,
c2 deep = the "old self") on a coarser grid. Its fast weight is already 0 after sleep and its
"used" flag is 0, so a cold synapse is: gen (4 bits, wiring, always exact) + c0/c1/c2 at
reduced precision + a tag that decays to 0 within ~3 nights.

Grid: b bits per level -> integer steps k in -m..m with m = 2^(b-1) - 1
  (b=6: the original 6-bit field, b=4: -7..7, b=3: -3..3, b=2: -1, 0, +1 (ternary), b=0: erased).
Scale: "global" = the step is w_max/m for every synapse (wastes levels: most weights are small);
       "block"  = one scale per (level, branch of 32 synapses): step = max|c| in that block / m.
                 The scale costs 5 bits per block = 0.16 bits per synapse per level.
Rounding: "stochastic" (unbiased: 0.3 of a step becomes 1 step 30% of the time) or "nearest".
The coarse value is kept inside the normal 6-bit field (so the neuron runs unchanged); the
quantizer works in 6-bit code space and is idempotent on its own grid, so re-fading a value
that is already on the grid does not add noise.

Timing:
  "eager": every time a cold neuron's levels are rewritten (each night's sleep, each rest), the
           write goes through the coarse grid. Pessimistic: rounding noise compounds nightly.
  "lazy":  the cold neuron is frozen once (faded, then never touched). Time passing is linear
           (cascade flow and downscaling), so it is accumulated as one 3x3 matrix and applied
           exactly when the neuron is next needed ("sharpened"). Pruning and tag decay are
           applied once at thaw. A cold neuron then costs NO work per night, and its frozen
           block can stay compressed. Re-fading to the next tier happens at sleep.
"""

from dataclasses import dataclass, replace

from common import np

from packed_neuron import LEVELS, Q, TAG_MAX, PackedPopulation, _flow_matrix, _get, _set

FULL = (6, 6, 6)


# ---------------------------------------------------------------- the quantizer

def fade_codes(s, bits, rng, scale="block", rounding="stochastic"):
    """s: signed int codes (..., S) in -31..31, blocks along the last axis.
    Returns codes on the coarser grid (still -31..31) and the grid index k."""
    s = s.astype(np.int32)
    if bits >= 6:
        return s, s
    if bits <= 0:
        z = np.zeros_like(s)
        return z, z
    m = (1 << (bits - 1)) - 1
    if scale == "global":
        step = np.full(s.shape[:-1] + (1,), Q / m, np.float64)
    else:
        step = np.abs(s).max(axis=-1, keepdims=True).astype(np.float64) / m
    safe = np.where(step > 0, step, 1.0)
    k_lo = np.floor(s / safe)
    code_lo = np.rint(k_lo * safe)
    code_hi = np.rint((k_lo + 1) * safe)
    gap = np.maximum(code_hi - code_lo, 1.0)
    p = (s - code_lo) / gap
    if rounding == "stochastic":
        up = rng.random(s.shape) < p
    else:
        up = p >= 0.5
    k = np.where(up, k_lo + 1, k_lo)
    k = np.clip(k, -m, m)
    out = np.where(step > 0, np.rint(k * safe), 0).astype(np.int32)
    k = np.where(step > 0, k, 0).astype(np.int32)
    return out, k


def fade_words(words, bits, rng, scale="block", rounding="stochastic", n_levels=3):
    """Apply per-level bits (b0, b1, b2) to the cascade fields of packed words (N, B, S).
    scale="vq<K>" / "holo<K>": the fringe codecs (fringe.py) instead of per-level scalar bits."""
    if scale.startswith("vq"):
        import fringe
        return fringe.apply_vq(words, scale[2:])
    if scale.startswith("holo"):
        import fringe
        return fringe.apply_holo(words, int(scale[4:]), rng)
    w = words
    for lvl in range(n_levels):
        b = bits[lvl]
        if b >= 6:
            continue
        s = _get(w, LEVELS[lvl]).astype(np.int32) - 32
        out, _ = fade_codes(s, b, rng, scale, rounding)
        w = _set(w, LEVELS[lvl], (out + 32).astype(np.uint32))
    return w


def stored_bits_per_synapse(bits, scale="block", slots=32, tag_bits=0):
    """Fixed-width (before entropy coding) bits per cold synapse: gen + levels (+ block scales)."""
    if scale.startswith("vq"):
        k = scale[2:].lstrip("m")
        return 4 + (np.log2(int(k)) if k.isdigit() else 10)
    if scale.startswith("holo"):
        return 4 + 3 * 16 / int(scale[4:])   # float16 trace shared by K neurons
    lv = sum(b for b in bits)
    sc = sum(5.0 / slots for b in bits if 0 < b < 6) if scale == "block" else 0.0
    return 4 + lv + sc + tag_bits


# ---------------------------------------------------------------- fading policies

@dataclass(frozen=True)
class Policy:
    """tiers: ((min nights unused, (b_c0, b_c1, b_c2)), ...) in increasing order of nights."""
    name: str
    tiers: tuple = ()
    scale: str = "block"
    rounding: str = "stochastic"
    timing: str = "eager"
    prune_cold: bool = True   # False: a cold neuron's wiring is frozen with it (no pruning while cold)

    def bits(self, nights):
        b = FULL
        for n, bb in self.tiers:
            if nights >= n:
                b = bb
        return b


POLICIES = {
    "none": Policy("none"),
    # The deep level ("old self") is the LAST thing to fade, and never below 4 bits.
    "deep-last": Policy("deep-last", ((1, (4, 6, 6)), (2, (4, 4, 6)), (3, (3, 4, 5)), (4, (2, 3, 4)),
                                      (6, (2, 2, 4)))),
    # The deep level gets special protection: it never fades at all.
    "deep-protected": Policy("deep-protected", ((1, (4, 6, 6)), (2, (4, 4, 6)), (3, (2, 3, 6)),
                                                (4, (2, 2, 6)))),
    # Everything fades together.
    "uniform": Policy("uniform", ((1, (4, 4, 4)), (3, (3, 3, 3)), (5, (2, 2, 2)))),
    # Wrong-headed control: the deep level fades first.
    "deep-first": Policy("deep-first", ((1, (6, 6, 4)), (2, (6, 4, 2)), (3, (4, 2, 2)), (5, (2, 2, 2)))),
}


def fixed(b_surface, b_mid, b_deep, onset=1, **kw):
    """Fade straight to fixed bits once a neuron has been unused for `onset` nights."""
    return Policy(f"fixed{b_surface}{b_mid}{b_deep}", ((onset, (b_surface, b_mid, b_deep)),), **kw)


# ---------------------------------------------------------------- behavior adapter

class FadingNeuron:
    """Same interface as PackedNeuron (forward / modulate / tick / rest / sleep / memory_bytes).
    Wraps a one-neuron PackedPopulation; fading uses its own RNG so the demo's shared RNG
    stream (which also generates the patterns) is consumed exactly as by PackedNeuron."""

    def __init__(self, ntype, n_inputs, rng, policy: Policy = POLICIES["deep-last"]):
        self.pop = PackedPopulation(ntype, 1, n_inputs, rng, first_id=int(rng.integers(1 << 31)))
        self.policy = policy
        self.frng = np.random.default_rng(self.pop.first_id ^ 0x5EED)
        self.nights_unused = 0
        self.bits = FULL
        self.bits_log = []          # stored bits per synapse after each night
        # lazy timing
        self.frozen = False
        self.op = np.eye(ntype.cascade_levels)
        self.nights_frozen = 0

        self.cold = False
        self.prunes_unused = 0      # synapses rewired on nights the neuron was not used

    # ----- helpers
    def _refade(self):
        p = self.policy
        if self.bits == FULL and p.scale in ("block", "global"):
            return
        self.pop.words = fade_words(self.pop.words, self.bits, self.frng, p.scale, p.rounding,
                                    self.pop.t.cascade_levels)

    def _levels_float(self):
        return self.pop._levels(self.pop.words).astype(np.float64)

    def _thawed_words(self, commit):
        """Apply the accumulated time evolution exactly; with commit, also prune and decay tags."""
        pop, t = self.pop, self.pop.t
        c = np.tensordot(self.op, self._levels_float(), axes=1)
        c = np.clip(c, -t.w_max, t.w_max)
        w = pop.words
        if not commit:   # a probe: deterministic nearest rounding, nothing stored
            for k in range(t.cascade_levels):
                w = _set(w, LEVELS[k], (np.clip(np.rint(c[k] * Q / t.w_max), -Q, Q) + 32).astype(np.uint32))
            return w
        tag = _get(w, "tag") / TAG_MAX * t.tag_decay ** self.nights_frozen
        tag = np.floor(tag * TAG_MAX + self.frng.random(tag.shape)) / TAG_MAX
        weak = (np.abs(c).max(axis=0) < t.prune_eps) & (_get(w, "used") == 0) & (tag < 0.05)
        if not self.policy.prune_cold:
            weak[:] = False
        n = int(weak.sum())
        self.prunes_unused += n
        if n:
            w = _set(w, "gen", np.where(weak, _get(w, "gen") + 1, _get(w, "gen")))
            c[:, weak] = 0.0
            c[0][weak] = self.frng.uniform(0.0, 0.1, n)
        for k in range(t.cascade_levels):
            x = np.floor(c[k] * Q / t.w_max + self.frng.random(c[k].shape))
            w = _set(w, LEVELS[k], (np.clip(x, -Q, Q) + 32).astype(np.uint32))
        w = _set(w, "tag", (tag * TAG_MAX).astype(np.uint32))
        return w

    def _thaw(self):
        if self.frozen:
            self.pop.words = self._thawed_words(commit=True)
            self.frozen = False
            self.op = np.eye(self.pop.t.cascade_levels)
            self.nights_frozen = 0

    def _freeze(self):
        self._refade()
        self.frozen = True
        self.op = np.eye(self.pop.t.cascade_levels)
        self.nights_frozen = 0

    # ----- interface
    def forward(self, x, context=0.0, learn_trace=True):
        if self.frozen and not learn_trace:          # probe a frozen neuron without waking it
            keep = self.pop.words
            self.pop.words = self._thawed_words(commit=False)
            r = float(self.pop.forward(x, context, False)[0])
            self.pop.words = keep
            return r
        self._thaw()
        r = float(self.pop.forward(x, context, learn_trace)[0])
        if learn_trace and _get(self.pop.words, "used").any():
            self.nights_unused, self.bits, self.cold = 0, FULL, False  # used again: sharpen
        return r

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        self._thaw()
        self.pop.modulate(dopamine, ach, arousal)

    def tick(self):
        self._thaw()
        self.pop.tick()
        if self.cold and self.pop.pending == 0:
            self._refade()

    def rest(self, steps):
        if self.frozen:
            self.op = _flow_matrix(self.pop.t, steps).astype(np.float64) @ self.op
            return
        self.pop.rest(steps)
        if self.cold:
            self._refade()

    def sleep(self):
        pop, t = self.pop, self.pop.t
        used = bool(_get(pop.words, "used").any()) and not self.frozen
        self.nights_unused = 0 if used else self.nights_unused + 1
        new_bits = self.policy.bits(self.nights_unused)
        tiers = self.policy.tiers
        n = 0
        if self.frozen:
            # A frozen neuron is not processed: its night is one more matrix in the product.
            D = np.eye(t.cascade_levels)
            D[0, 0] = 1.0 - t.downscale
            self.op = D @ _flow_matrix(t, t.sleep_flow_steps).astype(np.float64) @ self.op
            self.nights_frozen += 1
            pop.age += 1
            pop.plasticity = (t.plasticity_floor + (pop.plasticity - t.plasticity_floor) * t.maturation
                              ).astype(np.float16)
            if new_bits != self.bits:                # next tier: a sleep-time re-fade
                self._thaw()
                self.bits = new_bits
                self._freeze()
        else:
            tiers_on = bool(tiers) and self.nights_unused >= tiers[0][0]
            if tiers_on and not self.policy.prune_cold:
                pop.t = replace(t, prune_eps=-1.0)    # no synapse counts as weak tonight
                n = pop.sleep()
                pop.t = t
            else:
                n = pop.sleep()
            if self.nights_unused >= 1:
                self.prunes_unused += n
            self.bits = new_bits
            self.cold = tiers_on
            if self.cold:
                if self.policy.timing == "lazy":
                    self._freeze()
                else:
                    self._refade()
        self.bits_log.append(stored_bits_per_synapse(self.bits, self.policy.scale) if self.cold else 32)
        return n

    def memory_bytes(self):
        return int(self.pop.bytes_per_neuron())


def factory(policy):
    def make(ntype, n_inputs, rng):
        return FadingNeuron(ntype, n_inputs, rng, policy)
    return make

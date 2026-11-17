"""
A single human-style neuron for the LittleBrain project.

Template: the cortical/hippocampal pyramidal neuron ("workhorse"), using lessons
from the fly mushroom body's Kenyon cell. Design notes: notes/13-our-neuron.md

Written for clarity, not speed. Everything is float32 numpy so it is easy to inspect.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class NeuronType:
    """A neuron 'type' is just a set of parameters (see notes/12-neuron-catalog.md)."""

    name: str
    n_branches: int = 8           # basal dendrite branches (feedforward input)
    slots: int = 32               # synapse slots per branch (fixed budget)
    cascade_levels: int = 3       # hidden synaptic copies at slower and slower timescales
    sign: float = 1.0             # +1 excitatory, -1 inhibitory

    # Learning
    lr: float = 0.15              # fast-weight learning rate
    fast_decay: float = 0.995     # per step; fast weights fade ("fire and forget")
    elig_decay: float = 0.5       # per step; eligibility trace fades
    oneshot_level: float = 0.9    # dopamine * ACh above this also writes straight into cascade[0]
    w_max: float = 1.0

    # Cascade (Benna & Fusi style): level k has capacity 2^k, coupling g * 2^-k to level k+1
    cascade_g: float = 0.005
    deep_leak: float = 1e-5

    # Dendrites and soma
    branch_threshold: float = 0.8   # clustered input needed for a dendritic spike
    spike_bonus: float = 1.0
    context_gain: float = 1.0       # apical (top-down) multiplicative gain
    threshold_init: float = 6.0
    target_rate: float = 0.1        # homeostatic target firing rate (sparsity)
    threshold_lr: float = 0.002
    soma_slope: float = 0.5

    # Development
    plasticity_start: float = 1.0
    plasticity_floor: float = 0.3
    maturation: float = 0.97        # per sleep, plasticity moves toward the floor by this factor

    # Sleep
    sleep_flow_steps: int = 100
    downscale: float = 0.02
    prune_eps: float = 0.02
    tag_decay: float = 0.5

    # Learning rules v2 (notes/16). Defaults here reproduce v1; see WORKHORSE_V2 below.
    fast_range: float = 1.0          # fast weight may accumulate beyond w_max ("how much was learned")
    consolidation_rate: float = 1.0  # fraction of the day's fast weight written into cascade[0] at night
    post_offset: float = 0.5         # eligibility uses (post_offset + rate); 0 = only learn if the cell fired
    rewire_budget: float = 1.0       # max fraction of slots rewired per night (1.0 = unlimited)
    need_gated_rewiring: bool = False  # rewire only when the neuron is under-active (needs new input)
    rate_avg_tau: float = 0.01       # running average of firing rate, for rewiring need

    # Per-region learning (brain prototype; notes/22). Defaults keep the rules above unchanged.
    arousal_gated: bool = False      # amygdala: learning itself scales with arousal (noradrenaline)
    cooperativity: float = 0.0       # synapses learning together on one branch help each other (clustering)
    branch_lr: float = 0.0           # whole-branch excitability learns (branch strength plasticity)
    encoding_learning: float = 0.0   # hippocampus: learns whatever arrives while acetylcholine is high (encoding
                                     # mode), not only at dopamine moments; this much signal per unit ACh above 0.5


WORKHORSE = NeuronType("workhorse")  # pyramidal neuron: the template (learning rules v1)

# Learning rules v2: consolidate how much was learned, learn only when the cell fired,
# rewire only when the neuron needs new input, tags last until sleep only.
WORKHORSE_V2 = replace(WORKHORSE, name="workhorse-v2", fast_range=2.0, consolidation_rate=1.0,
                       post_offset=0.1, tag_decay=0.0, rewire_budget=0.03,
                       need_gated_rewiring=True)

# Other catalog types as parameter presets. Not tested yet.
CORE = replace(WORKHORSE, name="core", lr=0.03, plasticity_start=0.3, plasticity_floor=0.05,
               cascade_levels=4, downscale=0.005)
TRANSIENT = replace(WORKHORSE, name="transient", lr=0.6, fast_decay=0.9, cascade_levels=1,
                    slots=8, target_rate=0.03)
HUB = replace(WORKHORSE, name="hub", n_branches=16)
INHIBITORY = replace(WORKHORSE, name="inhibitory", sign=-1.0, lr=0.3, cascade_levels=2)


class Neuron:
    def __init__(self, ntype: NeuronType, n_inputs: int, rng: np.random.Generator):
        self.t = ntype
        self.n_inputs = n_inputs
        self.rng = rng
        B, S, L = ntype.n_branches, ntype.slots, ntype.cascade_levels

        # Synapses: one row per branch, one column per slot
        self.src = rng.integers(0, n_inputs, size=(B, S))
        self.fast = np.zeros((B, S), np.float32)
        self.cascade = np.zeros((L, B, S), np.float32)
        self.cascade[0] = rng.uniform(0.0, 0.1, size=(B, S))
        self.elig = np.zeros((B, S), np.float32)
        self.tag = np.zeros((B, S), np.float32)
        self.usage = np.zeros((B, S), np.float32)

        # Branches and soma
        self.gate = np.ones(B, np.float32)  # 1 = open, 0 = silenced by SST-like inhibition
        self.threshold = ntype.threshold_init
        self.plasticity = ntype.plasticity_start
        self.age = 0
        self.avg_rate = ntype.target_rate

        # Cascade constants
        self.capacity = 2.0 ** np.arange(L, dtype=np.float32)
        self.coupling = ntype.cascade_g * 2.0 ** -np.arange(max(L - 1, 0), dtype=np.float32)

        self._last = None

    # ---------- weights ----------

    def weights(self):
        return np.clip(self.fast, -self.t.w_max, self.t.w_max) + self.cascade[0]

    def _flow(self, steps: int):
        """Let the cascade levels exchange value: the surface sinks slowly, deep levels pull it back."""
        c = self.cascade
        for _ in range(steps):
            for k in range(len(self.coupling)):
                f = self.coupling[k] * (c[k] - c[k + 1])
                c[k] -= f / self.capacity[k]
                c[k + 1] += f / self.capacity[k + 1]
            c[-1] *= 1.0 - self.t.deep_leak

    # ---------- acting ----------

    def forward(self, x: np.ndarray, context: float = 0.0, learn_trace: bool = True,
                post_rate: float | None = None) -> float:
        """x: presynaptic activity in [0, 1]. context: apical top-down signal.
        post_rate: the firing rate after population inhibition, if a population decided it."""
        xs = x[self.src]                                         # (B, S) input seen by each synapse
        drive = (self.weights() * xs).sum(axis=1)                # (B,)
        spike = (drive > self.t.branch_threshold).astype(np.float32)
        branch_out = (np.maximum(drive, 0.0) + self.t.spike_bonus * spike) * self.gate
        soma = branch_out.sum() * (1.0 + self.t.context_gain * context)
        rate = float(1.0 / (1.0 + np.exp(-self.t.soma_slope * (soma - self.threshold))))
        if post_rate is not None:
            rate = post_rate

        if learn_trace:
            # Eligibility: "this synapse just took part" (input x open branch x postsynaptic firing)
            post = self.t.post_offset + rate
            self.elig = self.elig * self.t.elig_decay + xs * self.gate[:, None] * post
            self.usage += (xs > 0) * self.gate[:, None] * rate
            self.threshold += self.t.threshold_lr * (rate - self.t.target_rate)
            self.avg_rate += self.t.rate_avg_tau * (rate - self.avg_rate)
            self._last = rate
        return rate * self.t.sign

    # ---------- learning ----------

    def modulate(self, dopamine: float = 0.0, ach: float = 0.0, arousal: float = 0.0):
        """Three-factor learning. dopamine in [-1, 1]; ach and arousal in [0, 1]."""
        open_ = self.gate[:, None]
        delta = self.t.lr * self.plasticity * ach * dopamine * self.elig * open_
        self.fast = np.clip(self.fast + delta, -self.t.fast_range, self.t.fast_range)
        self.tag = np.clip(self.tag + arousal * abs(dopamine) * self.elig * open_, 0.0, 1.0)
        if abs(dopamine) * ach > self.t.oneshot_level:  # BTSP-like one-shot write
            self.cascade[0] = np.clip(self.cascade[0] + 0.5 * delta, -self.t.w_max, self.t.w_max)

    def tick(self):
        """One awake time step of background decay."""
        self.fast *= self.t.fast_decay
        self._flow(1)

    def rest(self, steps: int):
        """Awake but idle: no input, time passes."""
        self.fast *= self.t.fast_decay ** steps
        self._flow(steps)

    # ---------- sleeping ----------

    def sleep(self):
        t = self.t
        # 1. Consolidate: tagged learning moves into the surface level, in proportion to how much was learned
        self.cascade[0] = np.clip(self.cascade[0] + t.consolidation_rate * self.fast * self.tag,
                                  -t.w_max, t.w_max)
        self.fast[:] = 0.0
        self.elig[:] = 0.0
        # 2. Let the night's changes sink into deeper levels
        self._flow(t.sleep_flow_steps)
        # 3. Downscale the surface (only the strong survive)
        self.cascade[0] *= 1.0 - t.downscale
        # 4. Prune and regrow: weak at every level, rarely used, never tagged
        weak = (np.abs(self.cascade).max(axis=0) < t.prune_eps) & (self.usage < 1.0) & (self.tag < 0.05)
        if t.need_gated_rewiring:
            # Structural plasticity follows need: only an under-active neuron looks for new inputs,
            # and only a few slots per night, fewer as it matures
            need = float(np.clip(1.0 - self.avg_rate / t.target_rate, 0.0, 1.0))
            budget = int(t.rewire_budget * weak.size * need * self.plasticity)
            idx = np.flatnonzero(weak)
            if idx.size > budget:
                keep = self.rng.choice(idx, size=budget, replace=False)
                weak = np.zeros_like(weak)
                weak.flat[keep] = True
        n = int(weak.sum())
        if n:
            self.src[weak] = self.rng.integers(0, self.n_inputs, size=n)
            self.cascade[:, weak] = 0.0
            self.cascade[0][weak] = self.rng.uniform(0.0, 0.1, size=n)
        # 5. Tags and usage fade; the neuron matures
        self.tag *= t.tag_decay
        self.usage *= 0.5
        self.age += 1
        self.plasticity = t.plasticity_floor + (self.plasticity - t.plasticity_floor) * t.maturation
        return n

    def memory_bytes(self) -> int:
        arrays = [self.src, self.fast, self.cascade, self.elig, self.tag, self.usage, self.gate]
        return sum(a.nbytes for a in arrays)

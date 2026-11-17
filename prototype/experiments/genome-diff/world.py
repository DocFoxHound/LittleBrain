"""
A sparse, structured world for measuring how much of a brain experience actually changes.

Scenario (stated plainly, all knobs in WorldCfg):
  - 4,096 inputs in 64 regions of 64 ("cortical areas").
  - One column of neurons per region. A neuron's 256 synapses draw from its own region, with a
    10% chance per synapse of a neighbouring region (local connectivity via the wiring hash).
  - 2,048 "things" in the world. Each lives in one region and lights up 12 of its 64 inputs.
    How often each thing turns up follows a Zipf law (a few are everyday, most are rare), so new
    things keep appearing over a lifetime.
  - Things carry valence: ~80% neutral, ~13% good, ~7% bad. Halfway through life, the 10 most
    familiar good things turn bad ("hard times").
  - A day: 50 episodes x 4 steps. Each episode shows 7 things at once (84 inputs, ~2% of inputs,
    plus optional background noise). At the episode's last step, if any thing present has valence,
    dopamine = +-1 (ACh 0.8, arousal 0.8) -> three-factor learning on what was just active.
    A midday rest (100 idle steps), an evening rest (100), then sleep.
"""

import time
from dataclasses import dataclass, field

import numpy as np

import core
from neuron import WORKHORSE


@dataclass
class WorldCfg:
    n_regions: int = 64
    region_size: int = 64
    neurons_per_col: int = 256
    p_neighbor: float = 0.1
    n_things: int = 2048
    thing_k: int = 12
    zipf: float = 1.1
    p_good: float = 0.13
    p_bad: float = 0.07
    episodes: int = 50
    steps_per_episode: int = 4
    things_per_episode: int = 7
    noise: float = 0.0          # fraction of ALL inputs randomly active each step
    targeted: bool = False      # dopamine reaches only the columns of the valenced things (not broadcast)
    grow: tuple = ()            # (start_regions, regions_added_per_day): a world that opens up over life
    days: int = 40
    flip_day: int = 20
    seed: int = 7
    pop: dict = field(default_factory=dict)  # GenomePop kwargs (mode, need_activity, ...)
    snapshot_days: tuple = (1, 3, 7, 14, 20, 30, 40, 60)


class World:
    def __init__(self, cfg: WorldCfg):
        self.cfg = c = cfg
        rng = np.random.default_rng(c.seed)
        self.rng = rng
        self.n_inputs = c.n_regions * c.region_size
        self.thing_region = rng.integers(0, c.n_regions, c.n_things)
        self.thing_inputs = np.stack([r * c.region_size + rng.choice(c.region_size, c.thing_k, replace=False)
                                      for r in self.thing_region])
        ranks = rng.permutation(c.n_things) + 1
        p = 1.0 / ranks ** c.zipf
        self.p = p / p.sum()
        u = rng.random(c.n_things)
        self.valence = np.where(u < c.p_good, 1.0, np.where(u < c.p_good + c.p_bad, -1.0, 0.0))
        self.familiar_good = np.argsort(-self.p * (self.valence > 0))[:10]

    def flip(self):
        self.valence[self.familiar_good] = -1.0

    def active_neurons(self, x_idx):
        """Neurons that can possibly receive input: columns whose own or neighbouring region is active."""
        c = self.cfg
        regs = np.unique(x_idx // c.region_size)
        cols = np.unique(np.concatenate([(regs - 1) % c.n_regions, regs, (regs + 1) % c.n_regions]))
        return (cols[:, None] * c.neurons_per_col + np.arange(c.neurons_per_col)[None, :]).ravel()


def diff_stats(pop: core.GenomePop, G=None):
    """How far the persistent state is from the genome (after a night: fast = used = 0)."""
    w = pop.words
    if G is None:
        G = pop.genome_words()
    pm = core.PERSIST_MASK
    differs = (w & pm) != (G & pm)
    gen = (w & 15) != 0
    lev = [((w >> s) & 63).astype(np.int16) - ((G >> s) & 63).astype(np.int16) for s in (10, 16, 22)]
    absmax = np.maximum(np.maximum(np.abs(lev[0]), np.abs(lev[1])), np.abs(lev[2]))
    tag = ((w >> 28) & 7) != 0
    g0 = ~gen
    n_syn = w.size
    return {
        "syn_any": differs.sum() / n_syn,
        "syn_rewired": gen.sum() / n_syn,
        "syn_learned": (g0 & (absmax >= 1)).sum() / n_syn,
        "syn_learned2": (g0 & (absmax >= 2)).sum() / n_syn,
        "syn_deep": (g0 & ((lev[1] != 0) | (lev[2] != 0))).sum() / n_syn,
        "syn_tagonly": (g0 & (absmax == 0) & tag).sum() / n_syn,
        "neu_any": differs.any(axis=1).mean(),
        "neu_learned2": (g0 & (absmax >= 2)).any(axis=1).mean(),
        "per_dirty_neuron": differs.sum() / max(differs.any(axis=1).sum(), 1),
    }


def simulate(cfg: WorldCfg, verbose=True, keep_snapshots=True):
    c = cfg
    world = World(c)
    lay = core.layout_local(c.n_regions, c.region_size, c.neurons_per_col, c.p_neighbor)
    n = c.n_regions * c.neurons_per_col
    pop = core.GenomePop(WORKHORSE, n, lay, seed=c.seed, **c.pop)
    G = pop.genome_words() if pop.mode == core.BASE else None
    rng = world.rng
    x = np.zeros(world.n_inputs, np.float32)
    log, snaps = [], {}
    seen_cols = np.zeros(c.n_regions, bool)
    learned_cols = np.zeros(c.n_regions, bool)
    t_awake = t_sleep = 0.0
    evaluated = modulated = steps = mods = 0
    for day in range(1, c.days + 1):
        if day == c.flip_day + 1:
            world.flip()
        p_today = world.p
        if c.grow:
            avail = min(c.n_regions, int(c.grow[0] + c.grow[1] * (day - 1)))
            p_today = world.p * (world.thing_region < avail)
            p_today = p_today / p_today.sum()
        t0 = time.perf_counter()
        for ep in range(c.episodes):
            things = rng.choice(c.n_things, c.things_per_episode, p=p_today)
            base_idx = world.thing_inputs[things].ravel()
            val = world.valence[things]
            for st in range(c.steps_per_episode):
                if c.noise > 0:
                    k = rng.binomial(world.n_inputs, c.noise)
                    idx = np.concatenate([base_idx, rng.integers(0, world.n_inputs, k)])
                else:
                    idx = base_idx
                x[:] = 0.0
                x[idx] = 1.0
                act = world.active_neurons(idx)
                pop.forward(x, active=act)
                evaluated += act.size
                steps += 1
                seen_cols[np.unique(act // c.neurons_per_col)] = True
                if st == c.steps_per_episode - 1 and np.any(val != 0):
                    da = float(np.clip(val.sum(), -1, 1)) or float(val[val != 0][0])
                    sub = None
                    if c.targeted:
                        cols = np.unique(world.thing_region[things[val != 0]])
                        sub = (cols[:, None] * c.neurons_per_col + np.arange(c.neurons_per_col)[None, :]).ravel()
                    pop.modulate(dopamine=da, ach=0.8, arousal=0.8, subset=sub)
                    mods += 1
                    learned_cols[np.unique(act // c.neurons_per_col)] = True
                pop.tick()
            if ep == c.episodes // 2 - 1:
                pop.rest(100)
        pop.rest(100)
        t1 = time.perf_counter()
        pruned = pop.sleep()
        t2 = time.perf_counter()
        t_awake += t1 - t0
        t_sleep += t2 - t1
        st = diff_stats(pop, G)
        st.update(day=day, pruned=pruned, cols_seen=seen_cols.mean(), cols_learned=learned_cols.mean(),
                  awake_s=t1 - t0, sleep_s=t2 - t1)
        log.append(st)
        if keep_snapshots and day in c.snapshot_days:
            snaps[day] = pop.words.copy()
        if verbose and (day in c.snapshot_days or day == c.days):
            print(f"    day {day:3d}: syn differ {st['syn_any']:6.1%} = rewired {st['syn_rewired']:5.1%}"
                  f" + learned {st['syn_learned']:5.1%} (>=2q {st['syn_learned2']:5.1%}) + tag-only "
                  f"{st['syn_tagonly']:5.1%} | neurons {st['neu_any']:5.1%} | pruned/night {pruned / n:5.1f}/neuron"
                  f" | awake {t1 - t0:4.1f}s sleep {t2 - t1:5.2f}s", flush=True)
    meta = dict(n=n, steps=steps, mods=mods, evaluated_frac=evaluated / (steps * n), t_awake=t_awake,
                t_sleep=t_sleep, first_id=pop.first_id, lay=lay, birth_q=pop.birth_q, mode=pop.mode)
    return log, snaps, meta, pop

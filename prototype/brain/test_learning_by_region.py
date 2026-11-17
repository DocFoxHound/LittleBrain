"""
Step 6: learning rules by region, and dendritic clustering (notes/22-learning-by-region.md).

Run: PYTHONPATH=prototype python3 prototype/brain/test_learning_by_region.py
"""

from dataclasses import replace

import numpy as np

import celltypes as C
from cells import ExcitatoryGroup
from neuron import WORKHORSE_V2

np.seterr(over="ignore")
RESULTS = []
N_IN = 128


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))


def population(ctype, n=30, seed=0):
    rng = np.random.default_rng(seed)
    g = ExcitatoryGroup(ctype, n, rng)
    g.basal.src[:] = rng.integers(0, N_IN, g.basal.src.shape)
    g.basal.set_surface(0.04)
    g.threshold[:] = 2.0
    return g


def pattern(seed, k=24):
    x = np.zeros(N_IN, np.float32)
    x[np.random.default_rng(seed).choice(N_IN, k, replace=False)] = 1.0
    return x


def respond(g, x, steps=6):
    for _ in range(steps):
        g.update(g.compute(x), learn=False)
    r = float(g.rate.mean())
    g.rate[:] = 0
    return r


def train(g, x, dopamine, trials, ach=0.8, arousal=0.8, delay=0):
    for _ in range(trials):
        for _ in range(3):
            g.update(g.compute(x), learn=True)
        for _ in range(delay):  # time passes before the outcome arrives
            g.update(g.compute(0 * x), learn=True)
        g.modulate(dopamine, ach, arousal)
        g.tick()
        g.rate[:] = 0


def test_branch_eras():
    """Layered personality at the level of one cell: two eras of life on different dendritic branches."""
    out = {}
    for gated in (False, True):
        g = population(C.L23_IT)
        A = pattern(1)
        half = g.nb // 2  # first half of the branches vs the second half
        ctx1 = np.r_[np.ones(half), np.zeros(g.nb - half)].astype(np.float32)
        ctx2 = 1 - ctx1
        g.branch_open[:] = ctx1 if gated else 1.0
        train(g, A, +1.0, 15)            # era 1: A is good
        g.sleep()
        g.branch_open[:] = ctx2 if gated else 1.0
        train(g, A, -1.0, 15)            # era 2: A is bad
        g.sleep()
        g.branch_open[:] = ctx1 if gated else 1.0
        old = respond(g, A)
        g.branch_open[:] = ctx2 if gated else 1.0
        new = respond(g, A)
        out[gated] = (old, new)
    check("Branch gating keeps an old era alongside a new one (layered personality, one cell)",
          out[True][0] > out[True][1] + 0.1 and out[True][0] > out[False][0] + 0.1,
          f"no gating: response to A {out[False][0]:.2f} (old self overwritten); "
          f"gated: era-1 context {out[True][0]:.2f}, era-2 context {out[True][1]:.2f}")


def test_clustering():
    """Cooperativity + branch strength: learning concentrates on fewer branches."""
    def concentration(rules):
        g = population(replace(C.L23_IT, rules=rules))
        A = pattern(2)
        w0 = g.basal.weights().copy()
        train(g, A, +1.0, 3)  # few trials, before synapses saturate
        gain = (g.basal.weights() - w0).clip(0).sum(axis=2) * g.branch_gain   # (cells, branches)
        k = max(1, g.nb // 4)
        top = np.sort(gain, axis=1)[:, -k:].sum(1) / (gain.sum(1) + 1e-9)
        return float(top.mean()), float(g.branch_gain.std())
    plain = concentration(WORKHORSE_V2)
    clustered = concentration(replace(WORKHORSE_V2, cooperativity=0.5, branch_lr=0.05))
    check("Clustered plasticity: learning concentrates on a few branches", clustered[0] > plain[0] + 0.03,
          f"share of learning on the top quarter of branches: plain {plain[0]:.2f}, clustered {clustered[0]:.2f}; "
          f"branch-strength spread {plain[1]:.2f} -> {clustered[1]:.2f}")


def test_amygdala_arousal():
    res = {}
    for arousal in (0.1, 0.9):
        g = population(C.BLA_NEGATIVE)
        A = pattern(3)
        before = respond(g, A)
        train(g, A, +1.0, 3, arousal=arousal)   # 3 pairings of a cue with something important
        res[arousal] = respond(g, A) - before
    check("BLA: few-trial learning, gated by arousal (noradrenaline)", res[0.9] > 3 * max(res[0.1], 1e-3),
          f"response gained after 3 pairings: calm {res[0.1]:+.2f}, aroused {res[0.9]:+.2f}")


def test_striatum_delayed_reward():
    res = {}
    for ctype in (C.D1_MSN, replace(C.D1_MSN, rules=replace(C.D1_MSN.rules, elig_decay=0.5))):
        g = population(ctype)
        g.threshold[:] = 1.0
        A = pattern(4)
        w0 = g.basal.weights().copy()
        train(g, A, +1.0, 10, delay=4)          # reward arrives 4 steps after the action
        res[ctype.rules.elig_decay] = float((g.basal.weights() - w0).clip(0).sum())
    check("Striatum: a long eligibility trace lets late rewards still teach", res[0.85] > 5 * res[0.5],
          f"learning from a reward 4 steps late: striatal trace {res[0.85]:.2f}, cortical-length trace {res[0.5]:.2f}")


def test_ca3_fast():
    res = {}
    for ctype in (C.CA3_PYR, C.L23_IT):
        g = population(ctype)
        A = pattern(5)
        w0 = g.basal.weights().copy()
        train(g, A, +1.0, 1)
        res[ctype.name] = float((g.basal.weights() - w0).clip(0).sum() / g.n)
    check("CA3: fast storage (much more learned from a single trial)", res["CA3 pyramidal"] > 2 * res["L2/3 IT"],
          f"potentiation per cell after 1 trial: CA3 {res['CA3 pyramidal']:.2f}, cortex L2/3 {res['L2/3 IT']:.2f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print(f"\n{sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)} passed")

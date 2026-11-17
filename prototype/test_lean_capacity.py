"""
Lean neuron capacity (notes/24): how much more can a 4096- or 8192-synapse neuron hold than a 256 one,
does it need 32 levels instead of 16, and do smart dendrites help?

Task: P patterns (5% of 2048 inputs each), half good (dopamine +1) and half bad (-1), 4 days of
practice with sleep, then 6 days with no practice. Score: AUC, the chance that the neuron responds
more to a random good pattern than to a random bad one (0.5 = chance, 1 = perfect).

Larger neurons get proportionally scaled thresholds, so a synapse means the same everywhere.
Settings: chemical tags on, gentle (surface-only) sleep reset on, as adopted in notes/24.

Run: PYTHONPATH=prototype python3 prototype/test_lean_capacity.py
"""

import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from lean_neuron import LeanPopulation
from neuron import WORKHORSE_V2 as T

# Env knobs (read here so worker processes see them): N_INPUTS, SIZES ("syn:branches:slots,..."), VARIANTS
N_INPUTS = int(os.environ.get("N_INPUTS", "2048"))
ACTIVE, NEURONS, DAYS, REPS = int(round(0.05 * N_INPUTS)), 8, 4, 3
SIZES = {int(a): (int(b), int(c)) for a, b, c in
         (x.split(":") for x in os.environ.get("SIZES", "256:8:32,4096:32:128,8192:64:128").split(","))}
PATTERNS = (8, 32, 128, 512)
VARIANTS = (("16 lv", 7, "spikes"), ("32 lv", 15, "spikes"), ("16 lv linear dendrites", 7, "linear"),
            ("16 lv branch learning", 7, "spikes+learning"))
VARIANTS = tuple(VARIANTS[int(i)] for i in os.environ.get("VARIANTS", "0,1,2,3").split(","))
WORLDS = (0, 1)


def scaled(syn):
    B, S = SIZES[syn]
    f, g = B * S / 256, S / 32
    return replace(T, n_branches=B, slots=S, threshold_init=T.threshold_init * f,
                   threshold_lr=T.threshold_lr * f, soma_slope=T.soma_slope / f,
                   branch_threshold=T.branch_threshold * g, spike_bonus=T.spike_bonus * g)


def auc(good, bad):
    """Per neuron: chance a good pattern beats a bad one (ties count half)."""
    g, b = good[:, :, None], bad[:, None, :]
    return (g > b).mean(axis=(1, 2)) + 0.5 * (g == b).mean(axis=(1, 2))


def job(args):
    syn, P, (label, q, dend), world = args
    rng = np.random.default_rng(1000 * world + P)
    pats = np.zeros((P, N_INPUTS), np.float32)
    for p in pats:
        p[rng.choice(N_INPUTS, ACTIVE, replace=False)] = 1.0
    value = np.where(np.arange(P) % 2 == 0, 1.0, -1.0)
    pop = LeanPopulation(scaled(syn), NEURONS, N_INPUTS, np.random.default_rng(world), first_id=100 * world,
                         q=q, chem=True, dendrites=dend)

    def score():
        r = np.array([pop.forward(p, learn_trace=False) for p in pats]).T  # (neurons, P)
        return auc(r[:, value > 0], r[:, value < 0])

    for _ in range(DAYS):
        for i in rng.permutation(np.repeat(np.arange(P), REPS)):
            pop.forward(pats[i])
            pop.modulate(dopamine=value[i], ach=0.8, arousal=0.8)
            pop.tick()
            noise = np.zeros(N_INPUTS, np.float32)
            noise[rng.choice(N_INPUTS, ACTIVE, replace=False)] = 1.0
            pop.forward(noise)
            pop.tick()
        pop.sleep(reset_below=-1)
    learned = score()
    store = pop.storage()
    for _ in range(6):
        pop.rest(200)
        pop.sleep(reset_below=-1)
    return syn, P, label, world, learned, score(), store["changed %"], store["+5 gap-coded"]


def main():
    jobs = [(s, P, v, w) for s in SIZES for P in PATTERNS for v in VARIANTS for w in WORLDS]
    with Pool(int(os.environ.get("PROCS", "14"))) as pool:
        out = pool.map(job, sorted(jobs, key=lambda j: -j[0] * j[1]), chunksize=1)
    print("AUC good vs bad (0.5 chance, 1 perfect), mean over 8 neurons x 2 worlds; after practice / 6 days later")
    print("Storage after practice: synapses changed, long-term bytes per neuron (all ideas)\n")
    for syn in SIZES:
        print(f"== {syn} synapses ({SIZES[syn][0]} branches x {SIZES[syn][1]}) ==")
        print(f"   {'':<24}" + "".join(f"{f'P={P}':>14}" for P in PATTERNS) + "   storage at P=128")
        for label, _, _ in VARIANTS:
            cells, store = [], ""
            for P in PATTERNS:
                r = [o for o in out if o[:3] == (syn, P, label)]
                a = np.concatenate([o[4] for o in r])
                b = np.concatenate([o[5] for o in r])
                cells.append(f"{a.mean():.2f}/{b.mean():.2f}")
                if P == 128:
                    store = f"{np.mean([o[6] for o in r]):5.1f}%  {np.mean([o[7] for o in r]):6.0f} B"
            print(f"   {label:<24}" + "".join(f"{c:>14}" for c in cells) + f"   {store}")


if __name__ == "__main__":
    main()

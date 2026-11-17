"""
Smart dendrites, attempt 2 (notes/24): seeded choice with 64 candidates, in a world with structure.

4096 synapses, 16 levels, chem tags, gentle sleep reset, constant turnover (10% of slots per night,
3-night grace for newborn synapses), 12 days of practice, then 6 days without.
Worlds: "parts" (objects built from 32 shared parts, world.py) and "random" (unrelated patterns, as before).
Half the objects are good (dopamine +1), half bad (-1).

Measured:
  AUC      chance the neuron responds more to a good object than a bad one (after / 6 days later)
  cluster  for each part, the most of its synapses found on any one branch (random wiring: ~4-5)
  drive95  95th percentile of branch drive, vs the branch spike threshold (3.2)
  spikes   share of good / bad objects that make at least one branch spike

Run: PYTHONPATH=prototype python3 prototype/test_dendrites2.py
"""

import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import test_lean_capacity as C
from lean_neuron import LeanPopulation
from world import PartsWorld

SYN, DAYS, BUDGET, GRACE = int(os.environ.get("SYN", "4096")), 12, 0.10, 3
PATTERNS = (128, 512)
VARIANTS = (  # label, rewiring, dendrites, newborn strength, guide
    ("random rewiring, spikes", "random", "spikes", 0.1, "cofire"),
    ("seeded 64, spikes", "seeded", "spikes", 0.1, "cofire"),
    ("seeded 64, linear", "seeded", "linear", 0.1, "cofire"),
    ("seeded 64, spikes, strong newborn", "seeded", "spikes", 0.4, "cofire"),
    ("value-guided 64, spikes", "seeded", "spikes", 0.1, "value"),
    ("value-guided 64, linear", "seeded", "linear", 0.1, "value"),
    ("value-guided 64, spikes, strong newborn", "seeded", "spikes", 0.4, "value"),
)
SEEDS = (0, 1)


def job(args):
    world_kind, P, (label, rewiring, dend, newborn, guide), seed = args
    rng = np.random.default_rng(1000 * seed + P)
    f = C.N_INPUTS // 2048  # bigger worlds: parts and details scale with the input count (5% active)
    world = PartsWorld(rng, n_inputs=C.N_INPUTS, part_size=24 * f, details=30 * f, noise=4 * f)
    if world_kind == "parts":
        parts, sets = world.make_objects(P)
    else:
        parts, sets = None, [rng.choice(C.N_INPUTS, C.ACTIVE, replace=False) for _ in range(P)]
    value = np.where(np.arange(P) % 2 == 0, 1.0, -1.0)
    t = replace(C.scaled(SYN), rewire_budget=BUDGET)
    pop = LeanPopulation(t, C.NEURONS, C.N_INPUTS, np.random.default_rng(seed), first_id=100 * seed, q=7,
                         chem=True, dendrites=dend, rewiring=rewiring, turnover=True, newborn=newborn,
                         guide=guide)
    pop.grace, pop.anchor = GRACE, True
    exact = [world.see(s, exact=True) for s in sets]

    def score():
        r = np.array([pop.forward(x, learn_trace=False) for x in exact]).T
        return C.auc(r[:, value > 0], r[:, value < 0])

    for _ in range(DAYS):
        for i in rng.permutation(np.repeat(np.arange(P), C.REPS)):
            pop.forward(world.see(sets[i]))
            pop.modulate(dopamine=value[i], ach=0.8, arousal=0.8)
            pop.tick()
            noise = np.zeros(C.N_INPUTS, np.float32)
            noise[rng.choice(C.N_INPUTS, C.ACTIVE, replace=False)] = 1.0
            pop.forward(noise)
            pop.tick()
        pop.sleep(reset_below=-1)
    learned = score()
    member = np.zeros((len(world.parts), C.N_INPUTS), bool)
    for k, p in enumerate(world.parts):
        member[k, p] = True
    cluster = np.mean([member[k][pop.src].sum(axis=2).max(axis=1).mean() for k in range(len(world.parts))])
    w = pop._weights()
    drive = np.stack([(w * x[pop.src]).sum(axis=2) for x in exact])
    spikes = drive > t.branch_threshold
    store = pop.storage()
    for _ in range(6):
        pop.rest(200)
        pop.sleep(reset_below=-1)
    return (world_kind, P, label, seed, learned, score(), cluster, np.percentile(drive, 95),
            spikes[value > 0].any(axis=2).mean(), spikes[value < 0].any(axis=2).mean(), store["+5 gap-coded"])


def main():
    jobs = [("parts", P, v, s) for P in PATTERNS for v in VARIANTS for s in SEEDS]
    jobs += [("random", P, v, s) for P in PATTERNS for v in VARIANTS[:2] + VARIANTS[4:5] for s in SEEDS]
    with Pool(int(os.environ.get("PROCS", "14"))) as pool:
        out = pool.map(job, sorted(jobs, key=lambda j: -j[1]), chunksize=1)
    print(f"{SYN} synapses, 16 levels, {DAYS} days, turnover {BUDGET:.0%}/night, grace {GRACE} nights, 8 neurons x 2 seeds")
    print(f"   {'world':<7}{'':<42}{'P':>5}{'AUC':>12}{'cluster':>9}{'drive95':>9}{'spike good':>12}{'bad':>6}{'bytes':>8}")
    for world_kind in ("parts", "random"):
        for P in PATTERNS:
            for label, *_ in VARIANTS:
                r = [o for o in out if o[:3] == (world_kind, P, label)]
                if not r:
                    continue
                m = lambda k: np.mean([o[k] for o in r])
                a = np.concatenate([o[4] for o in r]).mean()
                z = np.concatenate([o[5] for o in r]).mean()
                print(f"   {world_kind:<7}{label:<42}{P:>5}{f'{a:.2f}/{z:.2f}':>12}{m(6):9.1f}{m(7):9.2f}"
                      f"{100 * m(8):11.0f}%{100 * m(9):5.0f}%{m(10):8.0f}")


if __name__ == "__main__":
    main()

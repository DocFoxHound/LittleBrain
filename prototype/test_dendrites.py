"""
Smart dendrites with seeded-choice rewiring (notes/24).

Same task as test_lean_capacity (4096 synapses, 16 levels, chem tags, gentle sleep reset), run longer so
rewiring has time to cluster inputs. Compared:
  random rewiring + branch spikes      (round 2: spikes never fired)
  seeded choice + branch spikes        (does clustering let branches spike, and does it help?)
  seeded choice + linear dendrites     (clustering alone, no branch spikes)
Also measured: how often a pattern makes at least one branch spike, and how clustered branches are.

Run: PYTHONPATH=prototype python3 prototype/test_dendrites.py
"""

import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import test_lean_capacity as C
from lean_neuron import LeanPopulation

SYN, DAYS = 4096, (4, 12)
PATTERNS = (128, 512)
VARIANTS = (("random rewiring, spikes", "random", "spikes"), ("seeded choice, spikes", "seeded", "spikes"),
            ("seeded choice, linear", "seeded", "linear"))
WORLDS = (0, 1)


def job(args):
    P, days, (label, rewiring, dend), world, budget = args
    rng = np.random.default_rng(1000 * world + P)
    pats = np.zeros((P, C.N_INPUTS), np.float32)
    for p in pats:
        p[rng.choice(C.N_INPUTS, C.ACTIVE, replace=False)] = 1.0
    value = np.where(np.arange(P) % 2 == 0, 1.0, -1.0)
    t = replace(C.scaled(SYN), rewire_budget=budget)
    pop = LeanPopulation(t, C.NEURONS, C.N_INPUTS, np.random.default_rng(world), first_id=100 * world,
                         q=7, chem=True, dendrites=dend, rewiring=rewiring, turnover=True)
    pop.grace = int(os.environ.get("GRACE", "0"))       # nights a newborn synapse is safe from turnover
    pop.anchor = os.environ.get("ANCHOR", "0") == "1"   # theme = the branch's single strongest input

    def score():
        r = np.array([pop.forward(p, learn_trace=False) for p in pats]).T
        return C.auc(r[:, value > 0], r[:, value < 0])

    for _ in range(days):
        for i in rng.permutation(np.repeat(np.arange(P), C.REPS)):
            pop.forward(pats[i])
            pop.modulate(dopamine=value[i], ach=0.8, arousal=0.8)
            pop.tick()
            noise = np.zeros(C.N_INPUTS, np.float32)
            noise[rng.choice(C.N_INPUTS, C.ACTIVE, replace=False)] = 1.0
            pop.forward(noise)
            pop.tick()
        pop.sleep(reset_below=-1)
    learned = score()
    # Clustering: for each pattern, the most active synapses any one branch receives, and branch spikes
    w = pop._weights()
    act = np.stack([p[pop.src].sum(axis=2) for p in pats])              # (P, N, B) active synapses per branch
    drive = np.stack([(w * p[pop.src]).sum(axis=2) for p in pats])
    spikes = drive > t.branch_threshold
    good = value > 0
    store = pop.storage()
    for _ in range(6):
        pop.rest(200)
        pop.sleep(reset_below=-1)
    return (P, days, label, world, budget, learned, score(), act.max(axis=2).mean(),
            spikes[good].any(axis=2).mean(), spikes[~good].any(axis=2).mean(),
            store["rewired %"], store["+5 gap-coded"])


def main():
    budgets = (0.03, 0.10)
    jobs = [(P, d, v, w, b) for P in PATTERNS for d in DAYS for v in VARIANTS for w in WORLDS for b in budgets]
    with Pool(int(os.environ.get("PROCS", "14"))) as pool:
        out = pool.map(job, sorted(jobs, key=lambda j: -j[0] * j[1]), chunksize=1)
    print(f"{SYN} synapses, 16 levels. AUC after practice / 6 days later (0.5 chance, 1 perfect).")
    print("max act = most active synapses one branch gets from a pattern (random wiring ~ 13);")
    print("spike good/bad = share of good/bad patterns that make at least one branch spike\n")
    for b in budgets:
        print(f"== rewiring budget {b:.0%} of slots per night ==")
        print(f"   {'':<26}{'P':>5}{'days':>6}{'AUC':>12}{'max act':>9}{'spike good':>12}{'spike bad':>11}"
              f"{'rewired':>9}{'bytes':>8}")
        for P in PATTERNS:
            for d in DAYS:
                for label, _, _ in VARIANTS:
                    r = [o for o in out if o[:3] == (P, d, label) and o[4] == b]
                    a = np.concatenate([o[5] for o in r]).mean()
                    z = np.concatenate([o[6] for o in r]).mean()
                    m = lambda k: np.mean([o[k] for o in r])
                    print(f"   {label:<26}{P:>5}{d:>6}{f'{a:.2f}/{z:.2f}':>12}{m(7):9.1f}{100 * m(8):11.0f}%"
                          f"{100 * m(9):10.0f}%{m(10):8.1f}%{m(11):8.0f}")


if __name__ == "__main__":
    main()

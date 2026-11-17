"""
Single-neuron demo: does an overwritten "old self" come back?

Story:
  1. Childhood: pattern A is good (rewarded) for several days, sleeping each night.
  2. Hard times: pattern A becomes bad (punished) for a couple of days.
  3. Time passes: no more A, just ordinary days and nights.
  4. Relearning: A becomes good again. How many trials until it's liked again?

Compared: our workhorse neuron (3 cascade levels) vs a "plain" neuron (1 level, no deep memory).

Run: python3 demo_single_neuron.py
"""

from dataclasses import replace

import numpy as np

from neuron import WORKHORSE, Neuron

N_INPUTS = 64
TRIALS_PER_DAY = 40


def make_pattern(rng, k=8):
    x = np.zeros(N_INPUTS, np.float32)
    x[rng.choice(N_INPUTS, k, replace=False)] = 1.0
    return x


def probe(n, x):
    return n.forward(x, learn_trace=False)


def live_day(n, rng, pattern, dopamine, trials=TRIALS_PER_DAY):
    """Interleave the pattern with random background experiences, then sleep."""
    for _ in range(trials):
        n.forward(pattern)
        if dopamine:
            n.modulate(dopamine=dopamine, ach=0.8, arousal=0.8)
        n.tick()
        n.forward(make_pattern(rng))  # ordinary, unremarkable moment
        n.tick()
    n.sleep()


def trials_to_like(n, rng, pattern, criterion=0.7, max_trials=400):
    for i in range(max_trials):
        if probe(n, pattern) >= criterion:
            return i
        n.forward(pattern)
        n.modulate(dopamine=1.0, ach=0.8, arousal=0.8)
        n.tick()
        n.forward(make_pattern(rng))
        n.tick()
    return max_trials


def run(label, ntype, seed=1, make=Neuron, verbose=True):
    rng = np.random.default_rng(seed)
    A = make_pattern(np.random.default_rng(0))
    n = make(ntype, N_INPUTS, rng)
    log = [("start", probe(n, A))]

    for _ in range(6):
        live_day(n, rng, A, dopamine=+1.0)
    log.append(("after childhood (A good)", probe(n, A)))

    for _ in range(2):
        live_day(n, rng, A, dopamine=-1.0)
    log.append(("after hard times (A bad)", probe(n, A)))

    for day in range(1, 7):
        live_day(n, rng, A, dopamine=0.0, trials=0)
        n.rest(200)
        if day in (1, 3, 6):
            log.append((f"time passes: day {day}", probe(n, A)))

    relearn = trials_to_like(n, rng, A)

    if verbose:
        print(f"\n== {label} ==")
        for name, r in log:
            print(f"  {name:<28} response to A = {r:5.2f}  {'#' * int(r * 30)}")
        print(f"  trials to like A again:      {relearn}")
    return log, relearn


if __name__ == "__main__":
    fresh = trials_to_like(Neuron(WORKHORSE, N_INPUTS, np.random.default_rng(1)),
                           np.random.default_rng(1), make_pattern(np.random.default_rng(0)))
    print(f"A brand-new neuron needs {fresh} trials to learn to like A.")
    n = Neuron(WORKHORSE, N_INPUTS, np.random.default_rng(1))
    print(f"Memory per neuron: {n.memory_bytes():,} bytes "
          f"({WORKHORSE.n_branches * WORKHORSE.slots} synapses)")

    run("Workhorse neuron (3 cascade levels)", WORKHORSE)
    run("Plain neuron (no deep memory)", replace(WORKHORSE, name="plain", cascade_levels=1))

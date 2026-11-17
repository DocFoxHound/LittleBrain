"""
Learning rules v1 vs v2 (notes/16-learning-rules-v2.md).

  1. Story:    "childhood -> hard times -> time passes -> relearn" must still work.
  2. Dose:     one hard day of N trials. More experience should consolidate more (v1 saturated).
  3. Spacing:  the same 80 hard trials over 2 nights vs 3. Spacing over more sleeps should
               consolidate somewhat more (the spacing effect), not overwhelmingly more.
  4. Sparsity: a population in a sparse world with competition (only the top 10% may fire).
               Learning should concentrate in the neurons that fired.

Run: python3 test_learning_rules.py
"""

import numpy as np

from demo_single_neuron import live_day, make_pattern, probe, trials_to_like
from neuron import WORKHORSE, WORKHORSE_V2, Neuron

N_INPUTS = 64
SEEDS = range(1, 17)
RULES = [("v1", WORKHORSE), ("v2", WORKHORSE_V2)]


def trial(n, rng, pattern, dopamine):
    n.forward(pattern)
    if dopamine:
        n.modulate(dopamine=dopamine, ach=0.8, arousal=0.8)
    n.tick()
    n.forward(make_pattern(rng))
    n.tick()


def childhood(ntype, seed):
    rng = np.random.default_rng(seed)
    A = make_pattern(np.random.default_rng(0))
    n = Neuron(ntype, N_INPUTS, rng)
    for _ in range(6):
        live_day(n, rng, A, dopamine=+1.0)
    return n, rng, A


def life(ntype, seed, hard_split=(40, 40)):
    """Childhood, hard times split into nights as given, 6 days of rest. Returns hard, day 6, relearn."""
    n, rng, A = childhood(ntype, seed)
    for trials in hard_split:
        for _ in range(trials):
            trial(n, rng, A, -1.0)
        n.sleep()
    after_hard = probe(n, A)
    for _ in range(6):
        n.sleep()
        n.rest(200)
    return after_hard, probe(n, A), trials_to_like(n, rng, A)


def test_story(rules):
    print("1. Story (16 seeds): response to A after hard times, 6 days later; trials to relearn")
    for name, t in rules:
        r = np.array([life(t, s) for s in SEEDS])
        print(f"   {name:<4} hard {r[:, 0].mean():.2f}   day 6 {r[:, 1].mean():.2f} "
              f"[{r[:, 1].min():.2f}-{r[:, 1].max():.2f}]   relearn median {np.median(r[:, 2]):.0f} "
              f"[{r[:, 2].min():.0f}-{r[:, 2].max():.0f}]")


def test_dose(rules, doses=(5, 10, 20, 40, 80)):
    print("\n2. Dose: surface weight on A's synapses after one hard day of N trials and a night "
          "(starts near +1)")
    print("   " + " " * 5 + "".join(f"{f'N={d}':>9}" for d in doses))
    for name, t in rules:
        row = []
        for d in doses:
            vals = []
            for s in SEEDS:
                n, rng, A = childhood(t, s)
                for _ in range(d):
                    trial(n, rng, A, -1.0)
                n.sleep()
                vals.append(n.cascade[0][A[n.src] > 0].mean())
            row.append(np.mean(vals))
        print(f"   {name:<5}" + "".join(f"{v:9.2f}" for v in row))


def test_spacing(rules):
    print("\n3. Spacing: 80 hard trials over 2 nights (40+40) vs 3 nights (20+40+20); day-6 recovery")
    for name, t in rules:
        two = np.array([life(t, s, (40, 40))[1] for s in SEEDS])
        three = np.array([life(t, s, (20, 40, 20))[1] for s in SEEDS])
        print(f"   {name:<4} 2 nights {two.mean():.2f}   3 nights {three.mean():.2f}   "
              f"ratio {three.mean() / two.mean():.2f}")


def test_sparsity(rules, n_neurons=64, n_inputs=1024, active=20, k_frac=0.1, days=3):
    print(f"\n4. Sparsity: {n_neurons} neurons, {n_inputs} inputs, {active} active "
          f"({100 * active / n_inputs:.0f}%), only the top {100 * k_frac:.0f}% may fire, "
          f"dopamine reaches everyone, {days} days")
    k = max(1, int(k_frac * n_neurons))
    for name, t in rules:
        rng = np.random.default_rng(7)
        pop = [Neuron(t, n_inputs, np.random.default_rng(100 + i)) for i in range(n_neurons)]
        # Twins with the same birth state live the same days and nights with no experience, so
        # comparing against them counts only what experience changed (sleep alone moves values too)
        twins = [Neuron(t, n_inputs, np.random.default_rng(100 + i)) for i in range(n_neurons)]
        src0 = [n.src.copy() for n in pop]

        def step(x, dopamine):
            rates = np.array([n.forward(x, learn_trace=False) for n in pop])
            winners = np.argsort(rates)[-k:]  # population inhibition: only the strongest fire
            post = np.zeros(n_neurons)
            post[winners] = rates[winners]
            for n, r in zip(pop, post):
                n.forward(x, post_rate=float(r))
                if dopamine:
                    n.modulate(dopamine=dopamine, ach=0.8, arousal=0.8)
                n.tick()
            return set(winners.tolist())

        A = np.zeros(n_inputs, np.float32)
        A[rng.choice(n_inputs, active, replace=False)] = 1.0
        fired_for_A, rewired = set(), 0
        for _ in range(days):
            for _ in range(40):
                fired_for_A |= step(A, 1.0)
                noise = np.zeros(n_inputs, np.float32)
                noise[rng.choice(n_inputs, active, replace=False)] = 1.0
                step(noise, 0.0)
            rewired += sum(n.sleep() for n in pop)
            for tw in twins:
                tw.rest(80)
                tw.sleep()
        changed = np.array([((np.abs(n.cascade - tw.cascade).max(axis=0) > 0.05) & (n.src == s)).mean()
                            for n, tw, s in zip(pop, twins, src0)])
        win = np.array([i in fired_for_A for i in range(n_neurons)])
        total = n_neurons * pop[0].src.size
        print(f"   {name:<4} synapses changed {100 * changed.mean():5.1f}%  "
              f"(neurons that fired for A: {100 * changed[win].mean():5.1f}%, "
              f"never fired: {100 * changed[~win].mean():5.1f}%)   rewired {100 * rewired / total:4.1f}% "
              f"of slots   neurons that ever fired for A: {win.sum()}/{n_neurons}")


if __name__ == "__main__":
    test_story(RULES)
    test_dose(RULES)
    test_spacing(RULES)
    test_sparsity(RULES)

"""
Lean neuron tests (notes/24), 256 synapses per neuron.

  1. Story at each precision: does the old self still come back? (packed neuron = reference)
  2. Dose: do small experiences survive low precision?
  3. A life in a sparse world: what fraction of synapses experience changes, how big the day
     buffer is, and how many bytes each storage format needs (ideas 1, 2, 4, 5 and 6 of notes/24).

Run: PYTHONPATH=prototype python3 prototype/test_lean.py
"""

from functools import partial

import numpy as np

from demo_single_neuron import N_INPUTS, live_day, make_pattern, probe, run
from lean_neuron import LeanPopulation, LeanNeuron
from neuron import WORKHORSE_V2 as T
from packed_neuron import PackedNeuron

SEEDS = range(1, 9)
VARIANTS = [("packed (reference)", PackedNeuron)] + [
    (f"lean {name}{' chem' if chem else ''}", partial(LeanNeuron, q=q, chem=chem))
    for name, q in (("64 lv", 31), ("32 lv", 15), ("16 lv", 7), ("3 lv", 1)) for chem in (False, True)]


def test_story():
    print("1. Story (8 seeds): response to A after hard times, 6 days later; trials to relearn")
    for name, make in VARIANTS:
        rows = [run("", T, seed=s, make=make, verbose=False) for s in SEEDS]
        logs = [dict(log) for log, _ in rows]
        hard = np.mean([l["after hard times (A bad)"] for l in logs])
        day6 = np.array([l["time passes: day 6"] for l in logs])
        relearn = np.array([r for _, r in rows])
        print(f"   {name:<22} hard {hard:.2f}   day 6 {day6.mean():.2f} [{day6.min():.2f}-{day6.max():.2f}]"
              f"   relearn median {np.median(relearn):.0f} [{relearn.min()}-{relearn.max()}]")


def test_dose(doses=(2, 5, 20, 80)):
    print("\n2. Dose: response to A after childhood + one hard day of N trials + a night (lower = learned more)")
    print("   " + " " * 22 + "".join(f"{f'N={d}':>8}" for d in doses))
    for name, make in VARIANTS:
        row = []
        for dose in doses:
            vals = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                A = make_pattern(np.random.default_rng(0))
                n = make(T, N_INPUTS, rng)
                for _ in range(6):
                    live_day(n, rng, A, +1.0)
                live_day(n, rng, A, -1.0, trials=dose)
                vals.append(probe(n, A))
            row.append(np.mean(vals))
        print(f"   {name:<22}" + "".join(f"{v:8.2f}" for v in row))


def test_life(q=7, chem=True, n_neurons=128, n_inputs=1024, active=20, k_frac=0.1, days=12,
              reset_below=0):
    """Recurring 'places' (some rewarded) among random noise; population competition (top 10% fire)."""
    rng = np.random.default_rng(7)
    pop = LeanPopulation(T, n_neurons, n_inputs, np.random.default_rng(1), q=q, chem=chem)
    k = max(1, int(k_frac * n_neurons))
    places = np.zeros((6, n_inputs), np.float32)
    for p in places:
        p[rng.choice(n_inputs, active, replace=False)] = 1.0
    value = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0]

    def step(x, dopamine):
        rates = pop.forward(x, learn_trace=False)
        post = np.zeros(n_neurons, np.float32)
        win = np.argsort(rates)[-k:]
        post[win] = rates[win]
        pop.forward(x, post=post)
        if dopamine:
            pop.modulate(dopamine=dopamine, ach=0.8, arousal=0.8)
        pop.tick()
        return win

    fired = np.zeros(n_neurons, bool)
    for day in range(days):
        for _ in range(60):
            i = rng.integers(len(places))
            fired[step(places[i], value[i])] = True
            noise = np.zeros(n_inputs, np.float32)
            noise[rng.choice(n_inputs, active, replace=False)] = 1.0
            step(noise, 0.0)
        buf_bits, entries = pop.day_buffer_bits()
        pop.sleep(reset_below=reset_below)
    s = pop.storage()
    ch = (pop.d != 0).any(axis=0).reshape(n_neurons, -1).mean(axis=1) * 100
    return s, buf_bits.mean() / 8, entries.mean(), ch[fired].mean(), ch[~fired].mean()


def report_life():
    print("\n3. A life: 128 neurons x 256 synapses, 12 days, 6 recurring places + noise, top 10% fire")
    print("   bytes per neuron of long-term storage, cumulative ideas left to right")
    keys = ["packed today (32 bits each)", "seed+changes, 32 bits each", "+1 day-only out", "+2 settled",
            "+4 gen list", "+5 gap-coded", "+5 zlib"]
    short = ["packed", "s+c", "+1", "+2", "+4", "+5gap", "+5zlib"]
    print(f"   {'':<24}{'changed':>8}{'settled':>8}{'±1':>6}" + "".join(f"{h:>8}" for h in short)
          + f"{'buffer':>8}")
    for label, q, chem, reset in [("64 lv", 31, False, 0), ("32 lv", 15, False, 0), ("16 lv", 7, False, 0),
                                  ("16 lv chem", 7, True, 0), ("3 lv chem", 1, True, 0),
                                  ("16 lv chem +6 reset≤1", 7, True, 1)]:
        s, buf, entries, ch_fired, ch_quiet = test_life(q, chem, reset_below=reset)
        print(f"   {label:<24}{s['changed %']:7.1f}%{s['settled % of changed']:7.0f}%"
              f"{s['settled±1 % of changed']:5.0f}%" + "".join(f"{s[k]:8.0f}" for k in keys)
              + f"{buf:8.0f}   (changed in cells that fired {ch_fired:.0f}%, never fired {ch_quiet:.0f}%)")


if __name__ == "__main__":
    test_story()
    test_dose()
    report_life()

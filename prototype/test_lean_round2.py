"""
Lean neuron, round 2 (notes/24): after the rewiring and chemical-tag fixes.

  1. Idea 6 (sleep resets differences of <= 1 step): is the drop in the old self's comeback significant?
     Same seeds with and without the reset (paired), sign-flip permutation test.
  2. Storage again, 16 levels, with the fixes.

Run: PYTHONPATH=prototype python3 prototype/test_lean_round2.py
"""

from functools import partial

import numpy as np

import test_lean as L
from demo_single_neuron import run
from lean_neuron import LeanNeuron
from neuron import WORKHORSE_V2 as T

SEEDS = range(1, 33)


def paired(a, b, rng=np.random.default_rng(0), n=20000):
    d = np.asarray(b) - np.asarray(a)
    flips = rng.choice([-1, 1], size=(n, len(d)))
    p = np.mean(np.abs((flips * d).mean(axis=1)) >= abs(d.mean()))
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), p


def test_reset():
    print(f"1. Idea 6, 16 levels, chem ({len(SEEDS)} seeds, paired)")
    res = {}
    for name, reset in (("no reset", 0), ("reset <= 1 step", 1), ("surface-only reset", -1)):
        rows = [run("", T, seed=s, make=partial(LeanNeuron, q=7, chem=True, reset_below=reset), verbose=False)
                for s in SEEDS]
        logs = [dict(log) for log, _ in rows]
        res[name] = {k: np.array([l[k] for l in logs]) for k in
                     ("after childhood (A good)", "after hard times (A bad)", "time passes: day 6")}
        res[name]["relearn"] = np.array([r for _, r in rows], float)
        print(f"   {name:<16} childhood {res[name]['after childhood (A good)'].mean():.2f}   "
              f"hard {res[name]['after hard times (A bad)'].mean():.2f}   "
              f"day 6 {res[name]['time passes: day 6'].mean():.2f}   relearn mean {res[name]['relearn'].mean():.1f}")
    for other in ("reset <= 1 step", "surface-only reset"):
        for k in ("time passes: day 6", "relearn"):
            m, se, p = paired(res["no reset"][k], res[other][k])
            print(f"   {other}: difference in {k:<20} {m:+.3f} ± {se:.3f}   p = {p:.3f}")


def test_storage():
    print("\n2. Storage after fixes (128 neurons x 256 synapses, 12 days)")
    keys = ["packed today (32 bits each)", "+5 gap-coded"]
    for label, q, chem, reset in (("16 lv", 7, False, 0), ("16 lv chem", 7, True, 0),
                                  ("16 lv chem +6", 7, True, 1), ("16 lv chem +6 surface", 7, True, -1),
                                  ("32 lv chem", 15, True, 0)):
        s, buf, entries, ch_f, _ = L.test_life(q, chem, reset_below=reset)
        print(f"   {label:<16} changed {s['changed %']:5.1f}%  rewired {s['rewired %']:4.1f}%   "
              f"packed {s[keys[0]]:5.0f} B -> lean {s[keys[1]]:4.0f} B per neuron   day buffer {buf:4.0f} B")


if __name__ == "__main__":
    test_reset()
    test_storage()

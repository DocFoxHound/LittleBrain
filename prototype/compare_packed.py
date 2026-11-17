"""
Compare the original float neuron with the packed (32 bits per synapse) neuron:
  1. Real memory used, measured with tracemalloc (includes Python overhead).
  2. Behavior: the same "childhood -> hard times -> time passes -> relearn" story, over several seeds.

Run: python3 compare_packed.py
"""

import tracemalloc
from dataclasses import replace

import numpy as np

from demo_single_neuron import N_INPUTS, run
from neuron import WORKHORSE, WORKHORSE_V2, Neuron
from packed_neuron import PackedNeuron, PackedPopulation

SEEDS = range(1, 9)


def measured(build):
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    obj = build()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return obj, sum(s.size_diff for s in after.compare_to(before, "filename"))


def memory_report():
    print("== Memory ==")
    k = 1000
    _, b = measured(lambda: [Neuron(WORKHORSE, N_INPUTS, np.random.default_rng(i)) for i in range(k)])
    print(f"  float neuron, one Python object each:   {b / k:8,.0f} bytes per neuron (measured)")
    _, b = measured(lambda: [PackedNeuron(WORKHORSE, N_INPUTS, np.random.default_rng(i)) for i in range(k)])
    print(f"  packed neuron, one Python object each:  {b / k:8,.0f} bytes per neuron (measured)")
    n = 100_000
    pop, b = measured(lambda: PackedPopulation(WORKHORSE, n, N_INPUTS, np.random.default_rng(0)))
    print(f"  packed population of {n:,}:         {b / n:8,.0f} bytes per neuron (measured)")
    print(f"    = {pop.bytes_per_neuron():,.0f} bytes of state per neuron "
          f"+ {pop.shared_bytes():,} bytes shared by all")
    per = pop.bytes_per_neuron()
    print(f"  1,000,000 neurons x 256 synapses: float ~{9248 * 1e6 / 2**30:.1f} GiB, "
          f"packed ~{per * 1e6 / 2**30:.2f} GiB")


def behavior_report():
    print("\n== Behavior (mean over seeds 1-8; min-max in brackets) ==")
    cols = ["after hard times (A bad)", "time passes: day 3", "time passes: day 6"]
    print(f"  {'':<26}{'hard times':>16}{'day 3':>16}{'day 6':>16}{'relearn trials':>18}")
    for label, ntype in [("v1 workhorse", WORKHORSE), ("v1 plain", replace(WORKHORSE, cascade_levels=1)),
                         ("v2 workhorse", WORKHORSE_V2)]:
        for impl, make in [("float", Neuron), ("packed", PackedNeuron)]:
            rows = [run("", ntype, seed=s, make=make, verbose=False) for s in SEEDS]
            stats = []
            for c in cols:
                v = np.array([dict(log)[c] for log, _ in rows])
                stats.append(f"{v.mean():.2f} [{v.min():.2f}-{v.max():.2f}]")
            r = np.array([relearn for _, relearn in rows])
            print(f"  {label + ' / ' + impl:<26}" + "".join(f"{s:>16}" for s in stats)
                  + f"{np.median(r):>10.0f} median [{r.min()}-{r.max()}]")


if __name__ == "__main__":
    memory_report()
    behavior_report()

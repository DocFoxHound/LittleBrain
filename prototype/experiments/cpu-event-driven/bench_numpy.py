"""
The NumPy baseline (prototype/packed_neuron.py) measured in the same session, same scenario as
bench_layouts.py: 100,000 neurons, 4,096 inputs, 2% binary activity. Also 64 inputs (the brief's setup).

Run: .venv/bin/python prototype/experiments/cpu-event-driven/bench_numpy.py
"""

import common
from common import MiB, peak_tracemalloc, timeit

import json

import numpy as np

from neuron import WORKHORSE
from packed_neuron import PackedPopulation

N = 100_000


def main():
    out = []
    for nin in (64, 4096):
        pop = PackedPopulation(WORKHORSE, N, nin, np.random.default_rng(0))
        rng = np.random.default_rng(1)
        xs = [(rng.random(nin) < max(0.02, 2 / nin)).astype(np.float32) for _ in range(6)]
        it = iter(range(10 ** 9))
        tf = timeit(lambda: pop.forward(xs[next(it) % 6]), reps=3)
        tm = timeit(lambda: pop.modulate(1.0, 0.8, 0.8), reps=3, setup=pop.tick)
        ts = timeit(pop.sleep, reps=3)
        _, pf = peak_tracemalloc(lambda: pop.forward(xs[0]))
        pop.tick()
        _, pm = peak_tracemalloc(lambda: pop.modulate(1.0, 0.8, 0.8))
        _, psl = peak_tracemalloc(pop.sleep)
        rec = dict(n_inputs=nin, forward_s=tf[0], modulate_s=tm[0], sleep_s=ts[0],
                   peak_forward_MiB=pf / MiB, peak_modulate_MiB=pm / MiB, peak_sleep_MiB=psl / MiB,
                   bytes_per_neuron=pop.bytes_per_neuron())
        out.append(rec)
        print(f"NumPy baseline, n_inputs={nin}: forward {tf[0] * 1e3:.0f} ms ({pf / MiB:.0f} MiB temp), "
              f"modulate {tm[0] * 1e3:.0f} ms ({pm / MiB:.0f} MiB), sleep {ts[0] * 1e3:.0f} ms ({psl / MiB:.0f} MiB)",
              flush=True)
        del pop
    return out


if __name__ == "__main__":
    res = main()
    with open(f"{common.HERE}/results_numpy.json", "w") as f:
        json.dump(res, f, indent=1)

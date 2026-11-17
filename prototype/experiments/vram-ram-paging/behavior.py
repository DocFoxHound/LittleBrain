"""
Behavior check: the demo story ("childhood -> hard times -> time passes -> relearn") with the
neuron running on the GPU kernels (per-block lazy clocks, counter-based stochastic rounding),
compared with PackedNeuron (NumPy) over the same seeds.

Run: .venv/bin/python prototype/experiments/vram-ram-paging/behavior.py
"""

import os
import time
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cupy as cp

from gpu_population import GPUPopulation, pn
from neuron import WORKHORSE
from demo_single_neuron import run

SEEDS = range(1, 17)


class GPUNeuron:
    """Same interface as PackedNeuron; the state lives on the GPU."""

    def __init__(self, ntype, n_inputs, rng):
        first_id = int(rng.integers(1 << 31))
        ref = pn.PackedPopulation(ntype, 1, n_inputs, rng, first_id=first_id)  # same rng use as PackedNeuron
        self.g = GPUPopulation(ntype, 1, n_inputs=n_inputs, block_size=1, mode="resident", init=False,
                               ids=[first_id], seed=first_id)
        self.g.load_words(ref.words.reshape(1, -1))

    def forward(self, x, context=0.0, learn_trace=True):
        return float(self.g.forward(x, context, learn_trace)[0].get())

    def modulate(self, dopamine=0.0, ach=0.0, arousal=0.0):
        self.g.modulate(dopamine, ach, arousal)

    def tick(self):
        self.g.tick()

    def rest(self, steps):
        self.g.rest(steps)

    def sleep(self):
        return self.g.sleep()

    def memory_bytes(self):
        st = self.g.vram_state_bytes()
        return int(st["words"] + st["per_neuron"])


def report():
    cols = ["after hard times (A bad)", "time passes: day 3", "time passes: day 6"]
    print(f"== Behavior (mean over seeds {SEEDS.start}-{SEEDS.stop - 1}; min-max in brackets) ==")
    print(f"  {'':<24}{'hard times':>18}{'day 3':>18}{'day 6':>18}{'relearn trials':>24}")
    out = {}
    for label, ntype in [("workhorse", WORKHORSE), ("plain", replace(WORKHORSE, name="plain", cascade_levels=1))]:
        for impl, make in [("packed numpy", pn.PackedNeuron), ("packed GPU", GPUNeuron)]:
            t0 = time.perf_counter()
            rows = [run("", ntype, seed=s, make=make, verbose=False) for s in SEEDS]
            stats = []
            for c in cols:
                v = np.array([dict(log)[c] for log, _ in rows])
                stats.append(f"{v.mean():.2f} [{v.min():.2f}-{v.max():.2f}]")
            r = np.array([relearn for _, relearn in rows])
            ok = int((r < 400).sum())
            print(f"  {label + ' / ' + impl:<24}" + "".join(f"{s:>18}" for s in stats)
                  + f"{np.median(r):>8.0f} median [{r.min()}-{r.max()}], {ok}/{len(r)} relearned"
                  + f"   ({time.perf_counter() - t0:.1f} s)")
            out[(label, impl)] = rows
    return out


if __name__ == "__main__":
    report()

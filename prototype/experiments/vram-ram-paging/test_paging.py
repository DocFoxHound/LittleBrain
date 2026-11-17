"""
Paging must not change what the neurons compute. Run the same structured scenario (forward +
modulate + tick, then sleep) with every storage mode from the same initial state, with a tiny
VRAM cache (8 blocks, below the working set, so eviction and thrashing paths run), and require
bit-identical words, thresholds and rates.

Run: .venv/bin/python prototype/experiments/vram-ram-paging/test_paging.py
"""

import numpy as np
import cupy as cp

from gpu_population import GPUPopulation
from neuron import WORKHORSE
from scenario import AssemblyScenario

N = 64 * 1024
STEPS = 30


def run(mode, K, init_words, ids):
    pop = GPUPopulation(WORKHORSE, N, col_size=64, npc=128, mode=mode, cache_blocks=K, ids=ids, init=False, seed=5)
    pop.load_words(init_words)
    scen = AssemblyScenario(N, active_frac=0.15, churn=0.2, seed=3)
    scen.set_layout(np.random.default_rng(9).permutation(scen.n_cols))
    rates = []
    for s in range(STEPS):
        cols, blocks = scen.step()
        rates.append(cp.asnumpy(pop.forward(cols=cols, active=blocks)).copy())
        if s % 3 == 0:
            pop.modulate(0.6, 0.7, 0.5)
        if s == 17:
            pop.modulate(1.0, 0.95, 0.9)  # one-shot path
        pop.tick()
        if s == 20:
            pop.rest(300)
    pruned = pop.sleep()
    cols, blocks = scen.step()
    rates.append(cp.asnumpy(pop.forward(cols=cols, active=blocks)).copy())
    out = dict(words=pop.words_of().copy(), thr=cp.asnumpy(pop.thr), rates=np.array(rates), pruned=pruned,
               h2d=pop.h2d, d2h=pop.d2h)
    del pop
    return out


if __name__ == "__main__":
    scen = AssemblyScenario(N, seed=3)
    scen.set_layout(np.random.default_rng(9).permutation(scen.n_cols))
    ids = scen.storage_ids()
    ref_pop = GPUPopulation(WORKHORSE, N, col_size=64, npc=128, mode="resident", ids=ids, seed=1)
    init_words = ref_pop.words_of().copy()
    del ref_pop
    ref = run("resident", 0, init_words, ids)
    print(f"resident: pruned {ref['pruned']}, mean rate {ref['rates'].mean():.4f}")
    for mode, K in [("paged", 8), ("paged", 40), ("zerocopy", 0), ("managed", 8), ("managed", 0),
                    ("managed_lazy", 8)]:
        r = run(mode, K, init_words, ids)
        same = (np.array_equal(r["words"], ref["words"]) and np.array_equal(r["thr"], ref["thr"])
                and np.array_equal(r["rates"], ref["rates"]) and r["pruned"] == ref["pruned"])
        print(f"{mode:>12} K={K:<3}: bit-identical to resident: {same}   "
              f"(PCIe in {r['h2d'] / 1e6:7.1f} MB, out {r['d2h'] / 1e6:7.1f} MB)")
        assert same, mode
    print("OK")

"""
Strategy 4: cache awareness. Two experiments, 1 thread, 4,096 inputs, interleaved timing.

  (a) Population size sweep across the 24 MB L3: does per-synapse throughput change when all the
      synapses fit in cache? Includes a plain streaming read of the same bytes as the memory ceiling.
  (b) Axon-major block ("tile") size: neurons are processed block by block; the block's dendritic
      accumulator (BLK x 8 floats) must stay in L1/L2 while events are scattered into it, but every
      block also needs a bucket index over all inputs (12 bytes x inputs per block).

Run: .venv/bin/python prototype/experiments/cpu-event-driven/bench_cache.py
"""

import common
from common import set_threads

import json
import time

import numpy as np
from numba import njit, prange

from axon_major import AxonPopulation
from bitslice import BitSlicedPopulation
from dm_fused import FusedPopulation
from neuron import WORKHORSE
from packed_neuron import PackedPopulation

NIN = 4096
S = WORKHORSE.n_branches * WORKHORSE.slots


@njit(parallel=True, cache=True)
def stream_read(words, nthreads):
    n = words.shape[0]
    acc = np.uint64(0)
    step = (n + nthreads - 1) // nthreads
    for t in prange(nthreads):
        lo, hi = min(n, t * step), min(n, (t + 1) * step)
        ws = words[lo:hi]
        for j in range(hi - lo):
            acc += np.uint64(ws[j])
    return acc


def med_interleaved(fns, reps):
    ts = {k: [] for k in fns}
    for f in fns.values():
        f()
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            ts[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in ts.items()}


def size_sweep(out):
    set_threads(1)
    rng = np.random.default_rng(7)
    for n in (2_000, 8_000, 23_000, 64_000, 200_000):
        base = PackedPopulation(WORKHORSE, n, NIN, np.random.default_rng(0))
        dm = FusedPopulation(WORKHORSE, n, NIN, np.random.default_rng(0))
        am = AxonPopulation(WORKHORSE, n, NIN, np.random.default_rng(0), block=4096, from_packed=base, nthreads=1)
        bs = BitSlicedPopulation(WORKHORSE, n, NIN, np.random.default_rng(0), nthreads=1)
        del base
        xs = [(rng.random(NIN) < 0.10).astype(np.float32) for _ in range(8)]
        Xs = [bs.bitset(x) for x in xs]
        c = {"i": 0}

        def nxt():
            c["i"] += 1
            return c["i"] % 8
        reps = int(np.clip(4e8 / (n * S), 5, 200))
        t = med_interleaved({
            "stream_read": lambda: stream_read(dm.words.reshape(-1), 1),
            "dendrite_fwd": lambda: dm.forward(xs[nxt()]),
            "axon_fwd": lambda: am.forward(xs[nxt()]),
            "bitslice_fwd": lambda: bs.forward(None, X=Xs[nxt()]),
            "dendrite_cu": lambda: (setattr(dm, "pending", 300), dm._catch_up()),
            "bitslice_cu": lambda: (setattr(bs, "pending", 300), bs._catch_up()),
        }, reps)
        mb = dm.words.nbytes / 2 ** 20
        rec = dict(n_neurons=n, synapse_MiB=mb, **{k: v for k, v in t.items()})
        out["size_sweep"].append(rec)
        print(f"N={n:>7} ({mb:6.1f} MiB of synapses) ns/synapse: "
              + "  ".join(f"{k} {v / (n * S) * 1e9:.3f}" for k, v in t.items()), flush=True)
        del dm, am, bs


def block_sweep(out):
    set_threads(1)
    n = 100_000
    base = PackedPopulation(WORKHORSE, n, NIN, np.random.default_rng(0))
    pops = {blk: AxonPopulation(WORKHORSE, n, NIN, np.random.default_rng(0), block=blk, from_packed=base, nthreads=1)
            for blk in (256, 1024, 4096, 8192)}
    del base
    rng = np.random.default_rng(8)
    for act in (0.02, 0.20):
        xs = [(rng.random(NIN) < act).astype(np.float32) for _ in range(8)]
        c = {"i": 0}

        def f(p):
            def g():
                c["i"] += 1
                p.forward(xs[c["i"] % 8])
            return g
        tf = med_interleaved({b: f(p) for b, p in pops.items()}, 7)
        for p in pops.values():
            for x in xs[:4]:
                p.forward(x)

        def m(p):
            def g():
                p.tick()
                p.modulate(1.0, 0.8, 0.8)
            return g
        tm = med_interleaved({b: m(p) for b, p in pops.items()}, 5)
        for b, p in pops.items():
            out["block_sweep"].append(dict(block=b, activity=act, forward_s=tf[b], modulate_s=tm[b],
                                           index_bytes_per_neuron=(p.off.nbytes + p.stamp.nbytes) / n,
                                           accumulator_KiB=b * WORKHORSE.n_branches * 5 / 1024))
        print(f"act {act:.0%}: " + "  ".join(f"BLK={b}: fwd {tf[b] * 1e3:.2f} ms, mod {tm[b] * 1e3:.2f} ms"
                                             for b in pops), flush=True)


if __name__ == "__main__":
    out = {"size_sweep": [], "block_sweep": []}
    size_sweep(out)
    block_sweep(out)
    with open(f"{common.HERE}/results_cache.json", "w") as f:
        json.dump(out, f, indent=1)

"""
Layouts head to head at different input activity levels (strategies 1-3).

Scenario: 100,000 workhorse neurons x 256 synapses (25.6 M synapses), 4,096 inputs, random wiring,
binary spikes, a NEW random spike pattern every step (worst case for the 4-step eligibility union).
  dendrite : fused Numba kernel on the baseline layout (hash wiring)
  axon     : axon-major, event-driven (same hash wiring, stored by input)
  bitslice : bit-sliced branches, local 128-input window per branch (C, AVX2 + BMI2)
Implementations are measured interleaved (one rep of each in turn) because the machine is shared
and its load drifts; numbers are medians. The modulate protocol calls tick() first, as the demo
loop does, so the baseline's global fast-weight clock has to be folded into every synapse.

Run: .venv/bin/python prototype/experiments/cpu-event-driven/bench_layouts.py [--quick]
"""

import common
from common import MiB, peak_extra_rss, peak_tracemalloc, set_threads

import json
import sys
import time

import numpy as np

from axon_major import AxonPopulation
from bitslice import BitSlicedPopulation
from dm_fused import FusedPopulation
from neuron import WORKHORSE
from packed_neuron import PackedPopulation

N = 100_000
NIN = 4096
SYN = N * WORKHORSE.n_branches * WORKHORSE.slots
ACTS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]


def set_pop_threads(pops, th):
    set_threads(th)
    for p in pops.values():
        if hasattr(p, "nthreads"):
            p.nthreads = th


def interleaved(fns, reps, setup=None):
    """fns: {name: fn}. Returns {name: median seconds}, alternating implementations each rep."""
    ts = {k: [] for k in fns}
    for k, f in fns.items():      # warm-up
        if setup:
            setup(k)
        f()
    for _ in range(reps):
        for k, f in fns.items():
            if setup:
                setup(k)
            t0 = time.perf_counter()
            f()
            ts[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in ts.items()}


def main(quick=False, repeats=1):
    reps = 3 if quick else 5
    acts = [0.02, 0.2] if quick else ACTS
    res = {"N": N, "n_inputs": NIN, "synapses": SYN, "speed": [], "fixed": [], "memory": {}}
    t0 = time.time()
    base = PackedPopulation(WORKHORSE, N, NIN, np.random.default_rng(0))
    pops = {"dendrite": FusedPopulation(WORKHORSE, N, NIN, np.random.default_rng(0)),
            "axon": AxonPopulation(WORKHORSE, N, NIN, np.random.default_rng(0), block=4096, from_packed=base),
            "bitslice": BitSlicedPopulation(WORKHORSE, N, NIN, np.random.default_rng(0))}
    del base
    print(f"built in {time.time() - t0:.0f} s", flush=True)
    for k, p in pops.items():
        res["memory"][k] = {"bytes_per_neuron": p.bytes_per_neuron()}
    a = pops["axon"]
    res["memory"]["axon"].update(words_post_bytes_per_synapse=(a.words.nbytes + a.post.nbytes) / SYN,
                                 bucket_index_bytes_per_neuron=(a.off.nbytes + a.stamp.nbytes) / N)
    print("bytes/neuron:", {k: round(v["bytes_per_neuron"]) for k, v in res["memory"].items()}, flush=True)

    rng = np.random.default_rng(42)
    for rep in range(repeats):
        for th in (1, 3):
            set_pop_threads(pops, th)
            for act in acts:
                xs = [(rng.random(NIN) < act).astype(np.float32) for _ in range(8)]
                Xs = [pops["bitslice"].bitset(x) for x in xs]
                cnt = {"i": 0}

                def fwd(name):
                    def f():
                        cnt["i"] += 1
                        i = cnt["i"] % 8
                        if name == "bitslice":
                            pops[name].forward(None, X=Xs[i])
                        else:
                            pops[name].forward(xs[i])
                    return f

                tf = interleaved({k: fwd(k) for k in pops}, reps)
                for i in range(4):          # the eligibility union = inputs active in the last 4 steps
                    for k in pops:
                        fwd(k)()
                union = float(pops["axon"].pre_hist.any(axis=0).mean())
                tm = interleaved({k: (lambda k=k: pops[k].modulate(1.0, 0.8, 0.8)) for k in pops}, reps,
                                 setup=lambda k: pops[k].tick())
                for k in pops:
                    res["speed"].append(dict(rep=rep, layout=k, threads=th, activity=act, elig_union=union,
                                             forward_s=tf[k], modulate_s=tm[k]))
                print(f"th={th} act={act:4.0%} union={union:5.1%} | forward ms: "
                      + "  ".join(f"{k} {tf[k] * 1e3:7.2f}" for k in pops) + " | modulate ms: "
                      + "  ".join(f"{k} {tm[k] * 1e3:7.2f}" for k in pops), flush=True)

            # steps that touch every synapse: cascade catch-up (every 512 ticks) and sleep
            def cu(k):
                return lambda: (setattr(pops[k], "pending", 300), pops[k]._catch_up())
            tc = interleaved({k: cu(k) for k in pops}, reps)
            ts = interleaved({k: pops[k].sleep for k in pops}, reps)
            zero = np.zeros(NIN, np.float32)
            tz = interleaved({"axon": lambda: pops["axon"].forward(zero)}, reps)
            for k in pops:
                res["fixed"].append(dict(rep=rep, layout=k, threads=th, catch_up_s=tc[k], sleep_s=ts[k]))
            res["fixed"].append(dict(rep=rep, layout="axon", threads=th, soma_floor_s=tz["axon"]))
            print(f"th={th} catch-up ms: " + "  ".join(f"{k} {tc[k] * 1e3:.1f}" for k in pops)
                  + " | sleep ms: " + "  ".join(f"{k} {ts[k] * 1e3:.1f}" for k in pops)
                  + f" | axon forward, zero input: {tz['axon'] * 1e3:.2f}", flush=True)

    # peak temporary memory (resident high-water mark above the starting RSS), 3 threads, 2% activity
    set_pop_threads(pops, 3)
    xs = [(rng.random(NIN) < 0.02).astype(np.float32) for _ in range(5)]
    res["peak_temp_MiB"] = {}
    for k, p in pops.items():
        for x in xs[:4]:
            p.forward(x)
        _, pf = peak_extra_rss(lambda: p.forward(xs[4]))
        _, pm = peak_extra_rss(lambda: p.modulate(1.0, 0.8, 0.8))
        _, pc = peak_extra_rss(lambda: (setattr(p, "pending", 300), p._catch_up()))
        _, psl = peak_extra_rss(p.sleep)
        for x in xs[:4]:
            p.forward(x)
        _, tf_ = peak_tracemalloc(lambda: p.forward(xs[4]))
        _, tm_ = peak_tracemalloc(lambda: p.modulate(1.0, 0.8, 0.8))
        _, ts_ = peak_tracemalloc(p.sleep)
        res["peak_temp_MiB"][k] = dict(rss_forward=pf / MiB, rss_modulate=pm / MiB, rss_catch_up=pc / MiB,
                                       rss_sleep=psl / MiB, py_forward=tf_ / MiB, py_modulate=tm_ / MiB,
                                       py_sleep=ts_ / MiB)
        print(f"peak temporaries {k}: RSS high-water forward {pf / MiB:.2f} MiB, modulate {pm / MiB:.2f}, "
              f"catch-up {pc / MiB:.2f}, sleep {psl / MiB:.2f} | NumPy-side (tracemalloc) forward "
              f"{tf_ / MiB:.2f}, modulate {tm_ / MiB:.2f}, sleep {ts_ / MiB:.2f}", flush=True)
    print(f"total {time.time() - t0:.0f} s")
    return res


if __name__ == "__main__":
    reps_arg = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--repeats=")]
    out = main(quick="--quick" in sys.argv, repeats=reps_arg[0] if reps_arg else 2)
    with open(f"{common.HERE}/results_layouts.json", "w") as f:
        json.dump(out, f, indent=1)

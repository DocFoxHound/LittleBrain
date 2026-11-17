"""
Throughput benchmark. One population (host-backed, or managed), several storage modes run
*interleaved* in rounds on the same scenario sequence, so that load from other processes on the
shared GPU hits every mode alike. Appends one JSON line per (mode, config) to results/bench.jsonl.

Example (8M neurons, does not fit the VRAM cap):
  .venv/bin/python prototype/experiments/vram-ram-paging/bench.py --n 8388608 \
      --modes paged,zerocopy --cache 1536 --frac 0.01,0.05 --churn 0.02,1.0
"""

import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
import numpy as np
import cupy as cp

from gpu_population import GPUPopulation
from neuron import WORKHORSE
from scenario import AssemblyScenario

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
VRAM_CAP = 2_500_000_000
CONTEXT_RESERVE = 450_000_000  # CUDA context + cuBLAS etc., outside the memory pool


def rss_mb():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k + "_MB"] = int(v.split()[0]) / 1024
    return out


def gpu_mb_this_process():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
        for line in q.strip().splitlines():
            pid, mem = [s.strip() for s in line.split(",")]
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:
        pass
    return float("nan")


def run_config(pop, scen, steps, warmup, mod=True):
    rec = {k: [] for k in ("f", "m", "act", "h2d_f", "d2h_f", "h2d_m", "d2h_m")}
    direct = pop.mode in ("zerocopy",) or (pop.mode == "managed" and not pop._paging())
    for s in range(warmup + steps):
        cols, blocks = scen.step()
        h0, d0, l0 = pop.h2d, pop.d2h, pop.lines_written()
        t0 = time.perf_counter()
        pop.forward(cols=cols, active=blocks)
        t1 = time.perf_counter()
        h1, d1, l1 = pop.h2d, pop.d2h, pop.lines_written()
        nrec = len(pop.recent_blocks())
        if mod:
            pop.modulate(0.5, 0.5, 0.5)
        t2 = time.perf_counter()
        l2 = pop.lines_written()
        pop.tick()
        if s >= warmup:
            rec["f"].append(t1 - t0)
            rec["m"].append(t2 - t1)
            rec["act"].append(len(blocks))
            if direct:  # the kernel itself crosses PCIe: every word of a processed block is read,
                # and every 128-byte line it changes is written back
                rec["h2d_f"].append(len(blocks) * pop.block_bytes)
                rec["d2h_f"].append((l1 - l0) * 128)
                rec["h2d_m"].append(nrec * pop.block_bytes if mod else 0)
                rec["d2h_m"].append((l2 - l1) * 128)
            else:
                rec["h2d_f"].append(h1 - h0)
                rec["d2h_f"].append(d1 - d0)
                rec["h2d_m"].append(pop.h2d - h1)
                rec["d2h_m"].append(pop.d2h - d1)
    return {k: np.array(v, float) for k, v in rec.items()}


def summarize(pop, recs, mod):
    """recs: one record per round. Median over all measured steps of all rounds."""
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    f, m = float(np.median(cat["f"])), float(np.median(cat["m"]))
    act = float(cat["act"].mean()) * pop.bs
    return {
        "fwd_ms": f * 1e3, "mod_ms": m * 1e3 if mod else None,
        "fwd_ms_rounds": [float(np.median(r["f"])) * 1e3 for r in recs],
        # other processes share the GPU (time-slicing) and only ever add time: best round = least disturbed
        "fwd_ms_best": min(float(np.median(r["f"])) for r in recs) * 1e3,
        "mod_ms_best": min(float(np.median(r["m"])) for r in recs) * 1e3 if mod else None,
        "mod_ms_rounds": [float(np.median(r["m"])) * 1e3 for r in recs] if mod else None,
        "active_neurons": act, "active_frac_actual": act / pop.N,
        "Mneuron_steps_per_s": pop.N / f / 1e6, "Mactive_neuron_steps_per_s": act / f / 1e6,
        "mod_Mactive_per_s": act / m / 1e6 if mod else None,
        "h2d_MB_fwd": float(cat["h2d_f"].mean()) / 1e6, "d2h_MB_fwd": float(cat["d2h_f"].mean()) / 1e6,
        "h2d_MB_mod": float(cat["h2d_m"].mean()) / 1e6, "d2h_MB_mod": float(cat["d2h_m"].mean()) / 1e6,
    }


def time_sleep(pop, reps=2):
    pop.reset_cache()
    h0, d0 = pop.h2d, pop.d2h
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pop.sleep()
        pop.flush()
        ts.append(time.perf_counter() - t0)
    return {"op": "sleep", "sleep_s": float(np.median(ts)), "sleep_s_all": ts,
            "h2d_MB": (pop.h2d - h0) / reps / 1e6, "d2h_MB": (pop.d2h - d0) / reps / 1e6,
            "Mneurons_per_s": pop.N / float(np.median(ts)) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20)
    ap.add_argument("--modes", default="paged,zerocopy", help="first mode decides the storage")
    ap.add_argument("--cache", type=str, default="0", help="comma list of VRAM cache sizes (blocks)")
    ap.add_argument("--frac", default="0.01,0.05,0.2")
    ap.add_argument("--churn", default="0.02,1.0")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--layout", default="clustered")
    ap.add_argument("--no-mod", action="store_true")
    ap.add_argument("--sleep", action="store_true", help="also time full sleeps (per mode)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cp.get_default_memory_pool().set_limit(size=VRAM_CAP - CONTEXT_RESERVE)
    os.makedirs(RESULTS, exist_ok=True)
    modes = a.modes.split(",")
    caches = [int(c) for c in a.cache.split(",")]
    scen0 = AssemblyScenario(a.n, seed=1)
    if a.layout == "scattered":
        scen0.set_layout(np.random.default_rng(99).permutation(scen0.n_cols))
    ids = scen0.storage_ids()

    t0 = time.perf_counter()
    pop = GPUPopulation(WORKHORSE, a.n, col_size=64, npc=128, mode=modes[0], cache_blocks=max(caches), ids=ids)
    t_init = time.perf_counter() - t0
    pool = cp.get_default_memory_pool()
    print(f"[N={a.n:,} modes={modes}] init {t_init:.1f} s, VRAM pool {pool.total_bytes() / 1e6:.0f} MB, "
          f"host words {pop.host_bytes() / 1e9:.2f} GB, {rss_mb()}", flush=True)

    def out(r):
        with open(os.path.join(RESULTS, "bench.jsonl"), "a") as fh:
            fh.write(json.dumps(r) + "\n")

    for K in caches:
        for frac in [float(f) for f in a.frac.split(",")]:
            for churn in [float(c) for c in a.churn.split(",")]:
                recs = {m: [] for m in modes}
                for rnd in range(a.rounds):
                    for mode in modes:
                        pop.switch_mode(mode) if mode in ("resident", "paged", "zerocopy") else None
                        if K and hasattr(pop, "slot_of"):
                            pop.set_cache_size(K)
                        pop.reset_cache()
                        scen = AssemblyScenario(a.n, active_frac=frac, churn=churn, seed=2 + rnd)
                        scen.set_layout(scen0.col_slot)
                        recs[mode].append(run_config(pop, scen, a.steps, a.warmup, mod=not a.no_mod))
                ref = modes[0]
                for mode in modes:
                    pop.switch_mode(mode) if mode in ("resident", "paged", "zerocopy") else None
                    r = summarize(pop, recs[mode], not a.no_mod)
                    ratio = [float(np.median(x["f"]) / np.median(y["f"])) for x, y in zip(recs[mode], recs[ref])]
                    st = pop.vram_state_bytes()
                    words_vram = K * pop.block_bytes / a.n if mode in ("paged", "managed", "managed_lazy") \
                        else st["words"]
                    r.update({"n": a.n, "mode": mode, "cache_blocks": K if mode != "zerocopy" else 0,
                              "frac": frac, "churn": churn, "layout": a.layout, "tag": a.tag, "init_s": t_init,
                              "fwd_time_vs_" + ref: float(np.median(ratio)),
                              "vram_pool_MB": pool.total_bytes() / 1e6, "vram_process_MB": gpu_mb_this_process(),
                              "vram_state_B_per_neuron": st["per_neuron"] + st["per_block"] + st["shared_inputs"],
                              "vram_words_B_per_neuron": words_vram,
                              "host_B_per_neuron": pop.host_bytes() / a.n, **rss_mb()})
                    print(f"  {mode:>12} K={K:5d} frac={frac:.3f} churn={churn:.2f}: fwd {r['fwd_ms']:8.2f} ms "
                          f"{['%.1f' % v for v in r['fwd_ms_rounds']]} mod {r['mod_ms'] or 0:8.2f} ms | "
                          f"{r['Mneuron_steps_per_s']:8.1f} M n-steps/s, {r['Mactive_neuron_steps_per_s']:6.1f} M active/s"
                          f" | PCIe/step fwd in {r['h2d_MB_fwd']:6.1f} out {r['d2h_MB_fwd']:6.1f} MB, "
                          f"mod in {r['h2d_MB_mod']:6.1f} out {r['d2h_MB_mod']:6.1f} MB | x{r['fwd_time_vs_' + ref]:.2f} vs {ref}",
                          flush=True)
                    out(r)

    if a.sleep:
        for mode in modes:
            pop.switch_mode(mode) if mode in ("resident", "paged", "zerocopy") else None
            if hasattr(pop, "slot_of"):
                pop.set_cache_size(max(caches))
            r = time_sleep(pop)
            r.update({"n": a.n, "mode": mode, "cache_blocks": max(caches), "tag": a.tag})
            print(f"  {mode:>12} sleep: {r['sleep_s']:.3f} s ({['%.3f' % t for t in r['sleep_s_all']]}) -> "
                  f"{r['Mneurons_per_s']:.1f} M neurons/s; PCIe in {r['h2d_MB']:.0f} MB, out {r['d2h_MB']:.0f} MB",
                  flush=True)
            out(r)
    print(f"  peak {rss_mb()}, VRAM this process {gpu_mb_this_process():.0f} MB", flush=True)


if __name__ == "__main__":
    main()

"""
The alternative to paging: leave the words in RAM and compute forward on the CPU (numba,
3 threads), touching only the active blocks. Same math as forward_kernel.

Checked against the GPU kernel on the same state, then timed against GPU zero-copy and paging
on the same 8M-neuron population (same pinned host buffer).

Run: .venv/bin/python prototype/experiments/vram-ram-paging/cpu_forward.py
"""

import json
import os
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cupy as cp
from numba import njit, prange, set_num_threads

from gpu_population import GPUPopulation, pn
from neuron import WORKHORSE
from scenario import AssemblyScenario

set_num_threads(3)
HERE = os.path.dirname(os.path.abspath(__file__))
T = WORKHORSE
C1, C2, C3, C4 = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9),
                  np.uint64(0x94D049BB133111EB), np.uint64(0xD6E8FEB86659FD93))
DEC = np.float32(T.w_max / pn.Q)


@njit(parallel=True, cache=False)
def cpu_forward(words, blocks, bs, n, coef, ids, gate, thr, post, ring, x, npc, col_size, ctx_mult, rate,
                bthr, bonus, slope, thr_lr, target, used_rate, dec):
    for bi in prange(len(blocks)):
        b = blocks[bi]
        m0, m1, m2, fs = coef[b, 0], coef[b, 1], coef[b, 2], coef[b, 9]
        for j in range(bs):
            p = b * bs + j
            if p >= n:
                break
            nid = np.uint64(ids[p])
            colbase = (ids[p] // npc) * col_size
            gb = gate[p]
            base = p * 256
            soma = np.float32(0.0)
            hid = (nid + np.uint64(1)) * C1
            cmask = np.uint64(col_size - 1)  # col_size is a power of two (64)
            for br in range(8):
                drive = np.float32(0.0)
                for s in range(32):
                    w = words[base + br * 32 + s]
                    gen = np.uint64(w & np.uint32(15))
                    h = hid ^ (np.uint64(br * 32 + s) * C2) ^ (gen * C3)
                    h ^= h >> np.uint64(31)
                    h *= C4
                    h ^= h >> np.uint64(32)
                    xv = x[colbase + np.int64(h & cmask)]
                    if xv != 0.0:
                        c0 = (m0 * ((np.float32((w >> np.uint32(10)) & np.uint32(63)) - 32.0) * dec)
                              + m1 * ((np.float32((w >> np.uint32(16)) & np.uint32(63)) - 32.0) * dec)
                              + m2 * ((np.float32((w >> np.uint32(22)) & np.uint32(63)) - 32.0) * dec))
                        wt = (np.float32((w >> np.uint32(4)) & np.uint32(63)) - 32.0) * dec * fs + c0
                        drive += wt * xv
                g = np.float32((gb >> (7 - br)) & 1)
                spike = np.float32(1.0) if drive > bthr else np.float32(0.0)
                soma += (max(drive, np.float32(0.0)) + bonus * spike) * g
            soma *= ctx_mult
            t = thr[p]
            r = np.float32(1.0) / (np.float32(1.0) + np.exp(-slope * (soma - t)))
            if r >= used_rate:
                for br in range(8):
                    if (gb >> (7 - br)) & 1:
                        for s in range(32):
                            k = base + br * 32 + s
                            w = words[k]
                            gen = np.uint64(w & np.uint32(15))
                            h = hid ^ (np.uint64(br * 32 + s) * C2) ^ (gen * C3)
                            h ^= h >> np.uint64(31)
                            h *= C4
                            h ^= h >> np.uint64(32)
                            if x[colbase + np.int64(h & cmask)] > 0.0:
                                words[k] = w | np.uint32(0x80000000)
            thr[p] = t + thr_lr * (r - target)
            post[p, ring] = np.float32(0.5) + r  # float32 here: numba has no float16 arrays
            rate[p] = r


class CPUForward:
    """forward() for the active blocks of a GPUPopulation's host words, on the CPU."""

    def __init__(self, pop):
        self.pop = pop
        self.thr = cp.asnumpy(pop.thr)
        self.post = cp.asnumpy(pop.post).astype(np.float32)
        self.rate = np.zeros(pop.N, np.float32)
        self.gate = cp.asnumpy(pop.gate)
        self.ids = cp.asnumpy(pop.ids).astype(np.int64)
        self.coef = np.zeros((pop.nb, 16), np.float32)
        self.coef[:, [0, 9]] = 1.0  # clocks in step: no pending flow, no fast decay

    def forward(self, x, blocks, ring):
        p = self.pop
        cpu_forward(p.host, blocks.astype(np.int64), p.bs, p.N, self.coef, self.ids, self.gate, self.thr, self.post,
                    ring, x, p.npc, p.col_size, np.float32(1.0), self.rate, np.float32(T.branch_threshold),
                    np.float32(T.spike_bonus), np.float32(T.soma_slope), np.float32(T.threshold_lr),
                    np.float32(T.target_rate), np.float32(pn.USED_RATE), DEC)


def check(n=64 * 1024):
    pop = GPUPopulation(T, n, col_size=64, npc=128, mode="zerocopy")
    cf = CPUForward(pop)
    scen = AssemblyScenario(n, active_frac=0.2, seed=4)
    cols, blocks = scen.step()
    r_g = cp.asnumpy(pop.forward(cols=cols, active=blocks))
    x = cp.asnumpy(pop.x)
    pop.host[:] &= np.uint32(0x7FFFFFFF)  # undo the used bits the GPU just set, then redo them on the CPU
    w_gpu = None
    cf.forward(x, blocks, 1)
    act = np.concatenate([np.arange(b * pop.bs, (b + 1) * pop.bs) for b in blocks])
    d = np.abs(cf.rate[act] - r_g[act])
    print(f"  CPU vs GPU forward on {len(act):,} active neurons: max |rate diff| {d.max():.2e}")
    return float(d.max())


def bench(n=8 * 1024 * 1024, reps=5):
    pop = GPUPopulation(T, n, col_size=64, npc=128, mode="zerocopy", cache_blocks=1536)
    cf = CPUForward(pop)
    out = []
    for frac in (0.01, 0.05):
        scen = AssemblyScenario(n, active_frac=frac, churn=0.02, seed=2)
        cols, blocks = scen.step()
        pop.forward(cols=cols, active=blocks)  # sets x for these columns
        x = cp.asnumpy(pop.x)
        cf.forward(x, blocks, 1)  # compile / warm
        tc = []
        for _ in range(reps):
            t0 = time.perf_counter()
            cf.forward(x, blocks, 1)
            tc.append(time.perf_counter() - t0)
        res = {"frac": frac, "active_neurons": len(blocks) * pop.bs}
        for mode in ("zerocopy", "paged"):
            pop.switch_mode(mode)
            pop.forward(cols=cols, active=blocks)
            tg = []
            for _ in range(reps):
                t0 = time.perf_counter()
                pop.forward(cols=cols, active=blocks)
                tg.append(time.perf_counter() - t0)
            res[mode + "_ms"] = float(np.median(tg)) * 1e3
        pop.switch_mode("zerocopy")
        res["cpu_ms"] = float(np.median(tc)) * 1e3
        res["cpu_Mactive_per_s"] = res["active_neurons"] / res["cpu_ms"] / 1e3
        print(f"  {n:,} neurons, {frac:.0%} active ({res['active_neurons']:,}): CPU numba (3 threads) "
              f"{res['cpu_ms']:.1f} ms = {res['cpu_Mactive_per_s']:.2f} M active neuron-steps/s | "
              f"GPU zero-copy {res['zerocopy_ms']:.1f} ms | GPU paged, warm cache {res['paged_ms']:.1f} ms",
              flush=True)
        out.append(res)
    return out


if __name__ == "__main__":
    print("== CPU (numba) forward: correctness ==")
    diff = check()
    print("== CPU (numba) forward vs GPU, same host RAM ==")
    res = bench()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "cpu_forward.json"), "w") as fh:
        json.dump({"max_rate_diff": diff, "bench": res}, fh, indent=1)

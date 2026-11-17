"""
Experiment 1: a fast CPU sleep kernel on the packed words.

  - correctness: C == Numba == GPU bit for bit (same seed); vs the NumPy prototype, statistically
  - speed and peak temporary memory: NumPy prototype vs C (1-3 threads) vs Numba vs GPU kernel

Run: .venv/bin/python prototype/experiments/rolling-local-sleep/bench_sleep.py
"""

import json
import time
import tracemalloc

import common
from common import HERE, WORKHORSE, rss_bytes

import cupy as cp
import numpy as np

from cpu_sleep import cpu_sleep
from packed_neuron import PackedPopulation, _get
from population import Block, KernelPopulation

T = WORKHORSE
N_INPUTS = 64


def pattern(rng, k=8):
    x = np.zeros(N_INPUTS, np.float32)
    x[rng.choice(N_INPUTS, k, replace=False)] = 1.0
    return x


def lived_in(n, seed=0, trials=40):
    """A population after one demo-like day (A rewarded, background moments), awake on the GPU."""
    rng = np.random.default_rng(seed)
    kp = KernelPopulation(T, n, N_INPUTS, rng)
    A = pattern(np.random.default_rng(0))
    for _ in range(trials):
        kp.forward(A)
        kp.modulate(1.0, 0.8, 0.8)
        kp.tick()
        kp.forward(pattern(rng))
        kp.tick()
    cp.cuda.Device().synchronize()
    return kp


def median_time(fn, reps=5, setup=None):
    ts = []
    for _ in range(reps):
        if setup:
            setup()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(x, 4) for x in ts]


def to_packed(b_host):
    pp = PackedPopulation(T, 1, N_INPUTS, np.random.default_rng(1))
    pp.n, pp.first_id = b_host.n, b_host.first_id
    pp.words = b_host.words.reshape(b_host.n, T.n_branches, T.slots).copy()
    pp.threshold, pp.plasticity = b_host.threshold.copy(), b_host.plast.copy()
    pp.post_hist, pp.age = b_host.post_hist.copy(), b_host.age.copy()
    pp.gate_bits = b_host.gate_bits.copy()
    pp.pending, pp.fast_scale = b_host.pending, b_host.fast_scale
    return pp


def exact_levels(w, fast_scale):
    """Float (pre-rounding) cascade levels after one night, in quantization steps."""
    from packed_neuron import _dec, _flow_matrix
    tag = _get(w, "tag") / 7
    c = np.stack([_dec(_get(w, f), T.w_max) for f in ("c0", "c1", "c2")])
    c[0] = np.clip(c[0] + _dec(_get(w, "fast"), T.w_max) * fast_scale * tag, -T.w_max, T.w_max)
    c = np.tensordot(_flow_matrix(T, T.sleep_flow_steps), c, axes=1)
    c[0] *= 1 - T.downscale
    return c * (31 / T.w_max)


class Fixed:
    def __init__(self, s):
        self.s = s

    def next(self):
        return self.s


def correctness(n=20_000):
    out = {}
    host = lived_in(n).b.to(np)
    fields = ["gen", "fast", "c0", "c1", "c2", "tag", "used"]
    out["state_before"] = {f: float(_get(host.words, f).astype(float).mean()) for f in fields}
    out["state_before"]["frac_tagged"] = float((_get(host.words, "tag") > 0).mean())

    # (a) bit-identity C / Numba / GPU for the same seed, with and without a pending clock
    for pend in (0, 37):
        wc, wn = host.words.copy(), host.words.copy()
        nc = cpu_sleep(wc, T, pend, host.fast_scale, 777, impl="c")
        nn = cpu_sleep(wn, T, pend, host.fast_scale, 777, impl="numba")
        g = host.to(cp)
        g.seeds, g.pending = Fixed(777), pend
        ng = int(g.sleep_gpu().get()[0])
        out[f"bit_identical_pending{pend}"] = {
            "c==numba": bool(np.array_equal(wc, wn)), "c==gpu": bool(np.array_equal(wc, g.words.get())),
            "pruned_c_numba_gpu": [nc, nn, ng]}

    # (b) vs the NumPy prototype from the same state (pending caught up first, so the
    #     pre-rounding floats are identical and only the random rounding differs)
    pp = to_packed(host)
    pp._catch_up()
    caught = pp.words.reshape(n, -1).copy()
    fs = pp.fast_scale  # (sleep resets it, so read it first)
    exact = exact_levels(caught, fs)  # the pre-rounding values both versions should round
    n_np = pp.sleep()
    w_np = pp.words.reshape(n, -1)
    w_c = caught.copy()
    n_c = cpu_sleep(w_c, T, 0, fs, 4242)
    pruned_np = _get(w_np, "gen") != _get(caught, "gen")
    pruned_c = _get(w_c, "gen") != _get(caught, "gen")
    keep = ~pruned_np
    cmp = {"pruned_numpy": n_np, "pruned_c": n_c,
           "prune_mask_mismatches": int((pruned_np != pruned_c).sum())}
    for k, f in enumerate(["c0", "c1", "c2", "tag", "fast", "used"]):
        a, b = _get(w_c, f).astype(np.int64)[keep], _get(w_np, f).astype(np.int64)[keep]
        cmp[f] = {"max_abs_step_diff_c_vs_numpy": int(np.abs(a - b).max())}
        if k < 3:  # bias of each version against the exact value, in steps
            x = exact[k][keep]
            cmp[f]["bias_c_steps"] = float((a - 32 - x).mean())
            cmp[f]["bias_numpy_steps"] = float((b - 32 - x).mean())
    out["vs_numpy_same_state"] = cmp

    # (c) with a pending clock (extra rounding step): compare distributions only
    host2 = host
    dist = {}
    for name in ("numpy", "c"):
        rows = []
        for s in range(3):
            if name == "numpy":
                p = to_packed(host2)
                p.rng = np.random.default_rng(100 + s)
                p.pending = 37
                k = p.sleep()
                w = p.words.reshape(n, -1)
            else:
                w = host2.words.copy()
                k = cpu_sleep(w, T, 37, host2.fast_scale, 900 + s)
            lv = [(_get(w, f).astype(np.float64) - 32) / 31 for f in ("c0", "c1", "c2")]
            rows.append([k] + [float(v.mean()) for v in lv] + [float(v.std()) for v in lv])
        dist[name] = np.mean(rows, axis=0).round(6).tolist()
    dist["columns"] = ["pruned", "mean c0", "mean c1", "mean c2", "std c0", "std c1", "std c2"]
    out["distribution_pending37"] = dist
    return out


def speed():
    res = {}
    pool = cp.get_default_memory_pool()
    small = lived_in(100_000).b.to(np)
    for n in (100_000, 1_000_000):
        if n == 100_000:
            src = small
        else:  # tile the lived-in 100k state 10x (same statistics, no 1M GPU warm-up day)
            src = Block(T, n, N_INPUTS, 0, 1, xp=np)
            for k in ("words", "threshold", "plast", "post_hist", "age", "gate_bits"):
                setattr(src, k, np.concatenate([getattr(small, k)] * 10))
            src.pending, src.fast_scale = small.pending, small.fast_scale
        r = res[n] = {}
        # NumPy prototype (only at 100k: at 1M it would need ~12 GB of temporaries)
        if n == 100_000:
            pp = to_packed(src)
            tracemalloc.start()
            pp.sleep()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            pps = [to_packed(src) for _ in range(3)]
            it = iter(pps)
            t_np, raw = median_time(lambda: next(it).sleep(), reps=3)
            r["numpy"] = {"s": t_np, "runs": raw, "peak_temp_MiB": peak / 2**20}
            del pps
        # C and Numba kernels, from pinned or plain RAM (no difference expected for the CPU)
        for impl, threads in [("c", 1), ("c", 2), ("c", 3), ("numba", 3)]:
            w = src.words.copy()
            cpu_sleep(w[:1000], T, src.pending, src.fast_scale, 1, impl=impl, threads=threads)
            tracemalloc.start()
            rss0 = rss_bytes()
            cpu_sleep(w, T, src.pending, src.fast_scale, 1, impl=impl, threads=threads)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            drss = rss_bytes() - rss0
            reps = 5 if n == 100_000 else 3
            t_k, raw = median_time(lambda: cpu_sleep(w, T, src.pending, src.fast_scale, 2, impl=impl,
                                                     threads=threads), reps=reps)
            r[f"{impl}_{threads}t"] = {"s": t_k, "runs": raw, "peak_temp_MiB": peak / 2**20,
                                       "rss_growth_MiB": drss / 2**20}
        # GPU kernel (for comparison; this is what the synchronous GPU schedule uses)
        g = src.to(cp)
        g.sleep_gpu()
        cp.cuda.Device().synchronize()
        pool.free_all_blocks()
        base = pool.total_bytes()

        def gsleep():
            g.pending = src.pending
            g.sleep_gpu()
            cp.cuda.Device().synchronize()
        t_g, raw = median_time(gsleep, reps=5)
        r["gpu"] = {"s": t_g, "runs": raw, "peak_temp_MiB": (pool.total_bytes() - base) / 2**20}
        # PCIe round trip of the words (what CPU sleep costs the GPU side when the block is resident)
        hb = cp.cuda.alloc_pinned_memory(g.words.nbytes)
        hw = np.frombuffer(hb, np.uint32, g.words.size)
        t_d2h, _ = median_time(lambda: (g.words.reshape(-1).get(out=hw), cp.cuda.Device().synchronize()), 3)
        t_h2d, _ = median_time(lambda: (g.words.reshape(-1).set(hw), cp.cuda.Device().synchronize()), 3)
        r["pcie"] = {"d2h_s": t_d2h, "h2d_s": t_h2d, "MiB": g.words.nbytes / 2**20,
                     "d2h_GBps": g.words.nbytes / t_d2h / 1e9, "h2d_GBps": g.words.nbytes / t_h2d / 1e9}
        del g, hw, hb
        pool.free_all_blocks()
        r["bytes_per_neuron_stored"] = src.nbytes() / n
    return res


def gpu_awake_speed():
    """The awake kernels the GPU runs while the CPU sleeps (CUDA-event timed, median of 5x10 calls)."""
    res = {}
    pool = cp.get_default_memory_pool()
    rng = np.random.default_rng(3)
    for n in (100_000, 1_000_000):
        b = Block.fresh(T, n, N_INPUTS, rng, xp=cp)
        x = cp.zeros(N_INPUTS, cp.float32)
        x[:8] = 1
        pre = cp.zeros((4, N_INPUTS), cp.float16)
        pre[:, :8] = 1
        rate = cp.zeros(n, cp.float32)
        r = res[n] = {}
        for name, fn in [("forward", lambda: b.forward(x, rate)),
                         ("modulate", lambda: b.modulate(pre, N_INPUTS, 1.0, 0.8, 0.8)),
                         ("sleep (GPU)", lambda: b.sleep_gpu())]:
            fn()
            cp.cuda.Device().synchronize()
            pool.free_all_blocks()
            base = pool.total_bytes()
            ts = []
            for _ in range(5):
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record()
                for _ in range(10):
                    fn()
                e1.record()
                e1.synchronize()
                ts.append(cp.cuda.get_elapsed_time(e0, e1) / 10 / 1e3)
            sec = float(np.median(ts))
            r[name] = {"s": sec, "neuron_steps_per_s": n / sec, "synapse_updates_per_s": n * 256 / sec,
                       "vram_temp_MiB": (pool.total_bytes() - base) / 2**20,
                       "effective_GBps_words": n * 1024 / sec / 1e9}
        r["vram_bytes_per_neuron"] = b.nbytes() / n + 4  # + the rate output buffer
        del b, rate
        pool.free_all_blocks()
    return res


def main():
    t0 = time.time()
    out = {"correctness": correctness(), "speed": speed(), "gpu_awake": gpu_awake_speed()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sleep_kernel.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()

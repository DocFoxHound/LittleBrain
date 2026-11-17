"""
Check the CUDA kernels against PackedPopulation (NumPy) from identical states, and time both.

  forward : must match (rates within float tolerance; used bits, thresholds, history exactly)
  modulate: stochastic rounding uses a different RNG, so we check that every new value is one of
            the two neighbours of the exact (unrounded) value, and that the rounding is unbiased,
            for both implementations
  sleep   : same idea, plus prune counts

Run: .venv/bin/python prototype/experiments/vram-ram-paging/verify_kernel.py
"""

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cupy as cp

from gpu_population import GPUPopulation, pn
from neuron import WORKHORSE

cp.get_default_memory_pool().set_limit(size=2_000_000_000)
T = WORKHORSE
N_IN = 64


def scramble(pop, rng):
    """Put a NumPy population in a varied state that exercises every field and code path."""
    sh = pop.words.shape
    w = pop.words
    w = pn._set(w, "gen", rng.integers(0, 16, sh))
    w = pn._set(w, "fast", 32 + rng.integers(-6, 7, sh))
    w = pn._set(w, "c0", 32 + rng.integers(-3, 9, sh))
    w = pn._set(w, "c1", 32 + rng.integers(-4, 5, sh))
    w = pn._set(w, "c2", 32 + rng.integers(-4, 5, sh))
    w = pn._set(w, "tag", rng.integers(0, 8, sh))
    w = pn._set(w, "used", (rng.random(sh) < 0.3).astype(np.uint32))
    pop.words = w
    pop.threshold = rng.normal(6.0, 1.5, pop.n).astype(np.float32)
    pop.plasticity = rng.uniform(0.3, 1.0, pop.n).astype(np.float16)
    pop.post_hist = rng.uniform(0.5, 1.5, (pop.n, 4)).astype(np.float16)
    pop.pre_hist = (rng.random((4, pop.n_inputs)) < 0.3).astype(np.float16)
    gates = rng.random((pop.n, T.n_branches)) > 0.1
    pop.gate_bits = np.packbits(gates, axis=1)
    pop.pending = 37
    pop.fast_scale = T.fast_decay ** 25


def upload(g, pop):
    """Copy a NumPy population's state into a GPUPopulation (ring position 0)."""
    g.load_words(pop.words.reshape(pop.n, -1))
    g.thr[:] = cp.asarray(pop.threshold)
    g.plast[:] = cp.asarray(pop.plasticity)
    g.gate[:] = cp.asarray(pop.gate_bits[:, 0])
    g.ring = 0
    post = np.zeros((pop.n, 4), np.float16)
    pre = np.zeros((4, pop.n_inputs), np.float16)
    for k in range(4):
        post[:, (-k) & 3] = pop.post_hist[:, k]
        pre[(-k) & 3] = pop.pre_hist[k]
    g.post[:] = cp.asarray(post)
    g.pre[:] = cp.asarray(pre)
    fs_steps = int(round(np.log(pop.fast_scale) / np.log(T.fast_decay)))
    g.T = pop.pending
    g.flow_clock.fill(0)
    g.fast_clock.fill(g.T - fs_steps)
    g.last_active[:] = 1
    g.fwd_count = 1


def gpu_post_newest(g):
    return cp.asnumpy(g.post[:, g.ring])


def check_forward(n, steps=5, seed=3):
    rng = np.random.default_rng(seed)
    pop = pn.PackedPopulation(T, n, N_IN, np.random.default_rng(seed))
    scramble(pop, rng)
    g = GPUPopulation(T, n, n_inputs=N_IN, block_size=1024, mode="resident", init=False)
    upload(g, pop)
    worst = 0.0
    flips = 0
    for s in range(steps):
        x = (rng.random(N_IN) < 0.25).astype(np.float32) * rng.choice([1.0, 0.5, 0.8], N_IN).astype(np.float32)
        ctx = [0.0, 0.3, -0.2, 0.0, 1.0][s % 5]
        r_np = pop.forward(x, context=ctx)
        r_g = cp.asnumpy(g.forward(x, context=ctx))
        d = np.abs(r_np - r_g)
        worst = max(worst, float(d.max()))
        flips += int((d > 1e-4).sum())
        w_g = g.words_of()
        same_words = np.array_equal(w_g, pop.words.reshape(n, -1))
        thr_d = float(np.abs(cp.asnumpy(g.thr) - pop.threshold).max())
        post_eq = np.mean(gpu_post_newest(g) == pop.post_hist[:, 0])
        print(f"    step {s}: max |rate diff| {d.max():.2e}, neurons >1e-4: {(d > 1e-4).sum()}, "
              f"words identical: {same_words}, max |thr diff| {thr_d:.1e}, history equal: {post_eq:.4f}")
        # resync words if a branch-threshold flip changed a used bit, so later steps stay comparable
        if not same_words:
            g.load_words(pop.words.reshape(n, -1))
    return worst, flips


def rounding_check(name, exact, got_np, got_gpu, step):
    """exact: unrounded target values; got_*: decoded results. Each must be a neighbour of exact."""
    out = []
    for label, got in (("numpy", got_np), ("gpu", got_gpu)):
        err = (got - exact) / step
        ok = np.mean(np.abs(err) < 1.0 + 1e-3)
        out.append(f"{label}: within 1 step {ok * 100:.3f}%, mean err {err.mean():+.4f} steps")
    print(f"    {name:<14}" + " | ".join(out))


def check_modulate(n, seed=5, oneshot=False):
    rng = np.random.default_rng(seed)
    pop = pn.PackedPopulation(T, n, N_IN, np.random.default_rng(seed))
    scramble(pop, rng)
    g = GPUPopulation(T, n, n_inputs=N_IN, block_size=1024, mode="resident", init=False)
    upload(g, pop)
    dop, ach, ar = (1.0, 0.95, 0.8) if oneshot else (0.7, 0.6, 0.8)
    w0 = pop.words.copy()
    gate = pop._gate()[:, :, None]
    elig = pop._elig(pop._src(w0)) * gate
    delta = T.lr * pop.plasticity[:, None, None] * ach * dop * elig * gate
    fast_exact = np.clip(pn._dec(pn._get(w0, "fast"), T.w_max) * pop.fast_scale + delta, -T.w_max, T.w_max)
    tag_exact = np.clip(pn._get(w0, "tag") / 7 + ar * abs(dop) * elig * gate, 0, 1) * 7
    lv = np.tensordot(pn._flow_matrix(T, pop.pending), pop._levels(w0), axes=1)
    lv[0] += 0.5 * delta
    lv = np.clip(lv, -T.w_max, T.w_max)

    pop.modulate(dop, ach, ar)
    g.modulate(dop, ach, ar)
    wn, wg = pop.words, g.words_of().reshape(pop.words.shape)
    step = T.w_max / pn.Q
    print(f"  modulate ({'one-shot' if oneshot else 'normal'}; dopamine {dop}, ACh {ach}, arousal {ar})")
    rounding_check("fast", fast_exact, pn._dec(pn._get(wn, "fast"), 1.0), pn._dec(pn._get(wg, "fast"), 1.0), step)
    rounding_check("tag", tag_exact, pn._get(wn, "tag").astype(float), pn._get(wg, "tag").astype(float), 1.0)
    if oneshot:
        for k, nm in enumerate(pn.LEVELS):
            rounding_check(nm, lv[k], pn._dec(pn._get(wn, nm), 1.0), pn._dec(pn._get(wg, nm), 1.0), step)
    same_gen_used = np.array_equal(pn._get(wn, "gen"), pn._get(wg, "gen")) and \
        np.array_equal(pn._get(wn, "used"), pn._get(wg, "used"))
    print(f"    gen and used bits unchanged and equal: {same_gen_used}")


def check_sleep(n, seed=7):
    rng = np.random.default_rng(seed)
    pop = pn.PackedPopulation(T, n, N_IN, np.random.default_rng(seed))
    scramble(pop, rng)
    # make some synapses weak and unused so pruning has something to do
    w = pop.words
    weak = rng.random(w.shape) < 0.2
    for nm in pn.LEVELS:
        w = np.where(weak, pn._set(w, nm, np.full(w.shape, 32, np.uint32)), w)
    w = np.where(weak, pn._set(w, "used", np.zeros(w.shape, np.uint32)), w)
    w = np.where(weak, pn._set(w, "tag", np.zeros(w.shape, np.uint32)), w)
    pop.words = w
    g = GPUPopulation(T, n, n_inputs=N_IN, block_size=1024, mode="resident", init=False)
    upload(g, pop)
    w0 = pop.words.copy()
    pruned_np = pop.sleep()
    pruned_g = g.sleep()
    wn, wg = pop.words, g.words_of().reshape(pop.words.shape)
    print(f"  sleep: pruned numpy {pruned_np:,} vs gpu {pruned_g:,} "
          f"({(pruned_g - pruned_np) / max(pruned_np, 1) * 100:+.2f}%)")
    for nm in pn.LEVELS:
        a, b = pn._dec(pn._get(wn, nm), 1.0), pn._dec(pn._get(wg, nm), 1.0)
        print(f"    {nm}: mean numpy {a.mean():+.5f} gpu {b.mean():+.5f}; "
              f"identical {np.mean(a == b) * 100:.1f}%, within 1 step {np.mean(np.abs(a - b) <= 1 / 31 + 1e-6) * 100:.2f}%")
    print(f"    fast all zero: numpy {np.all(pn._get(wn, 'fast') == 32)}, gpu {np.all(pn._get(wg, 'fast') == 32)}; "
          f"used cleared: {np.all(pn._get(wg, 'used') == 0)}; "
          f"tag mean numpy {pn._get(wn, 'tag').mean():.4f} gpu {pn._get(wg, 'tag').mean():.4f}")
    gen_np = (pn._get(wn, "gen") != pn._get(w0, "gen")).mean()
    gen_g = (pn._get(wg, "gen") != pn._get(w0, "gen")).mean()
    print(f"    rewired fraction numpy {gen_np:.4f} gpu {gen_g:.4f}")


def med(f, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def timing(n_np=100_000, n_gpu=1_000_000):
    print(f"\n== Speed (median of 3) ==")
    rng = np.random.default_rng(0)
    x = (rng.random(N_IN) < 0.125).astype(np.float32)
    pop = pn.PackedPopulation(T, n_np, N_IN, np.random.default_rng(0))
    pop.forward(x)
    t_f = med(lambda: pop.forward(x))
    t_m = med(lambda: pop.modulate(0.5, 0.5, 0.5))
    t_s = med(lambda: pop.sleep(), reps=3)
    print(f"  numpy  {n_np:>9,} neurons: forward {t_f * 1e3:8.1f} ms  modulate {t_m * 1e3:8.1f} ms  "
          f"sleep {t_s * 1e3:8.1f} ms   -> forward {n_np / t_f / 1e6:7.3f} M neuron-steps/s")
    res = {"numpy": (n_np, t_f, t_m, t_s)}
    del pop
    for n in (n_np, n_gpu):
        pool = cp.get_default_memory_pool()
        g = GPUPopulation(T, n, n_inputs=N_IN, block_size=1024, mode="resident")
        xd = cp.asarray(x)

        def f():
            g.forward(xd)
            g.sync()

        def m():
            g.modulate(0.5, 0.5, 0.5)
            g.sync()

        def s():
            g.sleep()
            g.sync()

        f()
        m()
        used0 = pool.used_bytes()
        t_f, t_m, t_s = med(f, 5), med(m, 5), med(s, 3)
        f()
        tmp = pool.used_bytes() - used0
        st = g.vram_state_bytes()
        print(f"  gpu    {n:>9,} neurons: forward {t_f * 1e3:8.2f} ms  modulate {t_m * 1e3:8.2f} ms  "
              f"sleep {t_s * 1e3:8.2f} ms   -> forward {n / t_f / 1e6:7.1f} M neuron-steps/s "
              f"({n * 256 / t_f / 1e9:.1f} G synapses/s)")
        print(f"         VRAM: words {st['words']:.0f} B/neuron + state {st['per_neuron']:.0f} B/neuron "
              f"(+{st['per_block']:.2f} per-block, +{st['shared_inputs'] * n / 1e3:.1f} kB shared); "
              f"temporaries after a step: {tmp / 1e6:.2f} MB")
        res[f"gpu{n}"] = (n, t_f, t_m, t_s)
        del g
        pool.free_all_blocks()
    nn, tf, tm, ts = res["numpy"]
    for key in res:
        if key.startswith("gpu"):
            n, gf, gm, gs = res[key]
            print(f"  speed-up per neuron vs numpy ({key}): forward {(n / gf) / (nn / tf):,.0f}x, "
                  f"modulate {(n / gm) / (nn / tm):,.0f}x, sleep {(n / gs) / (nn / ts):,.0f}x")
    return res


if __name__ == "__main__":
    print("== forward: GPU kernel vs PackedPopulation.forward (20,000 neurons, 64 inputs) ==")
    worst, flips = check_forward(20_000)
    print(f"  worst |rate diff| over 5 steps: {worst:.2e}; neurons differing by >1e-4: {flips}")
    print("\n== modulate / sleep (20,000 neurons) ==")
    check_modulate(20_000, oneshot=False)
    check_modulate(20_000, oneshot=True)
    check_sleep(20_000)
    timing()

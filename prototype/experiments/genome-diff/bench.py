"""
Throughput and memory: forward / modulate / sleep for
  - the prototype's NumPy PackedPopulation (baseline),
  - the numba BASE kernel (same rules, procedural birth),
  - the numba ANCHORED kernel (genome + diff dynamics; sleep only visits the diff),
plus forward read directly from a CSR diff (no dense words at all).

Medians of 3+ repeats. Threads <= 3. Timings are noisy (shared machine): compare ratios.
"""

import time
import tracemalloc

import core  # noqa: F401  (first: sets thread caps before numba loads)
import numba as nb
import numpy as np

import formats
from core import PRISTINE, innate_q, syn_hash, target
from neuron import WORKHORSE
from packed_neuron import PackedPopulation

N = 65_536


def med(f, reps=3, setup=None):
    ts = []
    for _ in range(reps):
        if setup:
            setup()
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def peak_mib(f):
    tracemalloc.start()
    tracemalloc.reset_peak()
    f()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2 ** 20


def dense_inputs(rng, n_inputs=64, k=8):
    x = np.zeros(n_inputs, np.float32)
    x[rng.choice(n_inputs, k, replace=False)] = 1.0
    return x


def bench_dense():
    """Every neuron gets input every step (the prototype's 64-input wiring)."""
    rng = np.random.default_rng(0)
    xs = [dense_inputs(rng) for _ in range(8)]
    out = {}
    print(f"\n== A. Dense activity: {N:,} neurons x 256 synapses, 64 inputs, every neuron active every step")
    print(f"   {'implementation':<34}{'forward':>12}{'modulate':>12}{'sleep':>12}   peak temporaries (fwd/mod/sleep)")

    # NumPy baseline
    pop = PackedPopulation(WORKHORSE, N, 64, np.random.default_rng(0))
    for x in xs[:4]:
        pop.forward(x)
    tf = med(lambda: pop.forward(xs[0]))
    tm = med(lambda: pop.modulate(1.0, 0.8, 0.8))
    ts = med(lambda: pop.sleep(), setup=lambda: [pop.forward(x) for x in xs[:2]])
    pf = peak_mib(lambda: pop.forward(xs[1]))
    pm = peak_mib(lambda: pop.modulate(1.0, 0.8, 0.8))
    ps = peak_mib(lambda: pop.sleep())
    out["numpy"] = (tf, tm, ts)
    print(f"   {'prototype NumPy PackedPopulation':<34}{tf * 1e3:>10.0f}ms{tm * 1e3:>10.0f}ms{ts * 1e3:>10.0f}ms"
          f"   {pf:.0f} / {pm:.0f} / {ps:.0f} MiB")
    del pop

    for name, mode in [("numba BASE (same rules)", core.BASE), ("numba ANCHORED (genome+diff)", core.ANCHORED)]:
        gp = core.GenomePop(WORKHORSE, N, core.layout_global(64), seed=1, mode=mode, need_activity=True)
        for x in xs[:4]:
            gp.forward(x)
            gp.modulate(1.0, 0.8, 0.8)
            gp.tick()
        tf = med(lambda: gp.forward(xs[0]), 5)
        tm = med(lambda: gp.modulate(1.0, 0.8, 0.8), 5)

        def day():
            for x in xs[:4]:
                gp.forward(x)
                gp.modulate(1.0, 0.8, 0.8)
                gp.tick()
        ts = med(lambda: gp.sleep(), setup=day)
        pf = peak_mib(lambda: gp.forward(xs[1]))
        pm = peak_mib(lambda: gp.modulate(1.0, 0.8, 0.8))
        ps = peak_mib(lambda: gp.sleep())
        diff = float((gp.words != gp.genome_words()).mean())
        out[name] = (tf, tm, ts)
        print(f"   {name:<34}{tf * 1e3:>10.0f}ms{tm * 1e3:>10.0f}ms{ts * 1e3:>10.0f}ms"
              f"   {pf:.1f} / {pm:.1f} / {ps:.1f} MiB   (diff after 4 dense learning steps: {diff:.0%})")
        if mode == core.ANCHORED:
            fresh = core.GenomePop(WORKHORSE, N, core.layout_global(64), seed=2, mode=mode, need_activity=True)
            t0 = med(lambda: fresh.sleep(), 3)
            print(f"   {'  ANCHORED sleep, brain never used':<34}{'':>12}{'':>12}{t0 * 1e3:>10.1f}ms   (nothing to do)")
    b = out["numpy"]
    for k, v in out.items():
        if k != "numpy":
            print(f"   speed-up vs NumPy, {k}: forward {b[0] / v[0]:.1f}x, modulate {b[1] / v[1]:.1f}x, sleep {b[2] / v[2]:.1f}x")
    syn = N * 256
    for k, v in out.items():
        print(f"   {k:<34} synapse-updates/s: forward {syn / v[0] / 1e6:6.0f}M  modulate {syn / v[1] / 1e6:6.0f}M"
              f"  sleep {syn / v[2] / 1e6:6.0f}M")


@nb.njit(parallel=True, cache=True)
def k_forward_csr(off, slot, val, idx, first_id, lay, x, frow, birth_q, gcoef, thr, rate_out, P):
    """Read-only forward straight from the CSR diff: walk each neuron's sorted diff entries while
    regenerating every synapse from the genome hash. No dense words exist anywhere."""
    q2w = P[7] / 31.0
    n = idx.shape[0]
    for ii in nb.prange(n):
        i = idx[ii]
        nid = np.uint64(first_id + i)
        e = off[i]
        end = off[i + 1]
        soma = 0.0
        for b in range(8):
            drive = 0.0
            for s in range(32):
                j = b * 32 + s
                if e < end and slot[e] == j:
                    w = val[e]
                    e += 1
                else:
                    w = PRISTINE
                h = syn_hash(nid, np.uint64(j), np.uint64(w & 15))
                xi = x[target(h, i, lay)]
                if xi != 0.0:
                    ce = (frow[0] * (((w >> 10) & 63) - 32.0) + frow[1] * (((w >> 16) & 63) - 32.0)
                          + frow[2] * (((w >> 22) & 63) - 32.0))
                    if (w & 15) == 0:
                        ce += innate_q(h, birth_q) * gcoef
                    drive += ((((w >> 4) & 63) - 32.0) + ce) * q2w * xi
            spike = P[1] if drive > P[0] else 0.0
            soma += max(drive, 0.0) + spike
        rate_out[ii] = 1.0 / (1.0 + np.exp(-P[3] * (soma - thr[i])))


def bench_sparse():
    """Local wiring, 4,096 inputs, 2% active: event-driven forward; ANCHORED sleep vs diff size."""
    n_regions, rsize, npc = 64, 64, N // 64
    lay = core.layout_local(n_regions, rsize, npc, 0.1)
    rng = np.random.default_rng(3)
    print(f"\n== B. Sparse activity: {N:,} neurons in 64 columns, 4,096 inputs, 7 things x 12 inputs active (2%)")
    gp = core.GenomePop(WORKHORSE, N, lay, seed=3, mode=core.ANCHORED, need_activity=True, prune_prob=0.05,
                        min_tag=2, snap_q=1)
    regions = rng.integers(0, n_regions, 7)
    xidx = (regions[:, None] * rsize + rng.integers(0, rsize, (7, 12))).ravel()
    x = np.zeros(n_regions * rsize, np.float32)
    x[xidx] = 1.0
    regs = np.unique(regions)
    cols = np.unique(np.concatenate([(regs - 1) % n_regions, regs, (regs + 1) % n_regions]))
    act = (cols[:, None] * npc + np.arange(npc)[None, :]).ravel()
    gp.forward(x, active=act)
    gp.forward(x)
    t_ev = med(lambda: gp.forward(x, active=act), 5)
    t_all = med(lambda: gp.forward(x), 5)
    print(f"   forward, all {N:,} neurons: {t_all * 1e3:.1f} ms | event-driven, {act.size:,} neurons "
          f"({act.size / N:.0%}) that can see input: {t_ev * 1e3:.1f} ms ({t_all / t_ev:.1f}x faster, exact)")
    gp.modulate(1.0, 0.8, 0.8)
    t_mod = med(lambda: gp.modulate(1.0, 0.8, 0.8), 5)
    print(f"   modulate (event-driven, last 4 steps' active cells): {t_mod * 1e3:.1f} ms")

    print("   ANCHORED sleep cost vs how much of the brain differs from the genome "
          "(BASE always pays the full sweep):")
    base = core.GenomePop(WORKHORSE, N, lay, seed=4, mode=core.BASE)
    tb = med(lambda: base.sleep(), 3)
    del base
    learned = core.PRISTINE & ~np.uint32(63 << 10) | np.uint32((32 + 3) << 10)  # l0 = +3 steps
    print(f"   {'dirty neurons':>15}{'synapses differing':>20}{'sleep':>10}{'vs BASE':>10}")
    for nf in [0.0, 0.01, 0.1, 0.5, 1.0]:
        def setup():
            gp.words[:] = core.PRISTINE
            gp.dirty[:] = 0
            gp.active_today[:] = 0
            k = int(N * nf)
            sel = rng.choice(N, k, replace=False)
            sub = gp.words[sel]
            sub[:, :64] = learned  # 64 learned synapses per dirty neuron
            gp.words[sel] = sub
            gp.dirty[sel] = 1
        ts = med(lambda: gp.sleep(), 3, setup=setup)
        print(f"   {nf:>15.0%}{nf * 0.25:>20.1%}{ts * 1e3:>8.1f}ms{tb / max(ts, 1e-9):>9.1f}x")
    print(f"   {'BASE (any)':>15}{'(all swept)':>20}{tb * 1e3:>8.1f}ms")

    # forward straight from a CSR diff vs from dense words (read-only, all neurons)
    setup_frac = 0.3
    gp.words[:] = core.PRISTINE
    sel = rng.choice(N, int(N * setup_frac), replace=False)
    sub = gp.words[sel]
    sub[:, :64] = learned
    gp.words[sel] = sub
    enc = formats.enc_csr(gp.words)
    rate = np.empty(N, np.float64)
    idx = np.arange(N, dtype=np.int64)
    frow = np.array([1.0, 0.0, 0.0])
    gc = gp._gcoef(frow)
    k_forward_csr(enc["off"], enc["slot"], enc["val"], idx, gp.first_id, lay, x, frow, gp.birth_q, gc,
                  gp.threshold, rate, gp.P)
    t_csr = med(lambda: k_forward_csr(enc["off"], enc["slot"], enc["val"], idx, gp.first_id, lay, x, frow,
                                      gp.birth_q, gc, gp.threshold, rate, gp.P), 5)
    ref = gp.forward(x, learn_trace=False)
    assert np.allclose(ref, rate, atol=1e-9), "CSR forward must equal dense forward" 
    t_dense = med(lambda: gp.forward(x, learn_trace=False), 5)
    print(f"   forward of all {N:,} neurons (read-only), 30% of neurons with 64 diff entries:"
          f"\n     from dense words {t_dense * 1e3:.1f} ms ({N * 1024 / 2**20:.0f} MiB)"
          f" | straight from CSR diff {t_csr * 1e3:.1f} ms ({formats.nbytes(enc) / 2**20:.1f} MiB)"
          f" -> {t_csr / t_dense:.2f}x the time, {formats.nbytes(enc) / (N * 1024):.1%} of the memory")


def main():
    bench_dense()
    bench_sparse()


if __name__ == "__main__":
    main()

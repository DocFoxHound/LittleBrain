"""
Correctness checks for every kernel against prototype/packed_neuron.py.

  1. Fused dendrite-major: forward vs PackedPopulation.forward; modulate bit-for-bit (same uniforms).
  2. Axon-major: built from the same PackedPopulation; forward exact; modulate bit-for-bit when the
     fast clock is fresh; lazy fast decay statistically equal; sleep statistics.
  3. Bit-sliced: against the baseline NumPy code reading the same (bitmap) wiring explicitly.
  4. Production kernels (own RNG, float32 tag math) vs the exact paths: every synapse within one
     quantization step, no bias. (This check exists because a 32-bit hash that silently became 64-bit
     in Numba once produced 'uniforms' far above 1 and still passed checks 1-3.)

Run: .venv/bin/python prototype/experiments/cpu-event-driven/validate.py
"""

import common  # noqa: F401

import json

import numpy as np

from axon_major import AxonPopulation
from bitslice import BitSlicedPopulation, ExplicitWiringReference, lib, ptr
from dm_fused import FusedPopulation
from neuron import WORKHORSE
from packed_neuron import PackedPopulation, _get

N = 2000
OUT = {}


def same_uniforms(pop_rng, shape):
    st = pop_rng.bit_generator.state
    uf = pop_rng.random(shape, dtype=np.float32)
    ut = pop_rng.random(shape, dtype=np.float32)
    pop_rng.bit_generator.state = st
    return uf, ut


def real_fast(words, scale):
    return (_get(words, "fast").astype(np.float64) - 32) * scale   # in quantization steps


def check(name, ok, detail):
    OUT[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")


def patterns(nin, k, a, seed=1, graded=False):
    r = np.random.default_rng(seed)
    xs = []
    for _ in range(k):
        x = (r.random(nin) < a).astype(np.float32)
        if graded:
            x *= r.choice(np.array([1.0, 0.5], np.float32), nin)
        xs.append(x)
    return xs


def dendrite_major(nin):
    base = PackedPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7)
    ex = FusedPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7, exact=True)
    pr = FusedPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7)
    maxd, mism = 0.0, 0
    for step, x in enumerate(patterns(nin, 6, 0.15, graded=True)):
        for p in (base, ex, pr):
            p.pending, p.fast_scale = step * 37, 0.9 ** step
        r0, r1 = base.forward(x), ex.forward(x)
        pr.forward(x)
        maxd = max(maxd, float(np.abs(r0 - r1).max()))
        mism += int((base.words != ex.words).sum())
        for p in (ex, pr):     # re-sync float rounding so the next step starts identical
            p.words[:], p.threshold[:], p.post_hist[:], p.pre_hist[:] = base.words, base.threshold, base.post_hist, base.pre_hist
    check(f"fused DM forward (n_inputs={nin})", maxd < 1e-5 and mism == 0,
          f"max |rate diff| {maxd:.1e}, used-flag mismatches {mism}")
    diffs = []
    for da, ach, ar in [(1.0, 0.8, 0.8), (-1.0, 0.8, 0.8), (0.3, 0.5, 0.2)]:
        uf, ut = same_uniforms(base.rng, base.words.shape)
        ex.words[:] = base.words
        base.fast_scale = ex.fast_scale = 0.97
        base.modulate(da, ach, ar)
        ex.modulate(da, ach, ar, uniforms=(uf, ut))
        diffs.append(int((base.words != ex.words).sum()))
    check(f"fused DM modulate bit-exact (n_inputs={nin})", sum(diffs) == 0, f"differing words per call {diffs}")
    # production RNG vs exact
    pr.words[:] = ex.words = base.words.copy()
    pr.fast_scale = ex.fast_scale = 0.97
    uf, ut = same_uniforms(np.random.default_rng(9), base.words.shape)
    ex.modulate(1.0, 0.8, 0.8, uniforms=(uf, ut))
    pr.modulate(1.0, 0.8, 0.8)
    d = real_fast(pr.words, 1) - real_fast(ex.words, 1)
    dt = _get(pr.words, "tag").astype(float) - _get(ex.words, "tag").astype(float)
    check(f"fused DM production RNG unbiased (n_inputs={nin})", np.abs(d).max() <= 1 and abs(d.mean()) < 0.01
          and np.abs(dt).max() <= 1 and abs(dt.mean()) < 0.01,
          f"fast diff max {np.abs(d).max():.0f} step, mean {d.mean():+.4f}; tag diff mean {dt.mean():+.4f}")


def axon_major(nin):
    base = PackedPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7)
    ax = AxonPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7, block=512, track_origin=True)
    w, src, _ = ax.to_dendrite_major()
    check(f"axon-major build (n_inputs={nin})", (w == (base.words & 0xFFFFFFF0)).all() and (src == base._src(base.words)).all(),
          "same words and same inputs as the hash wiring")
    maxd, mism = 0.0, 0
    for step, x in enumerate(patterns(nin, 5, 0.1, graded=True)):
        base.pending = ax.pending = step * 37
        maxd = max(maxd, float(np.abs(base.forward(x) - ax.forward(x)).max()))
        w, _, _ = ax.to_dendrite_major()
        mism += int((w != (base.words & 0xFFFFFFF0)).sum())
    check(f"axon-major forward (n_inputs={nin})", maxd < 1e-5 and mism == 0 and (base.post_hist == ax.post_hist).all(),
          f"max |rate diff| {maxd:.1e}, used-flag mismatches {mism}, post_hist identical")
    diffs = []
    for da, ach, ar in [(1.0, 0.8, 0.8), (-1.0, 0.8, 0.8), (0.3, 0.5, 0.2)]:
        base.fast_scale = 1.0
        uf, ut = same_uniforms(base.rng, base.words.shape)
        base.modulate(da, ach, ar)
        ax.modulate(da, ach, ar, uniforms=(uf, ut, ut))
        w, _, _ = ax.to_dendrite_major()
        diffs.append(int((w != (base.words & 0xFFFFFFF0)).sum()))
    touched = float(np.isin(src, np.flatnonzero(ax.pre_hist.any(0))).mean())
    check(f"axon-major modulate bit-exact, fresh clock (n_inputs={nin})", sum(diffs) == 0,
          f"differing words per call {diffs}; only {touched:.0%} of synapses were visited")
    # production kernel vs exact kernel, on two separate copies in identical states
    pa = AxonPopulation(WORKHORSE, N, nin, np.random.default_rng(6), first_id=8, block=512, track_origin=True)
    pb = AxonPopulation(WORKHORSE, N, nin, np.random.default_rng(6), first_id=8, block=512, track_origin=True)
    for x in patterns(nin, 4, 0.1, seed=3):
        pa.forward(x)
        pb.forward(x)
    for _ in range(3):
        pa.tick()
        pb.tick()
    uf, ut = same_uniforms(np.random.default_rng(9), base.words.shape)
    pa.modulate(1.0, 0.8, 0.8, uniforms=(uf, ut, ut))
    pb.modulate(1.0, 0.8, 0.8)
    d = real_fast(pb.words, 1) - real_fast(pa.words, 1)
    dt = _get(pb.words, "tag").astype(float) - _get(pa.words, "tag").astype(float)
    check(f"axon-major production kernel unbiased (n_inputs={nin})",
          np.abs(d).max() <= 1 and abs(d.mean()) < 0.01 and np.abs(dt).max() <= 1 and abs(dt.mean()) < 0.01,
          f"fast diff max {np.abs(d).max():.0f} step, mean {d.mean():+.4f}; tag diff mean {dt.mean():+.4f}")
    del pa, pb
    # lazy fast decay: after 50 ticks, same expected fast weights as the eager baseline
    for _ in range(50):
        base.tick()
        ax.tick()
    x = patterns(nin, 1, 0.1, seed=4)[0]
    base.forward(x)
    ax.forward(x)
    base.modulate(1.0, 0.8, 0.8)
    ax.modulate(1.0, 0.8, 0.8)
    w, _, fs = ax.to_dendrite_major()
    fb, fa = real_fast(base.words, base.fast_scale), real_fast(w, fs)
    check(f"axon-major lazy fast decay (n_inputs={nin})", abs(fb.mean() - fa.mean()) < 0.02 and np.abs(fb - fa).mean() < 0.5,
          f"mean real fast {fb.mean():.4f} vs {fa.mean():.4f} steps; mean |diff| {np.abs(fb - fa).mean():.3f} step "
          "(baseline re-rounds every synapse each modulate, axon-major only touched ones)")
    nb, na = base.sleep(), ax.sleep()
    w, _, _ = ax.to_dendrite_major()
    c0b, c0a = (_get(base.words, "c0").astype(float) - 32).mean(), (_get(w, "c0").astype(float) - 32).mean()
    ok_buckets = all((np.diff(ax.off[k]) >= 0).all() for k in range(ax.nblk)) and ax.off[-1, -1] == ax.words.size
    rb, ra = base.forward(x), ax.forward(x)
    check(f"axon-major sleep + bucket rebuild (n_inputs={nin})",
          abs(nb - na) < 0.01 * nb + 50 and abs(c0b - c0a) < 0.01 and ok_buckets and abs(rb.mean() - ra.mean()) < 2e-3,
          f"pruned {nb} vs {na}; mean c0 {c0b:.4f} vs {c0a:.4f}; buckets consistent; rate after sleep "
          f"{rb.mean():.4f} vs {ra.mean():.4f}")


def bit_sliced(nin):
    bs = BitSlicedPopulation(WORKHORSE, N, nin, np.random.default_rng(5), first_id=7)
    ref = ExplicitWiringReference(bs)
    w, src = bs.unpack()
    distinct = all(len(set(r)) == 32 for r in (src.reshape(-1, 32) if nin > 64 else []))
    check(f"bit-sliced layout (n_inputs={nin})", bs.bytes_per_neuron() == 1041.0 and (nin == 64 or distinct),
          f"{bs.bytes_per_neuron():.0f} bytes/neuron; 32 distinct inputs per branch inside a 128-input window")
    maxd, mism = 0.0, 0
    for step, x in enumerate(patterns(nin, 6, 0.1)):
        ref.pending = bs.pending = step * 37
        ref.fast_scale = bs.fast_scale = 0.9 ** step
        maxd = max(maxd, float(np.abs(ref.forward(x) - bs.forward(x)).max()))
        mism += int((bs.unpack()[0] != ref.words).sum())
        bs.threshold[:] = ref.threshold
    check(f"bit-sliced forward (n_inputs={nin})", maxd < 1e-5 and mism == 0 and (ref.post_hist == bs.post_hist).all(),
          f"max |rate diff| {maxd:.1e}, used-flag mismatches {mism}")
    diffs = []
    for da, ach, ar in [(1.0, 0.8, 0.8), (-1.0, 0.8, 0.8), (0.3, 0.5, 0.2)]:
        ref.fast_scale = bs.fast_scale = 0.97
        uf, ut = same_uniforms(ref.rng, ref.words.shape)
        ref.modulate(da, ach, ar)
        bs.modulate(da, ach, ar, uniforms=(uf, ut, ut))
        diffs.append(int((bs.unpack()[0] != ref.words).sum()))
    check(f"bit-sliced modulate bit-exact (n_inputs={nin})", sum(diffs) == 0, f"differing words per call {diffs}")
    words_before = bs.unpack()[0]
    bs.fast_scale = 0.97
    bs.modulate(1.0, 0.8, 0.8)
    ref.words[:] = words_before
    ref.fast_scale = 0.97
    ref.modulate(1.0, 0.8, 0.8)
    d = real_fast(bs.unpack()[0], 1) - real_fast(ref.words, 1)
    check(f"bit-sliced production RNG unbiased (n_inputs={nin})", np.abs(d).max() <= 1 and abs(d.mean()) < 0.01,
          f"fast diff max {np.abs(d).max():.0f} step, mean {d.mean():+.4f}")
    nr, nbs = ref.sleep(), bs.sleep()
    w2, src2 = bs.unpack()
    c0r, c0b = (_get(ref.words, "c0").astype(float) - 32).mean(), (_get(w2, "c0").astype(float) - 32).mean()
    moved = float((src2 != src).any(axis=2).mean())
    ws = np.empty(N * WORKHORSE.n_branches, np.int64)
    lib.bs_windows(N, WORKHORSE.n_branches, bs.first_id, bs.nW, ptr(ws))
    win_ok = bool(((src2 - ws.reshape(N, -1, 1) * 64) % nin < 128).all())
    check(f"bit-sliced sleep + local rewiring (n_inputs={nin})",
          abs(nr - nbs) < 0.01 * nr + 50 and abs(c0r - c0b) < 0.01 and win_ok,
          f"pruned {nr} vs {nbs}; mean c0 {c0r:.4f} vs {c0b:.4f}; {moved:.0%} of branches rewired, "
          f"new inputs stay in the window: {win_ok}")


if __name__ == "__main__":
    for nin in (64, 4096):
        print(f"== n_inputs = {nin} ==")
        dendrite_major(nin)
        axon_major(nin)
        bit_sliced(nin)
    bad = [k for k, v in OUT.items() if not v["ok"]]
    print("ALL OK" if not bad else f"FAILED: {bad}")
    with open(f"{common.HERE}/results_validate.json", "w") as f:
        json.dump(OUT, f, indent=1)

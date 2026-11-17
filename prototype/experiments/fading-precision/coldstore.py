"""
The cold store: freeze / thaw blocks of neurons, with optional precision fading.

  freeze:  fade the cascade levels to the tier's bits -> "symbols" layout -> zstd-3  (one blob per block)
  night:   nothing is touched. Time passing (flow + downscale) is linear, so it is one 3x3 matrix
           per block, multiplied up across nights ("lazy evolution").
  re-fade: at a tier boundary (e.g. after 3 nights) a sleep-time job thaws, re-fades, re-freezes.
  thaw:    zstd -> inverse layout -> apply the accumulated matrix exactly -> stochastic rounding
           back to 6 bits -> tags decayed by tag_decay^nights; age and plasticity caught up.
           Wiring is frozen with the neuron (no pruning while cold; see README "fading x pruning").

Run: python coldstore.py      (benchmarks on the simulated day-16 brain + population behavior test)
"""

import pickle
import sys
import tracemalloc
from dataclasses import replace

from common import DATA, RESULTS, median_time, np, save_json, time

import zstandard as zstd

from brainsim import (GROUPS, IDLE_REST, LocalPopulation, background, group_input, load_day8, load_day16,
                      probe)
from fade import FULL, fade_words, stored_bits_per_synapse
from lossless import LAYOUTS, extras_bytes
from packed_neuron import LEVELS, Q, TAG_MAX, _dec, _flow_matrix, _get, _set

LAYOUT = "symbols"   # best on both raw and faded cold blocks (see bench_sizes)
LEVEL = 3
_C = {}


def _codec(level):
    if level not in _C:
        _C[level] = (zstd.ZstdCompressor(level=level), zstd.ZstdDecompressor())
    return _C[level]


def freeze_block(words, extras, bits, rng, layout=LAYOUT, level=LEVEL, scale="block"):
    """words (n, B, S) uint32, extras (n, 17) uint8 -> blob (bytes) + header.
    scale="vqm64" etc. uses a brain-wide synapse-state codebook (fringe.py) instead of per-level bits."""
    w = fade_words(words, bits, rng, scale=scale) if (bits != FULL or scale != "block") else words
    n = words.shape[0]
    flat = w.reshape(n, -1)
    payload = LAYOUTS[layout][0](flat) + extras.tobytes()
    cc, _ = _codec(level)
    return {"z": cc.compress(payload), "shape": words.shape, "ne": extras.shape[1], "layout": layout,
            "bits": bits}


def thaw_block(blob, t, op=None, nights=0, rng=None):
    """-> words (n, B, S), extras (n, ne). Applies the accumulated linear evolution `op` exactly."""
    _, dd = _codec(LEVEL)
    raw = dd.decompress(blob["z"])
    n, B, S = blob["shape"]
    nw = n * B * S
    wbytes = len(raw) - n * blob["ne"]
    flat = LAYOUTS[blob["layout"]][1](raw[:wbytes], (n, B * S))
    words = np.ascontiguousarray(flat).reshape(n, B, S)
    extras = np.frombuffer(raw[wbytes:], np.uint8).reshape(n, blob["ne"])
    if op is not None and not np.allclose(op, np.eye(len(op))):
        c = np.stack([_dec(_get(words, LEVELS[k]), t.w_max) for k in range(t.cascade_levels)])
        c = np.tensordot(op.astype(np.float32), c, axes=1)
        for k in range(t.cascade_levels):
            x = np.floor(np.clip(c[k], -t.w_max, t.w_max) * (Q / t.w_max) + rng.random(c[k].shape, dtype=np.float32))
            words = _set(words, LEVELS[k], (np.clip(x, -Q, Q) + 32).astype(np.uint32))
    if nights:
        tag = _get(words, "tag").astype(np.float32) * t.tag_decay ** nights
        words = _set(words, "tag", np.floor(tag + rng.random(tag.shape, dtype=np.float32)).astype(np.uint32))
    assert nw == words.size
    return words, extras


def night_op(t):
    D = np.eye(t.cascade_levels)
    D[0, 0] = 1.0 - t.downscale
    return D @ _flow_matrix(t, t.sleep_flow_steps).astype(np.float64)


def rest_op(t, steps):
    return _flow_matrix(t, steps).astype(np.float64)


def set_extras(pop, idx, extras):
    """Write back per-neuron state (inverse of lossless.extras_bytes)."""
    gb = pop.gate_bits.reshape(pop.n, -1).shape[1]
    o = 0
    pop.gate_bits[idx] = extras[:, o:o + gb].reshape(len(idx), -1); o += gb
    pop.threshold[idx] = extras[:, o:o + 4].copy().view(np.float32).ravel(); o += 4
    pop.plasticity[idx] = extras[:, o:o + 2].copy().view(np.float16).ravel(); o += 2
    ph = pop.post_hist.shape[1] * 2
    pop.post_hist[idx] = extras[:, o:o + ph].copy().view(np.float16).reshape(len(idx), -1); o += ph
    pop.age[idx] = extras[:, o:o + 2].copy().view(np.uint16).ravel()


# ---------------------------------------------------------------- benchmarks on realistic cold data

TIERS = [FULL, (4, 6, 6), (4, 4, 6), (3, 3, 4), (2, 2, 4), (2, 2, 2), (0, 0, 4), (0, 0, 2)]


def bench_sizes():
    """Bytes per cold neuron after fading, for several lossless layouts (zstd-3, 64-neuron blocks).
    Two moments: day 8 (A and B just went cold: levels still disagree) and day 16 (8 nights cold:
    the cascade flow has pulled c0, c1, c2 together)."""
    rng = np.random.default_rng(0)
    out = {}
    snaps = {"day16": load_day16()["pops"], "day8": load_day8()}
    print("[cold store] bytes per neuron (incl. 17 B per-neuron state) after fading, by layout; zstd-3, 64-neuron blocks")
    for when, pops in snaps.items():
        for g in ("A", "B", "E", "C"):
            if g not in pops:
                continue
            p = pops[g]
            ex = extras_bytes(p)
            n = min(p.n, 2048)
            for bits in TIERS:
                row = {}
                for lay in ("raw", "fields", "dfields", "symbols"):
                    tot = sum(len(freeze_block(p.words[i:i + 64], ex[i:i + 64], bits, rng, layout=lay)["z"])
                              for i in range(0, n, 64))
                    row[lay] = tot / n
                key = f"{when}/{g}/{''.join(map(str, bits))}"
                out[key] = row
                print(f"  {key:<16} " + "  ".join(f"{k}={v:5.0f}" for k, v in row.items())
                      + f"   fixed-width {stored_bits_per_synapse(bits) * 32 + 17 if bits != FULL else 22 * 32 + 17:.0f}")
            if (DATA / "fringe_library.npz").exists():   # codebook fading (built by fringe.py)
                for sc in ("vqm64", "vqm256"):   # (the full 2,945-state table equals lossless here)
                    tot = sum(len(freeze_block(p.words[i:i + 64], ex[i:i + 64], FULL, rng, scale=sc)["z"])
                              for i in range(0, n, 64))
                    out[f"{when}/{g}/{sc}"] = {"symbols": tot / n}
                    print(f"  {when + '/' + g + '/' + sc:<16} symbols={tot / n:5.0f}")
    return out


def bench_speed():
    """Freeze / thaw throughput and peak temporary memory, on the cold groups (A + B + E)."""
    d = load_day16()
    pops = d["pops"]
    t = pops["A"].t
    words = np.concatenate([pops[g].words for g in ("A", "B", "E")])
    ex = np.concatenate([extras_bytes(pops[g]) for g in ("A", "B", "E")])
    n = 4096
    words, ex = words[:n], ex[:n]
    rng = np.random.default_rng(1)
    op = night_op(t) @ rest_op(t, IDLE_REST)
    res = {}
    for nb in (64, 1024):
        for bits in (FULL, (4, 4, 6), (2, 2, 4)):
            f = lambda: [freeze_block(words[i:i + nb], ex[i:i + nb], bits, rng) for i in range(0, n, nb)]
            tf, blobs = median_time(f)
            th = lambda: [thaw_block(b, t, op, 8, rng) for b in blobs]
            tt, _ = median_time(th)
            th0 = lambda: [thaw_block(b, t) for b in blobs]
            tt0, _ = median_time(th0)
            tracemalloc.start()
            thaw_block(blobs[0], t, op, 8, rng)
            peak_thaw = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            tracemalloc.start()
            freeze_block(words[:nb], ex[:nb], bits, rng)
            peak_freeze = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            key = f"nb{nb}/{''.join(map(str, bits))}"
            res[key] = {"bytes_per_neuron": sum(len(b["z"]) for b in blobs) / n,
                        "freeze_neurons_per_s": n / tf, "thaw_evolve_neurons_per_s": n / tt,
                        "thaw_plain_neurons_per_s": n / tt0,
                        "freeze_GBps": n * 1041 / tf / 1e9, "thaw_plain_GBps": n * 1041 / tt0 / 1e9,
                        "peak_temp_freeze_bytes_per_neuron": peak_freeze / nb,
                        "peak_temp_thaw_bytes_per_neuron": peak_thaw / nb}
            r = res[key]
            print(f"  {key:<14} {r['bytes_per_neuron']:6.0f} B/neuron  freeze {r['freeze_neurons_per_s']:9,.0f} n/s"
                  f"  thaw {r['thaw_plain_neurons_per_s']:9,.0f} n/s (with 8-night evolution "
                  f"{r['thaw_evolve_neurons_per_s']:8,.0f})  peak temp {r['peak_temp_freeze_bytes_per_neuron']:.0f}"
                  f" / {r['peak_temp_thaw_bytes_per_neuron']:.0f} B/neuron")
    # for scale: the regular per-neuron step costs on the same machine (hot path, unchanged)
    p = pops["C"]
    x = group_input(d["pats"], p.cols, 0)
    pc = pickle.loads(pickle.dumps(p))
    tfwd, _ = median_time(lambda: pc.forward(x))
    tmod, _ = median_time(lambda: pc.modulate(1.0, 0.8, 0.8))
    tsl, _ = median_time(lambda: pc.sleep())
    res["hot_forward_neurons_per_s"] = p.n / tfwd
    res["hot_modulate_neurons_per_s"] = p.n / tmod
    res["hot_sleep_neurons_per_s"] = p.n / tsl
    print(f"  for scale: hot forward {p.n / tfwd:,.0f} neurons/s, modulate {p.n / tmod:,.0f}, sleep {p.n / tsl:,.0f} "
          f"neurons/s "
          f"(a frozen neuron's night costs nothing: one 3x3 matrix product per block)")
    return res


# ---------------------------------------------------------------- population behavior with fading

REFREEZE = FULL + ("refreeze",)   # same bits; a different tier forces a sleep-time thaw + re-freeze

POP_POLICIES = {
    "baseline": None,
    "lossless-freeze": [(1, FULL)],
    "lossless-refreeze": [(1, FULL), (4, REFREEZE), (8, FULL)],
    "deep-last": [(1, (4, 6, 6)), (2, (4, 4, 6)), (3, (3, 4, 5)), (4, (2, 3, 4)), (6, (2, 2, 4))],
    "ternary": [(1, (2, 2, 2))],
    "lossless-then-ternary": [(1, FULL), (4, (2, 2, 2))],
    "codebook64": [(1, "vqm64")],
    "codebook256": [(1, "vqm256")],
    # "codebook-all-2945" ([(1, "vqm_all")]) was run once: A 0.03 0.07 0.15 0.17, relearn 2, 338 B/neuron,
    # but its nearest-entry search took ~5 minutes, so it is left out of the default run.
    "lossless-refreeze-then-codebook64": [(1, FULL), (4, "vqm64")],
    "surface-erased": [(1, (0, 0, 4))],
    "all-erased-but-2bit-deep": [(1, (0, 0, 2))],
    "deep-erased": [(1, (4, 4, 0))],
}


def tier_bits(tiers, nights):
    b = None
    for n, bb in tiers:
        if nights >= n:
            b = bb
    return b


def pop_story(policy_name, mode="lazy", relearn_trials=12, seed=7):
    """Groups A and B from day 8; idle days 9..16 under a fading policy; then A relearns its pattern."""
    d = load_day8()
    pats = d["pats"]
    rng = np.random.default_rng(seed)
    tiers = POP_POLICIES[policy_name]
    pops = {g: d[g] for g in ("A", "B")}
    t = pops["A"].t
    state = {g: {"frozen": None, "op": np.eye(3), "nights": 0, "bits": None} for g in pops}
    log = {"A_main": [], "B_main": [], "B_second": [], "bytes_per_neuron": []}

    def view(g):
        """Response probes on a frozen group thaw a temporary copy (nothing stored)."""
        p, s = pops[g], state[g]
        if s["frozen"] is None:
            return p
        q = pickle.loads(pickle.dumps(p))
        q.words, _ = [np.concatenate(z) for z in zip(*[thaw_block(b, t, s["op"], 0, rng)
                                                        for b in s["frozen"]])]
        return q

    def thaw(g):
        p, s = pops[g], state[g]
        if s["frozen"] is None:
            return
        parts = [thaw_block(b, t, s["op"], s["nights"], rng) for b in s["frozen"]]
        p.words = np.concatenate([w for w, _ in parts])
        set_extras(p, np.arange(p.n), np.concatenate([e for _, e in parts]))
        p.age += s["nights"]
        p.plasticity = (t.plasticity_floor + (p.plasticity.astype(np.float32) - t.plasticity_floor)
                        * t.maturation ** s["nights"]).astype(np.float16)
        s.update(frozen=None, op=np.eye(3), nights=0)

    def freeze(g, bits):
        p, s = pops[g], state[g]
        ex = extras_bytes(p)
        if isinstance(bits, str):   # a brain-wide synapse-state codebook (fringe.py)
            s["frozen"] = [freeze_block(p.words[i:i + 64], ex[i:i + 64], FULL, rng, scale=bits)
                           for i in range(0, p.n, 64)]
        else:
            s["frozen"] = [freeze_block(p.words[i:i + 64], ex[i:i + 64], bits, rng) for i in range(0, p.n, 64)]
        s["bits"] = bits

    for day in range(9, 17):
        nights_unused = day - 8
        for g, p in pops.items():
            s = state[g]
            if tiers is None:
                p.rest(IDLE_REST)
                p.sleep()
                continue
            if s["frozen"] is not None:
                s["op"] = night_op(t) @ rest_op(t, IDLE_REST) @ s["op"]
                s["nights"] += 1
                bits = tier_bits(tiers, nights_unused)
                if bits != s["bits"]:            # sleep-time re-fade to the next tier
                    thaw(g)
                    freeze(g, bits)
            else:
                p.rest(IDLE_REST)
                p.t = replace(t, prune_eps=-1.0)  # first cold night: no pruning (wiring frozen)
                p.sleep()
                p.t = t
                freeze(g, tier_bits(tiers, nights_unused))
        if tiers is not None:
            log["bytes_per_neuron"].append(round(sum(len(b["z"]) for g in pops for b in state[g]["frozen"])
                                                 / sum(p.n for p in pops.values()), 1))
        log["A_main"].append(round(float(probe(view("A"), pats, 0).mean()), 3))
        vb = view("B")
        log["B_main"].append(round(float(probe(vb, pats, 0).mean()), 3))
        log["B_second"].append(round(float(probe(vb, pats, 1).mean()), 3))

    # Day 17: A meets its childhood pattern again, rewarded. Per neuron: trials until response >= 0.7.
    thaw("A")
    pA = pops["A"]
    x = group_input(pats, pA.cols, 0)
    reached = np.full(pA.n, relearn_trials + 1)
    curve = []
    for i in range(relearn_trials + 1):
        r = pA.forward(x, learn_trace=False)
        curve.append(round(float(r.mean()), 3))
        reached = np.where((r >= 0.7) & (reached > relearn_trials), i, reached)
        if i == relearn_trials:
            break
        pA.forward(x)
        pA.modulate(dopamine=1.0, ach=0.8, arousal=0.8)
        pA.tick()
        pA.forward(background(rng, pA.cols))
        pA.tick()
    log["relearn_curve_A"] = curve
    log["relearn_median_trials_A"] = float(np.median(reached))
    log["relearn_frac_by_2_A"] = round(float((reached <= 2).mean()), 3)
    return log


def pop_behavior():
    res = {}
    print("\n[population behavior] groups A (6,144 neurons, old self) and B (4,096, stable memories), idle days 9-16")
    for name in POP_POLICIES:
        t0 = time.time()
        r = pop_story(name)
        res[name] = r
        print(f"  {name:<26} A day9..16 {r['A_main'][0]:.2f} {r['A_main'][2]:.2f} {r['A_main'][5]:.2f} "
              f"{r['A_main'][7]:.2f} | B liked {r['B_main'][-1]:.2f} disliked {r['B_second'][-1]:.2f} | "
              f"A relearn median {r['relearn_median_trials_A']:.0f} (by 2: {r['relearn_frac_by_2_A']:.2f}) "
              f"curve {r['relearn_curve_A'][:4]} | bytes/neuron {r['bytes_per_neuron'][-1] if r['bytes_per_neuron'] else 1041}"
              f"  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if what in ("sizes", "all"):
        out["sizes"] = bench_sizes()
    if what in ("speed", "all"):
        out["speed"] = bench_speed()
    if what in ("pop", "all"):
        out["pop_behavior"] = pop_behavior()
    save_json(f"coldstore_{what}.json", out)

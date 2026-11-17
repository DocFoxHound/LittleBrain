"""
Step 4 (fringe): sharing structure ACROSS synapses and neurons.

  a) Synapse-state codebook (vector quantization): one brain-wide table of K "prototype" cascade
     states (c0, c1, c2); a cold synapse stores only an index (log2 K bits) for all three levels.
     Distance is capacity-weighted (1, 2, 4), because the surface eventually drifts to the
     capacity-weighted mean of the levels, which is what behavior depends on in the long run.
  b) Cross-neuron deduplication / prototype + residual: do neurons that learned the same thing
     have similar synapse vectors? Checked in slot order (how they are stored) and in input order.
  c) Holographic superposition: K cold neurons' level vectors bound with random +-1 keys and a
     random permutation, summed into ONE vector; a neuron is read back by unbinding with its key.
     How many fit before recall degrades?

The codebook and a library of realistic cold neurons (for crosstalk) come from the simulated
brain (day 16, groups A + B + E) and are saved in data/fringe_library.npz, so the behavior workers
can use them (scale="vq<K>" or "holo<K>" in a fade Policy).

Run: python fringe.py
"""

import sys

from common import DATA, np, save_json, time

LIB = DATA / "fringe_library.npz"
CAP = np.array([1.0, 2.0, 4.0])   # cascade capacities: weights for the VQ distance
_lib = None


def _f(w, off, bits=6):
    return ((w >> np.uint32(off)) & np.uint32((1 << bits) - 1)).astype(np.int32)


def triples(words):
    """(n_syn, 3) signed codes for c0, c1, c2."""
    w = words.reshape(-1)
    return np.stack([_f(w, 10) - 32, _f(w, 16) - 32, _f(w, 22) - 32], axis=1)


def nearest(x, cb, budget=2_000_000):
    """Index of the nearest codebook entry (capacity-weighted), in chunks of <= budget distances."""
    out = np.empty(len(x), np.int64)
    chunk = max(1, budget // len(cb))
    for i in range(0, len(x), chunk):
        d = (((x[i:i + chunk, None, :] - cb[None]) ** 2) * CAP).sum(-1)
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def kmeans_weighted(x, wts, k, iters=30, seed=0):
    """Lloyd's k-means on unique points with counts; capacity-weighted distance."""
    r = np.random.default_rng(seed)
    xs = x * np.sqrt(CAP)
    k = min(k, len(x))
    # k-means++ init
    cent = [xs[r.choice(len(x), p=wts / wts.sum())]]
    d2 = ((xs - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = wts * d2
        cent.append(xs[r.choice(len(x), p=p / p.sum())])
        d2 = np.minimum(d2, ((xs - cent[-1]) ** 2).sum(1))
    cent = np.array(cent)
    for _ in range(iters):
        a = \
            np.concatenate([np.argmin(((xs[i:i + 8192, None, :] - cent[None]) ** 2).sum(-1), axis=1)
                            for i in range(0, len(xs), 8192)])
        for j in range(k):
            m = a == j
            if m.any():
                cent[j] = (xs[m] * wts[m, None]).sum(0) / wts[m].sum()
    cb = np.rint(cent / np.sqrt(CAP)).astype(np.int32)
    return np.unique(cb, axis=0)


def build_library():
    from brainsim import load_day16
    d = load_day16()
    words = np.concatenate([d["pops"][g].words for g in ("A", "B", "E")])
    tr = triples(words)
    uniq, cnt = np.unique(tr, axis=0, return_counts=True)
    books = {f"cb{k}": kmeans_weighted(uniq.astype(float), cnt.astype(float), k) for k in (16, 64, 256)}
    # "mixed" codebooks: also see freshly cold states (day 8, A and B right after their last active day),
    # where the surface and the deep levels still disagree (the old self vs the hard times)
    from brainsim import load_day8
    d8 = load_day8()
    tr8 = np.concatenate([tr, triples(np.concatenate([d8["A"].words, d8["B"].words]))])
    um, cm = np.unique(tr8, axis=0, return_counts=True)
    books.update({f"cbm{k}": kmeans_weighted(um.astype(float), cm.astype(float), k) for k in (64, 256)})
    books["cbm_all"] = um.astype(np.int32)      # every state seen: a lossless brain-wide table
    r = np.random.default_rng(0)
    sample = words[r.choice(len(words), 2048, replace=False)]
    lv = np.stack([(_f(sample, o) - 32) / 31.0 for o in (10, 16, 22)], axis=1).astype(np.float32)  # (n,3,B,S)
    np.savez(LIB, cold_levels=lv, uniq=uniq, cnt=cnt, **books)
    return words, uniq, cnt, books


def lib():
    global _lib
    if _lib is None:
        _lib = dict(np.load(LIB))
    return _lib


# ------------------------------------------------------------------ fade hooks (used by fade.py)

_LUT = {}


def lut(name):
    """Codes are integers in -31..31, so there are only 63^3 possible states: precompute the
    nearest codebook entry for every one of them once, then quantizing is a table lookup."""
    if name not in _LUT:
        g = np.arange(-31, 32)
        allt = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
        _LUT[name] = nearest(allt, lib()[name]).astype(np.int32)
    return _LUT[name]


def state_key(tr):
    return ((tr[:, 0] + 31) * 63 + (tr[:, 1] + 31)) * 63 + (tr[:, 2] + 31)


def apply_vq(words, k, rng=None):
    """k: an int (day-16 cold codebook) or 'm<K>' / 'm_all' (codebook that also saw freshly cold states)."""
    from packed_neuron import LEVELS, _set
    cb = lib()[f"cb{k}"]
    q = cb[lut(f"cb{k}")[state_key(triples(words))]]
    w = words
    for i in range(3):
        w = _set(w, LEVELS[i], (q[:, i].reshape(words.shape) + 32).astype(np.uint32))
    return w


def holo_roundtrip(levels, others, rng):
    """levels (3, B, S) float; others (K-1, 3, B, S). Superpose with random +-1 keys and
    permutations, then unbind the first item. Returns the recovered (3, B, S)."""
    items = np.concatenate([levels[None], others]).reshape(len(others) + 1, -1)
    n, d = items.shape
    keys = rng.choice([-1.0, 1.0], size=(n, d))
    perms = np.stack([rng.permutation(d) for _ in range(n)])
    trace = np.zeros(d)
    for i in range(n):
        trace[perms[i]] += keys[i] * items[i]
    rec = trace[perms[0]] * keys[0]
    return rec.reshape(levels.shape)


def apply_holo(words, k, rng):
    from packed_neuron import LEVELS, _set
    L = lib()["cold_levels"]
    lv = np.stack([(_f(words, o) - 32) / 31.0 for o in (10, 16, 22)], axis=0)[:, 0]   # (3, B, S)
    others = L[rng.choice(len(L), k - 1, replace=False)] if k > 1 else np.zeros((0,) + lv.shape)
    rec = holo_roundtrip(lv, others, rng)
    w = words
    for i in range(3):
        x = np.clip(np.floor(rec[i] * 31 + rng.random(rec[i].shape)), -31, 31)
        w = _set(w, LEVELS[i], (x[None] + 32).astype(np.uint32))
    return w


# ------------------------------------------------------------------ analyses

def entropy(cnt):
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def vq_report(uniq, cnt, books):
    out = {"unique_triples": int(len(uniq)), "triple_entropy_bits": entropy(cnt)}
    tr = np.repeat(uniq, cnt, axis=0).astype(float)
    sub = tr[np.random.default_rng(1).choice(len(tr), min(len(tr), 200000), replace=False)]
    global _lib
    _lib = dict(np.load(LIB))
    for name, cb in books.items():
        a = lut(name)[state_key(sub.astype(np.int64))]
        q = cb[a]
        err = q - sub
        mean_err = ((q - sub) * CAP).sum(1) / CAP.sum()     # error of the capacity-weighted mean
        _, ic = np.unique(a, return_counts=True)
        out[name] = {"K": int(len(cb)), "rms_code_err_per_level": np.sqrt((err ** 2).mean(0)).round(3).tolist(),
                     "rms_err_weighted_mean_codes": float(np.sqrt((mean_err ** 2).mean())),
                     "bias_weighted_mean_codes": float(mean_err.mean()),
                     "index_entropy_bits": entropy(ic)}
    # scalar fading at similar budgets, same metric (block scale, stochastic), for comparison
    from fade import fade_codes
    r = np.random.default_rng(2)
    n = (len(sub) // 32) * 32
    s = sub[:n].astype(np.int32).reshape(-1, 32, 3)
    for bits in ((2, 2, 4), (2, 2, 2), (3, 3, 3), (4, 4, 6)):
        q = np.stack([fade_codes(s[..., i], bits[i], r)[0] for i in range(3)], axis=-1).reshape(-1, 3)
        err = q - sub[:n]
        mean_err = (err * CAP).sum(1) / CAP.sum()
        out[f"scalar{''.join(map(str, bits))}"] = {
            "rms_code_err_per_level": np.sqrt((err ** 2).mean(0)).round(3).tolist(),
            "rms_err_weighted_mean_codes": float(np.sqrt((mean_err ** 2).mean())),
            "bias_weighted_mean_codes": float(mean_err.mean())}
    return out


def dedup_report(words):
    """Exact duplicates and cross-neuron similarity, in slot order and in input order."""
    from brainsim import GROUPS, REGION, load_day16
    from packed_neuron import _get
    out = {}
    n = words.shape[0]
    rows = words.reshape(n, -1)
    out["duplicate_neurons_frac"] = 1 - len(np.unique(rows, axis=0)) / n
    br = words.reshape(-1, words.shape[-1])
    out["duplicate_branches_frac"] = 1 - len(np.unique(br, axis=0)) / len(br)
    lv = (rows >> np.uint32(10)) & np.uint32((1 << 18) - 1)
    out["duplicate_branch_levels_frac"] = 1 - len(np.unique(lv.reshape(-1, words.shape[-1]), axis=0)) / len(br)
    # similarity between neurons of the same column (group A: all learned the same pattern)
    d = load_day16()
    p = d["pops"]["A"]
    c2 = (_get(p.words, "c2").astype(np.float32) - 32).reshape(p.n, -1)
    src = (p._src(p.words) - p.region_base).reshape(p.n, -1)
    m = 256
    slot = np.corrcoef(c2[:m])[np.triu_indices(m, 1)]
    prof = np.zeros((m, REGION), np.float32)
    for i in range(m):
        np.add.at(prof[i], src[i], c2[i])
        prof[i] /= np.maximum(np.bincount(src[i], minlength=REGION), 1)
    inp = np.corrcoef(prof)[np.triu_indices(m, 1)]
    out["same_column_corr_slot_order"] = float(np.nanmean(slot))
    out["same_column_corr_input_order"] = float(np.nanmean(inp))
    # prototype + residual in slot order: residual vs raw entropy (c2 plane)
    proto = np.rint(c2[: p.n // 6].mean(0))
    res = c2[: p.n // 6] - proto
    h = lambda v: entropy(np.unique(v, return_counts=True)[1].astype(float))
    out["c2_entropy_raw_bits"] = h(c2[: p.n // 6])
    out["c2_entropy_residual_vs_column_prototype_bits"] = h(res)
    return out


def holo_report(words, ks=(1, 2, 4, 8, 16, 64)):
    L = lib()["cold_levels"]
    r = np.random.default_rng(3)
    out = {}
    for k in ks:
        errs, exact, corr = [], [], []
        for t in range(64):
            i = r.integers(len(L))
            own = L[i]
            others = L[r.choice(np.delete(np.arange(len(L)), i), k - 1, replace=False)] if k > 1 \
                else np.zeros((0,) + own.shape, np.float32)
            rec = holo_roundtrip(own, others, r)
            errs.append(np.sqrt(((rec - own) ** 2).mean()) * 31)
            exact.append((np.rint(rec * 31) == np.rint(own * 31)).mean())
            corr.append(np.corrcoef(rec.ravel(), own.ravel())[0, 1])
        out[k] = {"rms_err_codes": float(np.mean(errs)), "exact_frac": float(np.mean(exact)),
                  "corr": float(np.mean(corr)),
                  "bytes_per_neuron_float16_trace": 3 * 256 * 2 / k}
        print(f"  holographic K={k:<3} rms error {out[k]['rms_err_codes']:5.2f} codes   exact "
              f"{out[k]['exact_frac']:.2f}   corr {out[k]['corr']:.2f}   trace bytes/neuron "
              f"{out[k]['bytes_per_neuron_float16_trace']:.0f}")
    return out


def main():
    t = time.time()
    words, uniq, cnt, books = build_library()
    res = {"vq": vq_report(uniq, cnt, books)}
    print("[VQ] unique (c0,c1,c2) triples in cold set:", res["vq"]["unique_triples"],
          " joint entropy %.2f bits" % res["vq"]["triple_entropy_bits"])
    for k, v in res["vq"].items():
        if isinstance(v, dict):
            print(f"  {k:<12} {v}")
    res["dedup"] = dedup_report(words)
    print("[dedup]", res["dedup"])
    res["holo"] = holo_report(words)
    save_json("fringe.json", res)
    print(f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

"""
Step 2: lossless compression of cold blocks of packed neurons.

Data: the day-16 state of the simulated brain (brainsim.py), right after a night's sleep.
  cold  = groups A + B (unused for 8 nights, each holds an old self) + E (never used, 16 nights)
  hot   = groups C + D (used today)
A "cold block" is `nb` consecutive neurons: their 256 x 32-bit words plus 17 bytes of per-neuron
state (threshold, plasticity, recent firing, age, branch gates).

Layouts (all exactly invertible; checked):
  raw          the 32-bit words as they are
  byteshuffle  4 byte planes (byte 0 of every word, then byte 1, ...)
  bitshuffle   32 bit planes (bit 0 of every word, then bit 1, ...) = field bit-planes
  fields       7 byte planes, one per field (gen, fast, c0, c1, c2, tag, used)
  dfields      fields, with the cascade delta-coded: c2, c1-c2, c0-c1 (mod 64)
  dbitshuffle  delta-coded words, then 32 bit planes
  symbols      per block, a table of the distinct (c0, c1, c2) triples + one small index per synapse
               (synapse-state deduplication), plus gen / tag planes
  sortrows     dfields after sorting the block's neurons by their deep-level mean (+ permutation)
Codecs: zstd at levels -5, 1, 3, 9, 19 (with and without a trained dictionary), lz4 (fast, HC).

Run: python lossless.py
"""

import sys

from common import median_time, np, save_json, time

import lz4.block
import lz4.frame
import zstandard as zstd

from brainsim import GROUPS, load_day16
from packed_neuron import FIELDS

ORDER = ("gen", "fast", "c0", "c1", "c2", "tag", "used")
S = 256


# ------------------------------------------------------------------ data

def extras_bytes(pop):
    """Per-neuron state that is not in the synapse words (17 bytes per neuron)."""
    return np.concatenate([pop.gate_bits.reshape(pop.n, -1).view(np.uint8),
                           pop.threshold.view(np.uint8).reshape(pop.n, 4),
                           pop.plasticity.view(np.uint8).reshape(pop.n, 2),
                           pop.post_hist.view(np.uint8).reshape(pop.n, -1),
                           pop.age.view(np.uint8).reshape(pop.n, 2)], axis=1)


def load_sets():
    d = load_day16()
    pops, unused = d["pops"], d["unused"]
    sets = {}
    for g, p in pops.items():
        sets[g] = (p.words.reshape(p.n, S).copy(), extras_bytes(p), unused[g].copy())
    cat = lambda gs: tuple(np.concatenate([sets[g][i] for g in gs]) for i in range(3))
    sets["cold"] = cat(["A", "B", "E"])
    sets["hot"] = cat(["C", "D"])
    return sets, pops


# ------------------------------------------------------------------ layouts (words block (n, 256) uint32)

def _f(w, name):
    off, bits = FIELDS[name]
    return (w >> np.uint32(off)) & np.uint32((1 << bits) - 1)


def delta_levels(w):
    c0, c1, c2 = _f(w, "c0"), _f(w, "c1"), _f(w, "c2")
    d1, d0 = (c1 - c2) & np.uint32(63), (c0 - c1) & np.uint32(63)
    keep = w & ~np.uint32(((1 << 12) - 1) << 10)
    return keep | (d0 << np.uint32(10)) | (d1 << np.uint32(16)) | (c2 << np.uint32(22))


def undelta_levels(w):
    d0, d1, c2 = _f(w, "c0"), _f(w, "c1"), _f(w, "c2")
    c1 = (c2 + d1) & np.uint32(63)
    c0 = (c1 + d0) & np.uint32(63)
    keep = w & ~np.uint32(((1 << 12) - 1) << 10)
    return keep | (c0 << np.uint32(10)) | (c1 << np.uint32(16)) | (c2 << np.uint32(22))


def to_fields(w):
    return np.stack([_f(w, k).astype(np.uint8) for k in ORDER]).tobytes()


def from_fields(buf, shape):
    p = np.frombuffer(buf, np.uint8).reshape(len(ORDER), *shape).astype(np.uint32)
    w = np.zeros(shape, np.uint32)
    for i, k in enumerate(ORDER):
        w |= p[i] << np.uint32(FIELDS[k][0])
    return w


def bitshuffle(w):
    b = np.unpackbits(w.reshape(-1, 1).view(np.uint8), axis=1, bitorder="little")  # (n, 32)
    return np.packbits(b.T, axis=1, bitorder="little").tobytes()


def unbitshuffle(buf, shape):
    n = int(np.prod(shape))
    planes = np.frombuffer(buf, np.uint8).reshape(32, n // 8)
    b = np.unpackbits(planes, axis=1, bitorder="little")
    return np.packbits(np.ascontiguousarray(b.T), axis=1, bitorder="little").view(np.uint32).reshape(shape)


def byteshuffle(w):
    return np.ascontiguousarray(w.view(np.uint8).reshape(-1, 4).T).tobytes()


def unbyteshuffle(buf, shape):
    return np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(4, -1).T).view(np.uint32).reshape(shape)


def to_symbols(w):
    key = (w >> np.uint32(10)) & np.uint32((1 << 18) - 1)
    uniq, inv = np.unique(key, return_inverse=True)
    idx = inv.astype(np.uint8 if len(uniq) <= 256 else np.uint16)
    rest = np.stack([_f(w, k).astype(np.uint8) for k in ("gen", "fast", "tag", "used")])
    head = np.array([len(uniq)], np.uint32)
    return head.tobytes() + uniq.astype(np.uint32).tobytes() + idx.tobytes() + rest.tobytes()


def from_symbols(buf, shape):
    n = int(np.prod(shape))
    u = int(np.frombuffer(buf[:4], np.uint32)[0])
    uniq = np.frombuffer(buf[4:4 + 4 * u], np.uint32)
    o = 4 + 4 * u
    dt = np.uint8 if u <= 256 else np.uint16
    isz = np.dtype(dt).itemsize
    idx = np.frombuffer(buf[o:o + n * isz], dt).reshape(shape)
    rest = np.frombuffer(buf[o + n * isz:], np.uint8).reshape(4, *shape).astype(np.uint32)
    w = uniq[idx] << np.uint32(10)
    for i, k in enumerate(("gen", "fast", "tag", "used")):
        w |= rest[i] << np.uint32(FIELDS[k][0])
    return w


def to_sortrows(w):
    perm = np.argsort(_f(w, "c2").astype(np.int32).sum(axis=1), kind="stable").astype(np.uint16)
    return perm.tobytes() + to_fields(delta_levels(w[perm]))


def from_sortrows(buf, shape):
    perm = np.frombuffer(buf[:2 * shape[0]], np.uint16)
    ws = undelta_levels(from_fields(buf[2 * shape[0]:], shape))
    w = np.empty_like(ws)
    w[perm] = ws
    return w


LAYOUTS = {
    "raw": (lambda w: w.tobytes(), lambda b, s: np.frombuffer(b, np.uint32).reshape(s)),
    "byteshuffle": (byteshuffle, unbyteshuffle),
    "bitshuffle": (bitshuffle, unbitshuffle),
    "fields": (to_fields, from_fields),
    "dfields": (lambda w: to_fields(delta_levels(w)), lambda b, s: undelta_levels(from_fields(b, s))),
    "dbitshuffle": (lambda w: bitshuffle(delta_levels(w)), lambda b, s: undelta_levels(unbitshuffle(b, s))),
    "symbols": (to_symbols, from_symbols),
    "sortrows": (to_sortrows, from_sortrows),
}


# ------------------------------------------------------------------ codecs

def make_codecs(dict_data=None):
    c = {}
    for lv in (-5, 1, 3, 9, 19):
        cc, dd = zstd.ZstdCompressor(level=lv), zstd.ZstdDecompressor()
        c[f"zstd{lv}"] = (cc.compress, dd.decompress)
    if dict_data is not None:
        for lv in (1, 3, 19):
            cc = zstd.ZstdCompressor(level=lv, dict_data=dict_data)
            dd = zstd.ZstdDecompressor(dict_data=dict_data)
            c[f"zstd{lv}+dict"] = (cc.compress, dd.decompress)
    c["lz4"] = (lambda b: lz4.block.compress(b, mode="default"), lz4.block.decompress)
    c["lz4hc"] = (lambda b: lz4.block.compress(b, mode="high_compression", compression=9), lz4.block.decompress)
    return c


def train_dict(words, layout, nb, size=16384, n_samples=2000, seed=0):
    enc = LAYOUTS[layout][0]
    r = np.random.default_rng(seed)
    starts = r.integers(0, len(words) - nb, n_samples)
    samples = [enc(words[s:s + nb]) for s in starts]
    return zstd.train_dictionary(size, samples, level=3)


# ------------------------------------------------------------------ measurement

def blocks_of(words, extras, nb):
    n = (len(words) // nb) * nb
    return [(words[i:i + nb], extras[i:i + nb]) for i in range(0, n, nb)]


def measure(words, extras, layout, codec, nb, timing_neurons=2048, repeats=3, check=True):
    """Compressed size over all blocks; compress / decompress speed (incl. layout) on a sample."""
    enc, dec = LAYOUTS[layout]
    comp, decomp = codec
    bl = blocks_of(words, extras, nb)
    total = 0
    for w, e in bl:
        total += len(comp(enc(w))) + len(comp(e.tobytes()))
    n = len(bl) * nb
    sample = bl[: max(1, timing_neurons // nb)]
    blobs = [(comp(enc(w)), comp(e.tobytes()), w.shape) for w, e in sample]
    if check:
        for (cw, ce, sh), (w, e) in zip(blobs, sample):
            assert np.array_equal(dec(decomp(cw), sh), w), layout
            assert decomp(ce) == e.tobytes()
    raw = sum(w.nbytes + e.nbytes for w, e in sample)
    tc, _ = median_time(lambda: [(comp(enc(w)), comp(e.tobytes())) for w, e in sample], repeats)
    td, _ = median_time(lambda: [(dec(decomp(cw), sh), decomp(ce)) for cw, ce, sh in blobs], repeats)
    raw_all = n * (S * 4 + extras.shape[1])
    return {"bytes_per_neuron": total / n, "ratio": raw_all / total,
            "comp_GBps": raw / tc / 1e9, "decomp_GBps": raw / td / 1e9}


def entropy_report(words):
    """Order-0 entropy of each field (bits per synapse), and of the (c0, c1, c2) triple."""
    out = {}
    for k in ORDER:
        v = _f(words, k).ravel()
        p = np.bincount(v) / v.size
        p = p[p > 0]
        out[k] = float(-(p * np.log2(p)).sum())
    key = ((words >> np.uint32(10)) & np.uint32((1 << 18) - 1)).ravel()
    _, cnt = np.unique(key, return_counts=True)
    p = cnt / cnt.sum()
    out["triple(c0,c1,c2)"] = float(-(p * np.log2(p)).sum())
    out["sum_of_fields"] = sum(out[k] for k in ORDER)
    out["joint_bound"] = out["triple(c0,c1,c2)"] + out["gen"] + out["fast"] + out["tag"] + out["used"]
    return out


def effective(cold_bpn, hot_bpn=1041.0, cold_frac=0.9, index_bytes=8, nb=64):
    return (1 - cold_frac) * hot_bpn + cold_frac * (cold_bpn + index_bytes / nb)


def main(quick=False):
    t0 = time.time()
    sets, _ = load_sets()
    cw, ce, cu = sets["cold"]
    print(f"cold set: {len(cw):,} neurons (nights unused: {np.unique(cu).tolist()}); hot set: {len(sets['hot'][0]):,}")
    res = {"entropy": {g: entropy_report(sets[g][0]) for g in ("A", "B", "E", "cold", "hot")}}
    for g, e in res["entropy"].items():
        print(f"  entropy {g:<5} " + "  ".join(f"{k}={v:.2f}" for k, v in e.items()))

    # 1) layouts x codecs, 64-neuron blocks, cold set
    codecs = make_codecs()
    nb = 64
    res["layout_codec"] = {}
    print(f"\n[layouts x codecs] cold set, {nb}-neuron blocks: bytes/neuron  ratio  comp GB/s  decomp GB/s")
    for lay in LAYOUTS:
        for cn, cc in codecs.items():
            if quick and cn in ("zstd9", "zstd19", "lz4hc"):
                continue
            r = measure(cw, ce, lay, cc, nb, repeats=1 if cn == "zstd19" else 3)
            res["layout_codec"][f"{lay}/{cn}"] = r
            print(f"  {lay:<12} {cn:<8} {r['bytes_per_neuron']:7.1f}  {r['ratio']:5.2f}x"
                  f"  {r['comp_GBps']:6.3f}  {r['decomp_GBps']:6.3f}")
    print(f"  ({time.time() - t0:.0f}s)")

    # 2) block size, with and without a dictionary. The 16 KB dictionary is trained on samples of the
    #    cold set itself (a brain-wide dictionary for cold blocks; 16 KB vs 12 MB of data).
    res["block_size"] = {}
    print("\n[block size] cold set")
    for lay in ("raw", "dfields", "symbols"):
        for nbk in (1, 4, 16, 64, 1024):
            d = train_dict(cw, lay, nbk) if nbk <= 16 else None
            cdc = make_codecs(d)
            for cn in ("zstd3", "zstd3+dict", "lz4"):
                if cn not in cdc:
                    continue
                r = measure(cw, ce, lay, cdc[cn], nbk, timing_neurons=1024)
                res["block_size"][f"{lay}/{nbk}/{cn}"] = r
                print(f"  {lay:<10} nb={nbk:<5} {cn:<10} {r['bytes_per_neuron']:7.1f} B/neuron  {r['ratio']:5.2f}x"
                      f"  comp {r['comp_GBps']:.3f}  decomp {r['decomp_GBps']:.3f} GB/s")
    print(f"  ({time.time() - t0:.0f}s)")

    # 3) per group (how cold / what kind of history), best layouts
    res["per_group"] = {}
    print("\n[per group] 64-neuron blocks")
    for g in ("A", "B", "E", "C", "D", "hot", "cold"):
        w, e, u = sets[g]
        for lay, cn in (("raw", "zstd3"), ("raw", "lz4"), ("dfields", "zstd3"), ("symbols", "zstd3"),
                        ("symbols", "zstd19"), ("bitshuffle", "zstd3")):
            r = measure(w, e, lay, codecs[cn], 64, timing_neurons=1024, repeats=1, check=False)
            res["per_group"][f"{g}/{lay}/{cn}"] = r
            print(f"  {g:<5} {lay:<10} {cn:<6} {r['bytes_per_neuron']:7.1f} B/neuron  {r['ratio']:5.2f}x")
    # effective bytes per neuron for a 90%-cold brain
    best = min(res["layout_codec"].items(), key=lambda kv: kv[1]["bytes_per_neuron"])
    lc = res["layout_codec"]
    res["effective_90pct_cold"] = {
        "best_ratio": [best[0], effective(best[1]["bytes_per_neuron"])],
        **{k: effective(lc[k]["bytes_per_neuron"]) for k in ("symbols/zstd3", "dfields/zstd3", "raw/zstd3",
                                                              "raw/lz4")},
        "uncompressed": 1041.0,
    }
    print("\n90%-cold brain, bytes per neuron:", res["effective_90pct_cold"])
    save_json("lossless.json", res)
    print(f"({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    main(quick="quick" in sys.argv)

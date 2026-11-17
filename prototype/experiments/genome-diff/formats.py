"""
Diff formats: how to store "what experience changed" against the procedural genome.

Input: an ANCHORED population's words (N, 256) uint32. A synapse is in the diff iff its word is
not PRISTINE. Each diff entry's value is the 32-bit word itself (only 25 bits are persistent:
gen, l0, l1, l2, tag; a 3-byte value would be 25% smaller still).

Formats (bytes counted exactly from the arrays, averaged over ALL neurons, dirty or not):
  dense        : the baseline layout, 1,024 B per neuron, no genome needed.
  block32      : 8-bit "dirty branch" mask per neuron + rank directory + whole 32-synapse blocks.
  block8       : 32-bit mask per neuron (8-synapse blocks) + rank directory + 8-word blocks.
  bitmap       : dirty-neuron bit + rank directory; per dirty neuron a 256-bit map + offset; values.
  csr          : per-neuron uint32 offsets (dense) + uint8 slot + uint32 value per entry.
  coo          : one sorted global list: uint32 key (neuron*256+slot) + uint32 value.
  hash         : open-addressing hash map, key uint32 + value uint32, load <= 0.7.
  zstd(csr)    : csr compressed with zstd level 3 (a cold / SSD form, not directly usable).
  zstd(dense)  : the dense words compressed directly (control: is the diff better than zstd?).

Access cost = materialize(neuron ids) -> dense rows (genome + overlay), and encode (rebuild
after sleep). Forward / modulate / sleep then run on dense rows (a per-day working-set cache).
"""

import time

import core  # noqa: F401  (first: sets thread caps before numba loads)
import numba as nb
import numpy as np
import zstandard


P = core.PRISTINE


# ------------------------------------------------------------------ encoders

def enc_csr(words):
    n = words.shape[0]
    m = words != P
    cnt = m.sum(axis=1)
    off = np.zeros(n + 1, np.uint32)
    np.cumsum(cnt, out=off[1:])
    flat = np.flatnonzero(m.ravel())
    slot = (flat & 255).astype(np.uint8)
    val = words.ravel()[flat]
    return dict(off=off, slot=slot, val=val)


def enc_coo(words):
    flat = np.flatnonzero(words.ravel() != P)
    return dict(key=flat.astype(np.uint32), val=words.ravel()[flat])


def _rank_dir(counts, group):
    """uint32 running count at the start of each group of `group` neurons."""
    n = counts.shape[0]
    g = (n + group - 1) // group
    pad = np.zeros(g * group, np.int64)
    pad[:n] = counts
    d = np.zeros(g, np.uint32)
    d[1:] = np.cumsum(pad.reshape(g, group).sum(axis=1))[:-1]
    return d


def enc_block(words, bs):
    n = words.shape[0]
    nb_ = core.NSYN // bs
    dirty = (words.reshape(n, nb_, bs) != P).any(axis=2)          # (n, nb)
    mask = np.packbits(dirty, axis=1, bitorder="little")          # (n, nb/8)
    mask = mask.view(np.uint8 if nb_ == 8 else np.uint32).reshape(n)
    cnt = dirty.sum(axis=1)
    rdir = _rank_dir(cnt, 16)
    blocks = words.reshape(n, nb_, bs)[dirty]                     # (n_dirty_blocks, bs)
    return dict(mask=mask, rdir=rdir, blocks=np.ascontiguousarray(blocks), bs=bs)


def enc_bitmap(words):
    n = words.shape[0]
    m = words != P
    ndirty = m.any(axis=1)
    nbits = np.packbits(ndirty, bitorder="little")
    pad = np.zeros(((n + 63) // 64) * 8, np.uint8)
    pad[:nbits.size] = nbits
    nbits64 = pad.view(np.uint64)
    rdir = _rank_dir(ndirty.astype(np.int64), 64)
    bm = np.packbits(m[ndirty], axis=1, bitorder="little").view(np.uint64)  # (n_dirty, 4)
    cnt = m[ndirty].sum(axis=1)
    off = np.zeros(int(ndirty.sum()) + 1, np.uint32)
    np.cumsum(cnt, out=off[1:])
    val = words[m]
    return dict(nbits=nbits64, rdir=rdir, bm=np.ascontiguousarray(bm), off=off, val=val)


@nb.njit(cache=True)
def _hash_build(keys, vals, cap):
    hk = np.zeros(cap, np.uint32)
    hv = np.zeros(cap, np.uint32)
    mask = cap - 1
    for e in range(keys.shape[0]):
        k = keys[e] + np.uint32(1)
        h = np.int64((np.uint64(k) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) & mask
        while hk[h] != 0:
            h = (h + 1) & mask
        hk[h] = k
        hv[h] = vals[e]
    return hk, hv


def enc_hash(words, load=0.7):
    c = enc_coo(words)
    cap = 1
    while cap * load < max(c["key"].size, 1):
        cap *= 2
    hk, hv = _hash_build(c["key"], c["val"], cap)
    return dict(hk=hk, hv=hv)


def nbytes(d):
    return sum(v.nbytes for v in d.values() if isinstance(v, np.ndarray))


# ------------------------------------------------------------------ materialize (decode rows)

@nb.njit(parallel=True, cache=True)
def mat_dense(words, ids, out):
    for k in nb.prange(ids.shape[0]):
        out[k, :] = words[ids[k], :]


@nb.njit(parallel=True, cache=True)
def mat_csr(off, slot, val, ids, out):
    for k in nb.prange(ids.shape[0]):
        i = ids[k]
        out[k, :] = P
        for e in range(off[i], off[i + 1]):
            out[k, slot[e]] = val[e]


@nb.njit(parallel=True, cache=True)
def mat_coo(key, val, ids, out):
    n = key.shape[0]
    for k in nb.prange(ids.shape[0]):
        i = ids[k]
        out[k, :] = P
        lo_key = np.uint32(i * 256)
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) >> 1
            if key[mid] < lo_key:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        while e < n and key[e] < lo_key + np.uint32(256):
            out[k, key[e] - lo_key] = val[e]
            e += 1


@nb.njit(inline="always", cache=True)
def _popc8(x):
    x = np.uint64(x)
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@nb.njit(parallel=True, cache=True)
def mat_block(mask, rdir, blocks, bs, ids, out):
    nb_ = 256 // bs
    for k in nb.prange(ids.shape[0]):
        i = ids[k]
        r = np.int64(rdir[i // 16])
        for i2 in range((i // 16) * 16, i):
            r += _popc8(np.uint64(mask[i2]))
        m = np.uint64(mask[i])
        for b in range(nb_):
            if (m >> np.uint64(b)) & np.uint64(1):
                out[k, b * bs:(b + 1) * bs] = blocks[r, :]
                r += 1
            else:
                out[k, b * bs:(b + 1) * bs] = P


@nb.njit(parallel=True, cache=True)
def mat_bitmap(nbits, rdir, bm, off, val, ids, out):
    for k in nb.prange(ids.shape[0]):
        i = ids[k]
        out[k, :] = P
        word = nbits[i >> 6]
        bit = np.uint64(1) << np.uint64(i & 63)
        if not (word & bit):
            continue
        r = np.int64(rdir[i >> 6]) + _popc8(word & (bit - np.uint64(1)))
        e = off[r]
        for q in range(4):
            x = bm[r, q]
            if x == 0:
                continue
            for b in range(64):
                if (x >> np.uint64(b)) & np.uint64(1):
                    out[k, q * 64 + b] = val[e]
                    e += 1


@nb.njit(parallel=True, cache=True)
def mat_hash(hk, hv, ids, out):
    mask = hk.shape[0] - 1
    for k in nb.prange(ids.shape[0]):
        i = ids[k]
        for j in range(256):
            kk = np.uint32(i * 256 + j + 1)
            h = np.int64((np.uint64(kk) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) & mask
            v = P
            while hk[h] != 0:
                if hk[h] == kk:
                    v = hv[h]
                    break
                h = (h + 1) & mask
            out[k, j] = v


ENCODERS = {
    "csr": enc_csr, "coo": enc_coo, "block32": lambda w: enc_block(w, 32), "block8": lambda w: enc_block(w, 8),
    "bitmap": enc_bitmap, "hash": enc_hash,
}


def materialize(name, enc, words, ids, out):
    if name == "dense":
        mat_dense(words, ids, out)
    elif name == "csr":
        mat_csr(enc["off"], enc["slot"], enc["val"], ids, out)
    elif name == "coo":
        mat_coo(enc["key"], enc["val"], ids, out)
    elif name.startswith("block"):
        mat_block(enc["mask"], enc["rdir"], enc["blocks"], enc["bs"], ids, out)
    elif name == "bitmap":
        mat_bitmap(enc["nbits"], enc["rdir"], enc["bm"], enc["off"], enc["val"], ids, out)
    elif name == "hash":
        mat_hash(enc["hk"], enc["hv"], ids, out)


def med(f, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def analyze(words, frac_active=0.25, seed=0, verify=True, timing=True):
    """Bytes per neuron and access cost for every format, for one snapshot."""
    n = words.shape[0]
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(n, max(1, int(n * frac_active)), replace=False)).astype(np.int64)
    out = np.empty((ids.size, core.NSYN), np.uint32)
    ref = words[ids]
    rows = {}
    n_diff = int((words != P).sum())
    rows["dense"] = dict(bytes=words.nbytes / n, enc_ns=0.0)
    if timing:
        rows["dense"]["mat_ns"] = med(lambda: mat_dense(words, ids, out)) / ids.size * 1e9
    for name, fn in ENCODERS.items():
        t0 = time.perf_counter()
        enc = fn(words)
        t_enc = time.perf_counter() - t0
        r = dict(bytes=nbytes(enc) / n, enc_ns=t_enc / n * 1e9)
        if timing:
            materialize(name, enc, words, ids, out)  # warm-up / compile
            r["mat_ns"] = med(lambda: materialize(name, enc, words, ids, out)) / ids.size * 1e9
        if verify:
            materialize(name, enc, words, ids, out)
            assert np.array_equal(out, ref), name
        rows[name] = r
        if name == "csr":
            raw = enc["off"].tobytes() + enc["slot"].tobytes() + enc["val"].tobytes()
            cctx = zstandard.ZstdCompressor(level=3)
            t0 = time.perf_counter()
            z = cctx.compress(raw)
            t_c = time.perf_counter() - t0
            dctx = zstandard.ZstdDecompressor()
            t_d = med(lambda: dctx.decompress(z), 3)
            rows["zstd(csr)"] = dict(bytes=len(z) / n, enc_ns=(t_enc + t_c) / n * 1e9,
                                     mat_ns=t_d / n * 1e9, note="decompress-all / neuron")
    cctx = zstandard.ZstdCompressor(level=3)
    t0 = time.perf_counter()
    z = cctx.compress(words.tobytes())
    t_c = time.perf_counter() - t0
    dctx = zstandard.ZstdDecompressor()
    t_d = med(lambda: dctx.decompress(z), 3)
    rows["zstd(dense)"] = dict(bytes=len(z) / n, enc_ns=t_c / n * 1e9, mat_ns=t_d / n * 1e9,
                               note="decompress-all / neuron")
    meta = dict(n=n, diff_frac=n_diff / words.size, dirty_neurons=float((words != P).any(axis=1).mean()),
                per_dirty=n_diff / max(1, int((words != P).any(axis=1).sum())))
    return rows, meta


def synthetic(n, neuron_frac, syn_per_dirty, seed=0, clustered=True, pool=None):
    """A diff with a chosen fraction of dirty neurons and synapses per dirty neuron.
    clustered: dirty synapses bunch on a few branches (as learning on active inputs tends to)."""
    rng = np.random.default_rng(seed)
    w = np.full((n, core.NSYN), P, np.uint32)
    dn = rng.choice(n, int(n * neuron_frac), replace=False)
    k = int(syn_per_dirty)
    for i in dn:
        if clustered:
            nbr = max(1, min(8, int(np.ceil(k / 24))))
            br = rng.choice(8, nbr, replace=False)
            cand = (br[:, None] * 32 + np.arange(32)[None, :]).ravel()
            js = rng.choice(cand, min(k, cand.size), replace=False)
        else:
            js = rng.choice(256, k, replace=False)
        if pool is None:
            w[i, js] = P ^ np.uint32(1 << 28)  # a tag-only change, enough to differ
        else:  # values drawn from a real snapshot's diff entries
            w[i, js] = pool[rng.integers(0, pool.size, js.size)]
    return w

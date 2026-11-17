"""
Dormancy and recall.

A neuron (or a whole column) that has been unused for many nights is compressed with zstd,
written to a dormant archive on the SSD and dropped from RAM. Its nights are skipped: they pile
up as one 3x3 matrix. When something needs it again it is RECALLED: read back (cold, O_DIRECT),
decompressed, and all skipped nights are applied at once, then it can fire.

1. Compression: how small do aged neurons get? (a 64-column population lives 10 days first,
   so the words look like real, used neurons: varied cascades, tags, rewired generations)
2. Recall latency, cold from the SSD: one neuron, one column (1,024 neurons), a batch of columns;
   raw (uncompressed, straight from the brain file) vs per-column zstd frames vs per-neuron
   zstd frames with a trained dictionary.
"""

import os
import time

from common import COL, DATA, GiB, KiB, LEDGER, MiB, NEURON_BYTES, aio, aligned, med, open_direct, save

import numba
import numpy as np
import zstandard as zstd

try:
    import lz4.frame as lz4f
except ImportError:  # pragma: no cover
    lz4f = None

from brain import BRAIN, n_neurons
from kernels import deferred_night, k_forward, k_init, k_sleep, night_matrices
from neuron import WORKHORSE
from wake import COL_SHIFT, R_IN, Waker

F32 = np.float32
SKIPPED = 30           # nights a dormant neuron has slept through when it is recalled
REPEAT = 200           # latency samples


def live(ncols=64, days=10, steps=200, frac=0.05, seed=5):
    """A small population lives `days` days (sparse columnar activity, learning, nights)."""
    n = ncols * COL
    t = WORKHORSE
    rng = np.random.default_rng(seed)
    words = np.empty((n, 256), np.uint32)
    k_init(words, 0, np.uint64(seed))
    w = Waker(n, seed=seed)
    Mpre, Ms, keep0 = night_matrices(t, steps)
    for day in range(days):
        for s in range(steps):
            w.X[:] = rng.random(R_IN) < 0.03
            w.pre_hist = np.roll(w.pre_hist, 1, axis=0)
            w.pre_hist[0] = w.X
            cols = rng.choice(ncols, max(1, int(4 * frac * ncols)), replace=False)
            ids = np.sort((cols[:, None] * COL + np.arange(COL))[rng.random((len(cols), COL)) < 0.25])
            ids = ids.astype(np.int64)
            w.compute(words, ids, ids, modulate=True)
        k_sleep(words, np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), Mpre,
                F32(t.fast_decay ** steps), Ms, keep0, F32(t.prune_eps), F32(t.tag_decay), F32(t.w_max),
                np.uint64(1000 + day), True, np.zeros(n, np.int64))
        w.post_hist[:] = 0
    return words


def shuffle(b):
    """Byte-shuffle (all byte-0s, then all byte-1s, ...): groups similar bit fields together."""
    return np.frombuffer(b, np.uint8).reshape(-1, 4).T.copy().tobytes()


def compression_table(aged, fresh):
    rows = {}

    def ratio_speed(label, frames_raw, comp, decomp):
        t0 = time.perf_counter()
        cf = [comp(f) for f in frames_raw]
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        for f in cf:
            decomp(f)
        td = time.perf_counter() - t0
        raw = sum(len(f) for f in frames_raw)
        c = sum(len(f) for f in cf)
        rows[label] = {"ratio": raw / c, "bytes_per_neuron": c / (raw / NEURON_BYTES),
                       "compress_MBps": raw / tc / 1e6, "decompress_MBps": raw / td / 1e6}
        print(f"  {label:<46} ratio {raw / c:5.2f}x  {c / (raw / NEURON_BYTES):7.0f} B/neuron   "
              f"compress {raw / tc / 1e6:7.0f} MB/s  decompress {raw / td / 1e6:7.0f} MB/s (1 thread)")
        return cf

    cols = [aged[i * COL:(i + 1) * COL].tobytes() for i in range(aged.shape[0] // COL)]
    fcols = [fresh[i * COL:(i + 1) * COL].tobytes() for i in range(fresh.shape[0] // COL)]
    print("== Dormancy: compression of packed neurons ==")
    ratio_speed("newborn neurons, column frames, zstd-3", fcols, zstd.ZstdCompressor(level=3).compress,
                zstd.ZstdDecompressor().decompress)
    for lvl in (1, 3, 9, 19):
        ratio_speed(f"aged 10 days, column frames, zstd-{lvl}", cols, zstd.ZstdCompressor(level=lvl).compress,
                    zstd.ZstdDecompressor().decompress)
    ratio_speed("aged, column frames, byte-shuffled + zstd-3", cols,
                lambda b: zstd.ZstdCompressor(level=3).compress(shuffle(b)), zstd.ZstdDecompressor().decompress)
    if lz4f:
        ratio_speed("aged, column frames, lz4", cols, lz4f.compress, lz4f.decompress)
    neurons = [aged[i].tobytes() for i in range(0, aged.shape[0], 8)]
    ratio_speed("aged, one frame per neuron, zstd-3 (no dict)", neurons, zstd.ZstdCompressor(level=3).compress,
                zstd.ZstdDecompressor().decompress)
    d = zstd.train_dictionary(16 * KiB, [aged[i].tobytes() for i in range(1, aged.shape[0], 7)][:8000])
    ratio_speed("aged, one frame per neuron, zstd-3 + 16 KiB dict", neurons,
                zstd.ZstdCompressor(level=3, dict_data=d).compress,
                zstd.ZstdDecompressor(dict_data=d).decompress)
    return rows, d


class Archive:
    """Append-only file of compressed frames + an in-RAM index (offset, length)."""

    def __init__(self, path, frames):
        self.path = path
        blob = b"".join(frames)
        lens = np.array([len(f) for f in frames], np.int64)
        self.off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.len = lens
        # pad to 4 KiB, plus one max frame span, so fixed-size batch reads never run past the end
        pad = (-len(blob)) % 4096 + (int(lens.max()) // 4096 + 2) * 4096
        with open(path, "wb") as f:
            f.write(blob + b"\0" * pad)
            f.flush()
            os.fsync(f.fileno())
        LEDGER.add(len(blob) + pad)
        self.fd = open_direct(path)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self.buf = aligned(int(lens.max()) + 8192)

    def read(self, i):
        a = self.off[i] // 4096 * 4096
        b = -(-(self.off[i] + self.len[i]) // 4096) * 4096
        os.preadv(self.fd, [memoryview(self.buf)[:b - a]], int(a))
        s = self.off[i] - a
        return bytes(self.buf[s:s + self.len[i]])

    def close(self):
        os.close(self.fd)
        self.path.unlink()


def main():
    n_brain = n_neurons()
    t = WORKHORSE
    res = {"skipped_nights": SKIPPED}
    t0 = time.perf_counter()
    aged = live()
    res["live_s"] = time.perf_counter() - t0
    fresh = np.empty((64 * COL, 256), np.uint32)
    k_init(fresh, 0, np.uint64(3))
    res["compression"], zdict = compression_table(aged, fresh)

    # skipped nights composed into one 3x3 map
    M = np.eye(3)
    t0 = time.perf_counter()
    for _ in range(SKIPPED):
        M = deferred_night(t, M, 200)
    res["compose_us"] = (time.perf_counter() - t0) * 1e6
    M = M.astype(F32)
    eye = np.eye(3, dtype=F32)

    w = Waker(n_brain)
    X = w.X

    def catch_up_and_fire(words, ids):
        rows = np.arange(len(ids), dtype=np.int64)
        k_sleep(words, rows, ids, M, F32(0), eye, F32(1), F32(t.prune_eps), F32(t.tag_decay ** SKIPPED),
                F32(t.w_max), np.uint64(9), True, np.zeros(len(ids), np.int64))
        rate = np.empty(len(ids), F32)
        k_forward(words, rows, ids, 0, w.gate, w.thresh, w.post_hist, X, R_IN, COL_SHIFT, w.F0, F32(1),
                  F32(1), F32(t.branch_threshold), F32(t.spike_bonus), F32(1), F32(t.soma_slope),
                  F32(t.threshold_lr), F32(t.target_rate), False, rate)
        return rate

    # archives: per-column frames (2,048 columns = 2 GiB raw) and per-neuron dict frames (1 M neurons)
    zc = zstd.ZstdCompressor(level=3)
    col_frames = [zc.compress(aged[i * COL:(i + 1) * COL].tobytes()) for i in range(64)]
    colA = Archive(DATA / "dormant_columns.zst", [col_frames[i % 64] for i in range(2048)])
    zn = zstd.ZstdCompressor(level=3, dict_data=zdict)
    neu_frames = [zn.compress(aged[i].tobytes()) for i in range(aged.shape[0])]
    neuA = Archive(DATA / "dormant_neurons.zst", [neu_frames[i % len(neu_frames)] for i in range(1 << 20)])
    dz = zstd.ZstdDecompressor()
    dzn = zstd.ZstdDecompressor(dict_data=zdict)
    bfd = open_direct(str(BRAIN))
    rng = np.random.default_rng(21)
    numba.set_num_threads(3)

    # warm up kernels on tiny inputs
    tmp = np.empty((COL, 256), np.uint32)
    k_init(tmp, 0, np.uint64(1))
    catch_up_and_fire(tmp[:1], np.zeros(1, np.int64))
    catch_up_and_fire(tmp, np.arange(COL, dtype=np.int64))

    lat = {}
    page = aligned(4096, np.uint32)
    colbuf = aligned(MiB, np.uint32)
    small_span = int(neuA.off[-1] // 4096)      # same size as the per-neuron archive (~0.35 GB)

    def measure(label, fn):
        """fn() returns (read_s, decompress_s, catchup_fire_s) for one recall."""
        parts = np.array([fn() for _ in range(REPEAT)]) * 1e6
        tot = parts.sum(axis=1)
        lat[label] = {"p50_us": float(np.median(tot)), "p99_us": float(np.percentile(tot, 99)),
                      "read_p50_us": float(np.median(parts[:, 0])), "decompress_p50_us": float(np.median(parts[:, 1])),
                      "catchup_fire_p50_us": float(np.median(parts[:, 2]))}
        print(f"  {label:<58} p50 {np.median(tot):>7,.0f} us  p99 {np.percentile(tot, 99):>7,.0f} us   "
              f"[read {np.median(parts[:, 0]):>6,.0f} + unzip {np.median(parts[:, 1]):>5,.0f} + "
              f"catch-up&fire {np.median(parts[:, 2]):>6,.0f}]")

    def timed(f, *a):
        t0 = time.perf_counter()
        out = f(*a)
        return out, time.perf_counter() - t0

    print(f"== Dormancy: recall latency, cold from the SSD (O_DIRECT), including {SKIPPED} skipped nights "
          f"applied on recall, then one forward pass ==")

    def ram_fire_only():
        i = int(rng.integers(COL))
        rate = np.empty(1, F32)
        _, tf = timed(k_forward, tmp, np.array([i], np.int64), np.array([i], np.int64), 0, w.gate, w.thresh,
                      w.post_hist, X, R_IN, COL_SHIFT, w.F0, F32(1), F32(1), F32(t.branch_threshold),
                      F32(t.spike_bonus), F32(1), F32(t.soma_slope), F32(t.threshold_lr), F32(t.target_rate),
                      False, rate)
        return 0.0, 0.0, tf

    def ram_neuron():
        i = int(rng.integers(COL))
        _, tf = timed(catch_up_and_fire, tmp[i:i + 1], np.array([i], np.int64))
        return 0.0, 0.0, tf

    def one_raw(span_pages):
        def fn():
            i = int(rng.integers(span_pages)) * 4 + int(rng.integers(4))
            _, tr = timed(os.preadv, bfd, [page], i * NEURON_BYTES // 4096 * 4096)
            wds = page.reshape(4, 256)[(i % 4):(i % 4) + 1].copy()
            _, tf = timed(catch_up_and_fire, wds, np.array([i], np.int64))
            return tr, 0.0, tf
        return fn

    def one_neuron_frame():
        j = int(rng.integers(1 << 20))
        blob, tr = timed(neuA.read, j)
        raw, td = timed(dzn.decompress, blob)
        wds = np.frombuffer(raw, np.uint32).reshape(1, 256).copy()
        _, tf = timed(catch_up_and_fire, wds, np.array([j], np.int64))
        return tr, td, tf

    def one_from_column_frame():
        c = int(rng.integers(2048))
        blob, tr = timed(colA.read, c)
        raw, td = timed(dz.decompress, blob)
        k = int(rng.integers(COL))
        wds = np.frombuffer(raw, np.uint32).reshape(COL, 256)[k:k + 1].copy()
        _, tf = timed(catch_up_and_fire, wds, np.array([c * COL + k], np.int64))
        return tr, td, tf

    def col_raw():
        c = int(rng.integers(n_brain // COL))
        _, tr = timed(os.preadv, bfd, [colbuf], c * COL * NEURON_BYTES)
        _, tf = timed(catch_up_and_fire, colbuf.reshape(COL, 256), c * COL + np.arange(COL, dtype=np.int64))
        return tr, 0.0, tf

    def col_frame():
        c = int(rng.integers(2048))
        blob, tr = timed(colA.read, c)
        raw, td = timed(dz.decompress, blob)
        wds = np.frombuffer(raw, np.uint32).reshape(COL, 256).copy()
        _, tf = timed(catch_up_and_fire, wds, c * COL + np.arange(COL, dtype=np.int64))
        return tr, td, tf

    measure("awake neuron in RAM: fire only", ram_fire_only)
    measure("neuron in RAM: catch up 30 nights + fire", ram_neuron)
    measure("1 dormant neuron, raw, anywhere in the 24 GiB brain", one_raw(n_brain // 4))
    measure("1 dormant neuron, raw, within a 0.35 GB span", one_raw(small_span))
    measure("1 dormant neuron, own zstd+dict frame (0.35 GB archive)", one_neuron_frame)
    measure("1 dormant neuron, via its column's zstd frame", one_from_column_frame)
    measure("1 dormant column (1,024 neurons), raw (1 MiB read)", col_raw)
    measure("1 dormant column, zstd frame (~0.3 MiB read)", col_frame)

    # batch recall: 256 random columns (256 MiB of neurons)
    print("== Dormancy: batch recall of 256 random columns (256 MiB raw) ==")
    nb = 256
    batch = {}
    big = aligned(nb * MiB, np.uint32).reshape(nb * COL, 256)
    for rep in range(3):
        cols = np.sort(rng.choice(n_brain // COL, nb, replace=False)).astype(np.int64)
        ids = (cols[:, None] * COL + np.arange(COL)).ravel()
        t0 = time.perf_counter()
        aio(bfd, False, cols * MiB, MiB, big, np.arange(nb, dtype=np.int64) * MiB, qd=16)
        t1 = time.perf_counter()
        catch_up_and_fire(big, ids)
        t2 = time.perf_counter()
        batch.setdefault("raw_read_s", []).append(t1 - t0)
        batch.setdefault("raw_total_s", []).append(t2 - t0)
        # compressed: AIO read of frame spans (fixed max size), parallel decompress, catch up
        cs = np.sort(rng.choice(2048, nb, replace=False))
        a = colA.off[cs] // 4096 * 4096
        span = int((-(-(colA.off[cs] + colA.len[cs]) // 4096) * 4096 - a).max())
        fb = aligned(nb * span)
        t0 = time.perf_counter()
        aio(colA.fd, False, a, span, fb, np.arange(nb, dtype=np.int64) * span, qd=16)
        t1 = time.perf_counter()
        frames = [bytes(fb[k * span + (colA.off[c] - a[k]): k * span + (colA.off[c] - a[k]) + colA.len[c]])
                  for k, c in enumerate(cs)]
        out = dz.multi_decompress_to_buffer(frames, decompressed_sizes=np.full(nb, MiB, np.uint64).tobytes(),
                                            threads=3)
        words = np.empty((nb * COL, 256), np.uint32)
        for k in range(nb):
            words[k * COL:(k + 1) * COL] = np.frombuffer(out[k], np.uint32).reshape(COL, 256)
        t2 = time.perf_counter()
        catch_up_and_fire(words, (cs[:, None] * COL + np.arange(COL)).ravel().astype(np.int64))
        t3 = time.perf_counter()
        batch.setdefault("zstd_read_s", []).append(t1 - t0)
        batch.setdefault("zstd_decompress_s", []).append(t2 - t1)
        batch.setdefault("zstd_total_s", []).append(t3 - t0)
        batch.setdefault("catchup_fire_s", []).append(t3 - t2)
        del fb, words, out
    batch = {k: med(v) for k, v in batch.items()}
    batch["raw_GBps_recalled"] = nb * MiB / batch["raw_total_s"] / 1e9
    batch["zstd_GBps_recalled"] = nb * MiB / batch["zstd_total_s"] / 1e9
    print(f"  raw:  read {batch['raw_read_s'] * 1e3:.0f} ms + catch-up&fire = {batch['raw_total_s'] * 1e3:.0f} ms "
          f"({batch['raw_GBps_recalled']:.2f} GB/s of neurons recalled, "
          f"{nb * COL / batch['raw_total_s'] / 1e6:.2f} M neurons/s)")
    print(f"  zstd: read {batch['zstd_read_s'] * 1e3:.0f} ms + decompress {batch['zstd_decompress_s'] * 1e3:.0f} ms "
          f"+ catch-up&fire {batch['catchup_fire_s'] * 1e3:.0f} ms = {batch['zstd_total_s'] * 1e3:.0f} ms "
          f"({batch['zstd_GBps_recalled']:.2f} GB/s recalled)")
    res["latency"] = lat
    res["batch_256_columns"] = batch
    res["archive_bytes_per_neuron_columns"] = float(colA.len.mean() / COL)
    res["archive_bytes_per_neuron_neurons"] = float(neuA.len.mean())
    colA.close()
    neuA.close()
    os.close(bfd)
    save("dormant", res)
    return res


if __name__ == "__main__":
    main()

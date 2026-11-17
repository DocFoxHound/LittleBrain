"""
Sleep as a streaming sweep over the on-disk brain.

The night is per-synapse maintenance (consolidate tagged fast weights, cascade flow, downscale,
prune & regrow, clear, fade tags): every synapse is read, changed and written back, which is
exactly what an SSD likes if it is done in big sequential chunks.

Pipeline (one night):
    reader thread  --(read-ahead, O_DIRECT AIO, 16 MiB requests)-->  ring of 4 x 128 MiB buffers
    main thread    --  Numba k_sleep on the chunk (3 threads)
    writer thread  --(write-behind, O_DIRECT AIO, same offsets)-->  back into the brain file

Variants measured: serial (read, compute, write one after another), pipelined (all three
overlap), compute only (RAM), I/O only (read + write, no compute), and the LAZY night, which
sweeps only the columns touched today; untouched (dormant) neurons skip the night entirely and
catch up on recall (one 3x3 matrix for any number of skipped nights; behaviour checked in
behavior.py).
"""

import os
import queue
import sys
import threading
import time

from common import COL, GiB, LEDGER, MiB, NEURON_BYTES, Timer, aio, aligned, med, peak_rss_mib, save

import numba
import numpy as np

from brain import BRAIN, n_neurons, open_brain
from kernels import k_checksum, k_sleep, night_matrices
from neuron import WORKHORSE

CHUNK = 128 * MiB
NBUF = 4
PENDING = 1000       # awake steps since the last cascade catch-up
FAST_SCALE = 0.5     # fast-weight fade accumulated over the day


class Sweeper:
    def __init__(self, t=WORKHORSE, chunk=CHUNK, nbuf=NBUF):
        self.t = t
        self.chunk = chunk
        self.bufs = [aligned(chunk, np.uint32).reshape(-1, 256) for _ in range(nbuf)]
        self.rows = np.arange(chunk // NEURON_BYTES, dtype=np.int64)
        self.pruned = np.zeros(chunk // NEURON_BYTES, np.int64)
        self.Mpre, self.Ms, self.keep0 = night_matrices(t, PENDING)
        self.gpu = None

    def sleep_chunk(self, words, ids, seed):
        t = self.t
        n = len(ids)
        if self.gpu is not None:            # ids are consecutive in a full sweep
            self.gpu.run(words, int(ids[0]), seed)
            return 0
        k_sleep(words, self.rows[:n], ids, self.Mpre, np.float32(FAST_SCALE), self.Ms, self.keep0,
                np.float32(t.prune_eps), np.float32(t.tag_decay), np.float32(t.w_max), seed, True,
                self.pruned[:n])
        return int(self.pruned[:n].sum())

    def run(self, fd, offs, req, mode="pipelined", seed=np.uint64(2026), compute=True, write=True,
            qd=8):
        """Sweep the requests at byte offsets `offs` (each `req` bytes) through the night kernel."""
        offs = np.asarray(offs, np.int64)
        per = self.chunk // req
        groups = [offs[i:i + per] for i in range(0, len(offs), per)]
        npr = req // NEURON_BYTES
        stats = {"read_s": 0.0, "compute_s": 0.0, "write_s": 0.0, "pruned": 0}

        def ids_of(g):
            return ((g // NEURON_BYTES)[:, None] + np.arange(npr)).ravel()

        def do_read(b, g):
            t0 = time.perf_counter()
            aio(fd, False, g, req, self.bufs[b], np.arange(len(g), dtype=np.int64) * req, qd=qd)
            stats["read_s"] += time.perf_counter() - t0

        def do_write(b, g):
            t0 = time.perf_counter()
            aio(fd, True, g, req, self.bufs[b], np.arange(len(g), dtype=np.int64) * req, qd=qd)
            stats["write_s"] += time.perf_counter() - t0

        def do_compute(b, g):
            if compute:
                t0 = time.perf_counter()
                ids = ids_of(g)
                stats["pruned"] += self.sleep_chunk(self.bufs[b][:len(ids)], ids, seed)
                stats["compute_s"] += time.perf_counter() - t0

        with Timer() as tm:
            if mode == "serial":
                for g in groups:
                    do_read(0, g)
                    do_compute(0, g)
                    if write:
                        do_write(0, g)
            else:
                free, ready, done = queue.Queue(), queue.Queue(), queue.Queue()
                for b in range(len(self.bufs)):
                    free.put(b)

                def reader():
                    for g in groups:
                        b = free.get()
                        do_read(b, g)
                        ready.put((b, g))
                    ready.put(None)

                def writer():
                    while (item := done.get()) is not None:
                        if write:
                            do_write(*item)
                        free.put(item[0])

                th = [threading.Thread(target=reader), threading.Thread(target=writer)]
                for x in th:
                    x.start()
                while (item := ready.get()) is not None:
                    do_compute(*item)
                    done.put(item)
                done.put(None)
                for x in th:
                    x.join()
            if write:
                os.fsync(fd)
        nbytes = len(offs) * req
        stats.update({"wall_s": tm.dt, "cores": tm.cores, "bytes": nbytes, "neurons": nbytes // NEURON_BYTES,
                      "GBps_through": nbytes / tm.dt / 1e9,
                      "neurons_per_s": nbytes / NEURON_BYTES / tm.dt})
        return stats


def fmt(label, s):
    syn = s["neurons"] * 256
    print(f"  {label:<44} {s['bytes'] / GiB:>6.1f} GiB {s['wall_s']:>7.1f} s  {s['GBps_through']:>5.2f} GB/s "
          f"(read+write {2 * s['GBps_through'] if s['write_s'] else s['GBps_through']:.2f})  "
          f"{s['neurons_per_s'] / 1e6:>5.2f} M neurons/s  {syn / s['wall_s'] / 1e9:>5.2f} G syn/s  "
          f"busy: read {s['read_s']:.1f} / compute {s['compute_s']:.1f} / write {s['write_s']:.1f} s, "
          f"cores {s['cores']:.1f}", flush=True)


def verify(fd, before_words, sample_ids, sw, seed):
    """Night on the SSD == night computed in RAM on the same neurons (bit-exact)."""
    expect = before_words.copy()
    k_sleep(expect, np.arange(len(sample_ids), dtype=np.int64), sample_ids, sw.Mpre, np.float32(FAST_SCALE),
            sw.Ms, sw.keep0, np.float32(sw.t.prune_eps), np.float32(sw.t.tag_decay), np.float32(sw.t.w_max),
            seed, True, np.zeros(len(sample_ids), np.int64))
    got = read_neurons(fd, sample_ids)
    return bool(np.array_equal(expect, got))


def read_neurons(fd, ids):
    """Read individual neurons (their 4 KiB pages) with O_DIRECT."""
    pages = ids * NEURON_BYTES // 4096 * 4096
    buf = aligned(len(ids) * 4096, np.uint32)
    aio(fd, False, pages, 4096, buf, np.arange(len(ids), dtype=np.int64) * 4096, qd=64)
    within = (ids * NEURON_BYTES % 4096) // NEURON_BYTES
    return buf.reshape(len(ids), 4, 256)[np.arange(len(ids)), within].copy()


def main(sub_gib=4):
    n = n_neurons()
    fd = open_brain(write=True)
    sw = Sweeper()
    rng = np.random.default_rng(11)
    res = {"neurons": n, "chunk_MiB": CHUNK // MiB, "nbuf": NBUF}
    print(f"== Night: streaming sleep sweep ({n:,} neurons, {n * NEURON_BYTES / GiB:.0f} GiB on SSD, "
          f"{CHUNK // MiB} MiB chunks x {NBUF} buffers = {CHUNK * NBUF // MiB} MiB RAM) ==")

    # compile + compute-only ceiling (RAM, 3 threads and 1 thread)
    ids0 = np.arange(CHUNK // NEURON_BYTES, dtype=np.int64)
    os.preadv(fd, [sw.bufs[0]], 0)
    sw.sleep_chunk(sw.bufs[0][:1024], ids0[:1024], np.uint64(1))
    for th in (1, 2, 3):
        numba.set_num_threads(th)
        ts = []
        for r in range(5):
            os.preadv(fd, [sw.bufs[0]], 0)          # fresh copy each time
            t0 = time.perf_counter()
            sw.sleep_chunk(sw.bufs[0], ids0, np.uint64(1))
            ts.append(time.perf_counter() - t0)
        rate = len(ids0) / med(ts)
        res[f"compute_only_neurons_per_s_{th}t"] = rate
        print(f"  compute only (RAM, {th} thread{'s' if th > 1 else ''}): {rate / 1e6:.2f} M neurons/s = "
              f"{rate * 256 / 1e9:.2f} G synapse-updates/s = {rate * NEURON_BYTES / 1e9:.2f} GB/s of brain "
              f"-> {n / rate:.0f} s per night for {n * NEURON_BYTES / GiB:.0f} GiB")
    numba.set_num_threads(3)

    # the same night on the GPU: bit-identical? how fast (including PCIe copies both ways)?
    gpu = None
    try:
        from gpu_night import GPUNight
        gpu = GPUNight(CHUNK, sw.Mpre, sw.Ms, sw.keep0, FAST_SCALE, sw.t.prune_eps, sw.t.tag_decay,
                       sw.t.w_max, host_bufs=sw.bufs)
        os.preadv(fd, [sw.bufs[0]], 0)
        ref = sw.bufs[0].copy()
        sw.sleep_chunk(sw.bufs[0], ids0, np.uint64(1))
        cpu_out = sw.bufs[0].copy()
        sw.bufs[0][:] = ref
        gpu.run(sw.bufs[0], 0, np.uint64(1))
        same = float((sw.bufs[0] == cpu_out).mean())
        res["gpu_cpu_identical_word_fraction"] = same
        ts = []
        for r in range(3):
            sw.bufs[0][:] = ref
            t0 = time.perf_counter()
            gpu.run(sw.bufs[0], 0, np.uint64(1))
            ts.append(time.perf_counter() - t0)
        rate = len(ids0) / med(ts)
        res["gpu_compute_neurons_per_s"] = rate
        print(f"  compute only (GPU incl. copies to and from VRAM): {rate / 1e6:.2f} M neurons/s = "
              f"{rate * NEURON_BYTES / 1e9:.2f} GB/s of brain; identical to the CPU night: {same:.6%} of words")
        del ref, cpu_out
    except Exception as e:  # GPU is optional
        print(f"  (GPU night skipped: {e})")
        gpu = None

    sub = sub_gib * GiB
    req = 16 * MiB
    # a different region for each variant (O_DIRECT, so nothing is warm anywhere)
    regions = iter(rng.permutation(np.arange(0, n * NEURON_BYTES - sub + 1, sub)))

    def region():
        start = int(next(regions))
        return np.arange(start, start + sub, req, dtype=np.int64)

    print(f"  -- {sub_gib} GiB regions --")
    for label, kw in [("read only (no compute, no write)", dict(compute=False, write=False)),
                      ("I/O only (read + write back, no compute)", dict(compute=False)),
                      ("serial (read, then compute, then write)", dict(mode="serial")),
                      ("pipelined (read-ahead + write-behind)", dict())]:
        runs = [sw.run(fd, region(), req, **kw) for _ in range(1)]
        s = min(runs, key=lambda x: x["wall_s"]) if len(runs) == 1 else sorted(runs, key=lambda x: x["wall_s"])[len(runs) // 2]
        res[label] = s
        res[label + " (all runs wall_s)"] = [r["wall_s"] for r in runs]
        fmt(label, s)
    if gpu is not None:
        sw.gpu = gpu
        s = sw.run(fd, region(), req)
        sw.gpu = None
        res["pipelined, GPU compute"] = s
        fmt("pipelined, GPU compute", s)

    # the full night, with a bit-exact check on 2,000 random neurons (CPU, then GPU compute)
    for label, use_gpu in [("FULL NIGHT", False), ("FULL NIGHT, GPU compute", True)]:
        if use_gpu and gpu is None:
            continue
        sample = np.sort(rng.choice(n, 2000, replace=False)).astype(np.int64)
        before = read_neurons(fd, sample)
        c0 = int(k_checksum(before))
        seed = np.uint64(777 + use_gpu)
        sw.gpu = gpu if use_gpu else None
        s = sw.run(fd, np.arange(0, n * NEURON_BYTES, req, dtype=np.int64), req, seed=seed)
        sw.gpu = None
        res[label] = s
        fmt(f"{label}, whole brain ({n * NEURON_BYTES / GiB:.0f} GiB)", s)
        ok = verify(fd, before, sample, sw, seed)
        res[label + " bit_exact_vs_RAM"] = ok
        if not use_gpu:
            res["pruned_fraction"] = s["pruned"] / (n * 256)
        print(f"  night on SSD == same night computed in RAM by Numba (2,000 neurons, bit-exact): {ok}; "
              f"checksum changed: {int(k_checksum(read_neurons(fd, sample))) != c0}"
              + (f"; synapses regrown: {res['pruned_fraction']:.1%}" if not use_gpu else ""))
    if gpu is not None:
        gpu.close()

    # lazy night: only the columns touched today (clustered activity), dormant ones skip it
    res["lazy"] = {}
    ncol = n // COL
    for frac in (0.05, 0.20):
        cols = np.sort(rng.choice(ncol, round(frac * ncol), replace=False)).astype(np.int64)
        s = sw.run(fd, cols * COL * NEURON_BYTES, COL * NEURON_BYTES, qd=16)
        res["lazy"][str(frac)] = s
        fmt(f"LAZY night, {frac:.0%} of columns touched", s)

    # the soma part of a night (RAM, vectorized): plasticity, firing history, age, dormancy clock
    plast = np.full(n, 1.0, np.float32)
    hist = np.ones((n, 4), np.float32)
    t0 = time.perf_counter()
    plast[:] = 0.3 + (plast - 0.3) * 0.97
    hist[:] = 0
    res["soma_night_s"] = time.perf_counter() - t0
    del plast, hist
    print(f"  soma part of the night (RAM, {n:,} neurons): {res['soma_night_s']:.2f} s")
    res["peak_rss_MiB"] = peak_rss_mib()
    res["ssd_written_GB_total"] = LEDGER.total / 1e9
    print(f"  peak RSS {res['peak_rss_MiB']:.0f} MiB; SSD bytes written by this experiment so far: "
          f"{LEDGER.total / 1e9:.1f} GB")
    os.close(fd)
    save("night", res)
    return res


if __name__ == "__main__":
    main()

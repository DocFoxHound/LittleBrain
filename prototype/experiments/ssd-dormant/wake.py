"""
Waking on an SSD brain: how fast can we run active neurons whose synapses live on the drive?

Scenario (the brief's "needs locality" point, made explicit):
  * 25.2 M neurons x 256 synapses = 24 GiB on the SSD, 3x larger than our 8 GB RAM cap.
  * Columns of 1,024 neurons (1 MiB). Each column reads its own 4,096-input region (3% active).
  * A step evaluates only the ACTIVE neurons (event-driven: the rest receive no input this step).
    One step = forward (with learning traces) + modulate (three-factor update) per active neuron.
  * Active fraction 1% / 5% / 20%; locality:
      scattered   uniform random neurons anywhere in the brain
      columnar    25% of the neurons in 4x as many columns (sparse code inside active columns)
      clustered   whole columns active (activity confined to a few regions)
  * Cold: every step uses a fresh active set and nothing is cached, so every byte comes from the SSD.
    (Warm behaviour = RAM reference; the day simulation below measures realistic hit rates.)

Methods:
  ram         neurons already in RAM (2 GiB in-RAM sample), the upper bound
  aio-4K/64K/1M  explicit O_DIRECT block reads (QD 64, one I/O thread), double-buffered so
                 reading batch i+1 overlaps computing batch i
  scan        stream the WHOLE brain sequentially (16 MiB requests) and compute the active ones
  mmap-random np.memmap (copy-on-write) + MADV_RANDOM: page faults, 4 KiB at a time
  mmap-normal np.memmap with default readahead (4 MiB on this btrfs)
"""

import mmap
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import COL, GiB, KiB, MiB, NEURON_BYTES, Timer, aio, aligned, drop_cache, med, peak_rss_mib, \
    rss_mib, save

import numpy as np

from brain import BRAIN, n_neurons, open_brain
from kernels import k_forward, k_init, k_modulate
from neuron import WORKHORSE
from packed_neuron import ELIG_STEPS

F32 = np.float32
R_IN = 4096
COL_SHIFT = 977


def read_bytes_io():
    with open("/proc/self/io") as f:
        for line in f:
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    return 0


class Waker:
    """Soma state for the whole brain (RAM) + the compute for one waking step."""

    def __init__(self, n, t=WORKHORSE, seed=0):
        rng = np.random.default_rng(seed)
        self.t, self.n = t, n
        self.gate = np.full(n, 255, np.uint8)
        self.thresh = np.full(n, t.threshold_init, F32)
        self.plast = np.full(n, t.plasticity_start, F32)
        self.post_hist = np.zeros((n, ELIG_STEPS), F32)
        self.X = (rng.random(R_IN) < 0.03).astype(F32)
        self.pre_hist = (rng.random((ELIG_STEPS, R_IN)) < 0.03).astype(F32)
        self.elig_w = (t.elig_decay ** np.arange(ELIG_STEPS)).astype(F32)
        self.F0 = np.array([1.0, 0.0, 0.0], F32)
        self.seed = np.uint64(seed + 1)

    def soma_bytes(self):
        return self.gate.nbytes + self.thresh.nbytes + self.plast.nbytes + self.post_hist.nbytes

    def compute(self, words, rows, ids, modulate=True):
        t = self.t
        rate = np.empty(len(rows), F32)
        k_forward(words, rows, ids, 0, self.gate, self.thresh, self.post_hist, self.X, R_IN, COL_SHIFT,
                  self.F0, F32(1.0), F32(t.w_max), F32(t.branch_threshold), F32(t.spike_bonus), F32(1.0),
                  F32(t.soma_slope), F32(t.threshold_lr), F32(t.target_rate), True, rate)
        if modulate:
            self.seed += np.uint64(1)
            k_modulate(words, rows, ids, 0, self.gate, self.plast, self.post_hist, self.pre_hist,
                       self.elig_w, R_IN, COL_SHIFT, F32(1.0), F32(t.w_max), F32(t.lr), F32(0.5), F32(0.5),
                       F32(0.5), False, self.seed)
        return rate


def active_set(n, frac, locality, rng):
    ncol = n // COL
    if locality == "scattered":
        return np.flatnonzero(rng.random(n) < frac).astype(np.int64)
    if locality == "columnar":       # 25% of the neurons inside 4x as many columns
        cols = np.sort(rng.choice(ncol, max(1, round(4 * frac * ncol)), replace=False))
        inside = rng.random((len(cols), COL)) < 0.25
        return (cols[:, None] * COL + np.arange(COL))[inside].astype(np.int64)
    cols = np.sort(rng.choice(ncol, max(1, round(frac * ncol)), replace=False))  # clustered
    return (cols[:, None] * COL + np.arange(COL)).ravel().astype(np.int64)


# ----------------------------------------------------------------------------- methods

def step_ram(w, ram_words, ids, rng, scattered_rows):
    """Upper bound: the same neurons, already in RAM (a 2 GiB in-RAM sample; ids keep their real
    hashing/columns). Large active sets are sampled and the rate extrapolated."""
    k = min(len(ids), ram_words.shape[0])
    sub = ids[:k] if k == len(ids) else np.sort(rng.choice(ids, k, replace=False))
    rows = (np.sort(rng.choice(ram_words.shape[0], k, replace=False)) if scattered_rows
            else np.arange(k)).astype(np.int64)
    with Timer() as tm:
        w.compute(ram_words, rows, sub)
    return {"neurons": k, "s": tm.dt, "bytes": 0, "ios": 0, "cores": tm.cores, "io_wait_s": 0.0,
            "compute_s": tm.dt}


_staging = {}


def staging_bufs(nbytes):
    """Two DMA buffers, allocated (and pre-faulted) once and reused for every step."""
    if nbytes not in _staging:
        _staging.clear()
        _staging[nbytes] = [aligned(nbytes, np.uint32).reshape(-1, 256) for _ in range(2)]
    return _staging[nbytes]


def step_blocks(w, fd, ids, bk, pool, staging=256 * MiB, qd=64, scan=False, n=None):
    """Read the blocks holding active neurons (or, if scan, the whole brain) with O_DIRECT AIO,
    double-buffered, and compute on them."""
    bb = bk * NEURON_BYTES
    if scan:
        blk = np.arange(n // bk, dtype=np.int64)
    else:
        blk = np.unique(ids // bk)
    per = max(1, staging // bb)
    bufs = staging_bufs(per * bb)
    bufs = [b[:per * bk] for b in bufs]
    pos = np.searchsorted(blk, ids // bk)                       # which block slot each neuron uses
    starts = list(range(0, len(blk), per))
    nb = np.searchsorted(pos, starts + [len(blk)])              # neuron range per batch
    req = 16 * MiB if scan else bb
    io_wait = comp = 0.0

    def read(i, buf):
        s, e = starts[i], min(starts[i] + per, len(blk))
        if scan:   # contiguous: big sequential requests
            offs = np.arange(blk[s] * bb, blk[e - 1] * bb + bb, req, dtype=np.int64)
            return aio(fd, False, offs, req, buf, offs - blk[s] * bb, qd=8)
        return aio(fd, False, blk[s:e] * bb, bb, buf, np.arange(e - s, dtype=np.int64) * bb, qd=qd)

    with Timer() as tm:
        fut = pool.submit(read, 0, bufs[0])
        for i in range(len(starts)):
            t0 = time.perf_counter()
            fut.result()
            io_wait += time.perf_counter() - t0
            if i + 1 < len(starts):
                fut = pool.submit(read, i + 1, bufs[(i + 1) % 2])
            a, b = nb[i], nb[i + 1]
            if b > a:
                t0 = time.perf_counter()
                rows = (pos[a:b] - starts[i]) * bk + ids[a:b] % bk
                w.compute(bufs[i % 2], rows, ids[a:b])
                comp += time.perf_counter() - t0
    return {"neurons": len(ids), "s": tm.dt, "bytes": len(blk) * bb, "ios": len(blk) * bb // req,
            "cores": tm.cores, "io_wait_s": io_wait, "compute_s": comp}


def step_mmap(w, mm, ids, advice, batch=32768, cap_s=20.0):
    """np.memmap, copy-on-write so learning never writes the file. After each batch the mapping
    and the page cache are dropped, so the next batch is cold too and RAM stays bounded."""
    arr = np.asarray(mm)
    raw = mm._mmap
    done, rb0, comp = 0, read_bytes_io(), 0.0
    with Timer() as tm:
        for a in range(0, len(ids), batch):
            sub = ids[a:a + batch]
            w.compute(arr, sub, sub)
            done += len(sub)
            raw.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(mm_fd(mm), 0, 0, os.POSIX_FADV_DONTNEED)
            raw.madvise(advice)
            if time.perf_counter() - tm.t0 > cap_s:
                break
    return {"neurons": done, "s": tm.dt, "bytes": read_bytes_io() - rb0, "ios": 0, "cores": tm.cores,
            "io_wait_s": float("nan"), "compute_s": float("nan"), "partial": done < len(ids)}


_fds = {}


def mm_fd(mm):
    if id(mm) not in _fds:
        _fds[id(mm)] = os.open(BRAIN, os.O_RDONLY)
    return _fds[id(mm)]


# ----------------------------------------------------------------------------- driver

def main(fracs=(0.01, 0.05, 0.20), localities=("scattered", "columnar", "clustered"), steps=3,
         methods=("ram", "aio-4K", "aio-64K", "aio-1M", "scan", "mmap-random", "mmap-normal"),
         budget_s=40.0):
    n = n_neurons()
    rng = np.random.default_rng(3)
    w = Waker(n)
    ram_n = 2 * GiB // NEURON_BYTES
    ram_words = np.empty((ram_n, 256), np.uint32)
    k_init(ram_words, 0, np.uint64(99))
    fd = open_brain()
    drop_cache(str(BRAIN))
    mm = np.memmap(BRAIN, dtype=np.uint32, mode="c", shape=(n, 256))
    pool = ThreadPoolExecutor(1)
    # compile + warm up
    w.compute(ram_words, np.arange(1000, dtype=np.int64), np.arange(1000, dtype=np.int64))
    res = {"neurons": n, "soma_bytes_per_neuron": w.soma_bytes() / n, "rows": []}
    print(f"== Waking: {n:,} neurons on SSD ({n * NEURON_BYTES / GiB:.0f} GiB); soma state in RAM "
          f"{w.soma_bytes() / n:.0f} B/neuron; one step = forward + modulate on the active neurons ==")
    print(f"  {'frac':>5} {'locality':<10} {'method':<12} {'neuron-steps/s':>15} {'s/step':>8} "
          f"{'GB read/step':>12} {'GB/s':>6} {'IOPS':>9} {'amplif.':>8} {'io-wait':>8} {'cores':>6}")
    for frac in fracs:
        for loc in localities:
            for meth in methods:
                runs = []
                t_start = time.perf_counter()
                for s in range(steps):
                    ids = active_set(n, frac, loc, rng)
                    if meth == "ram":
                        r = step_ram(w, ram_words, ids, rng, scattered_rows=(loc != "clustered"))
                    elif meth.startswith("aio"):
                        bk = {"aio-4K": 4, "aio-64K": 64, "aio-1M": 1024}[meth]
                        r = step_blocks(w, fd, ids, bk, pool)
                    elif meth == "scan":
                        r = step_blocks(w, fd, ids, 16, pool, staging=256 * MiB, scan=True, n=n)
                    else:
                        adv = mmap.MADV_RANDOM if meth == "mmap-random" else mmap.MADV_NORMAL
                        mm._mmap.madvise(adv)
                        r = step_mmap(w, mm, ids, adv, cap_s=min(20.0, budget_s / steps),
                                      batch=32768 if meth == "mmap-random" else 8192)
                    r["active"] = len(ids)
                    runs.append(r)
                    if time.perf_counter() - t_start > budget_s:
                        break
                rate = med([r["neurons"] / r["s"] for r in runs])
                gb = med([r["bytes"] * (r["active"] / r["neurons"]) for r in runs]) / 1e9
                gbps = med([r["bytes"] / r["s"] for r in runs]) / 1e9
                iops = med([r["ios"] / r["s"] for r in runs])
                amp = med([r["bytes"] / (r["neurons"] * NEURON_BYTES) for r in runs])
                iow = med([r["io_wait_s"] / r["s"] for r in runs])
                cores = med([r["cores"] for r in runs])
                sps = med([r["s"] * r["active"] / r["neurons"] for r in runs])
                row = {"frac": frac, "locality": loc, "method": meth, "neuron_steps_per_s": rate,
                       "s_per_step": sps, "GB_read_per_step": gb, "GBps": gbps, "IOPS": iops,
                       "amplification": amp, "io_wait_frac": iow, "cores": cores, "steps": len(runs),
                       "partial": any(r.get("partial", False) for r in runs)}
                res["rows"].append(row)
                print(f"  {frac:>5.0%} {loc:<10} {meth:<12} {rate:>15,.0f} {sps:>8.2f} {gb:>12.2f} "
                      f"{gbps:>6.2f} {iops:>9,.0f} {amp:>8.1f} {iow:>8.0%} {cores:>6.1f}"
                      + ("  (time-capped, extrapolated)" if row["partial"] else ""), flush=True)
                save("wake", res)
    res["peak_rss_MiB"] = peak_rss_mib()
    save("wake", res)
    print(f"  peak RSS {res['peak_rss_MiB']:.0f} MiB")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        main(fracs=(0.01,), localities=("scattered", "clustered"), steps=1)
    else:
        main()

"""
The on-disk brain: one flat file of packed neurons, neuron i at byte offset i * 1 KiB.
Columns of 1,024 neurons = 1 MiB each (a column draws its input from its own region).

Per-neuron "soma" state (threshold, plasticity, firing history, gate, age, last-active night)
is ~20 bytes per neuron and stays in RAM; only the synapses (98% of the bytes) live on the SSD.
"""

import os
import threading

from common import COL, DATA, GiB, LEDGER, MiB, NEURON_BYTES, Timer, aio, aligned, drop_cache, med, \
    open_direct, save

import numpy as np

from kernels import k_init

BRAIN = DATA / "brain.words"
DEFAULT_GIB = 24          # 24 GiB = 25,165,824 neurons = 6.4 billion synapses (3x the 8 GB RAM cap)


def n_neurons(path=BRAIN):
    return os.path.getsize(path) // NEURON_BYTES


class Soma:
    """RAM-resident per-neuron state for the whole brain (about 20 bytes per neuron)."""

    def __init__(self, n, t):
        self.gate = np.full(n, 255, np.uint8)
        self.thresh = np.full(n, t.threshold_init, np.float32)
        self.plast = np.full(n, t.plasticity_start, np.float32)
        self.post_hist = np.zeros((n, 4), np.float16)
        self.last_night = np.zeros(n, np.uint16)    # last night this neuron was active -> dormancy

    def nbytes(self):
        return sum(a.nbytes for a in (self.gate, self.thresh, self.plast, self.post_hist, self.last_night))


def create(gib=DEFAULT_GIB, chunk_gib=1, req=4 * MiB, qd=8, seed=12345):
    """Write a fresh brain with O_DIRECT, generating chunk i+1 while chunk i is being written.
    Each chunk's write is timed on its own: that is our sequential-write measurement."""
    n = gib * GiB // NEURON_BYTES
    chunk_n = chunk_gib * GiB // NEURON_BYTES
    if BRAIN.exists():
        BRAIN.unlink()
    fd = os.open(BRAIN, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
    os.posix_fallocate(fd, 0, n * NEURON_BYTES)
    bufs = [aligned(chunk_n * NEURON_BYTES, np.uint32).reshape(chunk_n, 256) for _ in range(2)]
    times = []
    writer = None

    def write(buf, off):
        offs = off + np.arange(0, buf.nbytes, req, dtype=np.int64)
        with Timer() as tm:
            aio(fd, True, offs, req, buf, np.arange(0, buf.nbytes, req, dtype=np.int64), qd=qd)
        times.append(tm.dt)

    with Timer() as total:
        for i in range(n // chunk_n):
            b = bufs[i % 2]
            k_init(b, i * chunk_n, np.uint64(seed))
            if writer:
                writer.join()
            writer = threading.Thread(target=write, args=(b, i * chunk_n * NEURON_BYTES))
            writer.start()
        writer.join()
        os.fsync(fd)
    os.close(fd)
    drop_cache(str(BRAIN))
    gbps = [chunk_gib * GiB / t / 1e9 for t in times]
    res = {"neurons": n, "bytes": n * NEURON_BYTES, "chunk_GBps": gbps,
           "seq_write_GBps_median": med(gbps), "seq_write_GBps_min": min(gbps),
           "create_wall_s": total.dt, "create_wall_GBps": n * NEURON_BYTES / total.dt / 1e9,
           "request_bytes": req, "qd": qd}
    print(f"  brain: {n:,} neurons, {n * NEURON_BYTES / GiB:.0f} GiB; sequential write "
          f"median {res['seq_write_GBps_median']:.2f} GB/s (min {res['seq_write_GBps_min']:.2f}) "
          f"per 1 GiB chunk; whole file {total.dt:.1f} s ({res['create_wall_GBps']:.2f} GB/s incl. generation)")
    save("create", res)
    return res


def ensure(gib=DEFAULT_GIB):
    if BRAIN.exists() and os.path.getsize(BRAIN) == gib * GiB:
        return None
    return create(gib)


def open_brain(write=False):
    return open_direct(BRAIN, write)

"""Shared plumbing: thread caps, paths, the O_DIRECT/AIO helper, aligned buffers, a write ledger."""

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import json
import mmap
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent.parent
sys.path.insert(0, str(PROTO))
DATA = HERE / "data"            # chattr +C (no copy-on-write, no btrfs compression)
RESULTS = HERE / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SYN = 256                       # synapses per neuron (8 branches x 32 slots)
NEURON_BYTES = SYN * 4          # 1 KiB of packed synapse words per neuron
COL = 1024                      # neurons per column = 1 MiB on disk
GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10

# ---------------------------------------------------------------- AIO helper

def aio(fd, write, offs, size, buf, buf_offs=None, qd=32, lat=False):
    """Move len(offs) requests of `size` bytes. Returns (bytes, latencies_ns or None).
    The native extension drops the GIL during the call, so this overlaps with Numba (nogil) compute."""
    offs = np.ascontiguousarray(offs, np.int64)
    bo = None if buf_offs is None else np.ascontiguousarray(buf_offs, np.int64)
    if offs.ndim != 1 or size <= 0 or not 1 <= qd <= 65535 or np.any(offs < 0):
        raise ValueError("Invalid AIO request range or queue depth")
    if bo is None:
        if min(len(offs), qd) * size > buf.nbytes:
            raise ValueError("AIO ring exceeds the supplied buffer")
    elif (bo.ndim != 1 or len(bo) < len(offs) or np.any(bo[:len(offs)] < 0)
          or np.any(bo[:len(offs)] > buf.nbytes - size)):
        raise ValueError("AIO request exceeds the supplied buffer")
    lt = np.zeros(len(offs), np.int64) if lat else None
    from _lb_aio import ffi, lib
    r = lib.aio_batch(fd, int(bool(write)), ffi.from_buffer("int64_t[]", offs), len(offs), int(size),
                      ffi.from_buffer("char[]", buf, require_writable=not write), ffi.NULL if bo is None else ffi.from_buffer("int64_t[]", bo),
                      int(qd), ffi.NULL if lt is None else ffi.from_buffer("int64_t[]", lt))
    if r < 0:
        raise OSError(-r, os.strerror(-r))
    if write:
        LEDGER.add(r)
    return r, lt


def aligned(nbytes, dtype=np.uint8):
    """Page-aligned (4 KiB) anonymous memory, as needed by O_DIRECT. MAP_POPULATE pre-faults it:
    AIO into never-touched pages makes io_submit fault them in one by one (measured: 14K instead
    of 340K IOPS at 4 KiB), so DMA buffers must be populated once and reused."""
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE)
    return np.frombuffer(m, dtype=dtype)


def open_direct(path, write=False):
    flags = (os.O_RDWR if write else os.O_RDONLY) | os.O_DIRECT
    return os.open(path, flags)


def drop_cache(path_or_fd):
    """Evict a file's pages from the page cache (no root needed)."""
    if isinstance(path_or_fd, int):
        os.posix_fadvise(path_or_fd, 0, 0, os.POSIX_FADV_DONTNEED)
    else:
        fd = os.open(path_or_fd, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def cached_pages(path):
    """How much of a file is in the page cache right now (bytes), via mincore on a mapping."""
    from _lb_aio import lib
    with open(path, "rb") as f:
        size = lib.resident_bytes(f.fileno(), os.fstat(f.fileno()).st_size)
    if size < 0:
        raise OSError(-size, os.strerror(-size))
    return size


# ---------------------------------------------------------------- bookkeeping

class Ledger:
    """Counts every byte this experiment writes to the SSD (endurance budget: ~150 GB)."""
    path = RESULTS / "ssd_writes.json"

    def __init__(self):
        self.total = json.loads(self.path.read_text())["bytes"] if self.path.exists() else 0

    def add(self, n):
        self.total += int(n)
        self.path.write_text(json.dumps({"bytes": self.total, "GB": self.total / 1e9}))


LEDGER = Ledger()


def rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Timer:
    """Wall time plus CPU time, so we can report how many cores were busy on average."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.c0 = r.ru_utime + r.ru_stime
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu = r.ru_utime + r.ru_stime - self.c0
        self.cores = self.cpu / max(self.dt, 1e-9)


def save(name, obj):
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def med(xs):
    return float(np.median(np.asarray(xs, float)))

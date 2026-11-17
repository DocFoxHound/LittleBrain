"""Shared plumbing: thread caps, paths, the O_DIRECT/AIO helper, aligned buffers, a write ledger."""

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import ctypes
import json
import mmap
import resource
import subprocess
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

_lib = None


def lib():
    global _lib
    if _lib is None:
        so = HERE / "_aio.so"
        src = HERE / "aio.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(src), "-o", str(so)], check=True)
        _lib = ctypes.CDLL(str(so))
        _lib.aio_batch.restype = ctypes.c_int64
        _lib.aio_batch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p]
    return _lib


def _ptr(a):
    return None if a is None else a.ctypes.data


def aio(fd, write, offs, size, buf, buf_offs=None, qd=32, lat=False):
    """Move len(offs) requests of `size` bytes. Returns (bytes, latencies_ns or None).
    ctypes drops the GIL during the call, so this overlaps with Numba (nogil) compute."""
    offs = np.ascontiguousarray(offs, np.int64)
    bo = None if buf_offs is None else np.ascontiguousarray(buf_offs, np.int64)
    lt = np.zeros(len(offs), np.int64) if lat else None
    r = lib().aio_batch(fd, int(bool(write)), _ptr(offs), len(offs), int(size),
                        buf.ctypes.data, _ptr(bo), int(qd), _ptr(lt))
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
    size = os.path.getsize(path)
    if size == 0:
        return 0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        pages = (size + 4095) // 4096
        vec = np.zeros(pages, np.uint8)
        libc.mincore(ctypes.c_void_p(addr), size, vec.ctypes.data)
        libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)
    return int((vec & 1).sum()) * 4096


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

"""
Characterize the NVMe drive (WD_BLACK SN770 2 TB, btrfs, our folder is chattr +C = no CoW and
no transparent compression).

Everything below uses O_DIRECT (bypasses the page cache), so we measure the SSD, not RAM.
One CPU thread keeps `qd` requests in flight with Linux native AIO; "QD" = parallel requests.
The page-cache pitfall is shown separately (buffered reads, cold vs warm).
"""

import os
import threading
import time

from common import DATA, GiB, KiB, LEDGER, MiB, Timer, aio, aligned, cached_pages, drop_cache, med, \
    open_direct, save

import numpy as np

from brain import BRAIN

REPS = 3


def seq_read(fd, fsize, region, req, qd, start):
    offs = start + np.arange(0, region, req, dtype=np.int64)
    buf = aligned(qd * req)
    with Timer() as t:
        aio(fd, False, offs, req, buf, qd=qd)
    return region / t.dt / 1e9, t


def rand_io(fd, fsize, req, qd, n, rng, write=False, lat=False):
    offs = rng.integers(0, fsize // req, n, dtype=np.int64) * req
    buf = aligned(qd * req)
    if write:
        buf[:] = rng.integers(0, 256, buf.size, dtype=np.uint8)  # incompressible
    with Timer() as t:
        _, lt = aio(fd, write, offs, req, buf, qd=qd, lat=lat)
    return n / t.dt, n * req / t.dt / 1e9, lt, t


def threaded_4k(path, fsize, nthreads, per_thread, rng):
    """Plain blocking preadv from N Python threads (os.preadv drops the GIL): QD = N."""
    offs = [rng.integers(0, fsize // 4096, per_thread, dtype=np.int64) * 4096 for _ in range(nthreads)]

    def worker(o):
        fd = open_direct(path)
        buf = aligned(4096)
        for x in o:
            os.preadv(fd, [buf], int(x))
        os.close(fd)

    th = [threading.Thread(target=worker, args=(o,)) for o in offs]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.perf_counter() - t0
    return nthreads * per_thread / dt


def page_cache_pitfall(path, region=2 * GiB):
    """Buffered reads: first (cold, after DONTNEED) hits the SSD; second is served from RAM."""
    out = {}
    drop_cache(path)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, region, os.POSIX_FADV_SEQUENTIAL)
    buf = aligned(16 * MiB)
    for label in ("buffered_cold", "buffered_warm"):
        t0 = time.perf_counter()
        off = 0
        while off < region:
            off += os.preadv(fd, [buf], off)
        out[label + "_GBps"] = region / (time.perf_counter() - t0) / 1e9
    out["cached_after_GiB"] = cached_pages(path) / GiB
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    out["cached_after_dontneed_GiB"] = cached_pages(path) / GiB
    os.close(fd)
    return out


def main():
    path = str(BRAIN)
    fsize = os.path.getsize(path)
    rng = np.random.default_rng(7)
    drop_cache(path)
    fd = open_direct(path)
    res = {"file_bytes": fsize, "seq_read": {}, "rand_read": {}, "rand_write": {}}

    print("== Drive: sequential read (O_DIRECT, 1 GiB-aligned regions, median of 3) ==")
    for req, qd in [(128 * KiB, 1), (1 * MiB, 1), (1 * MiB, 4), (1 * MiB, 16), (4 * MiB, 8), (16 * MiB, 4)]:
        vals = [seq_read(fd, fsize, 3 * GiB, req, qd, int(rng.integers(0, fsize // GiB - 2)) * GiB)[0]
                for _ in range(REPS)]
        res["seq_read"][f"{req // KiB}K_qd{qd}"] = med(vals)
        print(f"  {req // KiB:>6} KiB requests, QD {qd:>3}: {med(vals):6.2f} GB/s  (runs {', '.join(f'{v:.2f}' for v in vals)})")

    print("== Drive: random read (O_DIRECT, uniform over the 24 GiB file, median of 3) ==")
    plan = {4 * KiB: [(1, 15000), (4, 40000), (16, 120000), (64, 250000), (256, 300000)],
            64 * KiB: [(1, 6000), (4, 15000), (16, 30000), (64, 50000)],
            1 * MiB: [(1, 1500), (4, 3000), (16, 5000), (64, 6000)]}
    for req, runs in plan.items():
        for qd, n in runs:
            iops, gbps, lats = [], [], None
            for r in range(REPS):
                a, b, lt, _ = rand_io(fd, fsize, req, qd, n, rng, lat=(qd == 1 and r == 0))
                iops.append(a)
                gbps.append(b)
                if lt is not None:
                    lats = lt
            key = f"{req // KiB}K_qd{qd}"
            res["rand_read"][key] = {"IOPS": med(iops), "GBps": med(gbps)}
            extra = ""
            if lats is not None:
                p = np.percentile(lats / 1e3, [50, 99, 99.9])
                res["rand_read"][key]["lat_us_p50_p99_p999"] = p.tolist()
                extra = f"   latency p50 {p[0]:.0f} us, p99 {p[1]:.0f} us, p99.9 {p[2]:.0f} us"
            print(f"  {req // KiB:>5} KiB, QD {qd:>3}: {med(iops):>9,.0f} IOPS  {med(gbps):6.2f} GB/s{extra}")

    th = [threaded_4k(path, fsize, 3, 8000, rng) for _ in range(REPS)]
    res["rand_read"]["4K_3threads_blocking"] = {"IOPS": med(th)}
    print(f"  4 KiB, 3 blocking Python threads (QD 3): {med(th):,.0f} IOPS")
    os.close(fd)

    print("== Drive: random write into a 2 GiB scratch file (incompressible data) ==")
    scratch = DATA / "scratch.bin"
    sfd = os.open(scratch, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
    os.posix_fallocate(sfd, 0, 2 * GiB)
    for req, qd, n in [(4 * KiB, 1, 5000), (4 * KiB, 32, 60000), (64 * KiB, 32, 12000), (1 * MiB, 16, 1500)]:
        iops, gbps = [], []
        for _ in range(REPS):
            a, b, _, _ = rand_io(sfd, 2 * GiB, req, qd, n, rng, write=True)
            iops.append(a)
            gbps.append(b)
        res["rand_write"][f"{req // KiB}K_qd{qd}"] = {"IOPS": med(iops), "GBps": med(gbps)}
        print(f"  {req // KiB:>5} KiB, QD {qd:>3}: {med(iops):>9,.0f} IOPS  {med(gbps):6.2f} GB/s")
    os.fsync(sfd)
    os.close(sfd)
    scratch.unlink()

    print("== Pitfall: buffered reads of a 2 GiB region (what you'd see without O_DIRECT) ==")
    pc = page_cache_pitfall(path)
    res["page_cache"] = pc
    print(f"  cold (after fadvise DONTNEED): {pc['buffered_cold_GBps']:.2f} GB/s; warm (page cache = RAM): "
          f"{pc['buffered_warm_GBps']:.2f} GB/s; cached after: {pc['cached_after_GiB']:.2f} GiB, "
          f"after DONTNEED: {pc['cached_after_dontneed_GiB']:.2f} GiB")
    save("drive", res)
    print(f"  SSD bytes written by this experiment so far: {LEDGER.total / 1e9:.1f} GB")
    return res


if __name__ == "__main__":
    main()

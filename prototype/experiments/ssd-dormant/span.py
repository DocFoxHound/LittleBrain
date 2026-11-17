"""Does the span of random reads matter on this DRAM-less SSD? QD1 4 KiB reads, back to back (C loop),
over spans from 64 MiB to the whole 24 GiB brain. (The SN770 keeps its address map in a small host-memory
buffer, so a compact dormant archive may answer faster than the same data scattered over a big file.)"""
import os

from common import GiB, MiB, aio, aligned, open_direct, save

import numpy as np

from brain import BRAIN


def main(n=6000, reps=3):
    fd = open_direct(str(BRAIN))
    fsize = os.path.getsize(BRAIN)
    rng = np.random.default_rng(4)
    buf = aligned(4096)
    out = {}
    print("== Span test: QD1 random 4 KiB reads, p50 latency (median of 3 runs) ==")
    for span in (64 * MiB, 350 * MiB, 2 * GiB, 8 * GiB, fsize):
        p50s, p99s = [], []
        for _ in range(reps):
            start = int(rng.integers(0, (fsize - span) // 4096 + 1)) * 4096
            offs = start + rng.integers(0, span // 4096, n, dtype=np.int64) * 4096
            _, lat = aio(fd, False, offs, 4096, buf, qd=1, lat=True)
            p50s.append(np.median(lat) / 1e3)
            p99s.append(np.percentile(lat, 99) / 1e3)
        out[str(span)] = {"p50_us": float(np.median(p50s)), "p99_us": float(np.median(p99s))}
        print(f"  span {span / GiB:6.2f} GiB: p50 {np.median(p50s):5.0f} us   p99 {np.median(p99s):5.0f} us")
    os.close(fd)
    save("span", out)


if __name__ == "__main__":
    main()

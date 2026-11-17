"""
PCIe ceilings on this machine (RTX 4070 Laptop, PCIe 4.0 x8):
  * DMA copies (cudaMemcpyAsync, pinned) by copy size = "page size", one direction and both at once
  * zero-copy: a kernel reading / writing mapped pinned host memory directly
    (full 128-byte lines vs scattered 4-byte writes)

Run: .venv/bin/python prototype/experiments/vram-ram-paging/pcie_micro.py
"""

import json
import os
import time

import numpy as np
import cupy as cp

from gpu_population import pinned_array, H2D, D2H

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL = 256 << 20

SRC = r"""
extern "C" __global__ void zc_read(const unsigned int* __restrict__ a, long long n, unsigned int* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = 0;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) s += a[i];
    if (s == 0x12345678u) out[0] = s;   // keep the loads alive
}
extern "C" __global__ void zc_write_full(unsigned int* a, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) a[i] = (unsigned int)i;
}
// one 4-byte write per 32-word line (like setting a few "used" bits)
extern "C" __global__ void zc_write_sparse(unsigned int* a, long long n, int every) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x)
        if ((i % every) == 0) a[i] = (unsigned int)i;
}
"""


def gbps(nbytes, f, reps=5):
    f()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    return nbytes / float(np.median(ts)) / 1e9


def main():
    host, _pin = pinned_array(TOTAL)
    host[:] = 1
    dev = cp.empty(TOTAL // 4, cp.uint32)
    hp, dp = host.ctypes.data, dev.data.ptr
    s1, s2 = cp.cuda.Stream(non_blocking=False), cp.cuda.Stream(non_blocking=False)
    res = {"copy": []}
    print("== DMA copies (pinned host <-> VRAM), 256 MB moved per measurement ==")
    print(f"  {'copy size':>10} {'H2D GB/s':>9} {'D2H GB/s':>9} {'both at once GB/s (sum)':>24}")
    for size in [4 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 32 << 20]:
        n = TOTAL // size

        def h2d():
            for i in range(n):
                cp.cuda.runtime.memcpyAsync(dp + i * size, hp + i * size, size, H2D, s1.ptr)

        def d2h():
            for i in range(n):
                cp.cuda.runtime.memcpyAsync(hp + i * size, dp + i * size, size, D2H, s2.ptr)

        def both():
            half = n // 2
            for i in range(half):
                cp.cuda.runtime.memcpyAsync(dp + i * size, hp + i * size, size, H2D, s1.ptr)
                cp.cuda.runtime.memcpyAsync(hp + (half + i) * size, dp + (half + i) * size, size, D2H, s2.ptr)

        a, b, c = gbps(TOTAL, h2d), gbps(TOTAL, d2h), gbps(TOTAL, both)
        print(f"  {size >> 10:>8} KB {a:9.2f} {b:9.2f} {c:24.2f}")
        res["copy"].append({"size": size, "h2d": a, "d2h": b, "both": c})

    mod = cp.RawModule(code=SRC)
    rd, wf, ws = mod.get_function("zc_read"), mod.get_function("zc_write_full"), mod.get_function("zc_write_sparse")
    out = cp.zeros(1, cp.uint32)
    n = np.int64(TOTAL // 4)
    grid, blk = (36 * 32,), (256,)
    print("\n== zero-copy kernels on mapped pinned host memory ==")
    zr = gbps(TOTAL, lambda: rd(grid, blk, (np.uint64(hp), n, out)))
    zw = gbps(TOTAL, lambda: wf(grid, blk, (np.uint64(hp), n)))
    t0 = time.perf_counter()
    ws(grid, blk, (np.uint64(hp), n, np.int32(32)))
    cp.cuda.Device().synchronize()
    t_sparse = time.perf_counter() - t0
    writes = TOTAL // 4 // 32
    vr = gbps(TOTAL, lambda: rd(grid, blk, (dev, n, out)))
    print(f"  read  (coalesced)        {zr:6.2f} GB/s   (same kernel on VRAM: {vr:6.1f} GB/s)")
    print(f"  write (full 128 B lines) {zw:6.2f} GB/s")
    print(f"  write (one 4-byte word per 128 B line): {writes / t_sparse / 1e6:6.1f} M writes/s "
          f"= {writes * 128 / t_sparse / 1e9:5.2f} GB/s of lines touched")
    res.update({"zc_read": zr, "zc_write_full": zw, "zc_sparse_writes_per_s": writes / t_sparse, "vram_read": vr})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "pcie_micro.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

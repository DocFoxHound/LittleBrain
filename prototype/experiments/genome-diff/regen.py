"""
What does it cost to regenerate a synapse's birth state on the fly instead of reading it?

Three ways to get (target input, weight) for every synapse in a forward-style sweep:
  stored     : target uint16 + weight int8 read from memory (3 B/synapse; no procedural wiring)
  packed     : the prototype: 32-bit word read from memory, target from the wiring hash (4 B/synapse)
  procedural : NOTHING read: target AND innate weight from one 64-bit hash (0 B/synapse)
Each computes drive = sum(weight * x[target]) per 32-synapse branch, 4,096 inputs, 2% active.
CPU: numba, 3 threads. GPU: CUDA C via cupy RawKernel, one thread per synapse + warp reduction.
"""

import time

import core  # noqa: F401  (first: sets thread caps before numba loads)
import numba as nb
import numpy as np

from core import PRISTINE, innate_q, syn_hash

N_INPUTS = 4096


def med(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


@nb.njit(parallel=True, cache=True)
def cpu_stored(src, w8, x, drive):
    n = src.shape[0]
    for i in nb.prange(n):
        for b in range(8):
            d = 0.0
            for s in range(32):
                j = b * 32 + s
                d += w8[i, j] * x[src[i, j]]
            drive[i, b] = d


@nb.njit(parallel=True, cache=True)
def cpu_packed(words, x, drive, first_id):
    n = words.shape[0]
    for i in nb.prange(n):
        nid = np.uint64(first_id + i)
        for b in range(8):
            d = 0.0
            for s in range(32):
                j = b * 32 + s
                w = words[i, j]
                h = syn_hash(nid, np.uint64(j), np.uint64(w & 15))
                d += (((w >> 10) & 63) - 32.0) * x[np.int64(h & np.uint64(N_INPUTS - 1))]
            drive[i, b] = d


@nb.njit(parallel=True, cache=True)
def cpu_procedural(n, x, drive, first_id, birth_q):
    for i in nb.prange(n):
        nid = np.uint64(first_id + i)
        for b in range(8):
            d = 0.0
            for s in range(32):
                j = b * 32 + s
                h = syn_hash(nid, np.uint64(j), np.uint64(0))
                d += innate_q(h, birth_q) * x[np.int64(h & np.uint64(N_INPUTS - 1))]
            drive[i, b] = d


CUDA_SRC = r"""
#define N_INPUTS 4096ULL
__device__ __forceinline__ unsigned long long syn_hash(unsigned long long nid, unsigned long long j,
                                                       unsigned long long gen) {
    unsigned long long h = ((nid + 1ULL) * 0x9E3779B97F4A7C15ULL) ^ (j * 0xBF58476D1CE4E5B9ULL)
                           ^ (gen * 0x94D049BB133111EBULL);
    h ^= h >> 31; h *= 0xD6E8FEB86659FD93ULL; h ^= h >> 32;
    return h;
}
__device__ __forceinline__ void reduce_store(float v, float* drive, long long t) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) drive[t >> 5] = v;
}
extern "C" __global__ void g_stored(const unsigned short* src, const signed char* w8, const float* x,
                                    float* drive, long long n_syn) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n_syn) return;
    reduce_store((float)w8[t] * x[src[t]], drive, t);
}
extern "C" __global__ void g_packed(const unsigned int* words, const float* x, float* drive,
                                    long long n_syn, long long first_id) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n_syn) return;
    unsigned int w = words[t];
    unsigned long long h = syn_hash((unsigned long long)(first_id + (t >> 8)), (unsigned long long)(t & 255),
                                    (unsigned long long)(w & 15u));
    reduce_store(((float)((w >> 10) & 63u) - 32.f) * x[h & (N_INPUTS - 1)], drive, t);
}
extern "C" __global__ void g_procedural(const float* x, float* drive, long long n_syn, long long first_id,
                                        float birth_q) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n_syn) return;
    unsigned long long h = syn_hash((unsigned long long)(first_id + (t >> 8)), (unsigned long long)(t & 255), 0ULL);
    float u1 = (float)((h >> 24) & 0xFFFFFULL) * (1.f / 1048576.f);
    float u2 = (float)((h >> 44) & 0xFFFFFULL) * (1.f / 1048576.f);
    reduce_store(floorf(u1 * birth_q + u2) * x[h & (N_INPUTS - 1)], drive, t);
}
"""


def cpu_part(n=65_536):
    rng = np.random.default_rng(0)
    x = np.zeros(N_INPUTS, np.float32)
    x[rng.choice(N_INPUTS, 82, replace=False)] = 1.0
    src = rng.integers(0, N_INPUTS, (n, 256)).astype(np.uint16)
    w8 = rng.integers(0, 5, (n, 256)).astype(np.int8)
    words = np.full((n, 256), PRISTINE, np.uint32)
    drive = np.empty((n, 8), np.float64)
    bq = 0.1 * 31
    t_s = med(lambda: cpu_stored(src, w8, x, drive))
    t_p = med(lambda: cpu_packed(words, x, drive, 0))
    t_g = med(lambda: cpu_procedural(n, x, drive, 0, bq))
    syn = n * 256
    print(f"\n== Regenerating the genome on the fly: CPU (numba, 3 threads), {n:,} neurons x 256 synapses")
    for name, t, b in [("stored target+weight (3 B/syn)", t_s, 3), ("packed word + wiring hash (4 B/syn)", t_p, 4),
                       ("procedural: hash only (0 B/syn)", t_g, 0)]:
        print(f"   {name:<38}{t * 1e3:8.1f} ms  {t / syn * 1e9:5.2f} ns/synapse  {syn / t / 1e9:5.2f} G syn/s"
              f"  reads {b * syn / 2**20:5.0f} MiB  ({t / t_s:.2f}x stored)")
    return t_s, t_p, t_g


def gpu_part(n=262_144):
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        print(f"   (GPU part skipped: {e})")
        return None
    cp.get_default_memory_pool().set_limit(size=1 * 2**30)
    mod = cp.RawModule(code=CUDA_SRC)
    g_stored, g_packed, g_proc = (mod.get_function(k) for k in ("g_stored", "g_packed", "g_procedural"))
    rng = np.random.default_rng(0)
    xs = np.zeros(N_INPUTS, np.float32)
    xs[rng.choice(N_INPUTS, 82, replace=False)] = 1.0
    x = cp.asarray(xs)
    n_syn = n * 256
    src = cp.random.randint(0, N_INPUTS, n_syn, dtype=cp.int32).astype(cp.uint16)
    w8 = cp.random.randint(0, 5, n_syn, dtype=cp.int32).astype(cp.int8)
    words = cp.full(n_syn, int(PRISTINE), cp.uint32)
    drive = cp.empty(n_syn // 32, cp.float32)
    thr, blk = 256, (n_syn + 255) // 256

    def timed(fn, args):
        fn((blk,), (thr,), args)
        cp.cuda.Device().synchronize()
        ts = []
        for _ in range(7):
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            fn((blk,), (thr,), args)
            e.record()
            e.synchronize()
            ts.append(cp.cuda.get_elapsed_time(s, e) / 1e3)
        return float(np.median(ts))

    t_s = timed(g_stored, (src, w8, x, drive, cp.int64(n_syn)))
    t_p = timed(g_packed, (words, x, drive, cp.int64(n_syn), cp.int64(0)))
    t_g = timed(g_proc, (x, drive, cp.int64(n_syn), cp.int64(0), cp.float32(3.1)))
    # check: procedural on GPU == procedural on CPU for a few neurons
    d_cpu = np.empty((64, 8), np.float64)
    cpu_procedural(64, xs, d_cpu, 0, 3.1)
    d_gpu = cp.asnumpy(drive[:64 * 8]).reshape(64, 8)
    ok = np.allclose(d_cpu, d_gpu, atol=1e-3)
    used = cp.get_default_memory_pool().used_bytes() / 2**20
    print(f"\n== Same on the GPU (RTX 4070 Laptop, CUDA via cupy), {n:,} neurons x 256 synapses "
          f"(VRAM used {used:.0f} MiB; GPU result == CPU result: {ok})")
    for name, t, b in [("stored target+weight (3 B/syn)", t_s, 3), ("packed word + wiring hash (4 B/syn)", t_p, 4),
                       ("procedural: hash only (0 B/syn)", t_g, 0)]:
        print(f"   {name:<38}{t * 1e3:8.2f} ms  {t / n_syn * 1e12:6.1f} ps/synapse  {n_syn / t / 1e9:6.1f} G syn/s"
              f"  ({t / t_s:.2f}x stored)")
    del src, w8, words, drive, x
    cp.get_default_memory_pool().free_all_blocks()
    return t_s, t_p, t_g


def main():
    cpu_part()
    gpu_part()


if __name__ == "__main__":
    main()

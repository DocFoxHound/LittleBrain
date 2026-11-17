"""
The night kernel on the GPU (cupy RawKernel, CUDA C), one thread per synapse word.

Same maths, same counter-based random numbers and the same float32/float64 promotions as the
Numba k_sleep, compiled without FMA contraction, so CPU and GPU nights are bit-identical
(checked in night.py). VRAM use: two 128 MiB chunk buffers (+ the CUDA context), capped at 1 GB.
"""

import numpy as np

_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
#define K1 0x9E3779B97F4A7C15ULL
#define K2 0xBF58476D1CE4E5B9ULL
#define K3 0x94D049BB133111EBULL
#define K4 0xD6E8FEB86659FD93ULL

__device__ __forceinline__ float urand(u64 seed, u64 nid, u64 j, u64 field) {
    u64 x = seed ^ (nid * K1) ^ (j * K2) ^ (field * K4);
    x ^= x >> 30; x *= K2; x ^= x >> 27; x *= K3; x ^= x >> 31;
    return (float)(x >> 40) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float decf(u32 s, float scale) { return ((float)s - 32.0f) * scale; }
__device__ __forceinline__ u32 encd(double v, float inv, float u) {
    double q = floor(v * (double)inv + (double)u);
    if (q > 31.0) q = 31.0;
    if (q < -31.0) q = -31.0;
    return (u32)((int)q + 32);
}
__device__ __forceinline__ double clampd(double v, float w) { return fmin(fmax(v, -(double)w), (double)w); }

extern "C" __global__ void night(u32 *w, long long n_words, long long id0, const float *P, u64 seed) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_words) return;
    const float *m = P, *n = P + 9;
    float keep0 = P[18], fast_scale = P[19], prune_eps = P[20], tag_mult = P[21], wmax = P[22];
    float scale = wmax / 31.0f, inv = 31.0f / wmax;
    u64 nid = (u64)(id0 + k / 256);
    u64 j = (u64)(k % 256);
    u32 x = w[k];
    u32 gen = x & 15u;
    float f = decf((x >> 4) & 63u, scale);
    float a0 = decf((x >> 10) & 63u, scale), a1 = decf((x >> 16) & 63u, scale), a2 = decf((x >> 22) & 63u, scale);
    u32 tq = (x >> 28) & 7u, used = x >> 31;
    float c0f = m[0] * a0 + m[1] * a1 + m[2] * a2;
    float c1 = m[3] * a0 + m[4] * a1 + m[5] * a2;
    float c2 = m[6] * a0 + m[7] * a1 + m[8] * a2;
    double tag = (double)(float)tq / 7.0;
    double c0 = clampd((double)c0f + (double)(f * fast_scale) * tag, wmax);
    double d0 = ((double)n[0] * c0 + (double)(n[1] * c1) + (double)(n[2] * c2)) * (double)keep0;
    double d1 = (double)n[3] * c0 + (double)(n[4] * c1) + (double)(n[5] * c2);
    double d2 = (double)n[6] * c0 + (double)(n[7] * c1) + (double)(n[8] * c2);
    double big = fmax(fabs(d0), fmax(fabs(d1), fabs(d2)));
    if (big < (double)prune_eps && used == 0u && tq == 0u) {
        gen = (gen + 1u) & 15u;
        d0 = (double)(0.1f * urand(seed, nid, j, 13));
        d1 = 0.0; d2 = 0.0;
    }
    u32 q0 = encd(clampd(d0, wmax), inv, urand(seed, nid, j, 10));
    u32 q1 = encd(clampd(d1, wmax), inv, urand(seed, nid, j, 11));
    u32 q2 = encd(clampd(d2, wmax), inv, urand(seed, nid, j, 12));
    double ntd = floor(tag * (double)tag_mult * 7.0 + (double)urand(seed, nid, j, 14));
    u32 nt = ntd > 7.0 ? 7u : (u32)ntd;
    w[k] = gen | (32u << 4) | (q0 << 10) | (q1 << 16) | (q2 << 22) | (nt << 28);
}
"""


class GPUNight:
    def __init__(self, chunk_bytes, Mpre, Ms, keep0, fast_scale, prune_eps, tag_mult, wmax, host_bufs=()):
        import cupy as cp
        self.cp = cp
        cp.get_default_memory_pool().set_limit(size=768 * 2**20)
        self.kern = cp.RawKernel(_SRC, "night", options=("--fmad=false",))
        self.dev = cp.empty(chunk_bytes // 4, cp.uint32)
        P = np.concatenate([np.asarray(Mpre, np.float32).ravel(), np.asarray(Ms, np.float32).ravel(),
                            np.array([keep0, fast_scale, prune_eps, tag_mult, wmax], np.float32)])
        self.P = cp.asarray(P)
        self.registered = []
        for b in host_bufs:     # pin the host ring buffers so copies run at full PCIe speed
            try:
                cp.cuda.runtime.hostRegister(b.ctypes.data, b.nbytes, 0)
                self.registered.append(b)
            except Exception as e:  # pragma: no cover
                print(f"  (could not pin host buffer: {e})")
        self.stream = cp.cuda.Stream(non_blocking=True)

    def run(self, words, id0, seed):
        """words: host (n, 256) uint32, consecutive neuron ids starting at id0. In place."""
        cp = self.cp
        n = words.size
        d = self.dev[:n]
        with self.stream:
            cp.cuda.runtime.memcpyAsync(d.data.ptr, words.ctypes.data, n * 4,
                                        cp.cuda.runtime.memcpyHostToDevice, self.stream.ptr)
            threads = 256
            self.kern(((n + threads - 1) // threads,), (threads,),
                      (d, np.int64(n), np.int64(id0), self.P, np.uint64(seed)))
            cp.cuda.runtime.memcpyAsync(words.ctypes.data, d.data.ptr, n * 4,
                                        cp.cuda.runtime.memcpyDeviceToHost, self.stream.ptr)
        self.stream.synchronize()

    def close(self):
        for b in self.registered:
            self.cp.cuda.runtime.hostUnregister(b.ctypes.data)
        self.registered = []

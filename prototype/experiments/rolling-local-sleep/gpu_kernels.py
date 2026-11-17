"""
GPU "awake" kernels (CuPy RawKernel, CUDA C): forward, modulate, cascade catch-up, and a GPU
sleep kernel (for the synchronous / micro-sleep comparisons).

Layout: one CUDA block per neuron, one warp per branch (slots = 32 = warp size), one thread per
synapse. Each thread unpacks its 32-bit word in registers, recomputes its input from the hash
wiring, and writes 32 bits back. No temporaries.
"""

import cupy as cp
import numpy as np

import common  # noqa: F401
from common import VRAM_CAP

cp.get_default_memory_pool().set_limit(VRAM_CAP)

_SRC = r'''
#include <cuda_fp16.h>
typedef unsigned int u32;
typedef unsigned long long u64;

// counter-based RNG, identical to csleep.c: lowbias32(seed + 4*index + k), two 16-bit uniforms each
__device__ __forceinline__ u32 h32(u32 x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float ulo(u32 r) { return ((float)(r & 0xFFFFu) + 0.5f) * (1.0f / 65536.0f); }
__device__ __forceinline__ float uhi(u32 r) { return ((float)(r >> 16) + 0.5f) * (1.0f / 65536.0f); }
__device__ __forceinline__ float dec(u32 s, float wq) { return ((float)(int)s - 32.f) * wq; }
__device__ __forceinline__ u32 enc(float v, float wmax, float qinv, float u) {
    v = fminf(fmaxf(v, -wmax), wmax);
    float q = floorf(v * qinv + u);
    q = fminf(fmaxf(q, -31.f), 31.f);
    return (u32)(q + 32.f);
}
__device__ __forceinline__ int target(u64 id, u64 slot, u32 gen, int n_inputs) {
    u64 h = ((id + 1ULL) * 0x9E3779B97F4A7C15ULL) ^ (slot * 0xBF58476D1CE4E5B9ULL)
          ^ ((u64)gen * 0x94D049BB133111EBULL);
    h ^= h >> 31;
    h *= 0xD6E8FEB86659FD93ULL;
    h ^= h >> 32;
    // same result as h % n_inputs; 64-bit modulo is slow on a GPU, so use a mask for powers of two
    return (n_inputs & (n_inputs - 1)) ? (int)(h % (u64)n_inputs) : (int)(h & (u64)(n_inputs - 1));
}

extern "C" __global__ void forward_k(
    u32* __restrict__ words, const float* __restrict__ x, int n_inputs, u64 first_id,
    const unsigned char* __restrict__ gate_bits, int gstride,
    float* __restrict__ threshold, __half* __restrict__ post_hist, float* __restrict__ rate_out,
    float m0, float m1, float m2, float fast_scale, float wq,
    float branch_thr, float spike_bonus, float ctx_mult, float slope,
    float thr_lr, float target_rate, float sign, int learn_trace, float used_rate)
{
    const int n = blockIdx.x, tid = threadIdx.x, b = tid >> 5, nb = blockDim.x >> 5;
    const size_t wi = (size_t)n * blockDim.x + tid;
    const u32 w = words[wi];
    const int src = target(first_id + (u64)n, (u64)tid, w & 15u, n_inputs);
    const float xs = x[src];
    const float c0 = m0 * dec((w >> 10) & 63u, wq) + m1 * dec((w >> 16) & 63u, wq)
                   + m2 * dec((w >> 22) & 63u, wq);
    float v = (dec((w >> 4) & 63u, wq) * fast_scale + c0) * xs;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int g = (gate_bits[(size_t)n * gstride + (b >> 3)] >> (7 - (b & 7))) & 1;
    __shared__ float s_out[32];
    __shared__ float s_rate;
    if ((tid & 31) == 0) {
        const float spike = v > branch_thr ? 1.f : 0.f;
        s_out[b] = g ? (fmaxf(v, 0.f) + spike_bonus * spike) : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
        float soma = 0.f;
        for (int i = 0; i < nb; ++i) soma += s_out[i];
        soma *= ctx_mult;
        const float r = 1.f / (1.f + expf(-slope * (soma - threshold[n])));
        s_rate = r;
        rate_out[n] = r * sign;
        if (learn_trace) {
            __half* ph = post_hist + (size_t)n * 4;
            ph[3] = ph[2]; ph[2] = ph[1]; ph[1] = ph[0];
            ph[0] = __float2half(0.5f + r);
            threshold[n] += thr_lr * (r - target_rate);
        }
    }
    if (learn_trace) {
        __syncthreads();
        if (xs > 0.f && g && s_rate >= used_rate && !(w >> 31)) words[wi] = w | 0x80000000u;
    }
}

extern "C" __global__ void modulate_k(
    u32* __restrict__ words, int n_inputs, u64 first_id,
    const unsigned char* __restrict__ gate_bits, int gstride,
    const __half* __restrict__ plast, const __half* __restrict__ post_hist,
    const __half* __restrict__ pre0, int pstride,
    float ew0, float ew1, float ew2, float ew3,
    float lr, float ach, float da, float ar_absda, float fast_scale,
    float wq, float wmax, float qinv, int oneshot,
    float p00, float p01, float p02, float p10, float p11, float p12, float p20, float p21, float p22,
    u32 seed)
{
    const int n = blockIdx.x, tid = threadIdx.x, b = tid >> 5;
    const size_t wi = (size_t)n * blockDim.x + tid;
    const u32 w = words[wi];
    const int src = target(first_id + (u64)n, (u64)tid, w & 15u, n_inputs);
    const __half* ph = post_hist + (size_t)n * 4;
    float e = ew0 * __half2float(pre0[src]) * __half2float(ph[0]);
    e += ew1 * __half2float(pre0[pstride + src]) * __half2float(ph[1]);
    e += ew2 * __half2float(pre0[2 * pstride + src]) * __half2float(ph[2]);
    e += ew3 * __half2float(pre0[3 * pstride + src]) * __half2float(ph[3]);
    const int g = (gate_bits[(size_t)n * gstride + (b >> 3)] >> (7 - (b & 7))) & 1;
    if (!g) e = 0.f;
    const float delta = lr * __half2float(plast[n]) * ach * da * e;
    const u32 ctr = (u32)wi * 4u + seed;
    const u32 r = h32(ctr);
    const float fast = fminf(fmaxf(dec((w >> 4) & 63u, wq) * fast_scale + delta, -wmax), wmax);
    const u32 fq = enc(fast, wmax, qinv, ulo(r));
    const float tag = fminf(fmaxf((float)((w >> 28) & 7u) / 7.f + ar_absda * e, 0.f), 1.f);
    const u32 tq = (u32)floorf(tag * 7.f + uhi(r)) & 7u;
    u32 nw = (w & ~((63u << 4) | (7u << 28))) | (fq << 4) | (tq << 28);
    if (oneshot) {  // BTSP-like write: catch up the cascade clock, then c0 += 0.5 * delta
        const u32 r1 = h32(ctr + 1u), r2 = h32(ctr + 2u);
        const float a0 = dec((w >> 10) & 63u, wq), a1 = dec((w >> 16) & 63u, wq), a2 = dec((w >> 22) & 63u, wq);
        u32 q0 = enc(p00 * a0 + p01 * a1 + p02 * a2, wmax, qinv, ulo(r1));
        const u32 q1 = enc(p10 * a0 + p11 * a1 + p12 * a2, wmax, qinv, uhi(r1));
        const u32 q2 = enc(p20 * a0 + p21 * a1 + p22 * a2, wmax, qinv, ulo(r2));
        q0 = enc(dec(q0, wq) + 0.5f * delta, wmax, qinv, uhi(r2));
        nw = (nw & ~(0x3FFFFu << 10)) | (q0 << 10) | (q1 << 16) | (q2 << 22);
    }
    words[wi] = nw;
}

extern "C" __global__ void catchup_k(
    u32* __restrict__ words, long long nwords, float wq, float wmax, float qinv,
    float p00, float p01, float p02, float p10, float p11, float p12, float p20, float p21, float p22,
    u32 seed)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords) return;
    const u32 w = words[i];
    const u32 ctr = (u32)i * 4u + seed;
    const u32 r0 = h32(ctr), r1 = h32(ctr + 1u);
    const float a0 = dec((w >> 10) & 63u, wq), a1 = dec((w >> 16) & 63u, wq), a2 = dec((w >> 22) & 63u, wq);
    const u32 q0 = enc(p00 * a0 + p01 * a1 + p02 * a2, wmax, qinv, ulo(r0));
    const u32 q1 = enc(p10 * a0 + p11 * a1 + p12 * a2, wmax, qinv, uhi(r0));
    const u32 q2 = enc(p20 * a0 + p21 * a1 + p22 * a2, wmax, qinv, ulo(r1));
    words[i] = (w & ~(0x3FFFFu << 10)) | (q0 << 10) | (q1 << 16) | (q2 << 22);
}

extern "C" __global__ void sleep_k(
    u32* __restrict__ words, long long nwords, int do_catch,
    float p00, float p01, float p02, float p10, float p11, float p12, float p20, float p21, float p22,
    float s00, float s01, float s02, float s10, float s11, float s12, float s20, float s21, float s22,
    float fast_scale, float wq, float wmax, float qinv, float keep, float prune_eps, float tag_decay,
    u32 seed, unsigned int* __restrict__ pruned)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords) return;
    const u32 w = words[i];
    const u32 ctr = (u32)i * 4u + seed;
    const u32 r0 = h32(ctr), r1 = h32(ctr + 1u), r2 = h32(ctr + 2u), r3 = h32(ctr + 3u);
    u32 gen = w & 15u;
    const u32 tagi = (w >> 28) & 7u, used = w >> 31;
    float a0 = dec((w >> 10) & 63u, wq), a1 = dec((w >> 16) & 63u, wq), a2 = dec((w >> 22) & 63u, wq);
    if (do_catch) {
        const float b0 = p00 * a0 + p01 * a1 + p02 * a2;
        const float b1 = p10 * a0 + p11 * a1 + p12 * a2;
        const float b2 = p20 * a0 + p21 * a1 + p22 * a2;
        a0 = dec(enc(b0, wmax, qinv, ulo(r2)), wq);
        a1 = dec(enc(b1, wmax, qinv, uhi(r2)), wq);
        a2 = dec(enc(b2, wmax, qinv, ulo(r3)), wq);
    }
    const float tag = (float)tagi / 7.f;
    const float c0 = fminf(fmaxf(a0 + dec((w >> 4) & 63u, wq) * fast_scale * tag, -wmax), wmax);
    float d0 = (s00 * c0 + s01 * a1 + s02 * a2) * keep;
    float d1 = s10 * c0 + s11 * a1 + s12 * a2;
    float d2 = s20 * c0 + s21 * a1 + s22 * a2;
    float m = fabsf(d0); m = fabsf(d1) > m ? fabsf(d1) : m; m = fabsf(d2) > m ? fabsf(d2) : m;
    const bool weak = m < prune_eps && used == 0u && tagi == 0u;
    if (weak) { gen = (gen + 1u) & 15u; d0 = uhi(r3) * 0.1f; d1 = 0.f; d2 = 0.f; }
    const unsigned int ballot = __ballot_sync(__activemask(), weak);
    if ((threadIdx.x & 31) == __ffs(__activemask()) - 1 && ballot) atomicAdd(pruned, __popc(ballot));
    const u32 q0 = enc(d0, wmax, qinv, ulo(r0)), q1 = enc(d1, wmax, qinv, uhi(r0)),
              q2 = enc(d2, wmax, qinv, ulo(r1));
    const u32 tq = (u32)floorf((float)tagi * tag_decay + uhi(r1)) & 7u;
    words[i] = gen | (32u << 4) | (q0 << 10) | (q1 << 16) | (q2 << 22) | (tq << 28);
}
'''

_mod = cp.RawModule(code=_SRC, options=("-std=c++14", "--fmad=false"))
forward_k = _mod.get_function("forward_k")
modulate_k = _mod.get_function("modulate_k")
catchup_k = _mod.get_function("catchup_k")
sleep_k = _mod.get_function("sleep_k")

f32 = np.float32
i32 = np.int32


def mat9(m):
    return tuple(f32(v) for v in np.asarray(m, np.float32).reshape(-1))

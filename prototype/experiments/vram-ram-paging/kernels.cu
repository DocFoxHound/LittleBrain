// CUDA kernels for the packed workhorse neuron (prototype/packed_neuron.py), written so that
// a kernel never builds a full-size temporary: one warp = one neuron, one lane = one synapse
// slot of a branch (S = 32 slots = one warp), and each lane keeps its NB packed words in
// registers.
//
// Every kernel that touches synapses takes a *pointer table*: for each work item (a block of
// `bsize` neurons) a device-usable address of that block's words. The address can point
// into a VRAM cache slot, into mapped pinned host RAM (zero-copy over PCIe) or into CUDA
// managed memory. One kernel serves every storage strategy.
//
// Constants (NB, DEC, ENCF, ...) are injected as #defines by gpu_kernels.py from a NeuronType.

#include <cuda_fp16.h>

typedef unsigned int u32;
typedef unsigned long long u64;

#define WPN (NB * 32)                         // packed words per neuron
#define FMASK(bits) ((1u << (bits)) - 1u)
#define GETF_(w, off, bits) (((w) >> (off)) & FMASK(bits))
#define PUTF_(w, off, bits, v) (((w) & ~(FMASK(bits) << (off))) | ((((u32)(v)) & FMASK(bits)) << (off)))
// forwarding so that F_xxx ("offset, bits") expands into two arguments
#define GETF(...) GETF_(__VA_ARGS__)
#define PUTF(...) PUTF_(__VA_ARGS__)
// bit layout (same as packed_neuron.FIELDS)
#define F_GEN 0, 4
#define F_FAST 4, 6
#define F_C0 10, 6
#define F_C1 16, 6
#define F_C2 22, 6
#define F_TAG 28, 3
#define USED_BIT 0x80000000u

__device__ __forceinline__ float dec6(u32 s) { return __fmul_rn((float)s - 32.0f, DEC); }

__device__ __forceinline__ u32 enc6(float v, float u) {
    // numpy: clip(floor(v * Q/w_max + r), -Q, Q) + 32, all float32
    float q = floorf(__fadd_rn(__fmul_rn(v, ENCF), u));
    q = fminf(fmaxf(q, -31.0f), 31.0f);
    return (u32)((int)q + 32);
}

__device__ __forceinline__ float clampw(float v) { return fminf(fmaxf(v, -WMAX), WMAX); }

// Same hash as packed_neuron._targets, restricted to the neuron's column of the input space.
__device__ __forceinline__ u32 syn_target(u32 id, u32 slot, u32 gen, u32 col_size, u32 col_mask) {
    u64 h = ((u64)id + 1ull) * 0x9E3779B97F4A7C15ull;
    h ^= (u64)slot * 0xBF58476D1CE4E5B9ull;
    h ^= (u64)gen * 0x94D049BB133111EBull;
    h ^= h >> 31;
    h *= 0xD6E8FEB86659FD93ull;
    h ^= h >> 32;
    return col_mask ? (u32)(h & (u64)col_mask) : (u32)(h % (u64)col_size);
}

// Counter-based random numbers for stochastic rounding (no RNG state to store).
__device__ __forceinline__ u32 mix32(u32 x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float urand(u32 seed, u32 idx, u32 salt) {
    u32 h = mix32(idx ^ mix32(seed * 0x9E3779B9u + salt * 0x85EBCA6Bu));
    h = mix32(h ^ (salt * 0xC2B2AE35u));
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ u32 warp_sum_u(u32 v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// packbits order: branch 0 is the most significant bit of the byte
__device__ __forceinline__ float gate_of(unsigned gb, int br) { return ((gb >> (7 - br)) & 1u) ? 1.0f : 0.0f; }

// Locate this warp's neuron. Returns false if the warp has nothing to do.
__device__ __forceinline__ bool locate(const u64* wptr, const int* wblk, int n_work, int bsize,
                                       long long n_neurons, int& wi, int& b, long long& p, u32*& words) {
    long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    wi = (int)(warp / bsize);
    if (wi >= n_work) return false;
    int j = (int)(warp - (long long)wi * bsize);
    b = wblk[wi];
    p = (long long)b * bsize + j;
    if (p >= n_neurons) return false;
    words = (u32*)(wptr[wi]) + (size_t)j * WPN;
    return true;
}

// ---------------------------------------------------------------------------------------
// Per-block lazy clocks -> coefficients. For each listed block: M = flow^pending (3x3, from
// a table of flow^(2^k)), and fast_scale = fast_decay^(T - fast_clock). Stored as 16 floats.
extern "C" __global__ void prep_kernel(const int* __restrict__ blks, int n, const int* __restrict__ flow_clock,
                                       const int* __restrict__ fast_clock, int T,
                                       const double* __restrict__ mpow, double log_decay,
                                       float* __restrict__ bcoef) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = blks[i];
    int pend = T - flow_clock[b];
    double M[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int k = 0; k < 31; ++k) {
        if ((pend >> k) & 1) {
            const double* P = mpow + 9 * k;
            double R[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    R[3 * r + c] = M[3 * r] * P[c] + M[3 * r + 1] * P[3 + c] + M[3 * r + 2] * P[6 + c];
            for (int q = 0; q < 9; ++q) M[q] = R[q];
        }
    }
    float* out = bcoef + (size_t)b * 16;
    for (int q = 0; q < 9; ++q) out[q] = (float)M[q];
    out[9] = (float)exp((double)(T - fast_clock[b]) * log_decay);
}

// ---------------------------------------------------------------------------------------
// forward: drive per branch, dendritic spike, soma, rate; with learn_trace also the "used"
// bits, the homeostatic threshold and this neuron's firing history (ring buffer).
extern "C" __global__ void forward_kernel(
    const u64* __restrict__ wptr, const int* __restrict__ wblk, const int* __restrict__ wslot,
    int n_work, int bsize, long long n_neurons,
    const float* __restrict__ bcoef, const u32* __restrict__ ids, const unsigned char* __restrict__ gate_bits,
    float* __restrict__ thr, __half* __restrict__ post_hist, int ring,
    const float* __restrict__ x, u32 npc, u32 col_size, u32 col_mask,
    float ctx_mult, int learn_trace,
    float* __restrict__ rate_out, u32* __restrict__ dirty_slot, u64* __restrict__ counters) {
    int wi, b; long long p; u32* words;
    if (!locate(wptr, wblk, n_work, bsize, n_neurons, wi, b, p, words)) return;
    const int lane = threadIdx.x & 31;
    const float* cf = bcoef + (size_t)b * 16;
    const float m0 = cf[0], m1 = cf[1], m2 = cf[2], fs = cf[9];
    const u32 id = ids[p];
    const u32 colbase = (id / npc) * col_size;
    const unsigned gb = gate_bits[p];

    u32 w[NB];
#pragma unroll
    for (int br = 0; br < NB; ++br) w[br] = words[br * 32 + lane];

    float xs[NB];
    float soma = 0.0f;
#pragma unroll
    for (int br = 0; br < NB; ++br) {
        const u32 ww = w[br];
        const float xv = x[colbase + syn_target(id, br * 32 + lane, GETF(ww, F_GEN), col_size, col_mask)];
        xs[br] = xv;
        const float c0 = m0 * dec6(GETF(ww, F_C0)) + m1 * dec6(GETF(ww, F_C1)) + m2 * dec6(GETF(ww, F_C2));
        const float wt = __fadd_rn(__fmul_rn(dec6(GETF(ww, F_FAST)), fs), c0);
        const float drive = warp_sum(__fmul_rn(wt, xv));
        const float spike = drive > BRANCH_THR ? 1.0f : 0.0f;
        soma = __fadd_rn(soma, __fmul_rn(__fadd_rn(fmaxf(drive, 0.0f), SPIKE_BONUS * spike), gate_of(gb, br)));
    }
    soma = __fmul_rn(soma, ctx_mult);
    const float t = thr[p];
    const float rate = 1.0f / (1.0f + expf(__fmul_rn(-SOMA_SLOPE, soma - t)));

    if (learn_trace) {
        u32 changed = 0, lines = 0;
        if (rate >= USED_RATE) {  // warp-uniform
#pragma unroll
            for (int br = 0; br < NB; ++br) {
                const bool ch = xs[br] > 0.0f && gate_of(gb, br) > 0.0f && !(w[br] & USED_BIT);
                // write the whole 128-byte line if any lane changed: partial-line writes into host
                // RAM over PCIe (zero-copy) cost ~10-40x more than full lines
                if (__any_sync(0xffffffffu, ch)) {
                    words[br * 32 + lane] = ch ? (w[br] | USED_BIT) : w[br];
                    lines += (lane == 0);
                }
                changed += ch;
            }
        }
        changed = warp_sum_u(changed);
        if (lane == 0) {
            thr[p] = __fadd_rn(t, __fmul_rn(THR_LR, __fsub_rn(rate, TARGET_RATE)));
            post_hist[p * 4 + ring] = __float2half_rn(0.5f + rate);
            if (changed) {
                if (dirty_slot) dirty_slot[wslot[wi]] = 1u;
                atomicAdd(counters, (u64)changed);
                atomicAdd(counters + 2, (u64)lines);  // 128-byte lines written
            }
        }
    }
    if (lane == 0) rate_out[p] = rate * SIGN;
}

// Neurons whose block received no input this step: drive is exactly 0, so the rate depends
// only on the threshold. No synapse word is read. One thread per neuron.
extern "C" __global__ void quiet_kernel(const unsigned char* __restrict__ blk_active, int bsize, long long n,
                                        float* __restrict__ thr, __half* __restrict__ post_hist, int ring,
                                        int learn_trace, float* __restrict__ rate_out) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    if (blk_active[p / bsize]) return;
    const float t = thr[p];
    const float rate = 1.0f / (1.0f + expf(__fmul_rn(-SOMA_SLOPE, 0.0f - t)));
    if (learn_trace) {
        thr[p] = __fadd_rn(t, __fmul_rn(THR_LR, __fsub_rn(rate, TARGET_RATE)));
        post_hist[p * 4 + ring] = __float2half_rn(0.5f + rate);
    }
    rate_out[p] = rate * SIGN;
}

// ---------------------------------------------------------------------------------------
// modulate: three-factor learning with the eligibility rebuilt from the last 4 steps.
extern "C" __global__ void modulate_kernel(
    const u64* __restrict__ wptr, const int* __restrict__ wblk, const int* __restrict__ wslot,
    int n_work, int bsize, long long n_neurons,
    const float* __restrict__ bcoef, const u32* __restrict__ ids, const unsigned char* __restrict__ gate_bits,
    const __half* __restrict__ plast, const __half* __restrict__ post_hist, const __half* __restrict__ pre_hist,
    long long n_inputs, int ring, u32 npc, u32 col_size, u32 col_mask,
    float ach_h, float dop_h, float tag_gain, int oneshot, u32 seed,
    u32* __restrict__ dirty_slot, u64* __restrict__ counters) {
    int wi, b; long long p; u32* words;
    if (!locate(wptr, wblk, n_work, bsize, n_neurons, wi, b, p, words)) return;
    const int lane = threadIdx.x & 31;
    const float* cf = bcoef + (size_t)b * 16;
    const float fs = cf[9];
    const u32 id = ids[p];
    const u32 colbase = (id / npc) * col_size;
    const unsigned gb = gate_bits[p];

    // numpy evaluates lr * plasticity * ach * dopamine in float16 (NEP 50 weak scalars)
    float c = __half2float(__float2half_rn(LR_H * __half2float(plast[p])));
    c = __half2float(__float2half_rn(c * ach_h));
    c = __half2float(__float2half_rn(c * dop_h));

    float ph[4];
    int rk[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        rk[k] = (ring - k) & 3;
        ph[k] = __half2float(post_hist[p * 4 + rk[k]]);
    }
    const float ew[4] = {EW0, EW1, EW2, EW3};

    u32 w[NB];  // issue all loads first (more bytes in flight: matters over PCIe)
#pragma unroll
    for (int br = 0; br < NB; ++br) w[br] = words[br * 32 + lane];

    u32 changed = 0, lines = 0;
#pragma unroll
    for (int br = 0; br < NB; ++br) {
        const u32 ww = w[br];
        const u32 slot = br * 32 + lane;
        const u32 src = colbase + syn_target(id, slot, GETF(ww, F_GEN), col_size, col_mask);
        float e = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            e = __fadd_rn(e, __fmul_rn(__fmul_rn(ew[k], __half2float(pre_hist[(size_t)rk[k] * n_inputs + src])), ph[k]));
        const float g = gate_of(gb, br);
        e = e * g;
        const float delta = __fmul_rn(c, e) * g;
        const u32 sidx = (u32)(p * WPN + slot);

        u32 nw = ww;
        const float fast = clampw(__fadd_rn(__fmul_rn(dec6(GETF(ww, F_FAST)), fs), delta));
        nw = PUTF(nw, F_FAST, enc6(fast, urand(seed, sidx, 1)));
        // tag in units of 1/7: clip(tag/7 + gain*elig, 0, 1) * 7, then stochastic rounding
        const float tg7 = fminf(fmaxf((float)GETF(ww, F_TAG) + 7.0f * tag_gain * e, 0.0f), 7.0f);
        nw = PUTF(nw, F_TAG, (u32)floorf(tg7 + urand(seed, sidx, 2)));
        if (oneshot) {  // BTSP-like: catch up the cascade and write half of delta into the surface
            const float l0 = dec6(GETF(ww, F_C0)), l1 = dec6(GETF(ww, F_C1)), l2 = dec6(GETF(ww, F_C2));
            float c0 = cf[0] * l0 + cf[1] * l1 + cf[2] * l2;
            const float c1 = cf[3] * l0 + cf[4] * l1 + cf[5] * l2;
            const float c2 = cf[6] * l0 + cf[7] * l1 + cf[8] * l2;
            c0 += 0.5f * delta;
            nw = PUTF(nw, F_C0, enc6(clampw(c0), urand(seed, sidx, 3)));
            nw = PUTF(nw, F_C1, enc6(clampw(c1), urand(seed, sidx, 4)));
            nw = PUTF(nw, F_C2, enc6(clampw(c2), urand(seed, sidx, 5)));
        }
        if (__any_sync(0xffffffffu, nw != ww)) {  // full-line writes
            words[br * 32 + lane] = nw;
            lines += (lane == 0);
        }
        changed += (nw != ww);
    }
    changed = warp_sum_u(changed);
    if (lane == 0 && changed) {
        if (dirty_slot) dirty_slot[wslot[wi]] = 1u;
        atomicAdd(counters, (u64)changed);
        atomicAdd(counters + 2, (u64)lines);
    }
}

// ---------------------------------------------------------------------------------------
// sleep: catch up the cascade, consolidate tagged fast weights, run the night's flow,
// downscale, prune & regrow, clear fast/used, fade tags.
extern "C" __global__ void sleep_kernel(
    const u64* __restrict__ wptr, const int* __restrict__ wblk, const int* __restrict__ wslot,
    int n_work, int bsize, long long n_neurons, const float* __restrict__ bcoef,
    float s00, float s01, float s02, float s10, float s11, float s12, float s20, float s21, float s22,
    u32 seed, u32* __restrict__ dirty_slot, u64* __restrict__ counters) {
    int wi, b; long long p; u32* words;
    if (!locate(wptr, wblk, n_work, bsize, n_neurons, wi, b, p, words)) return;
    const int lane = threadIdx.x & 31;
    const float* cf = bcoef + (size_t)b * 16;
    const float fs = cf[9];
    u32 pruned = 0;
    u32 w[NB];
#pragma unroll
    for (int br = 0; br < NB; ++br) w[br] = words[br * 32 + lane];
#pragma unroll
    for (int br = 0; br < NB; ++br) {
        const u32 ww = w[br];
        const u32 sidx = (u32)(p * WPN + br * 32 + lane);
        const float tag = (float)GETF(ww, F_TAG) * (1.0f / 7.0f);
        const float l0 = dec6(GETF(ww, F_C0)), l1 = dec6(GETF(ww, F_C1)), l2 = dec6(GETF(ww, F_C2));
        float a0 = cf[0] * l0 + cf[1] * l1 + cf[2] * l2;      // catch up pending awake flow
        float a1 = cf[3] * l0 + cf[4] * l1 + cf[5] * l2;
        float a2 = cf[6] * l0 + cf[7] * l1 + cf[8] * l2;
        a0 = clampw(a0 + dec6(GETF(ww, F_FAST)) * fs * tag);  // 1. consolidate
        float c0 = s00 * a0 + s01 * a1 + s02 * a2;             // 2. the night's flow
        float c1 = s10 * a0 + s11 * a1 + s12 * a2;
        float c2 = s20 * a0 + s21 * a1 + s22 * a2;
        c0 *= (1.0f - DOWNSCALE);                               // 3. downscale
        u32 gen = GETF(ww, F_GEN);
        const bool weak = fmaxf(fabsf(c0), fmaxf(fabsf(c1), fabsf(c2))) < PRUNE_EPS &&
                          !(ww & USED_BIT) && tag < 0.05f;     // 4. prune & regrow
        if (weak) {
            gen = (gen + 1u) & 15u;
            c1 = 0.0f; c2 = 0.0f;
            c0 = 0.1f * urand(seed, sidx, 6);
            ++pruned;
        }
        u32 nw = 0;
        nw = PUTF(nw, F_GEN, gen);
        nw = PUTF(nw, F_FAST, 32u);                             // 5. clear fast, fade tag, clear used
        nw = PUTF(nw, F_C0, enc6(clampw(c0), urand(seed, sidx, 3)));
        nw = PUTF(nw, F_C1, enc6(clampw(c1), urand(seed, sidx, 4)));
        nw = PUTF(nw, F_C2, enc6(clampw(c2), urand(seed, sidx, 5)));
        nw = PUTF(nw, F_TAG, (u32)floorf((float)GETF(ww, F_TAG) * TAG_DECAY + urand(seed, sidx, 2)));
        words[br * 32 + lane] = nw;
    }
    pruned = warp_sum_u(pruned);
    if (lane == 0) {
        if (dirty_slot) dirty_slot[wslot[wi]] = 1u;
        if (pruned) atomicAdd(counters + 1, (u64)pruned);
        atomicAdd(counters + 2, (u64)NB);
    }
}

// ---------------------------------------------------------------------------------------
// Fresh synapses: c0 ~ uniform(0, 0.1), everything else zero.
extern "C" __global__ void init_kernel(u32* __restrict__ words, long long n_words, long long first_word, u32 seed) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_words) return;
    const u32 idx = (u32)(first_word + i);
    const float u = 0.1f * urand(seed, idx, 7);
    const u32 c0 = enc6(u, urand(seed, idx, 8));
    words[i] = (32u << 4) | (c0 << 10) | (32u << 16) | (32u << 22);
}

// Structured input: set (mode 1) or clear (mode 0) the inputs of the listed columns,
// in x and in the newest row of the shared input history.
extern "C" __global__ void input_kernel(const int* __restrict__ cols, int n_cols, u32 col_size,
                                        float* __restrict__ x, __half* __restrict__ pre_row,
                                        int mode, u32 seed, float p_in) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n_cols * col_size) return;
    const u32 idx = (u32)cols[i / col_size] * col_size + (u32)(i % col_size);
    if (mode == 0) {
        if (x) x[idx] = 0.0f;
        if (pre_row) pre_row[idx] = __float2half_rn(0.0f);
    } else {
        const float v = urand(seed, idx, 11) < p_in ? 1.0f : 0.0f;
        if (x) x[idx] = v;
        if (pre_row) pre_row[idx] = __float2half_rn(v);
    }
}

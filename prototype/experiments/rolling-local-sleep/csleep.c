/*
 * CPU sleep kernel in C (built as a Python extension that releases the GIL).
 * Same math, same operation order and same counter-based RNG as sleep_k in gpu_kernels.py,
 * so for a given seed the CPU and GPU nights are bit-identical (FMA contraction is off in both).
 *
 * Build with ../build_native.py csleep.
 */
#include <stdint.h>
#include "csleep.h"
#include <math.h>

static inline uint32_t h32(uint32_t x) {            /* "lowbias32" integer hash */
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static inline float ulo(uint32_t r) { return ((float)(r & 0xFFFFu) + 0.5f) * (1.0f / 65536.0f); }
static inline float uhi(uint32_t r) { return ((float)(r >> 16) + 0.5f) * (1.0f / 65536.0f); }
static inline float clampf(float v, float lo, float hi) { v = v < lo ? lo : v; return v > hi ? hi : v; }
static inline uint32_t enc(float v, float wmax, float qinv, float u) {
    float q = floorf(clampf(v, -wmax, wmax) * qinv + u);
    return (uint32_t)((int32_t)clampf(q, -31.f, 31.f) + 32);
}
static inline float dec(uint32_t s, float wq) { return ((float)(int32_t)s - 32.f) * wq; }
static inline float fabs_(float v) { return v < 0.f ? -v : v; }

/* One synapse. do_catch is a compile-time constant at each call site, so the branch disappears. */
static inline __attribute__((always_inline)) uint32_t night(
    uint32_t w, uint32_t ctr, const int do_catch,
    float p00, float p01, float p02, float p10, float p11, float p12, float p20, float p21, float p22,
    float s00, float s01, float s02, float s10, float s11, float s12, float s20, float s21, float s22,
    float fast_scale, float wmax, float wq, float qinv, float keep, float prune_eps, float tag_decay,
    int32_t *weak_out)
{
    const uint32_t r0 = h32(ctr), r1 = h32(ctr + 1u), r2 = h32(ctr + 2u), r3 = h32(ctr + 3u);
    uint32_t gen = w & 15u;
    const uint32_t tagi = (w >> 28) & 7u, used = w >> 31;
    float a0 = dec((w >> 10) & 63u, wq), a1 = dec((w >> 16) & 63u, wq), a2 = dec((w >> 22) & 63u, wq);
    if (do_catch) {      /* lazy cascade clock catch-up, re-quantized like _catch_up() */
        const float b0 = p00 * a0 + p01 * a1 + p02 * a2;
        const float b1 = p10 * a0 + p11 * a1 + p12 * a2;
        const float b2 = p20 * a0 + p21 * a1 + p22 * a2;
        a0 = dec(enc(b0, wmax, qinv, ulo(r2)), wq);
        a1 = dec(enc(b1, wmax, qinv, uhi(r2)), wq);
        a2 = dec(enc(b2, wmax, qinv, ulo(r3)), wq);
    }
    const float tag = (float)tagi / 7.f;
    /* 1. consolidate tagged fast weights; 2. the night's flow; 3. downscale the surface */
    const float c0 = clampf(a0 + dec((w >> 4) & 63u, wq) * fast_scale * tag, -wmax, wmax);
    float d0 = (s00 * c0 + s01 * a1 + s02 * a2) * keep;
    float d1 = s10 * c0 + s11 * a1 + s12 * a2;
    float d2 = s20 * c0 + s21 * a1 + s22 * a2;
    /* 4. prune and regrow: weak at every level, unused, untagged */
    float m = fabs_(d0); m = fabs_(d1) > m ? fabs_(d1) : m; m = fabs_(d2) > m ? fabs_(d2) : m;
    const int32_t weak = (m < prune_eps) & (used == 0u) & (tagi == 0u);
    d0 = weak ? uhi(r3) * 0.1f : d0;
    d1 = weak ? 0.f : d1;
    d2 = weak ? 0.f : d2;
    gen = weak ? ((gen + 1u) & 15u) : gen;
    *weak_out = weak;
    const uint32_t q0 = enc(d0, wmax, qinv, ulo(r0)), q1 = enc(d1, wmax, qinv, uhi(r0)),
                   q2 = enc(d2, wmax, qinv, ulo(r1));
    /* 5. clear fast and used, fade the tag */
    const uint32_t tq = (uint32_t)(int32_t)floorf((float)tagi * tag_decay + uhi(r1)) & 7u;
    return gen | (32u << 4) | (q0 << 10) | (q1 << 16) | (q2 << 22) | (tq << 28);
}

#define NIGHT_LOOP(DO_CATCH)                                                                  \
    _Pragma("omp parallel for schedule(static) num_threads(n_threads)")                       \
    for (int64_t n = 0; n < n_neurons; ++n) {                                                 \
        uint32_t *restrict wp = words + n * syn;                                              \
        const uint32_t ctr0 = (uint32_t)(base_index + n * (int64_t)syn) * 4u + seed;          \
        int32_t cnt = 0;                                                                      \
        _Pragma("omp simd reduction(+:cnt)")                                                  \
        for (int32_t j = 0; j < syn; ++j) {                                                   \
            int32_t weak;                                                                     \
            wp[j] = night(wp[j], ctr0 + 4u * (uint32_t)j, DO_CATCH,                           \
                          p00, p01, p02, p10, p11, p12, p20, p21, p22,                        \
                          s00, s01, s02, s10, s11, s12, s20, s21, s22,                        \
                          fast_scale, wmax, wq, qinv, keep, prune_eps, tag_decay, &weak);     \
            cnt += weak;                                                                      \
        }                                                                                     \
        pruned[n] = cnt;                                                                      \
    }

/* P, S: row-major 3x3. words: n_neurons * syn uint32, modified in place. */
void sleep_words(uint32_t *restrict words, int64_t n_neurons, int32_t syn,
                 const float *restrict P, int32_t do_catch, const float *restrict S,
                 float fast_scale, float wmax, float keep, float prune_eps, float tag_decay,
                 uint32_t seed, int64_t base_index, int32_t *restrict pruned, int32_t n_threads)
{
    const float wq = wmax / 31.f, qinv = 31.f / wmax;
    const float p00 = P[0], p01 = P[1], p02 = P[2], p10 = P[3], p11 = P[4], p12 = P[5],
                p20 = P[6], p21 = P[7], p22 = P[8];
    const float s00 = S[0], s01 = S[1], s02 = S[2], s10 = S[3], s11 = S[4], s12 = S[5],
                s20 = S[6], s21 = S[7], s22 = S[8];
    if (do_catch) { NIGHT_LOOP(1) } else { NIGHT_LOOP(0) }
}

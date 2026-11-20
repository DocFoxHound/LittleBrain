/*
 * Strategy 3: bit-sliced ("SWAR") branches with a local receptive-field bitmap.
 *
 * One branch = 32 synapses = one 32x32 bit matrix stored as 32 uint32 "planes" (128 bytes):
 *   plane k (k = 0..31) holds bit k of all 32 synapse words, synapse s at bit s.
 *   Stored as rec[k]; rec is a 32x32 bit matrix and a bit transpose turns it into 32 baseline words.
 * Planes 4..31 are exactly the baseline word fields (fast, c0, c1, c2, tag, used).
 * Planes 0..3 (the baseline's 4 gen bits x 32 synapses = 128 bits) hold the WIRING: a bitmap over
 * a window of 128 consecutive inputs (two 64-bit words of the input bitset, window start is a hash
 * of (neuron, branch)). Exactly 32 bits are set; synapse slot s = the s-th set bit.
 * So the record is exactly 4 bytes per synapse, like the baseline, with no hash per synapse.
 *
 * forward : active mask = PEXT(input bits, wiring bitmap); drive = sum_k 2^k popcount(plane_k & mask)
 * modulate/catch-up/sleep : transpose the 32x32 matrix back to 32 baseline words (AVX2 byte shuffle
 *           + movemask), run the baseline per-synapse update, transpose again.
 * Inputs must be binary spikes (0/1).
 *
 * Build with ../build_native.py.
 */
#include <stdint.h>
#include "bitslice.h"
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define C1 0x9E3779B97F4A7C15ULL
#define QF 31.0f

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31; return h;
}
static inline float u24(uint64_t h) { return (float)(h >> 40) * (1.0f / 16777216.0f); }
static inline float u24b(uint64_t h) { return (float)((h >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* 32x32 bit transpose with AVX2: out[k] bit s = in[s] bit k. Its own inverse.
 * Byte-shuffle so each 256-bit register holds byte k of all 32 rows, then 8 movemasks per register. */
static inline void tr32(const uint32_t *in, uint32_t *out) {
    const __m256i sh = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i pm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i r0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(in + 0)), sh), pm);
    __m256i r1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(in + 8)), sh), pm);
    __m256i r2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(in + 16)), sh), pm);
    __m256i r3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(in + 24)), sh), pm);
    __m256i t0 = _mm256_unpacklo_epi64(r0, r1), t1 = _mm256_unpackhi_epi64(r0, r1);
    __m256i t2 = _mm256_unpacklo_epi64(r2, r3), t3 = _mm256_unpackhi_epi64(r2, r3);
    __m256i B[4];
    B[0] = _mm256_permute2x128_si256(t0, t2, 0x20);
    B[1] = _mm256_permute2x128_si256(t1, t3, 0x20);
    B[2] = _mm256_permute2x128_si256(t0, t2, 0x31);
    B[3] = _mm256_permute2x128_si256(t1, t3, 0x31);
    for (int k = 0; k < 4; k++) {
        __m256i x = B[k];
        for (int i = 7; i >= 0; i--) { out[8 * k + i] = (uint32_t)_mm256_movemask_epi8(x); x = _mm256_add_epi8(x, x); }
    }
}

/* planes -> W[s] = word of synapse s (bits 0-3 carry wiring-plane bits; callers ignore them) */
static inline void to_words(const uint32_t *rec, uint32_t W[32]) { tr32(rec, W); }
static inline void from_words(const uint32_t W[32], uint32_t *rec) { tr32(W, rec); }
static inline uint64_t wire_lo(const uint32_t *r) { return (uint64_t)r[0] | ((uint64_t)r[1] << 32); }
static inline uint64_t wire_hi(const uint32_t *r) { return (uint64_t)r[2] | ((uint64_t)r[3] << 32); }
static inline void set_wire(uint32_t *r, uint64_t lo, uint64_t hi) {
    r[0] = (uint32_t)lo; r[1] = (uint32_t)(lo >> 32); r[2] = (uint32_t)hi; r[3] = (uint32_t)(hi >> 32);
}
static inline int64_t window(int64_t nid, int b, int64_t nW) {
    return (int64_t)(mix64(((uint64_t)nid << 8) + (uint64_t)b + 0x51ED27ULL) % (uint64_t)nW);
}
static inline uint32_t active_mask(const uint64_t *X, int64_t ws, int64_t nW, uint64_t lo, uint64_t hi) {
    int64_t ws1 = ws + 1 == nW ? 0 : ws + 1;
    return (uint32_t)(_pext_u64(X[ws], lo) | (_pext_u64(X[ws1], hi) << __builtin_popcountll(lo)));
}
static inline int gate_of(const uint8_t *gb, int gbytes, int64_t n, int b) {
    return (gb[n * gbytes + (b >> 3)] >> (7 - (b & 7))) & 1;
}
static inline float dec6(uint32_t w, int off, float wscale) { return ((float)((w >> off) & 63u) - 32.0f) * wscale; }
static inline uint32_t enc6(float v, float wmax, float u) {
    v = clampf(v, -wmax, wmax);
    float f = floorf(v * (QF / wmax) + u);
    return (uint32_t)(clampf(f, -QF, QF) + 32.0f);
}

/* ---------- packing (build / inspect) ---------- */

void bs_pack(int64_t nbr, const uint32_t *words, const uint64_t *wire, uint32_t *rec) {
    for (int64_t i = 0; i < nbr; i++) {
        uint32_t W[32];
        for (int s = 0; s < 32; s++) W[s] = words[i * 32 + s] & 0xFFFFFFF0u;
        from_words(W, rec + i * 32);
        set_wire(rec + i * 32, wire[2 * i], wire[2 * i + 1]);
    }
}

void bs_unpack(int64_t nbr, const uint32_t *rec, uint32_t *words, uint64_t *wire) {
    for (int64_t i = 0; i < nbr; i++) {
        uint32_t W[32];
        to_words(rec + i * 32, W);
        for (int s = 0; s < 32; s++) words[i * 32 + s] = W[s] & 0xFFFFFFF0u;
        wire[2 * i] = wire_lo(rec + i * 32);
        wire[2 * i + 1] = wire_hi(rec + i * 32);
    }
}

void bs_windows(int64_t N, int B, int64_t first_id, int64_t nW, int64_t *out) {
    for (int64_t n = 0; n < N; n++)
        for (int b = 0; b < B; b++) out[n * B + b] = window(first_id + n, b, nW);
}

/* ---------- acting ---------- */

void bs_forward(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *X, uint32_t *rec,
                float af, float b0, float b1, float b2, const uint8_t *gb, int gbytes, float *thr,
                float bthr, float bonus, float ctx, float slope, int learn, float used_rate,
                float thr_lr, float target, float *rate, uint64_t *ph, int nthreads) {
#pragma omp parallel for schedule(static, 64) num_threads(nthreads)
    for (int64_t n = 0; n < N; n++) {
        uint32_t act[64];
        float soma = 0.0f;
        for (int b = 0; b < B; b++) {
            const uint32_t *r = rec + ((n * B + b) << 5);
            uint32_t m = active_mask(X, window(first_id + n, b, nW), nW, wire_lo(r), wire_hi(r));
            act[b] = m;
            float drive = 0.0f;
            if (m) {
                int cnt = __builtin_popcount(m), F = 0, A0 = 0, A1 = 0, A2 = 0;
                for (int k = 0; k < 6; k++) {
                    F  += __builtin_popcount(r[4 + k] & m) << k;
                    A0 += __builtin_popcount(r[10 + k] & m) << k;
                    A1 += __builtin_popcount(r[16 + k] & m) << k;
                    A2 += __builtin_popcount(r[22 + k] & m) << k;
                }
                drive = af * (float)(F - 32 * cnt) + b0 * (float)(A0 - 32 * cnt)
                      + b1 * (float)(A1 - 32 * cnt) + b2 * (float)(A2 - 32 * cnt);
            }
            if (gate_of(gb, gbytes, n, b))
                soma += fmaxf(drive, 0.0f) + (drive > bthr ? bonus : 0.0f);
        }
        soma *= ctx;
        float rt = 1.0f / (1.0f + expf(-slope * (soma - thr[n])));
        rate[n] = rt;
        if (learn) {
            if (rt >= used_rate)
                for (int b = 0; b < B; b++)
                    if (gate_of(gb, gbytes, n, b)) rec[((n * B + b) << 5) + 31] |= act[b];   /* plane 31 = used */
            thr[n] += thr_lr * (rt - target);
            ph[n] = (ph[n] << 16) | (uint64_t)_cvtss_sh(0.5f + rt, _MM_FROUND_TO_NEAREST_INT);
        }
    }
}

/* ---------- learning ---------- */

static inline uint32_t h32(uint32_t x) {   /* lowbias32: cheap, vectorizable counter-based RNG */
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x;
}

/* Production modulate: branch-free per-synapse loop (clang vectorizes it 8-wide). */
void bs_modulate(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *Xh, uint32_t *rec,
                 const uint8_t *gb, int gbytes, const uint16_t *ph, const float *ew, const float *coef,
                 float fs, float wmax, float wscale, float ar_da, uint64_t key, int oneshot, int nthreads) {
    const uint32_t k1 = (uint32_t)key, k2 = (uint32_t)(key >> 32);
#pragma omp parallel for schedule(static, 64) num_threads(nthreads)
    for (int64_t n = 0; n < N; n++) {
        float p[4];
        for (int k = 0; k < 4; k++) p[k] = _cvtsh_ss(ph[n * 4 + k]);
        const float cf = coef[n];
        for (int b = 0; b < B; b++) {
            uint32_t *r = rec + ((n * B + b) << 5);
            uint64_t lo = wire_lo(r), hi = wire_hi(r);
            int64_t ws = window(first_id + n, b, nW);
            uint32_t hm[4];
            for (int k = 0; k < 4; k++) hm[k] = active_mask(Xh + k * nW, ws, nW, lo, hi);
            if ((hm[0] | hm[1] | hm[2] | hm[3]) == 0 && fs == 1.0f && !oneshot)
                continue;                               /* nothing on this branch can change */
            const float g = (float)gate_of(gb, gbytes, n, b);
            const uint32_t base = (uint32_t)((n * B + b) << 5);
            uint32_t W[32];
            float D[32];
            to_words(r, W);
            for (int s = 0; s < 32; s++) {
                uint32_t w = W[s];
                float e = 0.0f;
                for (int k = 0; k < 4; k++) e += (ew[k] * (float)((hm[k] >> s) & 1u)) * p[k];
                e = e * g;
                float d = (cf * e) * g;
                D[s] = d;
                float fv = clampf(dec6(w, 4, wscale) * fs + d, -wmax, wmax);
                float u = (float)(h32(k1 + (base + s) * 0x9E3779B9u) >> 8) * (1.0f / 16777216.0f);
                float u2 = (float)(h32(k2 + (base + s) * 0x85EBCA6Bu) >> 8) * (1.0f / 16777216.0f);
                int fq = (int)clampf(floorf(fv * QF + u), -QF, QF) + 32;
                uint32_t told = (w >> 28) & 7u;
                float tg = clampf((float)told * (1.0f / 7.0f) + (ar_da * e) * g, 0.0f, 1.0f);
                uint32_t tq = e == 0.0f ? told : (uint32_t)(int)floorf(tg * 7.0f + u2);
                W[s] = (w & 0x8FFFFC0Fu) | ((uint32_t)fq << 4) | ((tq & 7u) << 28);
            }
            if (oneshot)
                for (int s = 0; s < 32; s++) {
                    float u3 = (float)(h32((k1 ^ 0x3C6EF372u) + (base + s) * 0x9E3779B9u) >> 8) * (1.0f / 16777216.0f);
                    W[s] = (W[s] & 0xFFFF03FFu) | (enc6(dec6(W[s], 10, wscale) + 0.5f * D[s], wmax, u3) << 10);
                }
            from_words(W, r);       /* bits 0-3 of W were untouched, so the wiring planes come back */
        }
    }
}

/* Validation modulate: baseline arithmetic order, NumPy's uniforms (uf, ut, uo indexed by synapse). */
void bs_modulate_exact(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *Xh, uint32_t *rec,
                       const uint8_t *gb, int gbytes, const uint16_t *ph, const float *ew, const float *coef,
                       float fs, float wmax, float wscale, float ar_da, int oneshot,
                       const float *uf, const float *ut, const float *uo, int nthreads) {
#pragma omp parallel for schedule(static, 64) num_threads(nthreads)
    for (int64_t n = 0; n < N; n++) {
        float p[4];
        for (int k = 0; k < 4; k++) p[k] = _cvtsh_ss(ph[n * 4 + k]);
        float cf = coef[n];
        for (int b = 0; b < B; b++) {
            uint32_t *r = rec + ((n * B + b) << 5);
            uint64_t lo = wire_lo(r), hi = wire_hi(r);
            int64_t ws = window(first_id + n, b, nW);
            uint32_t hm[4];
            for (int k = 0; k < 4; k++) hm[k] = active_mask(Xh + k * nW, ws, nW, lo, hi);
            float g = (float)gate_of(gb, gbytes, n, b);
            uint32_t W[32];
            to_words(r, W);
            for (int s = 0; s < 32; s++) {
                int64_t idx = ((n * B + b) << 5) + s;
                uint32_t w = W[s];
                float e = 0.0f;
                for (int k = 0; k < 4; k++) e += (ew[k] * (float)((hm[k] >> s) & 1u)) * p[k];
                e = e * g;
                float d = (cf * e) * g;
                float fv = clampf(dec6(w, 4, wscale) * fs + d, -wmax, wmax);
                float fq = clampf(floorf(fv * QF + uf[idx]), -QF, QF) + 32.0f;
                double tg = (double)((w >> 28) & 7u) / 7.0 + (double)((ar_da * e) * g);
                tg = tg < 0.0 ? 0.0 : (tg > 1.0 ? 1.0 : tg);
                uint32_t tq = (uint32_t)floor(tg * 7.0 + (double)ut[idx]);
                w = (w & 0x8FFFFC0Fu) | ((uint32_t)fq << 4) | ((tq & 7u) << 28);
                if (oneshot) w = (w & 0xFFFF03FFu) | (enc6(dec6(w, 10, wscale) + 0.5f * d, wmax, uo[idx]) << 10);
                W[s] = w;
            }
            from_words(W, r);
        }
    }
}

void bs_catch_up(int64_t nbr, uint32_t *rec, const float *M, float wmax, float wscale, uint64_t key,
                 int nthreads) {
#pragma omp parallel for schedule(static, 256) num_threads(nthreads)
    for (int64_t i = 0; i < nbr; i++) {
        uint32_t W[32];
        to_words(rec + i * 32, W);
        for (int s = 0; s < 32; s++) {
            uint64_t idx = (uint64_t)(i * 32 + s);
            uint32_t w = W[s];
            float l0 = dec6(w, 10, wscale), l1 = dec6(w, 16, wscale), l2 = dec6(w, 22, wscale);
            float c0 = M[0] * l0 + M[1] * l1 + M[2] * l2;
            float c1 = M[3] * l0 + M[4] * l1 + M[5] * l2;
            float c2 = M[6] * l0 + M[7] * l1 + M[8] * l2;
            uint64_t h = mix64(key + idx * C1), h2 = mix64((key ^ 0xA5A5ULL) + idx * C1);
            W[s] = (w & 0xF00003FFu) | (enc6(c0, wmax, u24(h)) << 10) | (enc6(c1, wmax, u24b(h)) << 16)
                 | (enc6(c2, wmax, u24(h2)) << 22);
        }
        from_words(W, rec + i * 32);
    }
}

/* ---------- sleeping ---------- */

static inline int nth_set(uint64_t lo, uint64_t hi, int r) {   /* position of the r-th set bit */
    int pl = __builtin_popcountll(lo);
    if (r < pl) return __builtin_ctzll(_pdep_u64(1ULL << r, lo));
    return 64 + __builtin_ctzll(_pdep_u64(1ULL << (r - pl), hi));
}

int64_t bs_sleep(int64_t N, int B, uint32_t *rec, const float *Ms, float fs, float wmax, float wscale,
                 float downscale, float prune_eps, double tag_decay, uint64_t key, int nthreads) {
    int64_t pruned = 0;
#pragma omp parallel for schedule(static, 256) num_threads(nthreads) reduction(+:pruned)
    for (int64_t i = 0; i < N * B; i++) {
        uint32_t *r = rec + i * 32;
        uint64_t lo = wire_lo(r), hi = wire_hi(r);
        uint32_t W[32];
        int pos[32], weak[32], nweak = 0;
        to_words(r, W);
        for (int s = 0; s < 32; s++) {
            uint64_t idx = (uint64_t)(i * 32 + s);
            uint32_t w = W[s];
            double tag = (double)((w >> 28) & 7u) / 7.0;
            double l0d = (double)dec6(w, 10, wscale) + (double)(dec6(w, 4, wscale) * fs) * tag;
            float l0 = (float)(l0d < -wmax ? -wmax : (l0d > wmax ? wmax : l0d));
            float l1 = dec6(w, 16, wscale), l2 = dec6(w, 22, wscale);
            float c0 = Ms[0] * l0 + Ms[1] * l1 + Ms[2] * l2;
            float c1 = Ms[3] * l0 + Ms[4] * l1 + Ms[5] * l2;
            float c2 = Ms[6] * l0 + Ms[7] * l1 + Ms[8] * l2;
            c0 *= 1.0f - downscale;
            uint64_t h = mix64(key + idx * C1), h2 = mix64((key ^ 0x5DEECE66DULL) + idx * C1);
            weak[s] = fmaxf(fabsf(c0), fmaxf(fabsf(c1), fabsf(c2))) < prune_eps && !(w >> 31) && tag < 0.05;
            if (weak[s]) { nweak++; c0 = 0.1f * u24b(h2); c1 = 0.0f; c2 = 0.0f; }
            uint32_t tq = (uint32_t)floor(tag * tag_decay * 7.0 + (double)u24b(h));
            W[s] = (32u << 4) | (enc6(c0, wmax, u24(h)) << 10) | (enc6(c1, wmax, u24(h2)) << 16)
                 | (enc6(c2, wmax, u24(mix64(h2))) << 22) | ((tq & 7u) << 28);
        }
        if (nweak) {
            /* move each weak synapse to a random FREE input inside the same 128-input window */
            for (int s = 0; s < 32; s++) pos[s] = nth_set(lo, hi, s);
            for (int s = 0; s < 32; s++) {
                if (!weak[s]) continue;
                if (pos[s] < 64) lo &= ~(1ULL << pos[s]); else hi &= ~(1ULL << (pos[s] - 64));
                uint64_t flo = ~lo, fhi = ~hi;
                int nfree = __builtin_popcountll(flo) + __builtin_popcountll(fhi);
                int rr = (int)(mix64((key ^ 0x7F4A7C15ULL) + (uint64_t)(i * 32 + s) * C1) % (uint64_t)nfree);
                int p = nth_set(flo, fhi, rr);
                if (p < 64) lo |= 1ULL << p; else hi |= 1ULL << (p - 64);
                pos[s] = p;
            }
            for (int s = 1; s < 32; s++) {   /* slot order = position order: insertion sort */
                int ps = pos[s]; uint32_t ws_ = W[s]; int t = s - 1;
                while (t >= 0 && pos[t] > ps) { pos[t + 1] = pos[t]; W[t + 1] = W[t]; t--; }
                pos[t + 1] = ps; W[t + 1] = ws_;
            }
            pruned += nweak;
        }
        from_words(W, r);
        set_wire(r, lo, hi);
    }
    return pruned;
}

void bs_pack(int64_t nbr, const uint32_t *words, const uint64_t *wire, uint32_t *rec);

void bs_unpack(int64_t nbr, const uint32_t *rec, uint32_t *words, uint64_t *wire);

void bs_windows(int64_t N, int B, int64_t first_id, int64_t nW, int64_t *out);

void bs_forward(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *X, uint32_t *rec,
                float af, float b0, float b1, float b2, const uint8_t *gb, int gbytes, float *thr,
                float bthr, float bonus, float ctx, float slope, int learn, float used_rate,
                float thr_lr, float target, float *rate, uint64_t *ph, int nthreads);

void bs_modulate(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *Xh, uint32_t *rec,
                 const uint8_t *gb, int gbytes, const uint16_t *ph, const float *ew, const float *coef,
                 float fs, float wmax, float wscale, float ar_da, uint64_t key, int oneshot, int nthreads);

void bs_modulate_exact(int64_t N, int B, int64_t nW, int64_t first_id, const uint64_t *Xh, uint32_t *rec,
                       const uint8_t *gb, int gbytes, const uint16_t *ph, const float *ew, const float *coef,
                       float fs, float wmax, float wscale, float ar_da, int oneshot,
                       const float *uf, const float *ut, const float *uo, int nthreads);

void bs_catch_up(int64_t nbr, uint32_t *rec, const float *M, float wmax, float wscale, uint64_t key,
                 int nthreads);

int64_t bs_sleep(int64_t N, int B, uint32_t *rec, const float *Ms, float fs, float wmax, float wscale,
                 float downscale, float prune_eps, double tag_decay, uint64_t key, int nthreads);

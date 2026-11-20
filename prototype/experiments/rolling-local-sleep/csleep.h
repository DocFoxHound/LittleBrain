void sleep_words(uint32_t *restrict words, int64_t n_neurons, int32_t syn,
                 const float *restrict P, int32_t do_catch, const float *restrict S,
                 float fast_scale, float wmax, float keep, float prune_eps, float tag_decay,
                 uint32_t seed, int64_t base_index, int32_t *restrict pruned, int32_t n_threads);

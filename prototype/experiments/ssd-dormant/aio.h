int64_t aio_batch(int fd, int is_write, const int64_t *offs, int64_t n, int64_t size,
                  char *buf, const int64_t *buf_offs, int qd, int64_t *lat_ns);

int64_t resident_bytes(int fd, int64_t size);

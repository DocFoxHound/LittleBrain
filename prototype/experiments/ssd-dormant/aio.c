/*
 * Tiny Linux-native-AIO helper (raw syscalls, no libaio needed).
 *
 * One CPU thread keeps up to `qd` O_DIRECT requests in flight, so we can measure the SSD at
 * high queue depth without spending more than one CPU thread on I/O.
 *
 * aio_batch(fd, is_write, offs[n], n, size, buf, buf_offs[n] or NULL, qd, lat_ns[n] or NULL)
 *   request i moves `size` bytes between file offset offs[i] and buf + buf_offs[i]
 *   (or, if buf_offs is NULL, a ring of qd slots: buf + slot * size).
 *   Returns bytes moved, or a negative errno.
 *
 * Build with ../build_native.py.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/aio_abi.h>
#include <stdint.h>
#include "aio.h"
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static long sys_io_setup(unsigned nr, aio_context_t *ctx) { return syscall(SYS_io_setup, nr, ctx); }
static long sys_io_destroy(aio_context_t ctx) { return syscall(SYS_io_destroy, ctx); }
static long sys_io_submit(aio_context_t ctx, long n, struct iocb **p) { return syscall(SYS_io_submit, ctx, n, p); }
static long sys_io_getevents(aio_context_t ctx, long mn, long mx, struct io_event *ev, struct timespec *t)
{ return syscall(SYS_io_getevents, ctx, mn, mx, ev, t); }

static int64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int64_t aio_batch(int fd, int is_write, const int64_t *offs, int64_t n, int64_t size,
                  char *buf, const int64_t *buf_offs, int qd, int64_t *lat_ns)
{
    if (qd < 1) qd = 1;
    if (qd > 65535) qd = 65535;
    aio_context_t ctx = 0;
    if (sys_io_setup((unsigned)qd, &ctx) < 0) return -errno;

    struct iocb *cbs = calloc((size_t)qd, sizeof(struct iocb));
    struct iocb **ptrs = calloc((size_t)qd, sizeof(struct iocb *));
    struct io_event *ev = calloc((size_t)qd, sizeof(struct io_event));
    int64_t *start = calloc((size_t)qd, sizeof(int64_t));
    int *freelist = calloc((size_t)qd, sizeof(int));
    int nfree = qd;
    for (int i = 0; i < qd; i++) freelist[i] = qd - 1 - i;

    int64_t next = 0, done = 0, inflight = 0, moved = 0, err = 0;
    while (done < n) {
        int nsub = 0;
        while (inflight + nsub < qd && next < n) {
            int s = freelist[--nfree];
            struct iocb *cb = &cbs[s];
            memset(cb, 0, sizeof *cb);
            cb->aio_fildes = (uint32_t)fd;
            cb->aio_lio_opcode = is_write ? IOCB_CMD_PWRITE : IOCB_CMD_PREAD;
            cb->aio_buf = (uint64_t)(uintptr_t)(buf + (buf_offs ? buf_offs[next] : (int64_t)s * size));
            cb->aio_nbytes = (uint64_t)size;
            cb->aio_offset = offs[next];
            cb->aio_data = ((uint64_t)next << 16) | (uint64_t)s;
            ptrs[nsub++] = cb;
            start[s] = now_ns();
            next++;
        }
        if (nsub) {
            long r = sys_io_submit(ctx, nsub, ptrs);
            if (r < 0) { err = -errno; if (err != -EAGAIN) goto out; r = 0; }
            for (long k = r; k < nsub; k++) {          /* give back what was not accepted */
                freelist[nfree++] = (int)(ptrs[k]->aio_data & 0xFFFF);
                next--;
            }
            inflight += r;
        }
        if (inflight == 0) continue;
        long got = sys_io_getevents(ctx, 1, qd, ev, NULL);
        if (got < 0) { if (errno == EINTR) continue; err = -errno; goto out; }
        int64_t t = now_ns();
        for (long k = 0; k < got; k++) {
            int s = (int)(ev[k].data & 0xFFFF);
            int64_t idx = (int64_t)(ev[k].data >> 16);
            /* a negative result is a real error; a short read (e.g. past end of file) is -ENODATA */
            if (ev[k].res != (int64_t)size) { err = ev[k].res < 0 ? ev[k].res : -ENODATA; }
            else moved += size;
            if (lat_ns) lat_ns[idx] = t - start[s];
            freelist[nfree++] = s;
            inflight--;
            done++;
        }
        if (err) goto out;
    }
out:
    /* drain anything still in flight before freeing the control blocks */
    while (inflight > 0) {
        long got = sys_io_getevents(ctx, 1, qd, ev, NULL);
        if (got <= 0) break;
        inflight -= got;
    }
    sys_io_destroy(ctx);
    free(cbs); free(ptrs); free(ev); free(start); free(freelist);
    return err ? err : moved;
}

/* Return resident bytes for a file mapping, or a negative errno. */
int64_t resident_bytes(int fd, int64_t size) {
    if (size == 0) return 0;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || size < 0) return -EINVAL;
    size_t pages = 1 + ((size_t)size - 1) / page;
    unsigned char *vec = calloc(pages, 1);
    if (!vec) return -ENOMEM;
    void *addr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    int64_t bytes = 0;
    if (addr == MAP_FAILED) {
        bytes = -errno;
    } else {
        if (mincore(addr, size, vec) == -1) bytes = -errno;
        else for (size_t i = 0; i < pages; i++) bytes += (vec[i] & 1) * page;
        munmap(addr, size);
    }
    free(vec);
    return bytes;
}

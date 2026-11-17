"""
A day and a night with a RAM cache of hot blocks in front of the SSD brain.

Activity model (stated so it can be argued with):
  * a habitual pool of 2,048 columns (2 GiB: things done every day);
  * the day is 10 episodes x 20 steps; each episode has a context of 1,024 columns (1 GiB):
    896 drawn from the habitual pool + 128 novel columns from anywhere in the brain;
  * each step, 25% of the neurons in the context columns are active (about 1% of the brain),
    one step = forward + modulate.

Cache: 4 GiB of 64 KiB blocks, least-recently-used eviction, CLEAN blocks evicted first.
Blocks touched today are dirty (they hold today's learning) and stay in RAM until the night;
their share of the cache is the SLEEP PRESSURE. If the cache is full of dirty blocks, the oldest
are written back to the SSD (counted).

Night: only the dirty blocks get the night kernel (in RAM, no reads needed) and are written back
in block order; everything untouched is dormant and skips the night (caught up on recall).
"""

import os
import time

from common import COL, GiB, KiB, LEDGER, MiB, NEURON_BYTES, Timer, aio, aligned, med, peak_rss_mib, save

import numpy as np

from brain import n_neurons, open_brain
from kernels import k_sleep, night_matrices
from neuron import WORKHORSE
from wake import Waker

BK = 64                     # neurons per block = 64 KiB
CACHE = 4 * GiB


class BlockCache:
    def __init__(self, fd, n, bk=BK, cache_bytes=CACHE):
        self.fd, self.bk, self.bb = fd, bk, bk * NEURON_BYTES
        self.nslots = cache_bytes // self.bb
        self.words = aligned(cache_bytes, np.uint32).reshape(-1, 256)
        self.slot_of = np.full(n // bk, -1, np.int32)
        self.block_in = np.full(self.nslots, -1, np.int64)
        self.last = np.full(self.nslots, -1, np.int64)
        self.dirty = np.zeros(self.nslots, bool)
        self.step_no = 0
        self.stats = {"read_B": 0, "spill_B": 0, "hits": 0, "misses": 0}

    def fetch(self, blk):
        """Make the (sorted, unique) blocks resident; returns their slots."""
        s = self.slot_of[blk].astype(np.int64)
        miss = blk[s < 0]
        self.stats["hits"] += int((s >= 0).sum())
        self.stats["misses"] += len(miss)
        if len(miss):
            if len(miss) > self.nslots:
                raise MemoryError("one step needs more blocks than the cache holds")
            free = np.flatnonzero(self.block_in < 0)
            need = len(miss) - len(free)
            if need > 0:          # evict: clean & least recently used first, never this step's hits
                key = self.last + self.dirty * (1 << 40)
                key[s[s >= 0]] = 1 << 62
                key[free] = 1 << 62
                victims = np.argpartition(key, need)[:need]
                dv = victims[self.dirty[victims]]
                if len(dv):       # spill dirty blocks (today's learning) back to the SSD
                    order = np.argsort(self.block_in[dv])
                    dv = dv[order]
                    aio(self.fd, True, self.block_in[dv] * self.bb, self.bb, self.words,
                        dv * self.bb, qd=32)
                    self.stats["spill_B"] += len(dv) * self.bb
                self.slot_of[self.block_in[victims]] = -1
                self.block_in[victims] = -1
                self.dirty[victims] = False
                free = np.concatenate([free, victims])
            slots = np.sort(free[:len(miss)])
            aio(self.fd, False, miss * self.bb, self.bb, self.words, slots * self.bb, qd=64)
            self.stats["read_B"] += len(miss) * self.bb
            self.slot_of[miss] = slots
            self.block_in[slots] = miss
            s = self.slot_of[blk].astype(np.int64)
        self.last[s] = self.step_no
        return s

    def step(self, waker, ids):
        blk = np.unique(ids // self.bk)
        s = self.fetch(blk)
        rows = self.slot_of[ids // self.bk].astype(np.int64) * self.bk + ids % self.bk
        waker.compute(self.words, rows, ids)
        self.dirty[s] = True
        self.step_no += 1

    def pressure(self):
        return self.dirty.sum() / self.nslots


def main(episodes=10, steps=20, habit_cols=2048, ctx_habit=896, ctx_novel=128, seed=8):
    n = n_neurons()
    ncol = n // COL
    rng = np.random.default_rng(seed)
    fd = open_brain(write=True)
    w = Waker(n)
    cache = BlockCache(fd, n)
    habit = rng.choice(ncol, habit_cols, replace=False)
    w.compute(cache.words, np.arange(64, dtype=np.int64), np.arange(64, dtype=np.int64))  # compile
    log = []
    print(f"== Day: {episodes} episodes x {steps} steps; context = {ctx_habit} habitual + {ctx_novel} novel "
          f"columns; 25% of their neurons active per step; cache {CACHE // GiB} GiB of {BK} KiB blocks ==")
    t_day = time.perf_counter()
    for ep in range(episodes):
        ctx = np.sort(np.concatenate([rng.choice(habit, ctx_habit, replace=False),
                                      rng.choice(ncol, ctx_novel, replace=False)]))
        ctx = np.unique(ctx)
        ep_t, ep_n = [], 0
        for s in range(steps):
            ids = (ctx[:, None] * COL + np.arange(COL))[rng.random((len(ctx), COL)) < 0.25].astype(np.int64)
            ids.sort()
            h0, m0, r0 = cache.stats["hits"], cache.stats["misses"], cache.stats["read_B"]
            with Timer() as tm:
                cache.step(w, ids)
            hits = cache.stats["hits"] - h0
            miss = cache.stats["misses"] - m0
            log.append({"episode": ep, "step": s, "neurons": len(ids), "s": tm.dt,
                        "hit_rate": hits / max(1, hits + miss), "read_MB": (cache.stats["read_B"] - r0) / 1e6,
                        "pressure": float(cache.pressure())})
            ep_t.append(tm.dt)
            ep_n += len(ids)
        first, rest = log[-steps], log[-steps + 1:]
        print(f"  episode {ep:>2}: first step {first['s'] * 1e3:6.0f} ms (hit {first['hit_rate']:5.1%}, "
              f"read {first['read_MB']:6.0f} MB); other steps median {med([r['s'] for r in rest]) * 1e3:5.0f} ms "
              f"(hit {med([r['hit_rate'] for r in rest]):5.1%}); {ep_n / sum(ep_t):>9,.0f} neuron-steps/s; "
              f"sleep pressure {log[-1]['pressure']:5.1%}", flush=True)
    day_s = time.perf_counter() - t_day
    tot_n = sum(r["neurons"] for r in log)
    tot_s = sum(r["s"] for r in log)
    hit = cache.stats["hits"] / (cache.stats["hits"] + cache.stats["misses"])
    print(f"  day: {tot_n / tot_s:,.0f} neuron-steps/s overall; block hit rate {hit:.1%}; read "
          f"{cache.stats['read_B'] / 1e9:.2f} GB; spilled {cache.stats['spill_B'] / 1e9:.2f} GB; "
          f"dirty at bedtime {cache.dirty.sum() * cache.bb / GiB:.2f} GiB ({cache.pressure():.0%} pressure)")

    # the night: only dirty blocks, already in RAM; then write them back in block order
    t = WORKHORSE
    Mpre, Ms, keep0 = night_matrices(t, episodes * steps)
    dv = np.flatnonzero(cache.dirty)
    dv = dv[np.argsort(cache.block_in[dv])]
    rows = (dv[:, None] * BK + np.arange(BK)).ravel()
    ids = (cache.block_in[dv][:, None] * BK + np.arange(BK)).ravel()
    with Timer() as tc:
        k_sleep(cache.words, rows, ids, Mpre, np.float32(0.5), Ms, keep0, np.float32(t.prune_eps),
                np.float32(t.tag_decay), np.float32(t.w_max), np.uint64(31), True, np.zeros(len(ids), np.int64))
    with Timer() as tw:
        aio(fd, True, cache.block_in[dv] * cache.bb, cache.bb, cache.words, dv * cache.bb, qd=32)
        os.fsync(fd)
    cache.dirty[:] = False
    print(f"  lazy night on the dirty set ({len(dv) * cache.bb / GiB:.2f} GiB = {len(ids):,} neurons): compute "
          f"{tc.dt:.1f} s + write-back {tw.dt:.1f} s ({len(dv) * cache.bb / tw.dt / 1e9:.2f} GB/s) = "
          f"{tc.dt + tw.dt:.1f} s; the other {n - len(ids):,} neurons stay dormant")
    res = {"log": log, "neuron_steps_per_s": tot_n / tot_s, "hit_rate": hit, "read_GB": cache.stats["read_B"] / 1e9,
           "spill_GB": cache.stats["spill_B"] / 1e9, "dirty_GiB": len(dv) * cache.bb / GiB, "day_wall_s": day_s,
           "night_compute_s": tc.dt, "night_write_s": tw.dt, "night_neurons": len(ids),
           "peak_rss_MiB": peak_rss_mib(), "params": dict(episodes=episodes, steps=steps, habit_cols=habit_cols,
                                                          ctx_habit=ctx_habit, ctx_novel=ctx_novel, block_KiB=BK,
                                                          cache_GiB=CACHE // GiB)}
    print(f"  peak RSS {res['peak_rss_MiB']:.0f} MiB")
    save("day", res)
    return res


if __name__ == "__main__":
    main()

"""
Experiment 2: sleep schedules. Who sleeps when, and on which processor?

Every schedule gives every neuron the same work per "day": D awake steps (forward every step,
modulate every other step, like the demo's trial = rewarded pattern + background moment) and one
night. They differ in when the night happens and where it runs:

  sync-gpu    everyone awake D steps, then everyone sleeps on the GPU (in place)
  sync-cpu    everyone awake D steps, then all blocks go to pinned RAM, the CPU sleeps them, back up
  micro-gpu   no offline period: every D/M steps one block gets a quick GPU night in place
  rolling     M blocks; S of them are asleep (in pinned RAM, CPU kernel) while M-S are awake on the
              GPU. Each block: awake D steps, then asleep S shifts. With a spare slot the next waker
              is uploaded in advance (VRAM = M-S+1 blocks); without, swaps are synchronous (M-S).

Measured on the GPU timeline (CUDA events): wall time per day, GPU busy (awake / sleep) and idle;
on the host: sleep-worker busy time, process CPU time, time the GPU waited for the CPU;
coverage: nights per block per day, awake-run length, night latency.

Run: .venv/bin/python prototype/experiments/rolling-local-sleep/bench_schedule.py [--n 1008000]
"""

import argparse
import json
import queue
import resource
import threading
import time

import common
from common import HERE, WORKHORSE

import cupy as cp
import numpy as np

from population import Block

T = WORKHORSE
N_INPUTS = 64
D = 84  # awake steps per block per day (42 trials of rewarded pattern + background moment);
#        divisible by 7, 6, 4, 2, 1 awake blocks and by the 28 micro-sleep slices


# ------------------------------------------------------------------ the world (same for all)

class World:
    def __init__(self, steps, seed=0):
        rng = np.random.default_rng(seed)
        X = np.zeros((steps + 3, N_INPUTS), np.float32)  # rows 0-2: empty history before t=0
        A = rng.choice(N_INPUTS, 8, replace=False)
        for t in range(steps):
            X[t + 3, A if t % 2 == 0 else rng.choice(N_INPUTS, 8, replace=False)] = 1.0
        self.xf = cp.asarray(X)
        self.xh = cp.asarray(X.astype(np.float16))
        self.rate = None

    def step(self, blk, t, stream):
        """One awake step for one block: forward x_t, learn on rewarded steps, tick."""
        if self.rate is None or self.rate.size < blk.n:
            self.rate = cp.zeros(blk.n, cp.float32)
        blk.forward(self.xf[t + 3], self.rate, stream=stream)
        if t % 2 == 0:
            blk.modulate(self.xh[t + 3], -N_INPUTS, 1.0, 0.8, 0.8, stream=stream)
        blk.tick(stream)


# ------------------------------------------------------------------ CPU sleep worker

class SleepWorker(threading.Thread):
    """Runs nights on the CPU. Waits (politely) for each block's download to land first."""

    def __init__(self, threads, reps=1):
        super().__init__(daemon=True)
        self.q, self.threads, self.reps = queue.Queue(), threads, reps
        self.log = []  # (block, t_queued, t_start, t_end)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            b, blk, ev, done, tq = item
            while not ev.done:
                time.sleep(0.0002)
            t0 = time.perf_counter()
            for r in range(self.reps):  # reps > 1 = a heavier night (more passes of the same work)
                blk.sleep_cpu(threads=self.threads)
            self.log.append((b, tq, t0, time.perf_counter()))
            done.set()


# ------------------------------------------------------------------ helpers

def make_host_blocks(n, m, seed=0):
    rng = np.random.default_rng(seed)
    nb = n // m
    return [Block.fresh(T, nb, N_INPUTS, rng, first_id=b * nb, xp=np, pinned=True) for b in range(m)]


def device_slots(k, nb):
    return [Block(T, nb, N_INPUTS, 0, 0, xp=cp) for _ in range(k)]


def gpu_ms(segs):
    return sum(cp.cuda.get_elapsed_time(a, b) for a, b in segs)


class Meter:
    """Measures one window: GPU timeline wall + busy, host wall, process CPU time."""

    def start(self, stream):
        cp.cuda.Device().synchronize()
        self.ev0 = stream.record()
        self.t0 = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu0 = ru.ru_utime + ru.ru_stime

    def stop(self, stream):
        self.ev1 = stream.record()
        cp.cuda.Device().synchronize()
        self.host_wall = time.perf_counter() - self.t0
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu = ru.ru_utime + ru.ru_stime - self.cpu0
        self.gpu_wall = cp.cuda.get_elapsed_time(self.ev0, self.ev1) / 1e3


def summarize(name, n, days, meter, awake_segs, sleep_segs, worker=None, stall=0.0, extra=None):
    wall = meter.gpu_wall
    awake = gpu_ms(awake_segs) / 1e3
    gsleep = gpu_ms(sleep_segs) / 1e3 if sleep_segs else 0.0
    out = {
        "schedule": name, "neurons": n, "days": days,
        "wall_per_day_s": wall / days,
        "awake_neuron_steps_per_s": n * D * days / wall,
        "awake_synapse_updates_per_s": n * D * days * T.n_branches * T.slots / wall,
        "gpu_awake_busy_frac": awake / wall,
        "gpu_sleep_busy_frac": gsleep / wall,
        "gpu_idle_frac": max(0.0, 1 - (awake + gsleep) / wall),
        "process_cpu_util_of_3_threads": meter.cpu / meter.host_wall / 3,
        "host_wait_for_cpu_sleep_s_per_day": stall / days,
    }
    if worker is not None:
        logs = [l for l in worker.log if l[2] >= meter.t0]
        busy = sum(e - s for _, _, s, e in logs)
        out["cpu_sleep_worker_busy_frac"] = busy / meter.host_wall
        out["cpu_night_s_median"] = float(np.median([e - s for _, _, s, e in logs])) if logs else None
        out["cpu_nights_in_window"] = len(logs)
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------------ schedules

def run_sync(host, n, days, warm, mode, threads, reps):
    m, nb = len(host), host[0].n
    world = World(D * (days + warm) + 8)
    compute, copy = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    dev = device_slots(m, nb)
    for b in range(m):
        host[b].copy_into(dev[b], copy)
    copy.synchronize()
    worker = None
    if mode == "cpu":
        worker = SleepWorker(threads, reps)
        worker.start()
    meter, awake_segs, sleep_segs, stall = Meter(), [], [], 0.0
    t = 0
    for day in range(warm + days):
        if day == warm:
            meter.start(compute)
            awake_segs, sleep_segs, stall = [], [], 0.0
        e0 = compute.record()
        for _ in range(D):
            for b in range(m):
                world.step(dev[b], t, compute)
            t += 1
        e1 = compute.record()
        awake_segs.append((e0, e1))
        if mode == "gpu":
            for b in range(m):
                for _ in range(reps):
                    dev[b].sleep_gpu(stream=compute)
            sleep_segs.append((e1, compute.record()))
        else:
            copy.wait_event(e1)
            dones = []
            for b in range(m):
                dev[b].copy_into(host[b], copy)
                done = threading.Event()
                worker.q.put((b, host[b], copy.record(), done, time.perf_counter()))
                dones.append(done)
            for b in range(m):
                ts = time.perf_counter()
                dones[b].wait()
                stall += time.perf_counter() - ts
                host[b].copy_into(dev[b], copy)
            compute.wait_event(copy.record())
    meter.stop(compute)
    if worker:
        worker.q.put(None)
    vram = sum(d.nbytes() for d in dev)
    res = summarize(f"sync-{mode}", n, days, meter, awake_segs, sleep_segs, worker, stall,
                    {"blocks": m, "vram_bytes_per_neuron": vram / n,
                     "pinned_ram_bytes_per_neuron": (sum(h.nbytes() for h in host) / n) if mode == "cpu" else 0,
                     "frac_asleep_at_a_time": 0.0, "nights_per_block_per_day": 1.0,
                     "awake_run_steps": D})
    del dev
    cp.get_default_memory_pool().free_all_blocks()
    return res


def run_micro(host, n, days, warm, reps):
    m, nb = len(host), host[0].n
    assert D % m == 0
    every = D // m
    world = World(D * (days + warm) + 8)
    compute = cp.cuda.Stream(non_blocking=True)
    dev = device_slots(m, nb)
    for b in range(m):
        host[b].copy_into(dev[b], compute)
    compute.synchronize()
    meter, awake_segs, sleep_segs = Meter(), [], []
    t = 0
    for day in range(warm + days):
        if day == warm:
            meter.start(compute)
            awake_segs, sleep_segs = [], []
        for s in range(D):
            e0 = compute.record()
            for b in range(m):
                world.step(dev[b], t, compute)
            t += 1
            e1 = compute.record()
            awake_segs.append((e0, e1))
            if (s + 1) % every == 0:  # one block's quick night, in place, nobody goes offline
                b = (s + 1) // every - 1
                for _ in range(reps):
                    dev[b].sleep_gpu(stream=compute)
                sleep_segs.append((e1, compute.record()))
    meter.stop(compute)
    vram = sum(d.nbytes() for d in dev)
    del dev
    cp.get_default_memory_pool().free_all_blocks()
    return summarize("micro-gpu", n, days, meter, awake_segs, sleep_segs, None, 0.0,
                     {"blocks": m, "vram_bytes_per_neuron": vram / n, "pinned_ram_bytes_per_neuron": 0,
                      "frac_asleep_at_a_time": 0.0, "nights_per_block_per_day": 1.0,
                      "awake_run_steps": D})


def run_rolling(host, n, days, warm, S, spare, threads, reps):
    m, nb = len(host), host[0].n
    awake_n = m - S
    assert D % awake_n == 0 and S >= 1
    assert spare or S >= 2, "without a spare slot and S=1 the whole night would be a GPU stall"
    sps = D // awake_n                     # global steps per shift
    shifts_per_day = m
    n_shifts = (warm + days) * shifts_per_day
    world = World(n_shifts * sps + 8)
    compute, copy = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    K = awake_n + spare
    slots = device_slots(K, nb)
    slot_of = {}
    for b in range(awake_n):
        host[b].copy_into(slots[b], copy)
        slot_of[b] = b
    free = list(range(awake_n, K))
    copy.synchronize()
    done = [threading.Event() for _ in range(m)]
    for b in range(awake_n, m):
        done[b].set()
    worker = SleepWorker(threads, reps)
    worker.start()
    ev_up, last_end = {}, None
    meter, segs, stall = Meter(), [], 0.0
    awake_steps = {b: 0 for b in range(m)}
    runs = []
    t = 0

    def upload(w):
        nonlocal stall
        ts = time.perf_counter()
        done[w].wait()
        stall += time.perf_counter() - ts
        s = free.pop()
        host[w].copy_into(slots[s], copy)
        ev_up[w] = copy.record()
        slot_of[w] = s

    for k in range(n_shifts):
        if k == warm * shifts_per_day:
            meter.start(compute)
            segs, stall = [], 0.0
            worker.log.clear()
        if k > 0:
            x = (k - 1) % m                # falls asleep now: its day (D awake steps) is over
            runs.append(awake_steps[x])
            awake_steps[x] = 0
            sx = slot_of.pop(x)
            copy.wait_event(last_end)
            slots[sx].copy_into(host[x], copy)
            done[x].clear()
            worker.q.put((x, host[x], copy.record(), done[x], time.perf_counter()))
            free.append(sx)
            w = (k - 1 + awake_n) % m      # wakes now
            if not spare:
                upload(w)
            compute.wait_event(ev_up.pop(w))
        awake = [(k + j) % m for j in range(awake_n)]
        e0 = compute.record()
        for _ in range(sps):
            for b in awake:
                world.step(slots[slot_of[b]], t, compute)
            t += 1
        for b in awake:
            awake_steps[b] += sps
        last_end = compute.record()
        segs.append((e0, last_end))
        if spare:
            upload((k + awake_n) % m)      # prefetch the block that wakes at the next shift
    meter.stop(compute)
    worker.q.put(None)
    vram = sum(s.nbytes() for s in slots)
    # night latency: from falling asleep to its CPU night being finished
    lat = [e - tq for _, tq, s, e in worker.log]
    res = summarize(f"rolling M={m} S={S} spare={spare}", n, days, meter, segs, [], worker, stall, {
        "blocks": m, "asleep_blocks": S, "frac_asleep_at_a_time": S / m, "spare_slot": spare,
        "vram_bytes_per_neuron": vram / n,
        "pinned_ram_bytes_per_neuron": sum(h.nbytes() for h in host) / n,
        "nights_per_block_per_day": len(worker.log) / days / m,
        "awake_run_steps": sorted(set(runs[m:])) if len(runs) > m else sorted(set(runs)),
        "night_latency_s_median": float(np.median(lat)) if lat else None,
        "night_latency_s_max": float(np.max(lat)) if lat else None,
        "omp_threads_for_sleep": threads,
    })
    # bring the population back into RAM (everything awake is in a slot)
    for b, s in list(slot_of.items()):
        slots[s].copy_into(host[b], copy)
    copy.synchronize()
    del slots
    cp.get_default_memory_pool().free_all_blocks()
    return res


# ------------------------------------------------------------------ main

def configs(reps):
    return [
        ("sync-gpu", dict(kind="sync", mode="gpu", m=8)),
        ("sync-cpu", dict(kind="sync", mode="cpu", m=8)),
        ("micro-gpu", dict(kind="micro", m=28)),
        ("rolling M=8 S=1 +spare", dict(kind="rolling", m=8, S=1, spare=1)),
        ("rolling M=8 S=2 +spare", dict(kind="rolling", m=8, S=2, spare=1)),
        ("rolling M=8 S=4 +spare", dict(kind="rolling", m=8, S=4, spare=1)),
        ("rolling M=8 S=4 nospare", dict(kind="rolling", m=8, S=4, spare=0)),
        ("rolling M=2 S=1 +spare", dict(kind="rolling", m=2, S=1, spare=1)),
    ]


def run_one(cfg, n, days, warm, reps, threads_rolling, threads_sync, pop_cache):
    m = cfg["m"]
    if m not in pop_cache:
        pop_cache.clear()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        pop_cache[m] = make_host_blocks(n, m)
    host = pop_cache[m]
    if cfg["kind"] == "sync":
        return run_sync(host, n, days, warm, cfg["mode"], threads_sync, reps)
    if cfg["kind"] == "micro":
        return run_micro(host, n, days, warm, reps)
    return run_rolling(host, n, days, warm, cfg["S"], cfg["spare"], threads_rolling, reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_008_000)  # divisible by 2, 8, 28
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1, help="night passes (1 = the real sleep; >1 = heavier)")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--out", default="schedules.json")
    a = ap.parse_args()
    t_start = time.time()
    results = {}
    pop_cache = {}
    cfgs = configs(a.reps)
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c[0] in keep]
    # threads: rolling = 2 OpenMP sleep threads + the GPU-driving main thread (cap 3);
    #          sync-cpu = 3 (the GPU is idle while the CPU sleeps)
    # Configs sharing a block count share one population (built once); repeats are interleaved.
    groups = {}
    for name, cfg in cfgs:
        groups.setdefault(cfg["m"], []).append((name, cfg))
    for m, group in groups.items():
        for rep in range(a.repeats):
            for name, cfg in group:
                r = run_one(cfg, a.n, a.days, a.warm, a.reps, 2, 3, pop_cache)
                results.setdefault(name, []).append(r)
                print(f"[{time.time() - t_start:5.0f}s] {name:26s} rep {rep}: day {r['wall_per_day_s']:.3f} s, "
                      f"GPU awake {r['gpu_awake_busy_frac']:.3f} sleep {r['gpu_sleep_busy_frac']:.3f} "
                      f"idle {r['gpu_idle_frac']:.3f}, CPU {r['process_cpu_util_of_3_threads']:.2f}, "
                      f"wait {r['host_wait_for_cpu_sleep_s_per_day']:.3f} s/day", flush=True)
    summary = {}
    for name, rs in results.items():
        med = {}
        for k in rs[0]:
            v = [r[k] for r in rs]
            med[k] = float(np.median(v)) if isinstance(v[0], float) else v[0]
        med["wall_per_day_runs"] = [round(r["wall_per_day_s"], 4) for r in rs]
        summary[name] = med
    base = summary.get("sync-gpu", {}).get("wall_per_day_s")
    if base:
        for s in summary.values():
            s["throughput_vs_sync_gpu"] = base / s["wall_per_day_s"]
    (HERE / "results").mkdir(exist_ok=True)
    out = {"n": a.n, "D": D, "reps": a.reps, "repeats": a.repeats, "summary": summary}
    (HERE / "results" / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"[total {time.time() - t_start:.0f} s]")


if __name__ == "__main__":
    main()

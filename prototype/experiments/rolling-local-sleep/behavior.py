"""
Experiments 3 and 4: does a rolling night still bring the "old self" back? And what should a
sleeping block do with input that keeps arriving?

Part A (adapter check): demo_single_neuron.run with KernelNeuron (GPU forward/modulate + C sleep
on the CPU) vs the NumPy PackedNeuron, same seeds. This validates the kernels against the brief's
baseline (day-6 recovery ~0.14-0.15, relearn median 2).

Part B (population story on a world clock): 5 blocks x 200 workhorse neurons, all hearing the same
world. A world day has 50 slots. Each block is awake for 40 slots (one demo trial per slot: pattern
+ background moment) and asleep for 10, so every block gets exactly the demo's 40 trials and one
night per day. Only the timing of the night differs:

  sync          every block sleeps in slots 40-49 (the demo's schedule: 40 trials, then night)
  rolling-gate  block b sleeps in slots 10b..10b+9; input arriving during its night is ignored
                (thalamic gating). Block 4 has the sync phase.
  rolling-defer same nights, but events arriving during a block's night are buffered and replayed
                (forward + learning) as soon as it wakes. Blocks then get 50 trials per day.
  micro         no offline time: a 40-slot day, block b consolidates instantly after slot 8b+7

Story: childhood (6 days, A rewarded) -> hard times (2 days, A punished) -> time passes (6 days,
no A, 200 rest ticks per day) -> relearn (A rewarded until each neuron's response >= 0.7).

Run: .venv/bin/python prototype/experiments/rolling-local-sleep/behavior.py
"""

import json
import time
from dataclasses import replace

import common
from common import HERE, WORKHORSE

import numpy as np

from population import KernelNeuron, KernelPopulation

N_INPUTS = 64
M_BLOCKS, PER_BLOCK = 5, 200
TRIALS = 40
CRITERION, MAX_RELEARN = 0.7, 400


def pattern(rng, k=8):
    x = np.zeros(N_INPUTS, np.float32)
    x[rng.choice(N_INPUTS, k, replace=False)] = 1.0
    return x


# ------------------------------------------------------------------ Part A

def adapter_check(seeds=range(1, 17)):
    from demo_single_neuron import run
    from packed_neuron import PackedNeuron
    cols = ["after childhood (A good)", "after hard times (A bad)", "time passes: day 1",
            "time passes: day 3", "time passes: day 6"]
    out = {}
    for label, ntype in [("workhorse", WORKHORSE), ("plain", replace(WORKHORSE, name="plain", cascade_levels=1))]:
        for impl, make in [("numpy-packed", PackedNeuron), ("gpu-awake+cpu-sleep", KernelNeuron)]:
            t0 = time.time()
            rows = [run("", ntype, seed=s, make=make, verbose=False) for s in seeds]
            r = {}
            for c in cols:
                v = np.array([dict(log)[c] for log, _ in rows])
                r[c] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
            rl = np.array([x for _, x in rows])
            r["relearn"] = {"median": float(np.median(rl)), "min": int(rl.min()), "max": int(rl.max()),
                            "all": rl.tolist()}
            r["seconds"] = time.time() - t0
            out[f"{label} / {impl}"] = r
            print(f"  {label:9s} {impl:20s} day6 {r['time passes: day 6']['mean']:.3f} "
                  f"[{r['time passes: day 6']['min']:.2f}-{r['time passes: day 6']['max']:.2f}]  "
                  f"day3 {r['time passes: day 3']['mean']:.3f}  relearn median {r['relearn']['median']:.0f} "
                  f"[{r['relearn']['min']}-{r['relearn']['max']}]", flush=True)
    return out


# ------------------------------------------------------------------ Part B

SCHEDULES = {
    # name: (slots per world day, {block: (sleep slot, night length)}, mode)
    "sync": (50, {b: (40, 10) for b in range(M_BLOCKS)}, "gate"),
    "rolling-gate": (50, {b: (10 * b, 10) for b in range(M_BLOCKS)}, "gate"),
    "rolling-defer": (50, {b: (10 * b, 10) for b in range(M_BLOCKS)}, "defer"),
    "micro": (40, {b: ((8 * b + 8) % 40, 0) for b in range(M_BLOCKS)}, "gate"),
}


def story(schedule, world_seed, offset=0):
    """offset: shift every life event (childhood -> hard times -> time passes) by this many slots
    relative to the day/night clock, so a synchronous night no longer falls on the transitions."""
    L, nights, mode = SCHEDULES[schedule]
    t = WORKHORSE
    rng = np.random.default_rng(world_seed)
    A = pattern(np.random.default_rng(0))
    pops = [KernelPopulation(t, PER_BLOCK, N_INPUTS, rng, first_id=1_000_000 * world_seed + b * PER_BLOCK)
            for b in range(M_BLOCKS)]
    backlog = [[] for _ in range(M_BLOCKS)]
    asleep = [False] * M_BLOCKS
    stats = {"events": 0, "events_seen_by_someone": 0, "block_trials": np.zeros(M_BLOCKS, int),
             "nights": np.zeros(M_BLOCKS, int)}
    relearn = {"count": np.zeros((M_BLOCKS, PER_BLOCK), int), "done": np.full((M_BLOCKS, PER_BLOCK), -1)}

    def live(b, ev):
        kind, x_bg, da = ev
        p = pops[b]
        if kind == "rest":
            p.rest(5)
            return
        if kind == "relearn":  # probe first, like trials_to_like
            r = p.forward(A, learn_trace=False)
            hit = (r >= CRITERION) & (relearn["done"][b] < 0)
            relearn["done"][b][hit] = relearn["count"][b][hit]
            relearn["count"][b] += 1
        p.forward(A)
        if da:
            p.modulate(dopamine=da, ach=0.8, arousal=0.8)
        p.tick()
        p.forward(x_bg)
        p.tick()
        stats["block_trials"][b] += 1

    def run_slot(slot_in_day, ev):
        stats["events"] += int(ev[0] != "rest")
        seen = False
        for b in range(M_BLOCKS):
            s0, ln = nights[b]
            if slot_in_day == s0:          # falling asleep (instant consolidation on the CPU)
                pops[b].sleep()
                stats["nights"][b] += 1
                asleep[b] = ln > 0
            if asleep[b] and not (s0 <= slot_in_day < s0 + ln):
                asleep[b] = False          # waking up: replay what arrived during the night
                for old in backlog[b]:
                    live(b, old)
                backlog[b].clear()
            if asleep[b]:
                if mode == "defer" and ev[0] != "rest":
                    backlog[b].append(ev)
                continue
            seen = seen or ev[0] != "rest"
            live(b, ev)
        if ev[0] != "rest":
            stats["events_seen_by_someone"] += int(seen)

    def probe():
        return np.stack([p.forward(A, learn_trace=False) for p in pops])  # (blocks, neurons)

    log = {"start": probe()}
    g = 0  # global slot

    def advance(n_slots, kind, da):
        nonlocal g
        for _ in range(n_slots):
            run_slot(g % L, (kind, pattern(rng) if kind != "rest" else None, da))
            g += 1

    advance(6 * L + offset, "trial", +1.0)
    log["after childhood"] = probe()
    advance(2 * L, "trial", -1.0)
    log["after hard times"] = probe()
    for d in range(1, 7):
        advance(L, "rest", 0.0)
        if d in (1, 3, 6):
            log[f"day {d}"] = probe()
    # relearn: keep the day structure (nights keep coming) until everyone liked A again
    s = 0
    while (relearn["done"] < 0).any() and relearn["count"].min() < MAX_RELEARN + 1:
        run_slot((g + s) % L, ("relearn", pattern(rng), +1.0))
        s += 1
    done = relearn["done"].copy()
    done[done < 0] = MAX_RELEARN
    return {"log": log, "relearn": done, "stats": stats, "relearn_world_slots": s}


VARIANTS = [("sync", "sync", 0), ("rolling-gate", "rolling-gate", 0), ("rolling-defer", "rolling-defer", 0),
            ("micro", "micro", 0),
            # control: synchronous nights, but life changes in the middle of the day (slot 20)
            ("sync, events shifted 20 slots", "sync", 20)]


def population_story(seeds=range(1, 9)):
    out = {}
    for sch, sched, offset in VARIANTS:
        t0 = time.time()
        runs = [story(sched, s, offset) for s in seeds]
        keys = list(runs[0]["log"].keys())
        res = {}
        for k in keys:
            v = np.stack([r["log"][k] for r in runs])          # (seeds, blocks, neurons)
            res[k] = {"mean": float(v.mean()), "per_block_mean": v.mean(axis=(0, 2)).round(3).tolist(),
                      "seed_means": v.mean(axis=(1, 2)).round(3).tolist(),
                      "frac_neurons_above_0.06": float((v > 0.06).mean())}
        rl = np.stack([r["relearn"] for r in runs])
        res["relearn"] = {"median": float(np.median(rl)), "mean": float(rl.mean()),
                          "p90": float(np.percentile(rl, 90)), "frac_cap": float((rl >= MAX_RELEARN).mean()),
                          "per_block_median": np.median(rl, axis=(0, 2)).tolist(),
                          "per_block_mean": rl.mean(axis=(0, 2)).round(2).tolist()}
        st = [r["stats"] for r in runs]
        res["trials_per_block_per_story"] = np.mean([s["block_trials"] for s in st], axis=0).tolist()
        res["nights_per_block"] = np.mean([s["nights"] for s in st], axis=0).tolist()
        res["frac_world_events_seen_by_some_block"] = float(
            np.sum([s["events_seen_by_someone"] for s in st]) / np.sum([s["events"] for s in st]))
        res["seconds"] = time.time() - t0
        out[sch] = res
        print(f"  {sch:30s} childhood {res['after childhood']['mean']:.3f}  hard {res['after hard times']['mean']:.3f}  "
              f"day1 {res['day 1']['mean']:.3f}  day3 {res['day 3']['mean']:.3f}  day6 {res['day 6']['mean']:.3f} "
              f"per-block {res['day 6']['per_block_mean']}  relearn median {res['relearn']['median']:.0f} "
              f"mean {res['relearn']['mean']:.2f}  seen {res['frac_world_events_seen_by_some_block']:.2f}  "
              f"[{res['seconds']:.0f}s]", flush=True)
    return out


def main():
    t0 = time.time()
    print("Part A: single-neuron demo, kernels vs NumPy prototype (seeds 1-16)")
    a = adapter_check()
    print("Part B: population story on a world clock (5 blocks x 200 neurons, seeds 1-8)")
    b = population_story()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "behavior.json").write_text(json.dumps({"adapter": a, "population": b}, indent=1))
    print(f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()

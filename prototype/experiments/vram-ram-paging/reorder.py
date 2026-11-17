"""
Fringe: "neurons that fire together page together".

Columns start in a SCATTERED storage layout (an assembly's 8 columns sit in ~8 different 1,024-
neuron blocks), so a few percent of active neurons touch a large share of blocks. During a day we
log which columns are active; at sleep we cluster columns by co-activation (Jaccard similarity of
their activity, greedy merge capped at one block = 8 columns) and MIGRATE them into the new
layout (a permutation of whole 128-neuron columns: words + per-neuron state). Next day: how many
blocks does a step touch, and how fast does paging run with the same VRAM cache?

Run: .venv/bin/python prototype/experiments/vram-ram-paging/reorder.py
"""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
import cupy as cp

from gpu_population import GPUPopulation
from neuron import WORKHORSE
from scenario import AssemblyScenario

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1 << 20
CPA = 8  # columns per block (and per assembly)


def day_log(scen, steps):
    A = np.zeros((steps, scen.n_cols), np.bool_)
    for t in range(steps):
        cols, _ = scen.step()
        A[t, cols] = True
    return A


def learn_layout(A, current, cap=CPA, top=12):
    """Cluster columns by co-activation (GPU matmul for the counts), pack clusters into blocks.
    Columns with no co-activation evidence keep their current relative order (`current` = slots)."""
    a = cp.asarray(A, cp.float32)
    n = a.sum(0)
    nbs, ws = [], []
    for r0 in range(0, A.shape[1], 1024):          # row chunks keep VRAM small
        co = a[:, r0:r0 + 1024].T @ a              # co-activation counts
        jac = co / cp.maximum(n[r0:r0 + 1024, None] + n[None, :] - co, 1.0)
        jac[cp.arange(jac.shape[0]), cp.arange(r0, r0 + jac.shape[0])] = 0.0
        nb = cp.argsort(-jac, axis=1)[:, :top]     # strongest partners of each column
        nbs.append(nb)
        ws.append(cp.take_along_axis(jac, nb, axis=1))
        del co, jac
    nb, w = cp.concatenate(nbs), cp.concatenate(ws)
    i = cp.repeat(cp.arange(A.shape[1]), top)
    edges = cp.stack([i, nb.ravel()], 1)
    w = w.ravel()
    keep = w > 0
    order = cp.argsort(-w[keep])
    edges = cp.asnumpy(edges[keep][order])
    del a
    # greedy union-find with a size cap: strongest pairs first
    parent = np.arange(A.shape[1])
    size = np.ones(A.shape[1], np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv and size[ru] + size[rv] <= cap:
            parent[rv] = ru
            size[ru] += size[rv]
    roots = np.array([find(c) for c in range(A.shape[1])])
    clusters = {}
    for c, r in enumerate(roots):
        clusters.setdefault(r, []).append(c)
    # pack clusters into blocks of `cap` columns, biggest first (first-fit decreasing)
    blocks = []
    by_pos = sorted(clusters.values(), key=lambda cl: min(current[c] for c in cl))  # current storage order
    for cl in sorted(by_pos, key=len, reverse=True):  # stable: equal sizes stay in current order
        for b in blocks:
            if len(b) + len(cl) <= cap:
                b.extend(cl)
                break
        else:
            blocks.append(list(cl))
    order = [c for b in blocks for c in b]
    # blocks may be partly filled; lay them out compactly but keep block alignment by padding
    col_slot = np.empty(A.shape[1], np.int64)
    slot = 0
    for b in blocks:
        for c in b:
            col_slot[c] = slot
            slot += 1
    assert slot == A.shape[1] and len(set(order)) == A.shape[1]
    return col_slot, len(blocks)


def working_set(scen, layout, steps, seed):
    s = AssemblyScenario(N, active_frac=scen["frac"], churn=scen["churn"], noise_frac=scen["noise"], seed=seed)
    s.set_layout(layout)
    nb, na = [], []
    for _ in range(steps):
        cols, blocks = s.step()
        nb.append(len(blocks))
        na.append(len(cols))
    return float(np.mean(nb)), float(np.mean(na))


def migrate(pop, old_slot, new_slot, npc=128):
    """Move whole columns from their old storage slot to the new one (words + per-neuron state)."""
    src_of_new = np.empty_like(old_slot)
    src_of_new[new_slot] = old_slot                # new storage slot s holds the column that was in slot src
    t0 = time.perf_counter()
    words = pop.host.reshape(len(old_slot), -1)    # one row = one column's words (128 neurons x 1 KB)
    words[:] = words[src_of_new]
    perm = (src_of_new[:, None] * npc + np.arange(npc)[None, :]).ravel()
    pd = cp.asarray(perm)
    for name in ("ids", "gate", "thr", "plast", "post", "age", "rate"):
        setattr(pop, name, getattr(pop, name)[pd].copy())
    cp.cuda.Device().synchronize()
    pop.reset_cache()
    return time.perf_counter() - t0


def paged_speed(pop, layout, frac, churn, noise, steps=12, warmup=8, seed=11):
    s = AssemblyScenario(N, active_frac=frac, churn=churn, noise_frac=noise, seed=seed)
    s.set_layout(layout)
    pop.reset_cache()
    tf, tm, h = [], [], []
    for t in range(warmup + steps):
        cols, blocks = s.step()
        h0 = pop.h2d
        t0 = time.perf_counter()
        pop.forward(cols=cols, active=blocks)
        t1 = time.perf_counter()
        pop.modulate(0.5, 0.5, 0.5)
        t2 = time.perf_counter()
        pop.tick()
        if t >= warmup:
            tf.append(t1 - t0)
            tm.append(t2 - t1)
            h.append(pop.h2d - h0)
    return float(np.median(tf)) * 1e3, float(np.median(tm)) * 1e3, float(np.mean(h)) / 1e6


def main():
    out = {"rows": []}
    base = AssemblyScenario(N, seed=1)
    clustered = np.arange(base.n_cols)
    scattered = np.random.default_rng(99).permutation(base.n_cols)
    print(f"== Working set in 1,024-neuron blocks per step ({N:,} neurons, {base.n_cols:,} columns of 128, "
          f"{base.n_asm:,} assemblies of 8 columns) ==")
    print(f"  {'scenario':<34}{'active neurons':>15}{'clustered':>11}{'scattered':>11}{'learned':>10}"
          f"{'learned/ideal':>15}{'scat/learned':>14}")
    learned_for = {}
    for frac, churn, noise in [(0.01, 0.02, 0.0), (0.05, 0.02, 0.0), (0.2, 0.02, 0.0), (0.05, 0.02, 0.1),
                               (0.05, 0.2, 0.1)]:
        sc = {"frac": frac, "churn": churn, "noise": noise}
        day = AssemblyScenario(N, active_frac=frac, churn=churn, noise_frac=noise, seed=5)
        day.set_layout(scattered)
        t0 = time.perf_counter()
        A = day_log(day, 3000)
        seen = A.any(0).mean()
        learned, nblk = learn_layout(A, scattered)
        t_learn = time.perf_counter() - t0
        learned_for[(frac, churn, noise)] = learned
        ws = {name: working_set(sc, lay, 300, seed=77) for name, lay in
              [("clustered", clustered), ("scattered", scattered), ("learned", learned)]}
        act = ws["clustered"][1] * 128
        ideal = ws["clustered"][0]
        row = {"frac": frac, "churn": churn, "noise": noise, "active_neurons": act,
               "blocks": {k: v[0] for k, v in ws.items()}, "cols_seen_during_day": float(seen),
               "learn_s": t_learn}
        out["rows"].append(row)
        label = f"{frac:.0%} active, churn {churn}, noise {noise:.0%}"
        print(f"  {label:<34}{act:>15,.0f}{ws['clustered'][0]:>11.1f}{ws['scattered'][0]:>11.1f}"
              f"{ws['learned'][0]:>10.1f}{ws['learned'][0] / ideal:>15.2f}{ws['scattered'][0] / ws['learned'][0]:>14.2f}"
              f"   (day: {seen:.0%} of columns seen; clustering {t_learn:.1f} s)", flush=True)

    # Real migration + paging speed with a VRAM cache of 256 blocks (25% of the population)
    frac, churn, noise = 0.05, 0.02, 0.1
    learned = learned_for[(frac, churn, noise)]
    scen0 = AssemblyScenario(N, seed=1)
    scen0.set_layout(scattered)
    pop = GPUPopulation(WORKHORSE, N, col_size=64, npc=128, mode="paged", cache_blocks=256,
                        ids=scen0.storage_ids(), seed=3)
    f_s, m_s, h_s = paged_speed(pop, scattered, frac, churn, noise)
    # Migrate right after sleep: sleep catches every block up (the lazy per-block clocks are equal),
    # so whole columns can move between blocks. Same input before and after: same rate per neuron.
    t0 = time.perf_counter()
    pop.sleep()
    pop.flush()
    t_sleep = time.perf_counter() - t0
    probe = AssemblyScenario(N, active_frac=0.05, seed=8)
    pcols, _ = probe.step()
    seed0 = pop.seed
    r_before = cp.asnumpy(pop.forward(cols=pcols, active=np.unique(scen0.col_block[pcols]), learn_trace=False))
    ids_before = cp.asnumpy(pop.ids)
    pop.flush()
    t_mig = migrate(pop, scattered, learned)
    probe.set_layout(learned)
    pop.seed = seed0
    r_after = cp.asnumpy(pop.forward(cols=pcols, active=np.unique(probe.col_block[pcols]), learn_trace=False))
    ids_after = cp.asnumpy(pop.ids)
    same = np.array_equal(r_before[np.argsort(ids_before)], r_after[np.argsort(ids_after)])
    f_l, m_l, h_l = paged_speed(pop, learned, frac, churn, noise)
    print(f"\n== Paging with a 256-block VRAM cache, {frac:.0%} active, churn {churn}, noise {noise:.0%} ==")
    print(f"  scattered layout: forward {f_s:6.2f} ms, modulate {m_s:6.2f} ms, PCIe in {h_s:6.1f} MB/step")
    print(f"  learned layout:   forward {f_l:6.2f} ms, modulate {m_l:6.2f} ms, PCIe in {h_l:6.1f} MB/step")
    print(f"  sleep sweep {t_sleep:.2f} s, then migration (1 GiB of words + per-neuron state, host numpy): "
          f"{t_mig:.2f} s; rates identical per neuron after moving: {same}")
    out["paging"] = {"frac": frac, "churn": churn, "noise": noise, "cache_blocks": 256,
                     "scattered": {"fwd_ms": f_s, "mod_ms": m_s, "h2d_MB": h_s},
                     "learned": {"fwd_ms": f_l, "mod_ms": m_l, "h2d_MB": h_l},
                     "migration_s": t_mig, "sleep_s": t_sleep, "rates_identical_after_migration": bool(same)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "reorder.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

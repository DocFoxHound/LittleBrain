"""
genome-diff: "a neuron that has learned nothing costs nothing."

One command runs everything, each part in its own short process (< ~2 min each):
    .venv/bin/python prototype/experiments/genome-diff/run.py            # all parts
    .venv/bin/python prototype/experiments/genome-diff/run.py world      # one part
Parts: behavior, world, formats, bench, regen.
Results go to results/ (small CSV / txt). Snapshots in data/ are deleted at the end.
"""

import csv
import os

for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "3")
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE / "data"
PY = sys.executable


def scenarios():
    import core
    SD = dict(mode=core.ANCHORED, need_activity=True, prune_prob=0.05, min_tag=2, snap_q=1)
    return {
        # name: (description, WorldCfg kwargs)
        "base_rules": ("BASE: baseline rules, genome = birth state", dict(pop=dict(mode=core.BASE))),
        "anchored": ("ANCHORED + genome clock, baseline pruning", dict(pop=dict(mode=core.ANCHORED))),
        "sparse_default": ("+ prune only active cells, 5%/night turnover, tag>=2 consolidation, snap<=1q",
                           dict(pop=SD)),
        "noise_1pct": ("sparse_default + 1% random background input (20 days: slow, most cells active)",
                       dict(pop=SD, noise=0.01, days=20)),
        "targeted": ("sparse_default + dopamine reaches only the valenced thing's column", dict(pop=SD, targeted=True)),
        "targeted_noprune": ("targeted, no pruning at all (pure learning diff)",
                             dict(pop=dict(SD, prune_prob=0.0), targeted=True)),
        "growing_world": ("targeted, world opens up: 4 regions on day 1, +1.5/day",
                          dict(pop=SD, targeted=True, grow=(4, 1.5))),
    }


SNAP_SCEN = ("base_rules", "sparse_default", "targeted", "targeted_noprune", "growing_world")
SNAP_DAYS = (1, 7, 20, 40)
PRINT_DAYS = (1, 2, 3, 5, 7, 10, 14, 20, 21, 25, 30, 40)


def part_world(names=None):
    import numpy as np
    import world
    RES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    sc = scenarios()
    for name in (names or sc):
        desc, kw = sc[name]
        print(f"\n== world scenario '{name}': {desc}")
        kw = dict(kw)
        cfg = world.WorldCfg(days=kw.pop("days", 40), snapshot_days=SNAP_DAYS if name in SNAP_SCEN else (), **kw)
        t0 = time.time()
        log, snaps, meta, pop = world.simulate(cfg, verbose=False)
        for st in log:
            if st["day"] in PRINT_DAYS:
                print(f"  day {st['day']:2d}: synapses differing {st['syn_any']:6.1%} = rewired {st['syn_rewired']:5.1%}"
                      f" + learned {st['syn_learned']:5.1%} (>=2 steps {st['syn_learned2']:5.1%}) + tag-only"
                      f" {st['syn_tagonly']:5.1%} | neurons {st['neu_any']:6.1%} (>=2 steps {st['neu_learned2']:6.1%})"
                      f" | regions seen {st['cols_seen']:4.0%}")
        print(f"  {meta['n']:,} neurons, {meta['steps']} awake steps, {meta['mods']} dopamine events, "
              f"{meta['evaluated_frac']:.1%} of neuron-steps evaluated (event-driven); "
              f"awake {meta['t_awake']:.1f}s, sleep {meta['t_sleep']:.1f}s, total {time.time() - t0:.0f}s")
        with open(RES / f"world_{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            wr.writeheader()
            wr.writerows(log)
        for d, wds in snaps.items():
            np.save(DATA / f"snap_{name}_d{d}.npy", wds)


def part_behavior():
    import behavior
    behavior.main()


def part_formats():
    import numpy as np
    import core
    import formats
    RES.mkdir(exist_ok=True)
    order = ["dense", "block32", "block8", "bitmap", "csr", "coo", "hash", "zstd(csr)", "zstd(dense)"]
    rows_out = []
    print("\n== Diff formats on real snapshots (16,384 neurons; bytes per neuron averaged over ALL neurons;"
          "\n   materialize = rebuild 25% of neurons' rows from genome + diff, ns per neuron, median of 5)")
    import zstandard
    import world
    from neuron import WORKHORSE
    z = zstandard.ZstdCompressor(level=3)
    cfg = world.WorldCfg()
    lay = core.layout_local(cfg.n_regions, cfg.region_size, cfg.neurons_per_col, cfg.p_neighbor)
    n = cfg.n_regions * cfg.neurons_per_col
    fresh_base = core.GenomePop(WORKHORSE, n, lay, seed=cfg.seed, mode=core.BASE).words
    fresh_anch = core.GenomePop(WORKHORSE, n, lay, seed=cfg.seed, mode=core.ANCHORED).words
    print("\n== Control: plain zstd on the dense words. Does the genome subtraction matter, or would any compressor do?")
    print(f"   newborn brain, BASE words (innate weight stored in c0):   {len(z.compress(fresh_base.tobytes())) / n:7.1f} B/neuron")
    print(f"   newborn brain, ANCHORED words (innate weight procedural): {len(z.compress(fresh_anch.tobytes())) / n:7.1f} B/neuron")
    for d in SNAP_DAYS:
        p = DATA / f"snap_base_rules_d{d}.npy"
        if p.exists():
            print(f"   BASE rules, day {d:2d} (everything has drifted):           "
                  f"{len(z.compress(np.load(p).tobytes())) / n:7.1f} B/neuron")
    pool = None
    for scen in SNAP_SCEN:
        if scen == "base_rules":
            continue
        for d in SNAP_DAYS:
            p = DATA / f"snap_{scen}_d{d}.npy"
            if not p.exists():
                continue
            w = np.load(p)
            if pool is None and scen == "targeted" and d == 20:
                pool = w[w != core.PRISTINE]
            rows, meta = formats.analyze(w, timing=(d in (1, 40)))
            print(f"\n  {scen} day {d}: {meta['diff_frac']:.1%} of synapses differ, {meta['dirty_neurons']:.1%} of "
                  f"neurons, {meta['per_dirty']:.0f} entries per dirty neuron")
            print("   " + "".join(f"{k:>12}" for k in order))
            print("   " + "".join(f"{rows[k]['bytes']:>11.0f}B" for k in order))
            if "mat_ns" in rows["csr"]:
                print("   " + "".join(f"{rows[k]['mat_ns']:>10.0f}ns" for k in order) + "   materialize")
                print("   " + "".join(f"{rows[k]['enc_ns']:>10.0f}ns" for k in order) + "   encode (rebuild)")
            for k in order:
                rows_out.append(dict(source=f"{scen}_d{d}", fmt=k, diff_frac=meta["diff_frac"],
                                     dirty_neurons=meta["dirty_neurons"], **{q: rows[k].get(q, "") for q in
                                                                             ("bytes", "mat_ns", "enc_ns")}))
    if pool is None:
        pool = np.array([core.PRISTINE ^ np.uint32(1 << 28)], np.uint32)
    print("\n== Synthetic sweep: bytes per neuron vs experience (values drawn from a real diff;"
          " dirty synapses clustered on branches)")
    print(f"   {'dirty neurons':>13}{'per dirty':>10}{'diff %':>8}" + "".join(f"{k:>12}" for k in order))
    for nf, k in [(0.001, 16), (0.01, 16), (0.01, 64), (0.05, 32), (0.1, 32), (0.1, 128), (0.3, 64), (1.0, 16),
                  (1.0, 64), (1.0, 128), (1.0, 230)]:
        w = formats.synthetic(16384, nf, k, pool=pool)
        rows, meta = formats.analyze(w, timing=False, verify=False)
        print(f"   {nf:>13.1%}{k:>10}{meta['diff_frac']:>8.2%}" + "".join(f"{rows[q]['bytes']:>11.1f}B" for q in order))
        for q in order:
            rows_out.append(dict(source=f"synthetic_{nf}_{k}", fmt=q, diff_frac=meta["diff_frac"],
                                 dirty_neurons=meta["dirty_neurons"], bytes=rows[q]["bytes"], mat_ns="", enc_ns=""))
    with open(RES / "formats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wr.writeheader()
        wr.writerows(rows_out)


def part_bench():
    import bench
    bench.main()


def part_regen():
    import regen
    regen.main()


PARTS = {"behavior": part_behavior, "world": part_world, "formats": part_formats, "bench": part_bench,
         "regen": part_regen}


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    args = sys.argv[1:]
    if args and args[0] in PARTS:
        if args[0] == "world" and len(args) > 1:
            part_world(args[1:])
        else:
            PARTS[args[0]]()
    elif not args:
        env = dict(os.environ, NUMBA_NUM_THREADS="3", OMP_NUM_THREADS="3")
        jobs = [["behavior"]] + [["world", n] for n in scenarios()] + [["formats"], ["bench"], ["regen"]]
        RES.mkdir(exist_ok=True)
        for j in jobs:
            t0 = time.time()
            r = subprocess.run([PY, "-W", "ignore", str(Path(__file__).resolve())] + j, env=env,
                               capture_output=True, text=True)
            print(r.stdout, end="", flush=True)
            if r.returncode:
                print(r.stderr)
                raise SystemExit(r.returncode)
            mode = "a" if j[0] == "world" and j[1] != next(iter(scenarios())) else "w"
            with open(RES / f"{j[0]}.txt", mode) as f:
                f.write(r.stdout)
            print(f"[{' '.join(j)}: {time.time() - t0:.0f}s]", flush=True)
        for p in DATA.glob("*.npy"):
            p.unlink()
        if DATA.exists():
            DATA.rmdir()
    else:
        print(__doc__)

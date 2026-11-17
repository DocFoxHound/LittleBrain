"""
Behavior check: the "childhood -> hard times -> time passes -> relearn" story from
demo_single_neuron.py, run through GenomeNeuron adapters for every dynamics variant.

Run: .venv/bin/python prototype/experiments/genome-diff/behavior.py
"""

import sys
import time
from dataclasses import replace

import numpy as np

import core  # sets thread caps and sys.path
from demo_single_neuron import run
from neuron import WORKHORSE
from packed_neuron import PackedNeuron

SEEDS = range(1, 17)

VARIANTS = [
    ("baseline PackedNeuron (prototype)", PackedNeuron),
    ("BASE: procedural birth, same rules", core.make_adapter(mode=core.BASE)),
    ("ANCHORED, static innate weight", core.make_adapter(mode=core.ANCHORED, genome_clock=False)),
    ("ANCHORED + genome clock", core.make_adapter(mode=core.ANCHORED)),
    ("  + prune only if active today", core.make_adapter(mode=core.ANCHORED, need_activity=True)),
    ("  + GC: consolidate tag>=2, snap<=1q", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                               min_tag=2, snap_q=1)),
    ("  + turnover-limited pruning (5%/night)", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                                 prune_prob=0.05)),
    ("  + both (sparse-world default)", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                         prune_prob=0.05, min_tag=2, snap_q=1)),
    ("  + Hebbian gate: learn only if rate>=0.1", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                                    post_gate=0.1)),
    ("  + GC: untagged decay 10%/night", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                           untagged_decay=0.10)),
    ("  + GC: untagged decay 30%/night", core.make_adapter(mode=core.ANCHORED, need_activity=True,
                                                           untagged_decay=0.30)),
]

COLS = ["after childhood (A good)", "after hard times (A bad)", "time passes: day 3", "time passes: day 6"]


def summarize(rows):
    out = {}
    for c in COLS:
        v = np.array([dict(log)[c] for log, _ in rows])
        out[c] = (v.mean(), v.min(), v.max())
    r = np.array([x for _, x in rows])
    out["relearn"] = (np.median(r), r.min(), r.max())
    out["ok"] = int((r < 400).sum())
    return out


def main(seeds=SEEDS, ntypes=(("workhorse", WORKHORSE),
                              ("plain", replace(WORKHORSE, name="plain", cascade_levels=1)))):
    results = {}
    for tname, ntype in ntypes:
        print(f"\n== {tname}: mean over seeds {seeds.start}-{seeds.stop - 1} [min-max] ==")
        print(f"  {'variant':<42}{'childhood':>10}{'hard':>7}{'day 3':>7}{'day 6 [min-max]':>20}"
              f"{'relearn median [min-max]':>27}{'relearned':>11}")
        for name, make in VARIANTS:
            if tname != "workhorse" and (name.startswith("  + GC: untagged") or "Hebbian" in name):
                continue
            t0 = time.time()
            rows = [run("", ntype, seed=s, make=make, verbose=False) for s in seeds]
            st = summarize(rows)
            results[(tname, name)] = st
            d6 = st["time passes: day 6"]
            rl = st["relearn"]
            print(f"  {name:<42}{st[COLS[0]][0]:>10.2f}{st[COLS[1]][0]:>7.2f}{st[COLS[2]][0]:>7.2f}"
                  f"{d6[0]:>8.2f} [{d6[1]:.2f}-{d6[2]:.2f}]{rl[0]:>14.1f} [{rl[1]}-{rl[2]}]"
                  f"{st['ok']:>8}/{len(rows)}   ({time.time() - t0:.1f}s)", flush=True)
    return results


if __name__ == "__main__":
    main()

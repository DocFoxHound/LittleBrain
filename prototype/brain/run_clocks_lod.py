"""
Separate clocks per region and level of detail (notes/24, ideas C and D).

  1. Clocks: a slow hormone-like group (AgRP-type endocrine cell, driven by cortex) updated every
     step vs every 10 steps. Does its output follow the same course? (It has no outputs, so the
     rest of the brain is untouched.)
  2. Level of detail: the stage-2 brain (thalamus, cortex, hippocampus) on the memory task, with
     regions going dormant when their incoming activity drops below half its usual level.
     Memory score, share of group updates skipped, wall time.

Run: PYTHONPATH=prototype python3 prototype/brain/run_clocks_lod.py
"""

import copy
import time

import numpy as np

import brainspec as S
import celltypes as C
from federation import Group, Pooled
from run_federation import memory_task, prepare

SEEDS = range(3)


def with_hormone(every):
    spec = copy.deepcopy(S.stage1())
    spec.groups.append(Group("HY.stress", C.AGRP, 2, "HY", every=every))
    spec.pooled.append(Pooled("test", "A.L23", "HY.stress", 3.0, sigma=1.0))
    return spec


def test_clocks():
    print("1. Clocks: slow hormone group, every step vs every 10 steps (seed 0)")
    traces = {}
    for every in (1, 10):
        b = prepare(with_hormone(every), 0, 0.0)
        g = b.g["HY.stress"]
        trace, step = [], b.step

        def logged(*a, **k):
            step(*a, **k)
            trace.append(g.level.mean())
        b.step = logged
        score = memory_task(b, 0, epochs=2)
        traces[every] = np.array(trace)
        print(f"   every {every:>2}: memory score {score:.3f}   hormone level mean {traces[every].mean():.4f}")
    a, c = traces[1], traces[10]
    print(f"   course of the hormone level: correlation {np.corrcoef(a, c)[0, 1]:.3f}, "
          f"mean difference {100 * np.abs(a - c).mean() / a.mean():.1f}% of its level")


def test_lod():
    print(f"\n2. Level of detail, stage 2, 6 presentations ({len(SEEDS)} seeds)")
    for frac in (0.0, 0.5, 0.8):
        scores, skipped, secs = [], [], []
        for seed in SEEDS:
            b = prepare(S.stage2(), seed, 0.0)
            b.lod_frac, b.lod_skipped, b.lod_total = frac, 0, 0
            t = time.time()
            scores.append(memory_task(b, seed, epochs=6))
            secs.append(time.time() - t)
            skipped.append(b.lod_skipped / b.lod_total)
        s = np.array(scores)
        print(f"   dormant below {frac:.0%} of usual input: score {s.mean():.3f} ± {s.std(ddof=1) / np.sqrt(len(s)):.3f}"
              f"   updates skipped {100 * np.mean(skipped):4.1f}%   task time {np.mean(secs):5.1f} s")


if __name__ == "__main__":
    import sys
    if "lod" not in sys.argv:
        test_clocks()
    test_lod()

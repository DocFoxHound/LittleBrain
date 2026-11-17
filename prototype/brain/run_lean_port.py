"""
Porting the brain to lean neurons (notes/25): does it still work, and what does it cost?

Brains compared (same wiring rules, same task):
  float      the original float synapse banks (reference)
  lean       lean storage at the reference sizes (seed + differences, per-type precision, day buffer,
             per-branch tags, gentle sleep reset)
  low        lean, low-synapse suite (biological sizes / 16; L2/3 IT = 256)
  standard   lean, standard suite (biological sizes; L2/3 IT = 4096)
Task: run_federation.memory_task (fill-in of the unseen half; stage 1 thalamus + cortex, stage 2 + hippocampus).

Run: PYTHONPATH=prototype:prototype/brain python3 prototype/brain/run_lean_port.py
"""

import os
import time
from multiprocessing import Pool

import numpy as np

MODES = {"float": {"LEAN": "0"}, "lean": {"LEAN": "1"}, "low": {"SUITE": "low"}, "standard": {"SUITE": "standard"}}
RUNS = (("stage1", 6), ("stage2", 1), ("stage2", 6))
SEEDS = range(int(os.environ.get("N_SEEDS", "4")))


def job(args):
    mode, stage, epochs, seed = args
    os.environ.pop("LEAN", None)
    os.environ.pop("SUITE", None)
    os.environ.update(MODES[mode])
    import brainspec as S
    from run_federation import memory_task, prepare
    t = time.time()
    b = prepare(getattr(S, stage)(), seed, 0.0)
    t_prep = time.time() - t
    score = memory_task(b, seed, epochs)
    st = b.storage()
    return mode, stage, epochs, seed, score, time.time() - t, t_prep, st


def main():
    jobs = [(m, s, e, seed) for m in MODES for s, e in RUNS for seed in SEEDS]
    jobs.sort(key=lambda j: j[0] != "standard")  # longest first
    with Pool(int(os.environ.get("PROCS", "16"))) as pool:
        out = pool.map(job, jobs, chunksize=1)
    print(f"Memory completion (fill-in of the unseen half, ± s.e., {len(SEEDS)} seeds) and cost per brain\n")
    print(f"   {'brain':<10}{'stage1 x6':>16}{'stage2 x1':>16}{'stage2 x6':>16}{'synapses':>11}"
          f"{'float MB':>10}{'packed MB':>11}{'lean MB':>9}{'min/run':>9}")
    for m in MODES:
        cells = []
        for s, e in RUNS:
            v = np.array([o[4] for o in out if o[:3] == (m, s, e)])
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}")
        st = [o[7] for o in out if o[0] == m and o[1] == "stage2" and o[2] == 6]
        mins = np.mean([o[5] for o in out if o[0] == m and o[1] == "stage2" and o[2] == 6]) / 60
        f = lambda k: np.mean([x[k] for x in st])
        print(f"   {m:<10}" + "".join(f"{c:>16}" for c in cells) + f"{f('synapses') / 1e6:10.2f}M"
              f"{f('float MB'):10.0f}{f('packed MB'):11.1f}{f('lean MB'):9.2f}{mins:9.1f}")
    print("\n(storage columns: stage 2 after 6 presentations and a night's sleep)")


if __name__ == "__main__":
    main()

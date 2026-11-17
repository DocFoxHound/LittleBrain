"""
The memory tests at a given brain size, on the official scale (notes/24-25): Biological Synapse Count Scale,
partner cap, lean neurons, seeded-choice turnover. Same tasks as run_lean_port.py:
  stage1 x6   thalamus + 2 cortical areas, 6 presentations
  stage2 x1   + hippocampal loop, 1 presentation (where a hippocampus should help)
  stage2 x6   + hippocampal loop, 6 presentations
Each seed's stage-2 brain is built and developed once, then copied for the x1 and x6 tests.
Results are appended per seed to a JSON-lines file, so a long run can be followed and resumed.

Run (fly size, GPU):  DEVICE=gpu PYTHONPATH=prototype:prototype/brain .venv/bin/python prototype/brain/run_fly.py
Env: NEURONS (default 140000), N_SEEDS (16), OUT (results file), PROCS (CPU only: parallel seeds)
"""

import copy
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import backend
import brainspec as S
from run_federation import memory_task, prepare

NEURONS = int(os.environ.get("NEURONS", "140000"))
SEEDS = range(int(os.environ.get("N_SEEDS", "16")))
OUT = os.environ.get("OUT", f"results_fly_{NEURONS}.jsonl")


def free():
    if backend.DEVICE == "gpu":
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


def one_seed(seed):
    row = {"seed": seed, "neurons": NEURONS, "device": backend.DEVICE}
    t = time.time()
    b = prepare(S.sized(S.stage1(), NEURONS * 3 // 7), seed, 0.0)  # stage 1 alone is ~3/7 of the stage-2 brain
    row["stage1 x6"] = memory_task(b, seed, 6)
    del b
    free()
    b = prepare(S.sized(S.stage2(), NEURONS), seed, 0.0)
    row["n_neurons"], row["synapses"] = b.n_neurons, b.storage()["synapses"]
    b.to("cpu")
    snap = copy.deepcopy(b)
    b.to(backend.DEVICE)
    row["stage2 x1"] = memory_task(b, seed, 1)
    del b
    free()
    snap.to(backend.DEVICE)
    row["stage2 x6"] = memory_task(snap, seed, 6)
    row["lean MB"] = snap.storage()["lean MB"]
    del snap
    free()
    row["minutes"] = (time.time() - t) / 60
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def summary(path=OUT):
    rows = [json.loads(line) for line in open(path)]
    print(f"{len(rows)} brains of ~{rows[0]['n_neurons']:,} neurons, {np.mean([r['synapses'] for r in rows]) / 1e6:.1f}M "
          f"synapses (stage 2), {np.mean([r['minutes'] for r in rows]):.1f} min per seed")
    for k in ("stage1 x6", "stage2 x1", "stage2 x6"):
        v = np.array([r[k] for r in rows])
        print(f"   {k:<10} fill-in {v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}   range {v.min():.2f} .. {v.max():.2f}")
    print(f"   lean long-term storage: {np.mean([r['lean MB'] for r in rows]):.1f} MB per brain")


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["seed"] for line in open(OUT)}
    todo = [s for s in SEEDS if s not in done]
    if backend.DEVICE == "gpu" or int(os.environ.get("PROCS", "1")) == 1:
        for s in todo:
            print(one_seed(s), flush=True)
    else:
        with Pool(int(os.environ["PROCS"])) as pool:
            for r in pool.imap_unordered(one_seed, todo):
                print(r, flush=True)
    summary()

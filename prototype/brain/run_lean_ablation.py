"""Which lean ingredient costs recall? Stage 1, 6 presentations; lean features switched off one at a time."""
import os
from multiprocessing import Pool

import numpy as np

VARIANTS = {
    "float (reference)": {"LEAN": "0"},
    "lean (all features)": {"LEAN": "1"},
    # "lean, high precision (255 levels)": {"LEAN": "1", "LEAN_Q": "127"},
    # "lean, per-synapse tags": {"LEAN": "1", "LEAN_TAGS": "synapse"},
    # "lean, no sleep reset": {"LEAN": "1", "LEAN_RESET": "0"},
    "lean, all three off (storage only)": {"LEAN": "1", "LEAN_Q": "127", "LEAN_TAGS": "synapse", "LEAN_RESET": "0"},
}
SEEDS = range(int(os.environ.get("N_SEEDS", "6")))


def job(args):
    name, seed = args
    for k in ("LEAN", "LEAN_Q", "LEAN_TAGS", "LEAN_RESET", "SUITE", "SCALE"):
        os.environ.pop(k, None)
    os.environ.update(VARIANTS[name])
    import brainspec as S
    from run_federation import memory_task, prepare
    return name, memory_task(prepare(S.stage1(), seed, 0.0), seed, 6)


if __name__ == "__main__":
    with Pool(18) as pool:
        out = pool.map(job, [(v, s) for v in VARIANTS for s in SEEDS], chunksize=1)
    print(f"Stage 1, 6 presentations, fill-in ± s.e. ({len(SEEDS)} seeds)")
    for v in VARIANTS:
        x = np.array([o[1] for o in out if o[0] == v])
        print(f"   {v:<38} {x.mean():.2f} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f}   [{x.min():.2f}..{x.max():.2f}]")

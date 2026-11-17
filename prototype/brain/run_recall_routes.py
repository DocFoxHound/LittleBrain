"""
Which pathway carries recall? Lesion tests on the wiring-test circuit (notes/19, notes/20).

Every brain develops normally (4 days). Lesions are applied before learning the 6 patterns
("learn + recall") or only at recall time ("recall only"), like acute lesions in experiments.
Waves are off (dropped after notes/19).

Run: PYTHONPATH=prototype python3 prototype/brain/run_recall_routes.py   (N_SEEDS, default 6)
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
from network import Brain  # noqa: E402
from run_wiring_test import encode, make_patterns, recall, score  # noqa: E402

SEEDS = range(int(os.environ.get("N_SEEDS", "6")))
CONDITIONS = {
    "intact": ([], []),
    "no apical feedback (B->A)": (["feedback"], []),
    "no top-down to VIP (B->A)": (["topdown"], []),
    "no B->A at all": (["feedback", "topdown"], []),
    "area B silenced": (["B"], []),
    "A's local synapses can't learn": (["learn_A_local"], []),
    "long-range synapses can't learn": (["learn_long"], []),
    "B->A cut at recall only": ([], ["feedback", "topdown"]),
    "area B silenced at recall only": ([], ["B"]),
}


def main():
    rows = {}
    for seed in SEEDS:
        base = Brain(seed=seed)
        base.calibrate()
        base.develop()
        pats, cues = make_patterns(base, np.random.default_rng(100 + seed))
        for name, (during, at_recall) in CONDITIONS.items():
            b = copy.deepcopy(base)
            b.ablate(*during)
            t = encode(b, pats, None, False)
            b.sleep()
            b.ablate(*at_recall)
            rows.setdefault(name, []).append(score(b, *recall(b, pats, cues, None, False, t)))
        print(f"seed {seed} done", flush=True)
    print(f"\nWhich pathway carries recall? ({len(SEEDS)} seeds; fill-in of the unseen right half; "
          f"± = standard error)")
    print(f"{'condition':<34}{'fill-in':>16}{'identify (A)':>14}{'identify (B)':>14}")
    for name, rs in rows.items():
        f = np.array([r["fill-in"] for r in rs])
        print(f"{name:<34}{f.mean():>9.3f} ± {f.std(ddof=1) / np.sqrt(len(f)):.3f}"
              f"{np.mean([r['identify A'] for r in rs]):>14.2f}{np.mean([r['identify B'] for r in rs]):>14.2f}")


if __name__ == "__main__":
    main()

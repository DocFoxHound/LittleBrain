"""
Does the amygdala help a character learn to avoid danger (notes/32)? Same brain, amygdala intact vs silenced.
Run: .venv/bin/python prototype/brain/run_danger_game.py   (env SEED, TURNS, LESION=1 for the silenced copy only)
"""

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

for k, v in {"HOMEO_RULE": "log", "L4_L23_GAIN": "3", "CORTEX_LRF": "0.05", "HOMEO_SLOW": "STR.SO:0.1",
             "BRAIN": "full"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
from agent import BrainAgent  # noqa: E402
from games import DangerGame  # noqa: E402
from run_signal_game import prepare  # noqa: E402

AMYGDALA = {"BLA", "CeL", "CeM"}


def blocks(log, key, block=50, when=None):
    out = []
    for i in range(0, len(log), block):
        L = [e for e in log[i:i + block] if when is None or when(e)]
        out.append(np.mean([e[key] for e in L]) if L else float("nan"))
    return " ".join(f"{v:.2f}" for v in out)


def main():
    seed, turns = int(os.environ.get("SEED", "0")), int(os.environ.get("TURNS", "300"))
    t0 = time.time()
    b = prepare(seed)
    print(f"{b.n_neurons} neurons, prepared in {time.time() - t0:.0f} s", flush=True)
    which = {"1": ("amygdala silenced",), "0": ("amygdala intact",)}.get(os.environ.get("LESION", ""),
                                                                        ("amygdala intact", "amygdala silenced"))
    for name in which:
        brain = copy.deepcopy(b)
        if "silenced" in name:
            brain.ablated |= AMYGDALA
        agent, log = BrainAgent(brain, 4, seed=seed), []
        agent.play(DangerGame(seed=seed), turns, log=log)
        for e in log:
            e["hurt"] = float(e["reward"] < 0)
            e["fed"] = float(e["reward"] > 0)
        hurt = [e["hurt"] for e in log if e["danger"]]
        print(f"{name:<18} hurt on danger, per 50 turns: {blocks(log, 'hurt', when=lambda e: e['danger'])}  "
              f"| fed on food: {blocks(log, 'fed', when=lambda e: not e['danger'])}  "
              f"| total hurt {int(sum(hurt))}, last 100 turns hurt rate {np.mean([e['hurt'] for e in log[-100:] if e['danger']]):.2f}",
              flush=True)


if __name__ == "__main__":
    main()

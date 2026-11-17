"""
Does a LittleBrain character learn which button goes with which picture (notes/29)?

A learning brain vs an identical copy with learning off, same game. Chance = 1 / n_actions.
Run: .venv/bin/python prototype/brain/run_signal_game.py   (env SEED, TURNS; HOMEO_RULE, L4_L23_GAIN, CORTEX_LRF,
     MSN_LR, MSN_DECAY)
"""

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

# The cortex fixes from notes/28 are on by default for games (override with env)
for k, v in {"HOMEO_RULE": "log", "L4_L23_GAIN": "3", "CORTEX_LRF": "0.05"}.items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
from agent import BrainAgent  # noqa: E402
from federation import Brain  # noqa: E402
from games import SignalGame  # noqa: E402
from run_wiring_test import blobs  # noqa: E402


def prepare(seed):
    b = Brain(S.stage_bg(), seed=seed)
    rng = np.random.default_rng(1000 + seed)
    zero = np.zeros(len(b.pos["MOT"]), np.float32)
    b.calibrate(lambda: {"retina": (rng.random(len(b.pos["retina"])) < 0.2).astype(np.float32), "MOT": zero})
    b.develop(lambda: {"retina": blobs(b.pos["retina"], rng), "MOT": zero}, days=2)
    x = None
    for t in range(360):   # wake up: the night's changes leave the cortex quiet for a while (notes/28)
        if t % 9 == 0:
            x = blobs(b.pos["retina"], rng)
        b.step({"retina": x if t % 9 < 6 else 0 * x, "MOT": zero}, ach=0.5)
    return b


def curve(r, block=50):
    r = np.array(r)
    return " ".join(f"{r[i:i + block].mean():.2f}" for i in range(0, len(r), block))


def main():
    seed, turns = int(os.environ.get("SEED", "0")), int(os.environ.get("TURNS", "400"))
    t0 = time.time()
    b = prepare(seed)
    print(f"{b.n_neurons} neurons, prepared in {time.time() - t0:.0f} s", flush=True)
    results = {}
    runs = (("learning", True),) if os.environ.get("NO_CONTROL") else (("learning", True), ("learning off", False))
    for name, learn in runs:
        t0 = time.time()
        agent = BrainAgent(copy.deepcopy(b), 4, seed=seed, learn=learn, critic=os.environ.get("CRITIC", "average"),
                           look=int(os.environ.get("LOOK", "20")), read=int(os.environ.get("READ", "16")))
        log = []
        r = agent.play(SignalGame(seed=seed), turns, log=log)
        results[name] = (r, log)
        print(f"{name:<13} reward per 50 turns: {curve(r)}   last 100: {np.mean(r[-100:]):.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    log = results["learning"][1]
    print("last turn: release", np.round(log[-1]["release"], 3), "D1", np.round(log[-1]["D1"], 3),
          "D2", np.round(log[-1]["D2"], 3))


if __name__ == "__main__":
    main()

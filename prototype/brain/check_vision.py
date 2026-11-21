"""
Vision checks for a brain with a retina (notes 72+). Prints JSON lines.

  orientation   drifting gratings at 8 angles: each V1 cell's orientation selectivity (OSI) and preferred angle
  motion        drifting gratings in 4 directions: direction selectivity (DSI) per area     (note 74)
  colour        red / green / blue / grey patches: colour preference per area                (note 74)

  .venv/bin/python prototype/brain/check_vision.py --load-brain PATH orientation [--groups A.L4 A.L23 ...]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import run_signal_game  # noqa: E402,F401  (sets the default environment)
from federation import Brain  # noqa: E402
from retina import Retina  # noqa: E402

SIZE = 64


def grating(theta, phase, sf=6, contrast=0.4, colour=None):
    yy, xx = (np.mgrid[0:SIZE, 0:SIZE] + .5) / SIZE
    g = 0.5 + contrast * np.sin(2 * np.pi * sf * (xx * np.cos(theta) + yy * np.sin(theta)) + phase)
    return g.astype(np.float32) if colour is None else (g[..., None] * np.asarray(colour, np.float32))


def responses(b, stimuli, groups, frames=10, blank=6, reps=2):
    """stimuli: list of callables frame(s) -> image. Returns {group: (n_stimuli, cells)} mean rates."""
    eye = Retina(b)
    quiet = {k: np.zeros(len(b.pos[k]), np.float32) for k in b.spec.inputs}
    grey = np.full((SIZE, SIZE), 0.5, np.float32)
    out = {k: np.zeros((len(stimuli), b.g[k].n)) for k in groups}
    for _ in range(reps):
        for i, stim in enumerate(stimuli):
            for s in range(blank):
                b.step({**quiet, **eye(grey)}, ach=0.5)
            for s in range(frames):
                b.step({**quiet, **eye(stim(s))}, ach=0.5)
                if s >= frames // 2:
                    for k in groups:
                        out[k][i] += np.asarray(b.g[k].rate)
    return {k: v / (reps * (frames - frames // 2)) for k, v in out.items()}


def orientation(b, groups, n=8):
    thetas = np.arange(n) * np.pi / n
    stims = [lambda s, th=th: grating(th, 0.7 * s) for th in thetas]
    R = responses(copy.deepcopy(b), stims, groups)
    rows = []
    for k, r in R.items():
        active = r.max(0) > 0.02
        z = (r * np.exp(2j * thetas)[:, None]).sum(0) / (r.sum(0) + 1e-9)
        osi = np.abs(z)[active]
        pref = (np.angle(z[active]) / 2) % np.pi
        hist = np.histogram(pref, bins=4, range=(0, np.pi))[0] / max(active.sum(), 1)
        rows.append({"check": "orientation", "group": k, "responding": float(active.mean()),
                     "osi_median": float(np.median(osi)) if osi.size else 0.0,
                     "tuned_fraction": float((osi > 0.3).mean()) if osi.size else 0.0,
                     "preferred_spread": [round(float(h), 2) for h in hist]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-brain", required=True)
    ap.add_argument("check", choices=["orientation"])
    ap.add_argument("--groups", nargs="*", default=["TH.TC", "A.L4", "A.L23", "A.L5IT", "B.L23"])
    a = ap.parse_args()
    b = Brain.load(a.load_brain)
    for row in {"orientation": orientation}[a.check](b, a.groups):
        print(json.dumps(row))


if __name__ == "__main__":
    main()

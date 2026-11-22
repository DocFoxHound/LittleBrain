"""
Place cells (plan phase 2, note 83). Prints JSON lines.

A brain is walked at random through the homing room (games/spatial_games.py) with learning on, seeing the view and
receiving self-motion. For every CA1 pyramidal cell: a rate map (mean rate per room cell) from the first and the
second half of the walk. A cell counts as place-like if it is active, its map is concentrated (peak > 3 x its mean)
and stable (split-half correlation > 0.5).
  intact        view + head direction + grid cells
  no self-motion  grid and head-direction cells silent: only the view tells where you are
  shuffled      intact, but the second half's positions shuffled (chance level of the measure)

  .venv/bin/python prototype/brain/check_spatial.py --load-brain PATH [--steps 800]
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
import run_signal_game  # noqa: E402,F401
from federation import Brain  # noqa: E402
from games.spatial_games import HomingGame, N  # noqa: E402
from retina import Retina  # noqa: E402
from spatial import SelfMotion  # noqa: E402


def walk(b, steps, seed, self_motion=True, hold=6):
    g = HomingGame(seed=seed)
    g.reset()
    eye, sm = Retina(b), SelfMotion(b, seed)
    sm.reset()
    x0 = g.pos
    sm.pos = np.array([x0[0] - 3.0, x0[1] - 3.0])
    quiet = {k: np.zeros(len(b.pos[k]), np.float32) for k in b.spec.inputs}
    rng = np.random.default_rng(seed + 7)
    pos, rates = [], []
    for t in range(steps):
        sm.move([g._walk(int(rng.integers(4)))])
        if g.pos == (3, 3):
            sm.pos = np.zeros(2)       # the home landmark fix, as in the game
        ch = {**quiet, **eye(g._view())}
        if self_motion:
            ch.update(sm.channels())
        acc = 0
        for s in range(hold):
            b.step(ch, learn=True, ach=0.6, dopamine=0.0, arousal=0.3)
            if s >= hold - 3:
                acc = acc + np.asarray(b.g["CA1.pyr"].rate)
        pos.append(g.pos[1] * N + g.pos[0])
        rates.append(acc / 3)
    return np.array(pos), np.array(rates)


def place_stats(pos, R, shuffle=False, seed=0):
    h = len(pos) // 2
    p2 = pos[h:].copy()
    if shuffle:
        p2 = np.random.default_rng(seed).permutation(p2)

    def ratemap(p, r):
        M = np.full((N * N, r.shape[1]), np.nan)
        for c in range(N * N):
            if (p == c).sum() >= 2:
                M[c] = r[p == c].mean(0)
        return M
    A, B = ratemap(pos[:h], R[:h]), ratemap(p2, R[h:])
    ok = ~np.isnan(A[:, 0]) & ~np.isnan(B[:, 0])
    A, B = A[ok], B[ok]
    active = R.mean(0) > 0.02
    corr = np.array([np.corrcoef(A[:, i], B[:, i])[0, 1] if A[:, i].std() > 1e-6 and B[:, i].std() > 1e-6 else 0.0
                     for i in range(R.shape[1])])
    m = (A + B) / 2
    peaked = m.max(0) > 3 * np.maximum(m.mean(0), 1e-6)
    place = active & peaked & (corr > 0.5)
    return dict(active=int(active.sum()), place_like=int(place.sum()),
                place_fraction=round(float(place.sum() / max(active.sum(), 1)), 3),
                median_stability=round(float(np.median(corr[active])) if active.any() else 0.0, 3),
                cells_seen=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-brain", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b = Brain.load(a.load_brain, quiet=True)
    pos, R = walk(copy.deepcopy(b), a.steps, a.seed)
    print(json.dumps(dict(check="place cells", condition="intact", **place_stats(pos, R))))
    print(json.dumps(dict(check="place cells", condition="shuffled", **place_stats(pos, R, shuffle=True))))
    pos, R = walk(copy.deepcopy(b), a.steps, a.seed, self_motion=False)
    print(json.dumps(dict(check="place cells", condition="no self-motion", **place_stats(pos, R))))


if __name__ == "__main__":
    main()

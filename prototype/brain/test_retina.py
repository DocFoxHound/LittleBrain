"""Retina signature behaviours (note 72): contrast, colour opponency, transients, direction selectivity, fovea."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
from federation import Brain  # noqa: E402
from retina import Retina  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


def disc(x, y, colour, bg=0.5, size=64, r=0.09):
    yy, xx = (np.mgrid[0:size, 0:size] + .5) / size
    img = np.full((size, size, 3), bg, np.float32)
    img[(xx - x) ** 2 + (yy - y) ** 2 < r * r] = colour
    return img


def main():
    b = Brain(S.full_brain_vision(), seed=0)
    eye = Retina(b)
    xy, _ = eye._where("RET.on", (64, 64))
    # a small spot covers a foveal cell's centre but not its surround (a large even area gives no response)
    inside = np.linalg.norm(xy - .5, axis=1) < .01
    o = eye(disc(.5, .5, [.9, .9, .9], r=.025))
    check("a small bright spot: ON cells centred on it fire, OFF cells there are silent",
          o["RET.on"][inside].mean() > .1 and o["RET.off"][inside].mean() < .02,
          f"ON {o['RET.on'][inside].mean():.2f}, OFF {o['RET.off'][inside].mean():.3f} (inside the spot)")
    o = Retina(b)(disc(.5, .5, [.1, .1, .1], r=.025))
    check("a dark spot: the reverse", o["RET.off"][inside].mean() > .1 and o["RET.on"][inside].mean() < .02,
          f"OFF {o['RET.off'][inside].mean():.2f}, ON {o['RET.on'][inside].mean():.3f}")
    o = Retina(b)(disc(.5, .5, [.85, .2, .2]))
    check("red excites red-green ON cells, not OFF", o["RET.rg_on"].max() > 3 * o["RET.rg_off"].max() + .05)
    o = Retina(b)(disc(.5, .5, [.2, .2, .95]))
    check("blue excites blue-yellow cells", o["RET.by"].max() > .3, f"{o['RET.by'].max():.2f}")
    grey = np.full((64, 64, 3), .5, np.float32)
    e = Retina(b)
    e(grey)
    o = e(disc(.5, .5, [.9, .9, .9]))
    p = e(disc(.5, .5, [.9, .9, .9]))
    check("transient cells respond when a spot appears, not while it stays",
          o["RET.ton"].max() > .2 and p["RET.ton"].max() < .02, f"onset {o['RET.ton'].max():.2f}, held {p['RET.ton'].max():.3f}")
    res = {}
    for name, (dx, dy) in {"right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}.items():
        e, acc = Retina(b), []
        for i in range(6):
            out = e(disc(.5 + dx * (i - 3) * .02, .5 + dy * (i - 3) * .02, [0, 0, 0]))
            if i:
                acc.append([out[f"RET.ds{j}"].mean() for j in range(4)])
        res[name] = np.mean(acc, 0)
    ok = all(int(np.argmax(res[n])) == j and res[n][j] > 5 * res[n][(j + 2) % 4]
             for j, n in enumerate(("right", "down", "left", "up")))
    check("direction-selective cells prefer their own direction (>5x the opposite)", ok,
          " ".join(f"{n}:{res[n].round(3).tolist()}" for n in res))
    off, sig = eye.geo["RET.on"]
    ecc = np.linalg.norm(off, axis=1)
    check("fovea: centre cells have smaller fields than peripheral cells",
          sig[ecc < .1].mean() * 3 < sig[ecc > .6].mean(), f"centre {sig[ecc < .1].mean():.4f}, periphery {sig[ecc > .6].mean():.4f}")
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

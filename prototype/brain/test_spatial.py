"""Head-direction and grid cells (spatial.py, note 83): tuning, path integration and the home fix."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spatial  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


class FakeBrain:
    def __init__(self):
        g8 = (np.stack(np.meshgrid(np.arange(8), np.arange(8)), -1).reshape(-1, 2) + 0.5) / 8
        g16 = (np.stack(np.meshgrid(np.arange(16), np.arange(16)), -1).reshape(-1, 2) + 0.5) / 16
        self.pos = {"ATN.hd": g8, "MEC.grid": g16}


def main():
    sm = spatial.SelfMotion(FakeBrain())
    sm.heading = 0.0
    a = sm.channels()["ATN.hd"]
    sm.heading = np.pi
    b = sm.channels()["ATN.hd"]
    check("head-direction cells: each fires for its own heading", int(a.argmax()) != int(b.argmax())
          and float(np.corrcoef(a, b)[0, 1]) < 0)
    spatial.DRIFT = 0.0
    sm.reset()
    sm.move([(0.0, 1.0), (np.pi / 2, 2.0)])
    check("path integration: east 1 then turn and go 2 lands at (1, 2)", np.allclose(sm.pos, [1, 2], atol=1e-6),
          str(sm.pos))
    sm.move([], {"home_scent": 1.0})
    check("smelling home resets the position estimate", np.allclose(sm.pos, 0))
    maps = []
    for x in np.linspace(-3, 3, 25):
        for y in np.linspace(-3, 3, 25):
            sm.pos = np.array([x, y])
            maps.append(sm.channels()["MEC.grid"])
    M = np.array(maps).reshape(25, 25, -1)
    peaks = [(M[:, :, i] > 0.6 * M[:, :, i].max()).sum() for i in range(M.shape[2])]
    check("grid cells: several separate firing fields per cell, covering a fraction of the room",
          0.05 < np.mean(peaks) / 625 < 0.5, f"mean field area {np.mean(peaks) / 625:.2f}")
    small, large = M[:, :, :5].std(), M[:, :, -5:].std()
    period = lambda i: np.abs(np.fft.rfft(M[12, :, i] - M[12, :, i].mean()))[1:].argmax() + 1
    check("grid spacing grows along the map's y axis (dorsal small, ventral large)",
          np.mean([period(i) for i in range(16)]) > np.mean([period(i) for i in range(240, 256)]))
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

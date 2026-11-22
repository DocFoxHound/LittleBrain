"""V1 edge cells (edges.py, note 85): orientation-selective, phase-invariant, quiet on blank fields."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import edges  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


class FakeBrain:
    def __init__(self):
        g32 = (np.stack(np.meshgrid(np.arange(32), np.arange(32)), -1).reshape(-1, 2) + 0.5) / 32
        g16 = (np.stack(np.meshgrid(np.arange(16), np.arange(16)), -1).reshape(-1, 2) + 0.5) / 16
        self.pos = {"RET.on": g32, "RET.off": g32, **{k: g16 for k in edges.EDGE_GROUPS}}


def channels(img):
    """A stand-in retina: ON = brighter than the mean, OFF = darker."""
    v = (img - img.mean()).ravel() * 4.0          # with the real retina's gain (retina.GAIN)
    return {"RET.on": np.tanh(np.maximum(v, 0)).astype(np.float32),
            "RET.off": np.tanh(np.maximum(-v, 0)).astype(np.float32)}


def main():
    e = edges.EdgeCells(FakeBrain())
    yy, xx = np.mgrid[0:32, 0:32]
    blank = e(channels(np.full((32, 32), 0.5)))
    check("a blank field drives no edge cell", max(float(v.max()) for v in blank.values()) == 0.0)
    resp = []
    for a in edges.ANGLES:
        stripes = 0.5 + 0.4 * np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / edges.WAVELENGTH)
        out = e(channels(stripes))
        resp.append([float(out[k].mean()) for k in edges.EDGE_GROUPS])
    R = np.array(resp)
    check("each orientation drives its own cells most", all(int(R[i].argmax()) == i for i in range(4)),
          np.array2string(R, precision=2))
    shifted = []
    for ph in (0.0, np.pi / 2, np.pi):
        stripes = 0.5 + 0.4 * np.sin(2 * np.pi * xx / edges.WAVELENGTH + ph)
        shifted.append(float(e(channels(stripes))["A.E0"].mean()))
    check("phase-invariant: shifting the stripes barely changes the response",
          max(shifted) - min(shifted) < 0.2 * max(shifted), str(np.round(shifted, 3)))
    sq = np.full((32, 32), 0.5)
    sq[10:22, 10:22] = 0.1
    out = e(channels(sq))
    check("a square's vertical and horizontal edges drive the 0 and 90 degree cells, not the diagonals",
          min(out["A.E0"].max(), out["A.E2"].max()) > 3 * max(out["A.E1"].max(), out["A.E3"].max(), 0.01))
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

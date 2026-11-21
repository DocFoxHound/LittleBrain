"""Per-structure map resolution (note 71): groups on 16x16 / 32x32 maps alongside the 8x8 brain."""
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
from federation import Brain, Pooled  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


def fine_spec():
    spec = S.stage1()
    spec.groups = [replace(g, grid=32, per_tile=1) if g.key == "retina" else
                   replace(g, grid=16, per_tile=1) if g.key == "TH.TC" else g for g in spec.groups]
    # a population-level connection between maps of different sizes: relay -> reticular nucleus (8x8)
    spec.pooled.append(Pooled("test fine->coarse", "TH.TC", "TH.TRN", 0.5, sigma=0))
    return spec


def main():
    b = Brain(fine_spec(), seed=0)
    n_ret, n_tc = len(b.pos["retina"]), len(b.pos["TH.TC"])
    check("sizes follow each map (32x32 retina, 16x16 relay)", (n_ret, n_tc) == (1024, 256), f"{n_ret}, {n_tc}")
    ok = True
    for k, gs in (("retina", 32), ("TH.TC", 16)):
        P = b.pos[k]
        ok &= np.array_equal(b.ftile[k], np.minimum((P[:, 0] * gs).astype(int), gs - 1) + np.minimum((P[:, 1] * gs).astype(int), gs - 1) * gs)
        ok &= np.array_equal(b.tile[k], np.minimum((P[:, 0] * 8).astype(int), 7) + np.minimum((P[:, 1] * 8).astype(int), 7) * 8)
    check("fine and coarse tiles agree with positions", bool(ok))
    K = np.asarray(b._Kx(32, 32, 0))
    check("same-place kernel between equal fine maps is nearly one-to-one", float(np.diag(K).mean()) > 0.85,
          f"mean diagonal weight {float(np.diag(K).mean()):.3f}")
    K2 = np.asarray(b._Kx(16, 8, 1.0))
    check("kernel rows sum to one across map sizes", np.allclose(K2.sum(1), 1, atol=1e-5))
    # a local spot of light activates the matching place downstream
    quiet = {}
    x = ((b.pos["retina"][:, 0] < 0.25) & (b.pos["retina"][:, 1] < 0.25)).astype(np.float32)
    for _ in range(6):
        b.step({"retina": x, **quiet})
    r = np.asarray(b.g["TH.TC"].rate)
    inside = (b.pos["TH.TC"][:, 0] < 0.25) & (b.pos["TH.TC"][:, 1] < 0.25)
    check("a spot of light drives the relay cells at the same place", r[inside].mean() > 2 * r[~inside].mean(),
          f"inside {r[inside].mean():.3f}, elsewhere {r[~inside].mean():.3f}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fine.brain")
        b.save(path)
        c = Brain.load(path)
        c.step({"retina": x})
        b.step({"retina": x})
        check("checkpoint round trip keeps fine maps and behaviour",
              np.array_equal(np.asarray(b.r), np.asarray(c.r)) and np.array_equal(c.ftile["retina"], b.ftile["retina"]))
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

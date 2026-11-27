"""World model and look-ahead (world_model.py, note 96) on its own, in a small grid: it learns where actions lead,
finds the food, and re-routes when the short way is walled off."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_model import WorldModel  # noqa: E402

passed = []
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


def code(p, n=5, width=0.7):
    c = np.arange(n)
    cx, cy = np.meshgrid(c, c)
    return np.exp(-((cx.ravel() - p[0]) ** 2 + (cy.ravel() - p[1]) ** 2) / (2 * width ** 2)).astype(np.float32)


def episode(wm, walls, rng, start=(0, 2), food=(4, 2), limit=20, explore=0.1, learn=True):
    p, steps = start, 0
    while steps < limit:
        phi = code(p)
        a = int(rng.integers(4)) if rng.random() < explore else int(np.argmax(wm.q(phi)))
        q = (p[0] + MOVES[a][0], p[1] + MOVES[a][1])
        if not (0 <= q[0] < 5 and 0 <= q[1] < 5) or q in walls:
            q = p
        steps += 1
        fed = q == food
        if learn:
            wm.observe(phi, a, 1.0 if fed else -0.02, None if fed else code(q))
        p = q
        if fed:
            return steps
    return None


def main():
    rng = np.random.default_rng(0)
    wm = WorldModel(25, 4, replay=20, seed=0)
    wall = {(2, y) for y in range(5) if y not in (0, 2)}           # a wall at x=2 with gaps at y=0 and y=2
    for _ in range(30):
        episode(wm, wall, rng, explore=0.5)                        # explore
    for _ in range(20):
        episode(wm, wall, rng)
    short = episode(wm, wall, rng, explore=0.0, learn=False)
    check("takes the short route once it knows the room", short == 4, f"steps {short}")
    blocked = wall | {(2, 2)}                                       # the middle gap closes: only y=0 is open
    found = [episode(wm, blocked, rng, explore=0.1) for _ in range(5)]
    detour = episode(wm, blocked, rng, explore=0.0, learn=False)
    check("re-routes around the new wall within a few episodes", detour is not None and detour <= 10,
          f"first episodes {found}, then {detour} steps (optimal 8)")
    check("the model has learned that walking into the wall goes nowhere",
          float(code((1, 2)) @ (wm.T[1] @ code((1, 2)))) > float(code((2, 2)) @ (wm.T[1] @ code((1, 2)))))
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

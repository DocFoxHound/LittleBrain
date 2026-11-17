"""
Honest brain-level checks (notes/26 showed the old memory score measured timing, not memory).

  reliability(b, ...)   test-retest per group: show each pattern twice, far apart and in a different order.
                        Score = (match with its own earlier response) - (match with other patterns' responses).
                        1 = perfectly reproducible and pattern-specific, 0 = unrelated to the input.
  memory(b, ...)        learn patterns, then show ALL full patterns first and the half-cues later in a scrambled
                        order, so a slowly drifting state can't fake a match. Always compare with an untrained
                        copy of the same brain (`memory_vs_control`).

Run: .venv/bin/python prototype/brain/diagnostics.py   (SEED, STAGE=1|2, DEVICE=cpu|gpu)
"""

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
from backend import to_host  # noqa: E402
from federation import ExcitatoryGroup  # noqa: E402
from run_wiring_test import BLANK, EPOCHS, HOLD, N_PATTERNS, blobs, corr  # noqa: E402


def present(b, x, n, groups, blank=6, inp="retina"):
    """Blank for `blank` steps, then show x for n steps; mean rate per group over the last 4 steps."""
    for _ in range(blank):
        b.step({inp: 0 * x}, ach=0.2)
    acc = {k: [] for k in groups}
    for s in range(n):
        b.step({inp: x}, ach=0.2)
        if s >= n - 4:
            for k in groups:
                acc[k].append(to_host(b.g[k].rate).copy())
    return {k: np.mean(v, 0) for k, v in acc.items()}


def _specificity(a, c):
    """a, c: (patterns, cells) responses. Own-minus-other correlation after removing the common response."""
    base = np.concatenate([a, c]).mean(0)
    a, c = a - base, c - base
    n = len(a)
    own = [corr(a[i], c[i]) for i in range(n)]
    other = [np.mean([corr(a[i], c[j]) for j in range(n) if j != i]) for i in range(n)]
    return float(np.mean(own) - np.mean(other))


def reliability(b, pats, groups=None, hold=8, seed=0):
    groups = groups or [k for k, g in b.g.items() if isinstance(g, ExcitatoryGroup)]
    rng = np.random.default_rng(seed)
    first = [present(b, p, hold, groups) for p in pats]
    order = rng.permutation(len(pats))
    second = [None] * len(pats)
    for i in order:
        second[i] = present(b, pats[i], hold, groups)
    out = {}
    for k in groups:
        A = np.array([f[k] for f in first])
        C = np.array([s[k] for s in second])
        out[k] = {"rate": float(A.mean()), "retest": _specificity(A, C)}
    return out


def learn(b, pats, epochs=EPOCHS, hold=HOLD):
    for _ in range(epochs):
        for p in pats:
            for s in range(hold + BLANK):
                b.step({"retina": p if s < hold else 0 * p}, learn=True, ach=0.8,
                       dopamine=1.0 if s == 2 else 0.0, arousal=0.6)
    b.sleep()


def wake(b, steps=360, seed=0):
    """Awake, no learning: unrelated new patterns, so homeostasis can settle after the night's changes."""
    rng = np.random.default_rng(10_000 + seed)
    x = None
    for t in range(steps):
        if t % (HOLD + BLANK) == 0:
            x = blobs(b.pos["retina"], rng)
        b.step({"retina": x if t % (HOLD + BLANK) < HOLD else 0 * x}, ach=0.5)


def memory(b, pats, cues, group="A.L23", recall=8, seed=0, region=None, rounds=2):
    """Full patterns first, then cues; each set shown `rounds` times in fresh random orders and averaged, so slow
    drift can't line up with the pattern order. Scored on `region` cells (default: the half of the map the cue
    leaves out, so the answer must be filled in). Returns own-minus-other match and identification accuracy."""
    rng = np.random.default_rng(seed + 7)
    sel = region if region is not None else b.pos[group][:, 0] > 0.55

    def show(xs):
        acc = np.zeros((len(xs), int(sel.sum())))
        for _ in range(rounds):
            for i in rng.permutation(len(xs)):
                acc[i] += present(b, xs[i], recall, [group])[group][sel]
        return acc / rounds

    full, cue = show(pats), show(cues)
    base = np.concatenate([full, cue]).mean(0)
    C = np.array([[corr(c - base, f - base) for f in full] for c in cue])
    return {"match": _specificity(full, cue), "accuracy": float(np.mean(C.argmax(1) == np.arange(len(pats))))}


def memory_vs_control(b, seed=0, epochs=EPOCHS, n=10, group="A.L23", wake_steps=360):
    rng = np.random.default_rng(100 + seed)
    pats = np.array([blobs(b.pos["retina"], rng) for _ in range(n)])
    cues = pats * (b.pos["retina"][:, 0] < 0.5)
    control = copy.deepcopy(b)
    control.sleep()  # same nightly changes, no learning of these patterns
    learn(b, pats, epochs)
    out = {}
    for name, brain in (("learned", b), ("untrained", control)):
        wake(brain, wake_steps, seed)
        out[name] = memory(brain, pats, cues, group, seed=seed)
    return out


def main():
    import brainspec as S
    from run_federation import prepare
    seed = int(os.environ.get("SEED", "0"))
    spec = S.stage2() if os.environ.get("STAGE") == "2" else S.stage1()
    t0 = time.time()
    b = prepare(spec, seed, float(os.environ.get("HOMEO", "0.0")))
    print(f"{b.n_neurons} neurons, prepared in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(500 + seed)
    pats = np.array([blobs(b.pos["retina"], rng) for _ in range(N_PATTERNS)])
    t0 = time.time()
    rel = reliability(copy.deepcopy(b), pats, seed=seed)
    print(f"reliability ({time.time() - t0:.0f} s)\n  {'group':<10} {'rate':>6} {'retest':>7}")
    for k, v in rel.items():
        print(f"  {k:<10} {v['rate']:6.3f} {v['retest']:7.2f}")
    t0 = time.time()
    m = memory_vs_control(b, seed)
    L, U = m["learned"], m["untrained"]
    print(f"memory, scrambled recall ({time.time() - t0:.0f} s): match learned {L['match']:+.3f} untrained "
          f"{U['match']:+.3f} (diff {L['match'] - U['match']:+.3f}); identified learned {L['accuracy']:.2f} "
          f"untrained {U['accuracy']:.2f} (chance 0.10)")


if __name__ == "__main__":
    main()

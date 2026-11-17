"""
Federated brain checks (roadmap step 7).

Stage 1 regression: the data-built cortex-thalamus brain must reproduce the hand-built one (notes/19-20):
same memory-completion task, same scoring.

Run: PYTHONPATH=prototype python3 prototype/brain/run_federation.py   (N_SEEDS, default 4)
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
from backend import to_host  # noqa: E402
from federation import Brain  # noqa: E402
from run_wiring_test import BLANK, EPOCHS, HOLD, N_PATTERNS, blobs, corr  # noqa: E402

SEEDS = range(int(os.environ.get("N_SEEDS", "4")))


def prepare(spec, seed, homeostasis):
    b = Brain(spec, seed=seed)
    b.homeostasis = homeostasis
    rng = np.random.default_rng(1000 + seed)
    b.calibrate(lambda: {"retina": (rng.random(len(b.pos["retina"])) < 0.2).astype(np.float32)})
    b.develop(lambda: {"retina": blobs(b.pos["retina"], rng)})
    return b


def memory_task(b, seed, epochs=EPOCHS, lesion_at_recall=(), hold=HOLD, recall=8):
    """hold: steps each pattern is shown while learning; recall: steps each probe is shown (the last 4 are
    read). The hippocampal loop needs ~12 steps to answer in area A (notes/26), so stage-2 tests use longer."""
    rng = np.random.default_rng(100 + seed)
    pats = np.array([blobs(b.pos["retina"], rng) for _ in range(N_PATTERNS)])
    cues = pats * (b.pos["retina"][:, 0] < 0.5)
    for _ in range(epochs):
        for p in pats:
            for s in range(hold + BLANK):
                b.step({"retina": p if s < hold else 0 * p}, learn=True, ach=0.8,
                       dopamine=1.0 if s == 2 else 0.0, arousal=0.6)
    b.sleep()
    b.ablated |= set(lesion_at_recall)

    def present(x, n=8):
        acc = []
        for s in range(n):
            b.step({"retina": x}, ach=0.2)
            if s >= n - 4:
                acc.append(to_host(b.g["A.L23"].rate).copy())
        return np.mean(acc, 0)

    full, cue = [], []
    for p, c in zip(pats, cues):
        present(0 * p, 6)
        full.append(present(p, recall))
        present(0 * p, 6)
        cue.append(present(c, recall))
    full, cue = np.array(full), np.array(cue)
    base = full.mean(0)
    full, cue = full - base, cue - base
    right = b.pos["A.L23"][:, 0] > 0.55
    own = [corr(cue[i][right], full[i][right]) for i in range(N_PATTERNS)]
    other = [np.mean([corr(cue[i][right], full[j][right]) for j in range(N_PATTERNS) if j != i])
             for i in range(N_PATTERNS)]
    return float(np.mean(own) - np.mean(other))


def main():
    rows = {}
    for seed in SEEDS:
        for name, homeo in (("stage 1, homeostasis off (as notes/19)", 0.0),):
            b = prepare(S.stage1(), seed, homeo)
            rows.setdefault(name, []).append(memory_task(b, seed))
        print(f"seed {seed} done", flush=True)
    print(f"\nMemory completion, data-built brain ({len(SEEDS)} seeds; notes/19-20 hand-built: 0.42-0.48)")
    for name, v in rows.items():
        v = np.array(v)
        print(f"  {name:<42} fill-in {v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}")


def stage2_main():
    """Does the hippocampal loop help? Fewer presentations is where a hippocampus should matter."""
    rows = {}
    for seed in SEEDS:
        for epochs in (1, 6):
            b1 = prepare(S.stage1(), seed, 0.0)
            rows.setdefault(f"cortex only, {epochs} presentation(s)", []).append(memory_task(b1, seed, epochs))
            b2 = prepare(S.stage2(), seed, 0.0)
            lesioned = copy.deepcopy(b2)
            rows.setdefault(f"+ hippocampus, {epochs} presentation(s)", []).append(memory_task(b2, seed, epochs))
            rows.setdefault(f"+ hippocampus, lesioned at recall, {epochs}", []).append(
                memory_task(lesioned, seed, epochs, lesion_at_recall=("EC", "DG", "CA3", "CA1", "SUB")))
        print(f"seed {seed} done", flush=True)
    print(f"\nStage 2: hippocampal loop ({len(SEEDS)} seeds; fill-in of the unseen half, ± s.e.)")
    for name, v in rows.items():
        v = np.array(v)
        print(f"  {name:<46} {v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}")


if __name__ == "__main__":
    stage2_main() if os.environ.get("STAGE") == "2" else main()

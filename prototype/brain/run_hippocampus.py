"""
Where does the hippocampal memory get lost? (notes/26)

The stage-2 one-shot task (run_fly.py, 1 presentation), measured at every station of the loop:
cortex B -> EC -> DG -> CA3 -> CA1 -> subiculum -> EC L5 -> back to B and A.

  rate      mean activity during the full patterns
  overlap   mean correlation between different patterns' responses (high = no pattern separation)
  complete  half cue vs full pattern: own - other correlation (> 0 = the cue brings back its own pattern)
            cortex: right half only (the half the cue doesn't show), as in the memory score

Each is measured in the learned brain and in a copy that saw nothing (naive), so learned - naive is what
one presentation added. Last line: the memory score with the hippocampus intact vs silenced at recall.

Run: PYTHONPATH=prototype:prototype/brain python3 prototype/brain/run_hippocampus.py
Env: NEURONS (7104), SEED (0), EPOCHS (1)
"""

import copy
import os
import time

import numpy as np

import backend
import brainspec as S
from backend import to_host
from run_federation import memory_task, prepare
from run_wiring_test import BLANK, N_PATTERNS, blobs, corr

NEURONS = int(os.environ.get("NEURONS", "7104"))
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
HOLD_HC, RECALL = int(os.environ.get("HOLD", "12")), int(os.environ.get("RECALL", "16"))
STATIONS = ["B.L23", "EC.stellate", "EC.L3", "DG.gc", "DG.mossy", "CA3.pyr", "CA1.pyr", "SUB.pyr", "EC.L5", "A.L23"]
HPC = ("EC", "DG", "CA3", "CA1", "SUB")


def learn(b, pats, epochs):
    for _ in range(epochs):
        for p in pats:
            for s in range(HOLD_HC + BLANK):
                b.step({"retina": p if s < HOLD_HC else 0 * p}, learn=True, ach=0.8,
                       dopamine=1.0 if s == 2 else 0.0, arousal=0.6)
    b.sleep()


def probe(b, pats, cues):
    """Mean late response of every station to each full pattern and each cue (same protocol as memory_task)."""
    def present(x, n=RECALL):
        acc = []
        for s in range(n):
            b.step({"retina": x}, ach=0.2)
            if s >= n - 4:
                acc.append({k: to_host(b.g[k].rate).copy() for k in STATIONS})
        return {k: np.mean([a[k] for a in acc], 0) for k in STATIONS}

    full, cue = [], []
    for p, c in zip(pats, cues):
        present(0 * p, 6)
        full.append(present(p))
        present(0 * p, 6)
        cue.append(present(c))
    return full, cue


def measures(b, full, cue):
    out = {}
    for k in STATIONS:
        f = np.array([x[k] for x in full])
        c = np.array([x[k] for x in cue])
        if k.startswith(("A.", "B.")):
            right = b.pos[k][:, 0] > 0.55
            f, c = f[:, right], c[:, right]
        rate, n = f.mean(), len(f)
        overlap = np.mean([corr(f[i], f[j]) for i in range(n) for j in range(i + 1, n)])
        base = f.mean(0)
        f, c = f - base, c - base
        own = np.mean([corr(c[i], f[i]) for i in range(n)])
        other = np.mean([corr(c[i], f[j]) for i in range(n) for j in range(n) if i != j])
        out[k] = (rate, overlap, own - other)
    return out


def main():
    t = time.time()
    b = prepare(S.sized(S.stage2(), NEURONS), SEED, 0.0)
    n_neurons = b.n_neurons
    rng = np.random.default_rng(100 + SEED)
    pats = np.array([blobs(b.pos["retina"], rng) for _ in range(N_PATTERNS)])
    cues = pats * (b.pos["retina"][:, 0] < 0.5)
    b.to("cpu")   # copies are kept on the host; one brain at a time goes to the device (fly size on the GPU)

    def fresh():
        free()
        return copy.deepcopy(b).to(backend.DEVICE)

    x = fresh()
    learn(x, pats, EPOCHS)
    learned = measures(x, *probe(x, pats, cues))
    x = fresh()
    before = measures(x, *probe(x, pats, cues))
    del x
    print(f"{n_neurons:,} neurons, seed {SEED}, {EPOCHS} presentation(s), shown {HOLD_HC} steps, recall {RECALL} steps")
    print(f"   {'station':<13}{'rate':>7}{'overlap':>9}{'complete':>10}   {'naive: rate':>12}{'overlap':>9}{'complete':>10}")
    for k in STATIONS:
        r, o, c = learned[k]
        r0, o0, c0 = before[k]
        print(f"   {k:<13}{r:7.3f}{o:9.2f}{c:10.2f}   {r0:12.3f}{o0:9.2f}{c0:10.2f}")
    scores = [memory_task(fresh(), SEED, EPOCHS, lesion_at_recall=les, hold=HOLD_HC, recall=RECALL)
              for les in ((), HPC, HPC[1:])]
    print(f"   memory score (A, unseen half): intact {scores[0]:.2f}; silenced at recall: entorhinal + hippocampus "
          f"{scores[1]:.2f}, hippocampus only (DG, CA3, CA1, subiculum) {scores[2]:.2f}")
    print(f"   {(time.time() - t) / 60:.1f} min")


def free():
    if backend.DEVICE == "gpu":
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()

"""
First wiring test (roadmap step 4): long-range connections and brain waves.

Task: learn 6 retina patterns, sleep, then show only the LEFT half of each pattern. Does area A's
layer 2/3 fill in the RIGHT half from memory (pattern completion), and can the brain tell which
pattern it was?

Conditions:
  untrained      developed (4 days of ordinary experience) but never shown these patterns
  no waves       learning, no waves
  forward wave   a back-to-front wave, always
  wave as mode   back-to-front while learning (encode), front-to-back while recalling
  mode, no FB    "wave as mode" with the feedback connections (B -> A apical) removed
  scheduled      "wave as mode", but only cells near the wave crest are updated each step
  long-range x   "wave as mode" with the long-range fraction scaled to 0 / 0.5 / 1

Also simulates paging: tiles of cortex as memory blocks, a cache of 25% of them, and counts
demand misses (stalls) with and without wave-guided prefetch.

Run: PYTHONPATH=prototype python3 prototype/brain/run_wiring_test.py   (N_SEEDS=6 for more seeds, ~7 min)
"""

import copy
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
from network import GRID, Brain, Wave  # noqa: E402

N_PATTERNS, EPOCHS, HOLD, BLANK = 6, 6, 6, 3
SEEDS = tuple(range(int(__import__("os").environ.get("N_SEEDS", "3"))))


def blobs(pos, rng, n_blobs=3, radius=0.13):
    """An object-like pattern: a few blobs at random places (at least one in each half of the field)."""
    centers = rng.random((n_blobs, 2))
    centers[0, 0], centers[1, 0] = 0.1 + 0.3 * rng.random(), 0.6 + 0.3 * rng.random()
    d = np.linalg.norm(pos[:, None, :] - centers[None], axis=2).min(axis=1)
    return (d < radius).astype(np.float32)


def make_patterns(brain, rng):
    pats = np.array([blobs(brain.pos["retina"], rng) for _ in range(N_PATTERNS)])
    left = (brain.pos["retina"][:, 0] < 0.5).astype(np.float32)
    return pats, pats * left


def encode(b, pats, wave, scheduled, t0=0):
    t = t0
    if wave is not None:
        b.set_wave(Wave(direction=+1))
    for _ in range(EPOCHS):
        for p in pats:
            for s in range(HOLD + BLANK):
                on = s < HOLD
                # Dopamine is phasic: one brief burst shortly after each (novel) pattern appears
                b.step(p if on else 0 * p, t, learn=True, ach=0.8,
                       dopamine=1.0 if s == 2 else 0.0, arousal=0.6, scheduled=scheduled)
                t += 1
    return t


def recall(b, pats, cues, wave_dir, scheduled, t0=0):
    if wave_dir is not None:
        b.set_wave(Wave(direction=wave_dir))
    else:
        b.set_wave(None)
    t, full, cue, fullB, cueB = t0, [], [], [], []

    def present(x, n=8):
        nonlocal t
        acc, accB = [], []
        for s in range(n):
            b.step(x, t, ach=0.2, scheduled=scheduled)
            t += 1
            if s >= n - 4:
                acc.append(b.groups["A.L23"].rate.copy())
                accB.append(b.groups["B.L23"].rate.copy())
        return np.mean(acc, 0), np.mean(accB, 0)

    for p, c in zip(pats, cues):
        present(0 * p, 6)
        f, fB = present(p)
        present(0 * p, 6)
        q, qB = present(c)
        full.append(f), cue.append(q), fullB.append(fB), cueB.append(qB)
    return np.array(full), np.array(cue), np.array(fullB), np.array(cueB)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score(b, full, cue, fullB, cueB):
    # Each cell's own baseline (its mean response over all patterns) is removed first, so only
    # pattern-specific activity counts
    base, baseB = full.mean(0), fullB.mean(0)
    full, cue, fullB, cueB = full - base, cue - base, fullB - baseB, cueB - baseB
    right = b.pos["A.L23"][:, 0] > 0.55
    own = [corr(cue[i][right], full[i][right]) for i in range(N_PATTERNS)]
    other = [np.mean([corr(cue[i][right], full[j][right]) for j in range(N_PATTERNS) if j != i])
             for i in range(N_PATTERNS)]
    ident = np.mean([np.argmax([corr(cue[i], f) for f in full]) == i for i in range(N_PATTERNS)])
    identB = np.mean([np.argmax([corr(cueB[i], f) for f in fullB]) == i for i in range(N_PATTERNS)])
    return {"fill-in": np.mean(own) - np.mean(other), "identify A": ident, "identify B": identB}


def run_condition(base, pats, cues, wave, recall_dir, scheduled=False, learn=True):
    b = copy.deepcopy(base)
    t = encode(b, pats, wave, scheduled) if learn else 0
    if learn:
        b.sleep()
    res = score(b, *recall(b, pats, cues, recall_dir, scheduled, t))
    res["updated"] = np.mean(list(b.updated.values())) if scheduled else 1.0
    return res


def paging(base, pats, cache_frac=0.25, steps=300):
    """Tiles of A and B as memory blocks. Event-driven (no waves): a block is needed when any cell in it
    is active. Wave-scheduled: a block is needed when the wave crest is on it, and is prefetched one step
    ahead. Counts demand misses (stalls) per step."""
    out = {}
    for mode in ("event-driven, LRU", "wave-scheduled, LRU", "wave-scheduled + prefetch"):
        b = copy.deepcopy(base)
        wave = mode.startswith("wave")
        b.set_wave(Wave(direction=+1) if wave else None)
        cache, cap = OrderedDict(), int(cache_frac * 2 * GRID * GRID)
        stalls, loads = 0, 0
        for t in range(steps):
            p = pats[(t // (HOLD + BLANK)) % N_PATTERNS]
            b.step(p if t % (HOLD + BLANK) < HOLD else 0 * p, t, ach=0.5, scheduled=wave)
            need = set()
            for area in ("A", "B"):
                for g in ("L4", "L23", "L5IT", "L5PT", "L6"):
                    name = f"{area}.{g}"
                    cells = b.crest(name, t) > 0 if wave else b.groups[name].rate > 0.3
                    need |= {(area, int(x)) for x in np.unique(b.tile[name][cells])}
            if mode.endswith("prefetch"):  # the wave says exactly which blocks come next
                for area in ("A", "B"):
                    nxt = b.crest(f"{area}.L23", t + 1) > 0
                    for x in np.unique(b.tile[f"{area}.L23"][nxt]):
                        key = (area, int(x))
                        if key not in cache:
                            loads += 1
                        cache[key] = True
                        cache.move_to_end(key)
                        while len(cache) > cap:
                            cache.popitem(last=False)
            for key in need:
                if key not in cache:
                    stalls += 1
                    loads += 1
                cache[key] = True
                cache.move_to_end(key)
                while len(cache) > cap:
                    cache.popitem(last=False)
        out[mode] = (stalls / steps, loads / steps)
    return out


def main():
    rows = {}
    page = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        base = Brain(seed=seed)
        base.calibrate()
        base.develop()
        pats, cues = make_patterns(base, rng)
        conds = {
            "untrained": (base, None, None, False, False),
            "no waves": (base, None, None, False, True),
            "forward wave": (base, True, +1, False, True),
            "wave as mode": (base, True, -1, False, True),
            "scheduled (mode)": (base, True, -1, True, True),
        }
        for lr in (0.0, 0.5):
            b2 = Brain(seed=seed, lr_scale=lr)
            b2.calibrate()
            b2.develop()
            conds[f"long-range x{lr} (mode)"] = (b2, True, -1, False, True)
        nofb = Brain(seed=seed, remove_feedback=True)
        nofb.calibrate()
        nofb.develop()
        conds["mode, no feedback"] = (nofb, True, -1, False, True)
        for name, (b, wave, rdir, sched, learn) in conds.items():
            rows.setdefault(name, []).append(run_condition(b, pats, cues, wave, rdir, sched, learn))
        page.append(paging(base, pats))
        print(f"seed {seed} done", flush=True)

    print(f"\nPattern completion ({len(SEEDS)} seeds; cue = left half of each learned pattern)")
    print(f"{'condition':<26}{'fill-in (right half)':>22}{'identify (A)':>14}{'identify (B)':>14}{'cells updated':>15}")
    for name, rs in rows.items():
        f = [r["fill-in"] for r in rs]
        print(f"{name:<26}{np.mean(f):>14.3f} ± {np.std(f):.3f}{np.mean([r['identify A'] for r in rs]):>14.2f}"
              f"{np.mean([r['identify B'] for r in rs]):>14.2f}{np.mean([r['updated'] for r in rs]):>15.2f}")
    print("\nPaging (cortex tiles as memory blocks, cache = 25% of blocks)")
    for mode in page[0]:
        s = np.mean([p[mode][0] for p in page]); l = np.mean([p[mode][1] for p in page])
        print(f"  {mode:<28} stalls/step {s:6.2f}   loads/step {l:6.2f}")


if __name__ == "__main__":
    main()

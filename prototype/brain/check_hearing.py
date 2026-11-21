"""
Hearing checks (plan track L1, note 81). Prints JSON lines.

  syllables     6 syllables x 5 repeats: can each stage (ear, MGB, A1 L4, A1 L2/3) tell them apart? (nearest centroid)
  statistics    statistical learning (Saffran, Aslin & Newport 1996): a brain hears a continuous stream made of four
                three-syllable "words" (tupiro golabu bidaku padoti); a twin copy hears the same syllables in random
                order. Then (learning off) both hear words and part-words (spanning word boundaries). Predictable syllables
                evoke less cortical activity, so a brain that picked up the words should respond less to words than to
                part-words. Effect = response(part-words) - response(words), exposed vs control.

  .venv/bin/python prototype/brain/check_hearing.py --load-brain PATH [syllables|statistics]
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
import cochlea  # noqa: E402
from federation import Brain  # noqa: E402

WORDS = ["tupiro", "golabu", "bidaku", "padoti"]
PARTWORDS = ["rogola", "bubida", "kupado", "titupi"]


def quiet_inputs(b):
    return {k: np.zeros(len(b.pos[k]), np.float32) for k in b.spec.inputs}


def hear(b, ear, sound, learn=False, record=(), pulse_every=0, start=0):
    """Play a cochleagram one frame per step; return {group: (frames, cells)} of rates for `record`."""
    q = quiet_inputs(b)
    out = {k: [] for k in record}
    for t in range(sound.shape[1]):
        da = 0.5 if (pulse_every and (start + t) % pulse_every == 2) else 0.0
        b.step({**q, **ear(sound[:, t])}, learn=learn, ach=0.6, dopamine=da, arousal=0.3)
        for k in record:
            out[k].append(np.asarray(b.g[k].rate).copy())
    return {k: np.array(v) for k, v in out.items()}


def syllables(b):
    ear = cochlea.Ear(b)
    sylls = ["ta", "ki", "mo", "bu", "se", "la"]
    G = [k for k in ("MGB.v", "AUD.L4", "AUD.L23") if k in b.g]
    order = np.random.default_rng(0).permutation(np.repeat(np.arange(len(sylls)), 5))
    X = {k: [] for k in G + ["EAR.sus"]}
    for i in order:
        hear(b, ear, np.zeros((cochlea.BANDS, 4), np.float32))
        snd = cochlea.text_sound(sylls[i])[:, :2]
        q = quiet_inputs(b)
        acc = {k: 0 for k in X}
        for t in range(4):
            ch = ear(snd[:, min(t, 1)])
            b.step({**q, **ch}, ach=0.6)
            acc["EAR.sus"] = acc["EAR.sus"] + ch["EAR.sus"]
            for k in G:
                acc[k] = acc[k] + np.asarray(b.g[k].rate)
        for k in X:
            X[k].append(acc[k])
    rows = []
    for k, A in X.items():
        A, ok = np.array(A), 0
        for i in range(len(order)):
            m = np.arange(len(order)) != i
            M = np.array([A[m][order[m] == c].mean(0) for c in range(len(sylls))])
            ok += int(np.argmin(np.linalg.norm(M - A[i], axis=1)) == order[i])
        rows.append({"check": "syllables", "group": k, "identified": ok / len(order), "chance": 1 / len(sylls)})
    return rows


def stream(words, n, rng, scramble=False):
    if scramble:   # the same syllables, random order: no words
        sylls = [w[i:i + 2] for w in words for i in (0, 2, 4)]
        return "".join(rng.choice(sylls) for _ in range(3 * n))
    seq, last = [], None
    for _ in range(n):
        w = rng.choice([x for x in words if x != last])
        seq.append(w)
        last = w
    return "".join(seq)


def statistics(b, n_words=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, scramble in (("exposed to words", False), ("control: random syllables", True)):
        c = copy.deepcopy(b)
        ear = cochlea.Ear(c)
        text = stream(WORDS, n_words, np.random.default_rng(seed + 1), scramble)
        hear(c, ear, cochlea.text_sound(text)[:, :-1], learn=True, pulse_every=9)
        resp = {"words": [], "part-words": []}
        for rep in range(6):
            for kind, items in (("words", WORDS), ("part-words", PARTWORDS)):
                for w in rng.permutation(items):
                    hear(c, ear, np.zeros((cochlea.BANDS, 3), np.float32))
                    r = hear(c, ear, cochlea.text_sound(w)[:, :-1], record=("AUD.L23",))["AUD.L23"]
                    resp[kind].append(float(r[2:].mean()))   # 2nd and 3rd syllables (predictable inside a word)
        effect = float(np.mean(resp["part-words"]) - np.mean(resp["words"]))
        rows.append({"check": "statistics", "brain": name, "words": float(np.mean(resp["words"])),
                     "part_words": float(np.mean(resp["part-words"])), "effect": effect})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-brain", required=True)
    ap.add_argument("check", choices=["syllables", "statistics"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b = Brain.load(a.load_brain, quiet=True)
    rows = syllables(b) if a.check == "syllables" else statistics(b, seed=a.seed)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()

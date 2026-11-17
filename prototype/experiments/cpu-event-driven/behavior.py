"""
Behavior check: the "childhood -> hard times -> time passes -> relearn" story from
demo_single_neuron.py, seeds 1-8, for each implementation (adapter with the PackedNeuron interface).

Run: .venv/bin/python prototype/experiments/cpu-event-driven/behavior.py
"""

import common  # noqa: F401

from dataclasses import replace

import numpy as np

from demo_single_neuron import run
from neuron import WORKHORSE
from packed_neuron import PackedNeuron

SEEDS = range(1, 9)
COLS = ["after hard times (A bad)", "time passes: day 3", "time passes: day 6"]


def summarize(make, ntype, seeds=SEEDS):
    rows = [run("", ntype, seed=s, make=make, verbose=False) for s in seeds]
    out = {}
    for c in COLS:
        v = np.array([dict(log)[c] for log, _ in rows])
        out[c] = (float(v.mean()), float(v.min()), float(v.max()))
    r = np.array([relearn for _, relearn in rows])
    out["relearn"] = (float(np.median(r)), int(r.min()), int(r.max()))
    out["day6_all"] = [float(dict(log)["time passes: day 6"]) for log, _ in rows]
    out["relearn_all"] = [int(x) for x in r]
    out["relearned_within_400"] = int((r < 400).sum())
    return out


def report(impls, seeds=SEEDS, printer=print):
    """impls: list of (name, make). Returns {(kind, name): summary}."""
    res = {}
    printer(f"  {'':<30}{'hard times':>18}{'day 3':>18}{'day 6':>18}{'relearn (median)':>22}{'relearned':>11}")
    for kind, ntype in [("workhorse", WORKHORSE), ("plain", replace(WORKHORSE, cascade_levels=1))]:
        for name, make in impls:
            s = summarize(make, ntype, seeds)
            res[(kind, name)] = s
            cells = [f"{s[c][0]:.2f} [{s[c][1]:.2f}-{s[c][2]:.2f}]" for c in COLS]
            printer(f"  {kind + ' / ' + name:<30}" + "".join(f"{c:>18}" for c in cells)
                    + f"{s['relearn'][0]:>10.0f} [{s['relearn'][1]}-{s['relearn'][2]}]"
                    + f"{s['relearned_within_400']:>8}/{len(s['relearn_all'])}")
    return res


def default_impls():
    from axon_major import AxonNeuron
    from bitslice import BitSlicedNeuron
    from dm_fused import FusedNeuron
    return [("packed (NumPy baseline)", PackedNeuron), ("fused kernel", FusedNeuron),
            ("axon-major", AxonNeuron), ("bit-sliced", BitSlicedNeuron)]


if __name__ == "__main__":
    import json
    res = report(default_impls(), seeds=range(1, 17) if "--16" in __import__("sys").argv else SEEDS)
    with open(f"{common.HERE}/results_behavior.json", "w") as f:
        json.dump({f"{k[0]} / {k[1]}": v for k, v in res.items()}, f, indent=1)

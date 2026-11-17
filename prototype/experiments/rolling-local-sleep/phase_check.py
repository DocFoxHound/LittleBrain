"""
Why does a rolling night change the "old self comes back" result? A one-population check of the
same world-clock story, with the night at different slots of a 50-slot day, using BOTH the NumPy
prototype (PackedPopulation) and the kernels. If both show the effect, it is a property of the
neuron model, not of the kernels.

Also traces the mean cascade levels of the synapses that listen to pattern A after every night.

Run: .venv/bin/python prototype/experiments/rolling-local-sleep/phase_check.py
"""

import json

import common
from common import HERE, WORKHORSE

import numpy as np

from packed_neuron import PackedPopulation, _dec, _get
from population import KernelPopulation

T = WORKHORSE
L, NIGHT = 50, 10


def pattern(rng, k=8):
    x = np.zeros(64, np.float32)
    x[rng.choice(64, k, replace=False)] = 1.0
    return x


def story(make, phase, seed=1, n=200, trace=False):
    rng = np.random.default_rng(seed)
    A = pattern(np.random.default_rng(0))
    p = make(rng, n)
    asleep, out, nights = False, [], []

    def slot(s, kind, da, day):
        nonlocal asleep
        if s == phase:
            if trace:  # fast weight of A-synapses at sleep onset, then the levels after the night
                src = p._src(p.words)
                on_a = A[src] > 0
                fast = float((_dec(_get(p.words, "fast"), 1.0) * p.fast_scale)[on_a].mean())
            p.sleep()
            asleep = True
            if trace:
                p._catch_up()
                lv = [round(float(_dec(_get(p.words, f), 1.0)[on_a].mean()), 3) for f in ("c0", "c1", "c2")]
                nights.append({"day": day, "fast_at_onset": round(fast, 3), "c0_c1_c2_after": lv})
        if asleep and not (phase <= s < phase + NIGHT):
            asleep = False
        if asleep:
            return
        if kind == "rest":
            p.rest(5)
            return
        p.forward(A)
        p.modulate(da, 0.8, 0.8)
        p.tick()
        p.forward(pattern(rng))
        p.tick()

    for d in range(14):  # 6 days A good, 2 days A bad, 6 days of time passing
        kind, da = ("trial", 1.0) if d < 6 else ("trial", -1.0) if d < 8 else ("rest", 0.0)
        for s in range(L):
            slot(s, kind, da, d)
        out.append(round(float(np.mean(p.forward(A, learn_trace=False))), 3))
    return out, nights


def main():
    mk_np = lambda rng, n: PackedPopulation(T, n, 64, rng, first_id=777)  # noqa: E731
    mk_k = lambda rng, n: KernelPopulation(T, n, 64, rng, first_id=777)  # noqa: E731
    res = {"response_to_A_end_of_day_7_to_13": {}, "night_trace_numpy": {}}
    for phase in (40, 0, 10, 20, 30):
        a, nights = story(mk_np, phase, trace=True)
        b, _ = story(mk_k, phase)
        res["response_to_A_end_of_day_7_to_13"][f"night at slot {phase}"] = {"numpy": a[7:], "kernels": b[7:]}
        res["night_trace_numpy"][f"night at slot {phase}"] = nights[5:]
        print(f"night at slot {phase:2d}: numpy {a[7:]}  kernels {b[7:]}")
    for k, v in res["night_trace_numpy"].items():
        if k in ("night at slot 0", "night at slot 20"):
            print(k)
            for row in v:
                print("   ", row)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "phase_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

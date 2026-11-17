"""
Step 3 behavior check: the demo story ("childhood -> hard times -> time passes -> relearn") with
precision fading during the "time passes" days. Uses run(...) from demo_single_neuron.py with
FadingNeuron as the adapter.

Fading starts the first night a neuron is unused (onset = 1 night), so all six "time passes"
days happen at reduced precision. That is the harshest schedule this story allows.

Extra controls (because noise in weights can fake "recovery" through the sigmoid):
  - ctrl:  response to a never-trained control pattern, probed alongside A at every probe;
  - plain: the same fading applied to the plain neuron (1 cascade level, no old self).

Run: python behavior.py [named|grid|ablation|all]
"""

import os
import sys
from dataclasses import replace

from common import np, save_json, time

from demo_single_neuron import make_pattern, run
from fade import POLICIES, FadingNeuron, Policy, fixed, stored_bits_per_synapse
from neuron import WORKHORSE
from packed_neuron import LEVELS, _dec, _get

PLAIN = replace(WORKHORSE, name="plain", cascade_levels=1)
FULL6 = (6, 6, 6)
CONTROL = make_pattern(np.random.default_rng(12345))
SEEDS = range(1, 33)
PROBES = ["start", "after childhood (A good)", "after hard times (A bad)",
          "time passes: day 1", "time passes: day 3", "time passes: day 6"]


class Probed:
    """Wraps an adapter; every probe of A also probes the control pattern (no side effects)."""

    def __init__(self, inner):
        self.inner, self.ctrl, self.contrast = inner, [], []

    def level_contrast(self, x):
        """Per cascade level: mean value on synapses that see x minus mean on the others."""
        inner, pop = self.inner, self.inner.pop
        w = inner._thawed_words(commit=False) if inner.frozen else pop.words
        on = x[pop._src(w)] > 0
        out = []
        for k in range(pop.t.cascade_levels):
            c = _dec(_get(w, LEVELS[k]), pop.t.w_max)
            out.append(float(c[on].mean() - c[~on].mean()))
        return out

    def forward(self, x, context=0.0, learn_trace=True):
        if not learn_trace:
            self.ctrl.append(self.inner.forward(CONTROL, context, False))
            self.contrast.append(self.level_contrast(x))
        return self.inner.forward(x, context, learn_trace)

    def __getattr__(self, k):
        return getattr(self.inner, k)


class Maker:
    def __init__(self, policy):
        self.policy, self.last = policy, None

    def __call__(self, ntype, n_inputs, rng):
        self.last = Probed(FadingNeuron(ntype, n_inputs, rng, self.policy))
        return self.last


def one(args):
    policy, ntype, seed = args
    mk = Maker(policy)
    log, relearn = run("", ntype, seed=seed, make=mk, verbose=False)
    d = dict(log)
    n = mk.last
    return {"seed": seed, **{k: d[k] for k in PROBES}, "relearn": relearn,
            "ctrl_day6": n.ctrl[5], "ctrl_hard": n.ctrl[2], "prunes_unused": n.inner.prunes_unused,
            "bits_day6": n.inner.bits_log[-1] if n.inner.bits_log else 32,
            "deep_contrast_day6": n.contrast[5][-1], "surface_contrast_day6": n.contrast[5][0],
            "deep_contrast_hard": n.contrast[2][-1]}


def summarize(rows):
    out = {}
    for k in PROBES[2:] + ["ctrl_day6", "bits_day6", "prunes_unused", "deep_contrast_day6",
                           "surface_contrast_day6", "deep_contrast_hard"]:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": round(v.mean(), 3), "min": round(v.min(), 3), "max": round(v.max(), 3)}
    r = np.array([x["relearn"] for x in rows])
    out["relearn"] = {"median": float(np.median(r)), "mean": round(float(r.mean()), 2), "max": int(r.max()),
                      "frac_over_5": round(float((r > 5).mean()), 3)}
    return out


_POOL = None


def pool():
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _POOL = mp.get_context("spawn").Pool(3)
    return _POOL


def evaluate(configs, seeds=SEEDS):
    """configs: list of (key, policy, ntype). Returns {key: summary}."""
    jobs = [(p, nt, s) for _, p, nt in configs for s in seeds]
    res = pool().map(one, jobs, chunksize=8)
    out, i = {}, 0
    for key, p, nt in configs:
        rows = res[i:i + len(seeds)]
        i += len(seeds)
        out[key] = summarize(rows)
        out[key]["policy"] = {"tiers": p.tiers, "scale": p.scale, "rounding": p.rounding, "timing": p.timing}
    return out


def line(key, s):
    d6, d3 = s["time passes: day 6"], s["time passes: day 3"]
    r = s["relearn"]
    return (f"  {key:<34} day3 {d3['mean']:.2f}  day6 {d6['mean']:.2f} [{d6['min']:.2f}-{d6['max']:.2f}]"
            f"  ctrl6 {s['ctrl_day6']['mean']:.2f}  relearn med {r['median']:.0f} mean {r['mean']:.1f}"
            f" max {r['max']}  >5: {r['frac_over_5']:.2f}  bits {s['bits_day6']['mean']:.1f}"
            f"  rewired {s['prunes_unused']['mean']:.1f}"
            f"  deepC {s['deep_contrast_hard']['mean']:+.3f}->{s['deep_contrast_day6']['mean']:+.3f}"
            f"  surfC6 {s['surface_contrast_day6']['mean']:+.3f}")


def named():
    cfgs = []
    for prune in (True, False):
        for timing in ("eager", "lazy"):
            for name, p in POLICIES.items():
                if name == "none":
                    p = Policy("lossless", ((1, (6, 6, 6)),))
                p = replace(p, timing=timing, prune_cold=prune)
                for nt in (WORKHORSE, PLAIN):
                    cfgs.append((f"{nt.name}/{timing}/{'prune' if prune else 'keepwiring'}/{p.name}", p, nt))
    out = evaluate(cfgs)
    for k, s in out.items():
        print(line(k, s))
    save_json("behavior_named.json", out)
    return out


GRID = (6, 4, 3, 2, 0)


def grid(timing="eager", prune=True):
    kw = dict(timing=timing, prune_cold=prune)
    cfgs = [(f"{b0}{b1}{b2}", fixed(b0, b1, b2, **kw), WORKHORSE)
            for b0 in GRID for b1 in GRID for b2 in GRID]
    cfgs += [(f"plain/{b0}", fixed(b0, 6, 6, **kw), PLAIN) for b0 in GRID]
    out = evaluate(cfgs)
    for k, s in out.items():
        print(line(k, s))
    save_json(f"behavior_grid_{timing}_{'prune' if prune else 'keepwiring'}.json", out)
    return out


def ablation():
    cfgs = []
    for bits in ((4, 4, 4), (3, 3, 3), (2, 2, 4), (2, 2, 2)):
        for timing in ("eager", "lazy"):
            for scale in ("block", "global"):
                for rounding in ("stochastic", "nearest"):
                    p = fixed(*bits, timing=timing, scale=scale, rounding=rounding, prune_cold=False)
                    cfgs.append((f"{''.join(map(str, bits))}/{timing}/{scale}/{rounding}", p, WORKHORSE))
    out = evaluate(cfgs)
    for k, s in out.items():
        print(line(k, s))
    save_json("behavior_ablation.json", out)
    return out


def fringe_behavior():
    cfgs = []
    for timing in ("eager", "lazy"):
        for sc in ("vq16", "vq64", "vq256", "vqm64", "vqm256", "vqm_all", "holo1", "holo2", "holo4", "holo8",
                   "holo16"):
            p = Policy(sc, ((1, FULL6),), scale=sc, timing=timing, prune_cold=False)
            cfgs.append((f"{timing}/{sc}", p, WORKHORSE))
    out = evaluate(cfgs)
    for k, s in out.items():
        print(line(k, s))
    save_json("behavior_fringe.json", out)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t = time.time()
    if what in ("named", "all"):
        named()
    if what in ("grid", "all"):
        grid("eager", prune=True)
    if what in ("gridkeep", "all"):
        grid("eager", prune=False)
    if what in ("gridlazy", "all"):
        grid("lazy", prune=False)
    if what in ("ablation", "all"):
        ablation()
    if what in ("fringe", "all"):
        fringe_behavior()
    print(f"({time.time() - t:.0f}s)")
    print("bits/synapse reference:", {b: stored_bits_per_synapse(b) for b in [(6, 6, 6), (4, 4, 6), (2, 2, 4), (2, 2, 2)]})

"""
Signature behaviors: each cell type must show the behavior that defines its real counterpart.

Run: python3 prototype/brain/test_cell_types.py
"""

import numpy as np

from cells import ExcitatoryGroup, InhibitoryGroup, Neuromodulators
from celltypes import (DG_GRANULE, DG_GRANULE_YOUNG, L23_IT, L5_PT, PV, SST, TC_RELAY)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))


def _layout(shape, ref_branches, ref_slots):
    """Inputs 0..63 laid out as on the reference cell (each branch its own run of inputs), at any size:
    branch b plays reference branch b * ref_branches // B, slot s plays reference slot s % ref_slots."""
    B, S = shape
    rb = np.arange(B)[:, None] * ref_branches // B
    return (rb * ref_slots + np.arange(S)[None, :] % ref_slots) % 64


def one_cell(ctype, weight=0.5, rng=None, apical_weight=None):
    """A single cell: basal synapses listen to inputs 0..63, apical to inputs 64..127."""
    g = ExcitatoryGroup(ctype, 1, rng or np.random.default_rng(0))
    b = g.basal
    b.src[:] = _layout(b.src.shape[1:], ctype.n_basal, ctype.basal_slots)
    b.set_surface(weight)
    if g.apical is not None:
        g.apical.src[:] = 64 + _layout(g.apical.src.shape[1:], ctype.n_apical, ctype.apical_slots)
        g.apical.set_surface(weight if apical_weight is None else apical_weight)
    return g


def settle(g, r, steps=30, **kw):
    out = []
    for _ in range(steps):
        g.update(g.compute(r, **kw), learn=False)
        out.append(float(g.rate[0]))
    return out


def inputs(basal=0.0, apical=0.0):
    r = np.zeros(128, np.float32)
    r[:16] = basal       # a cluster of basal inputs
    r[64:72] = apical    # a cluster of apical inputs
    return r


def test_l5_bac():
    res = {}
    for name, (b, a) in {"basal": (0.6, 0), "apical": (0, 1.0), "both": (0.6, 1.0)}.items():
        g = one_cell(L5_PT, weight=0.06, apical_weight=0.25)
        res[name] = settle(g, inputs(b, a))[-1]
    ok = res["both"] > res["basal"] + res["apical"] + 0.1
    check("L5 PT: coincidence burst (BAC firing)", ok,
          f"basal only {res['basal']:.2f}, apical only {res['apical']:.2f}, both {res['both']:.2f}")


def test_l23_adaptation():
    g = one_cell(L23_IT, weight=0.12)
    trace = settle(g, inputs(1.0, 0), steps=40)
    peak, late = max(trace), trace[-1]
    check("L2/3 IT: spike-frequency adaptation", late < peak - 0.05,
          f"peak {peak:.2f} then settles to {late:.2f} under constant input")


def test_pv_gain_control():
    def curve(pv):
        out = []
        for level in (0.3, 0.6, 0.9):
            g = one_cell(L23_IT, weight=0.12)
            g.c = g.c.__class__(**{**g.c.__dict__, "adapt": 0.0})
            out.append(settle(g, inputs(level, 0), pv_in=pv)[-1])
        return out
    off, on = curve(0.0), curve(1.5)
    ok = (on[2] - on[0]) < (off[2] - off[0]) and on[2] < off[2]
    check("PV: divisive gain control at the cell body", ok,
          f"response to 3 input levels without PV {[round(x, 2) for x in off]}, "
          f"with PV {[round(x, 2) for x in on]}")


def test_sst_facilitation_and_gating():
    s = InhibitoryGroup(SST, 1)
    rates = [float(s.step(np.array([1.0]))[0]) for _ in range(30)]
    ramp = rates[-1] > rates[2] + 0.2
    g_free, g_gated = one_cell(L23_IT, 0.12), one_cell(L23_IT, 0.12)
    free = settle(g_free, inputs(0.5, 1.0), sst_in=0.0)[-1]
    gated = settle(g_gated, inputs(0.5, 1.0), sst_in=1.5)[-1]
    check("SST: facilitates with sustained input, gates apical tufts", ramp and gated < free - 0.05,
          f"SST rate step 3 {rates[2]:.2f} -> step 30 {rates[-1]:.2f}; "
          f"cell with apical input: SST off {free:.2f}, SST on {gated:.2f}")


def test_vip_disinhibition():
    out = {}
    for vip_on in (False, True):
        s, v = InhibitoryGroup(SST, 1), InhibitoryGroup(__import__("celltypes").VIP, 1)
        g = one_cell(L23_IT, 0.12)
        for _ in range(30):
            vr = v.step(np.array([1.5 if vip_on else 0.0]))
            sr = s.step(np.array([1.0]) - 1.5 * vr)          # VIP cells inhibit SST cells
            g.update(g.compute(inputs(0.5, 1.0), sst_in=1.5 * sr), learn=False)
        out[vip_on] = (float(sr[0]), float(g.rate[0]))
    ok = out[True][0] < out[False][0] and out[True][1] > out[False][1]
    check("VIP: inhibits SST, so dendrites open (disinhibition)", ok,
          f"VIP off: SST {out[False][0]:.2f}, cell {out[False][1]:.2f}; "
          f"VIP on: SST {out[True][0]:.2f}, cell {out[True][1]:.2f}")


def test_relay_burst_and_trn():
    def run(hyper_steps, trn_during):
        g = one_cell(TC_RELAY, weight=0.15)
        settle(g, inputs(0.0), steps=hyper_steps, trn_in=1.0 if hyper_steps else 0.0)
        return max(settle(g, inputs(0.12), steps=10, trn_in=trn_during))
    tonic, burst, gated = run(0, 0.0), run(30, 0.0), run(0, 0.6)
    check("TC relay: burst after hyperpolarization (T-type Ca)", burst > tonic + 0.1,
          f"peak response to a weak input from rest {tonic:.2f}, after being held down {burst:.2f}")
    check("TRN: inhibitory gate on the relay", gated < tonic - 0.1,
          f"relay response {tonic:.2f}, with TRN active {gated:.2f}")


def test_dg_pattern_separation():
    rng = np.random.default_rng(1)
    n, n_in = 400, 200
    g = ExcitatoryGroup(DG_GRANULE, n, rng)
    g.basal.src[:] = rng.integers(0, n_in, g.basal.src.shape)
    g.basal.set_surface(rng.uniform(0, 0.2, g.basal.cascade[0].shape))
    a = (rng.random(n_in) < 0.3).astype(np.float32)
    b = a.copy()
    flip = rng.choice(n_in, 30, replace=False)
    b[flip] = 1 - b[flip]
    def code(x):
        drive = g.compute(x)
        k = max(1, int(0.02 * n))  # strong lateral inhibition: only ~2% of granule cells fire
        out = np.zeros(n)
        out[np.argsort(drive)[-k:]] = 1
        return out
    def overlap(u, v):
        return float((u * v).sum() / np.sqrt(u.sum() * v.sum()))
    oin, oout = overlap(a, b), overlap(code(a), code(b))
    check("DG granule: pattern separation", oout < oin - 0.1,
          f"input overlap {oin:.2f} -> granule output overlap {oout:.2f}")
    young, mature = DG_GRANULE_YOUNG, DG_GRANULE
    check("DG young granule: more excitable and more plastic",
          young.threshold < mature.threshold and young.rules.lr > mature.rules.lr,
          f"threshold {young.threshold} vs {mature.threshold}, learning rate {young.rules.lr} vs {mature.rules.lr}")


def test_neuromodulators():
    m = Neuromodulators()
    das = [m.dopamine("bell", 1.0) for _ in range(15)]
    omit = m.dopamine("bell", 0.0)
    check("VTA dopamine: reward prediction error", das[0] > 0.8 and das[-1] < 0.1 and omit < -0.5,
          f"first reward {das[0]:.2f}, 15th {das[-1]:.2f}, omitted reward {omit:.2f}")
    novel = [m.acetylcholine(0.8) for _ in range(5)][-1]
    familiar = [m.acetylcholine(0.05) for _ in range(5)][-1]
    check("Basal forebrain ACh: high for novelty (encode), low when familiar (recall)", novel > 0.8 > 0.2 > familiar,
          f"novel {novel:.2f}, familiar {familiar:.2f}")
    calm = [m.noradrenaline(0.1) for _ in range(5)][-1]
    shock = m.noradrenaline(1.0)
    check("Locus coeruleus NE: phasic burst on surprise", shock > calm + 0.3,
          f"calm {calm:.2f}, surprise {shock:.2f}")


if __name__ == "__main__":
    for t in (test_l5_bac, test_l23_adaptation, test_pv_gain_control, test_sst_facilitation_and_gating,
              test_vip_disinhibition, test_relay_burst_and_trn, test_dg_pattern_separation,
              test_neuromodulators):
        t()
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print(f"\n{sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)} passed")

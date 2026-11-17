"""
Signature behaviors for the cell types added with the brain regions (notes/regions/*.md).

Run: PYTHONPATH=prototype python3 prototype/brain/test_cell_types_regions.py
"""

import numpy as np

import celltypes as C
from cells import make_group
from test_cell_types import inputs, one_cell, settle

np.seterr(over="ignore")
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))


def pooled(ctype, drives):
    g = make_group(ctype, 1)
    return [float(np.asarray(g.step(np.array([d])))[0]) for d in drives]


# --- Cerebellum --------------------------------------------------------------------------------

def test_purkinje():
    g = one_cell(C.PURKINJE, weight=0.05)
    rest = settle(g, inputs(0.0))[-1]
    before = settle(g, inputs(0.8))[-1]
    for _ in range(40):  # parallel fibers active together with climbing-fiber error signals
        g.update(g.compute(inputs(0.8)), learn=True)
        g.climbing_fiber([1.0])
    after_ltd = settle(g, inputs(0.8))[-1]
    check("Purkinje: tonic firing; parallel fiber + climbing fiber -> LTD", rest > 0.5 and after_ltd < before - 0.05,
          f"at rest {rest:.2f}; response to the trained input {before:.2f} -> {after_ltd:.2f} after pairing with errors")


def test_cb_granule():
    rng = np.random.default_rng(3)
    n, n_mf = 2000, 100
    g = make_group(C.CB_GRANULE, n, rng)
    g.basal.src[:] = rng.integers(0, n_mf, g.basal.src.shape)  # 4 dendrites, one mossy fiber each
    g.basal.set_surface(0.5)
    a = (rng.random(n_mf) < 0.3).astype(np.float32)
    b = a.copy()
    f = rng.choice(n_mf, 15, replace=False)
    b[f] = 1 - b[f]
    def code(x):
        d = g.compute(x)
        out = np.zeros(n)
        out[np.argsort(d)[-int(0.05 * n):]] = 1  # Golgi feedback inhibition keeps ~5% active
        return out
    ov = lambda u, v: float((u * v).sum() / np.sqrt(u.sum() * v.sum()))
    check("Cerebellar granule: expansion recoding decorrelates inputs", ov(code(a), code(b)) < ov(a, b) - 0.1,
          f"mossy-fiber overlap {ov(a, b):.2f} -> granule overlap {ov(code(a), code(b)):.2f} (20x more cells)")


def test_dcn_rebound():
    g = one_cell(C.DCN_PROJ, weight=0.0)
    steady = settle(g, inputs(0.0))[-1]
    settle(g, inputs(0.0), steps=20, trn_in=1.0)  # Purkinje cells inhibit the nucleus
    peak = max(settle(g, inputs(0.0), steps=5))
    check("DCN projection: rebound burst after Purkinje inhibition ends", peak > steady + 0.1,
          f"steady {steady:.2f}, peak after inhibition is released {peak:.2f}")


def test_olive():
    g = make_group(C.OLIVE, 50)
    quiet = np.mean([g.step(0.0).mean() for _ in range(50)])
    err = np.mean([g.step(0.5).mean() for _ in range(50)])
    supp = np.mean([g.step(0.5, inhibition=0.4).mean() for _ in range(50)])
    check("Inferior olive: fires for errors; nucleo-olivary inhibition silences learned errors",
          err > quiet + 0.3 and supp < err - 0.3,
          f"no error {quiet:.2f}, error {err:.2f}, error + nucleo-olivary inhibition {supp:.2f}")


# --- Basal ganglia -----------------------------------------------------------------------------

def test_msn_upstate_and_dopamine():
    weak = settle(one_cell(C.D1_MSN, weight=0.1), inputs(0.3))[-1]
    strong = settle(one_cell(C.D1_MSN, weight=0.1), inputs(1.0))[-1]
    check("MSN: silent 'down state' until strong convergent input", weak < 0.1 and strong > 0.5,
          f"scattered weak input {weak:.2f}, strong convergent input {strong:.2f}")
    res = {}
    for ctype in (C.D1_MSN, C.D2_MSN):
        g = one_cell(ctype, weight=0.04)
        w0 = g.basal.weights()[0, 0, :16].mean()
        for _ in range(10):
            g.update(g.compute(inputs(1.0)), learn=True)
            g.modulate(dopamine=1.0, ach=1.0, arousal=0.5)  # dopamine burst: better than expected
        res[ctype.name] = g.basal.weights()[0, 0, :16].mean() - w0
    check("D1 vs D2 MSN: a dopamine burst strengthens D1 and weakens D2 synapses",
          res["D1 MSN"] > 0 > res["D2 MSN"],
          f"weight change D1 {res['D1 MSN']:+.3f}, D2 {res['D2 MSN']:+.3f}")


def test_pallidum_and_stn():
    gpi = pooled(C.GPI, [0.0] * 20 + [-1.5] * 20)
    check("GPi/SNr: tonic inhibitory output; pauses when striatum inhibits it (disinhibition)",
          gpi[19] > 0.4 and gpi[-1] < 0.1, f"tonic {gpi[19]:.2f}, with D1 MSN inhibition {gpi[-1]:.2f}")
    g = one_cell(C.STN, weight=0.0)
    steady = settle(g, inputs(0.0))[-1]
    settle(g, inputs(0.0), steps=20, trn_in=1.0)  # GPe inhibition
    peak = max(settle(g, inputs(0.0), steps=5))
    check("STN: excitatory, rebound burst after GPe inhibition", peak > steady + 0.1,
          f"steady {steady:.2f}, rebound peak {peak:.2f}")


def test_tan():
    r = pooled(C.TAN, [0.0] * 20 + [-1.5] * 4 + [0.0] * 6)
    tonic, pause, rebound = r[19], min(r[20:24]), max(r[24:])
    check("Cholinergic TAN: tonic, pauses on a salient event, then rebounds",
          pause < tonic - 0.3 and rebound > tonic, f"tonic {tonic:.2f}, pause {pause:.2f}, rebound {rebound:.2f}")


# --- Interneuron targets -----------------------------------------------------------------------

def test_chandelier_and_neurogliaform():
    free = settle(one_cell(C.L23_IT, 0.12), inputs(1.0))[-1]
    vetoed = settle(one_cell(C.L23_IT, 0.12), inputs(1.0), ais_in=1.0)[-1]
    check("Chandelier: axon-initial-segment synapses veto output even under strong input", vetoed < free / 3,
          f"strongly driven cell {free:.2f}, with chandelier input {vetoed:.2f}")
    ngf = pooled(C.LAMP5_NGF, [2.0] * 3 + [0.0] * 20)
    pv = pooled(C.PV, [2.0] * 3 + [0.0] * 20)
    check("Neurogliaform: slow volume inhibition outlasts its firing (vs PV)", ngf[15] > 0.1 and pv[15] < 0.01,
          f"10+ steps after a brief burst: neurogliaform output {ngf[15]:.2f}, PV {pv[15]:.2f}")


def test_ca1_btsp():
    out = {}
    for ctype in (C.CA1_PYR, C.L23_IT):
        g = one_cell(ctype, weight=0.05)
        g.update(g.compute(inputs(0.8)), learn=True)
        c0 = g.basal.cascade[0].copy()
        g.modulate(dopamine=0.75, ach=0.75, arousal=0.5)  # one strong "plateau" event
        out[ctype.name] = float((g.basal.cascade[0] - c0).max())
    check("CA1: one-shot plateau learning (BTSP) writes a memory from a single event",
          out["CA1 pyramidal"] > 0.01 and out["L2/3 IT"] == 0.0,
          f"immediate long-term change after one event: CA1 {out['CA1 pyramidal']:.3f}, L2/3 {out['L2/3 IT']:.3f}")


def test_von_economo():
    def rise(ctype):
        tr = settle(one_cell(ctype, 0.12), inputs(1.0), steps=20)
        return next(i for i, v in enumerate(tr) if v >= 0.9 * max(tr)) + 1
    ven, pt = rise(C.VON_ECONOMO), rise(C.L5_PT)
    check("von Economo: fast relay (reaches its response sooner than an L5 pyramid)", ven < pt,
          f"steps to 90%: von Economo {ven}, L5 PT {pt}")


# --- Amygdala ----------------------------------------------------------------------------------

def test_cea_microcircuit():
    def cem(sst_drive):
        sst, pkcd, out = make_group(C.CEL_SST, 1), make_group(C.CEL_PKCD, 1), make_group(C.CEM_OUTPUT, 1)
        for _ in range(30):
            sst.update(sst.compute(np.zeros(1, np.float32), external=sst_drive), learn=False)  # a learning cell
            s = sst.rate
            p = pkcd.step(-2.0 * s)        # SST "on" cells inhibit PKCδ "off" cells
            o = out.step(-2.0 * p)         # PKCδ cells inhibit the CeM output
        return float(o[0])
    calm, fear = cem(0.0), cem(2.0)
    check("Central amygdala: SST 'on' cells disinhibit the CeM fear output", fear > calm + 0.1,
          f"CeM output calm {calm:.2f}, when SST cells are driven {fear:.2f}")


# --- Neuromodulators, hypothalamus, pacemakers -------------------------------------------------

def test_modulators():
    da = make_group(C.DA_NEURON, 1)
    base = float(da.step(0.0)[0])
    burst = float(da.step(1.0)[0])
    dip = float(da.step(-1.0)[0])
    check("Dopamine cell: baseline firing, burst for better-than-expected, dip for worse", burst > base > dip,
          f"baseline {base:.2f}, burst {burst:.2f}, dip {dip:.2f}")
    ht = make_group(C.RAPHE_5HT, 1)
    ht_trace = [float(ht.step(1.0)[0]) for _ in range(20)]
    check("Serotonin cell: slow (tracks average reward, not single events)", ht_trace[0] < 0.35 and ht_trace[-1] > ht_trace[0],
          f"after 1 reward {ht_trace[0]:.2f}, after 20 {ht_trace[-1]:.2f}")


def test_endocrine():
    crh = pooled(C.CRH, [3.0] * 30 + [0.0] * 60)
    peak = max(crh)
    half = next((i for i, v in enumerate(crh[30:]) if v < peak / 2), 60)
    pv = pooled(C.PV, [3.0] * 30 + [0.0] * 60)
    check("Neuroendocrine (CRH): slow rise, output lingers long after the stressor", half > 20 and pv[40] < 0.01,  # PV inhibition decays over a few steps (notes/26)
          f"peak {peak:.2f}; still above half {half} steps after the stressor ends (a PV cell: gone within 10)")


def test_pacemakers():
    sep = make_group(C.SEPTUM_PACEMAKER, 1)
    tr = np.array([float(sep.step()[0]) for _ in range(32)])
    peaks = np.flatnonzero((tr[1:-1] > tr[:-2]) & (tr[1:-1] >= tr[2:])) + 1
    period = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0
    check("Septal pacemaker: theta-like rhythm", abs(period - C.SEPTUM_PACEMAKER.period) < 1,
          f"period {period:.1f} steps (set {C.SEPTUM_PACEMAKER.period:.0f})")
    shifts = {}
    for name, at in (("at dawn (phase 0)", 0.0), ("in the evening (phase pi/2)", np.pi / 2)):
        a, b = make_group(C.SCN_CLOCK, 1), make_group(C.SCN_CLOCK, 1)
        a.phase[:] = b.phase[:] = at
        for _ in range(100):
            a.step(light=1.0)
            b.step()
        shifts[name] = float(b.phase[0] - a.phase[0])
    check("SCN clock: light shifts the clock depending on when it arrives (phase response)",
          abs(shifts["in the evening (phase pi/2)"]) > 5 * abs(shifts["at dawn (phase 0)"]) + 1e-3,
          "phase shift from 100 steps of light: " + ", ".join(f"{k} {v:+.3f} rad" for k, v in shifts.items()))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}")
    print(f"\n{sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)} passed")

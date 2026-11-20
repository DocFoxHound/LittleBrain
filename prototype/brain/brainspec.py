"""
Brain descriptions for the federated brain (federation.py), built in the order of
notes/regions/connections.md section 8. Pathway IDs refer to that file.

  stage1()  sensory-thalamocortical core: retina -> thalamus -> area A -> area B (= notes/19 circuit)
  stage2()  + hippocampal memory loop: B -> entorhinal -> DG -> CA3 -> CA1 -> subiculum -> EC L5 -> B
"""

import os
from dataclasses import replace

import celltypes as C
from federation import GRID, Group, Pooled, Spec, Synaptic

AREA_CELLS = {"L4": (C.L4_STELLATE, 6), "L23": (C.L23_IT, 10), "L5IT": (C.L5_IT, 3), "L5PT": (C.L5_PT, 3),
              "L6": (C.L6_CT, 5), "PV": (C.PV, 4), "SST": (C.SST, 3), "VIP": (C.VIP, 2)}
EXC = ["L4", "L23", "L5IT", "L5PT", "L6"]
LOCAL = {
    "L4":   {"basal": [("L4", 0.45), ("L6", 0.30), ("L23", 0.25)]},
    "L23":  {"basal": [("L4", 0.35), ("L23", 0.50), ("L5IT", 0.15)], "apical": [("L23", 0.5), ("L5IT", 0.5)]},
    "L5IT": {"basal": [("L23", 0.5), ("L5IT", 0.2), ("L5PT", 0.1), ("L4", 0.2)], "apical": [("L23", 1.0)]},
    "L5PT": {"basal": [("L23", 0.5), ("L5IT", 0.2), ("L5PT", 0.1), ("L4", 0.2)], "apical": [("L23", 1.0)]},
    "L6":   {"basal": [("L4", 0.3), ("L6", 0.3), ("L5IT", 0.2), ("L23", 0.2)]},
}
SPREAD = {"A": 0.12, "B": 0.25}
# Strength of the L4 -> L2/3 feedforward synapses (notes/28). 1 = as before; 3 lets the sensory signal outweigh
# L2/3's own recurrent chatter (L4 fires sparsely here, so L2/3 heard it at ~1/5 the volume of its neighbours).
L4_L23_GAIN = float(os.environ.get("L4_L23_GAIN", "1"))
# Learning-rate factor for cortical excitatory cells (notes/28: one day of learning was ~25x a synapse's base
# strength; 0.05 keeps L2/3 active after sleep). 1 = as before.
CORTEX_LRF = float(os.environ.get("CORTEX_LRF", "1"))


def _cortex(ct):
    return replace(ct, rules=replace(ct.rules, lr=ct.rules.lr * CORTEX_LRF)) if CORTEX_LRF != 1 else ct


def _with_local(dst_area, group, comp, long_paths):
    """Long-range paths take their fraction first; local inputs share the rest (as in notes/19)."""
    long_total = sum(p.frac for p in long_paths)
    local = LOCAL[group][comp]
    tot = sum(f for _, f in local)
    paths = list(long_paths)
    for src, f in local:
        gain = L4_L23_GAIN if (src, group) == ("L4", "L23") else 1.0
        paths.append(Synaptic("local", f"{dst_area}.{src}", f"{dst_area}.{group}", comp,
                              (1 - long_total) * f / tot, gain, spread=SPREAD[dst_area], kind="local"))
    return paths


def _area_inhibition(area, topdown=None):
    E = [f"{area}.{g}" for g in EXC]
    p = [Pooled("PV<-E", E, f"{area}.PV", 6.0),
         Pooled("SST<-E", E, f"{area}.SST", 4.0),
         Pooled("SST<-VIP", f"{area}.VIP", f"{area}.SST", -2.0),
         Pooled("VIP<-ACh", None, f"{area}.VIP", 1.0, signal="ach"),
         Pooled("E<-PV", f"{area}.PV", E, 1.5, channel="pv"),
         Pooled("E<-SST", f"{area}.SST", E, 1.0, channel="sst")]
    if topdown:
        p.append(Pooled("NM-12 VIP<-top-down", topdown, f"{area}.VIP", 2.0, sigma=0))
    return p


def stage1():
    groups = [Group("retina", None, 4, "retina"),
              Group("TH.TC", C.TC_RELAY, 4, "TH"), Group("TH.TRN", C.TRN, 1, "TH")]
    for area in ("A", "B"):
        groups += [Group(f"{area}.{g}", _cortex(ct) if g in EXC else ct, k, area) for g, (ct, k) in AREA_CELLS.items()]
    syn = [Synaptic("CT-2 retina->relay (driver)", "retina", "TH.TC", "basal", 0.3, 3.0, 0.05),
           Synaptic("CT-3 L6 CT->relay (modulator)", "A.L6", "TH.TC", "basal", 0.7, 0.15, 0.08)]
    long_ = {
        ("A", "L4"): {"basal": [Synaptic("CT-1 core relay->L4 (driver)", "TH.TC", "A.L4", "basal", 0.10, 8.0, 0.06)]},
        ("A", "L6"): {"basal": [Synaptic("CT-1 relay->L6", "TH.TC", "A.L6", "basal", 0.10, 5.0, 0.06)]},
        ("A", "L23"): {"apical": [Synaptic("FB B->A L1 (apical)", "B.L5IT", "A.L23", "apical", 0.6, 1.0, 0.30)]},
        ("A", "L5IT"): {"apical": [Synaptic("FB B->A L1", "B.L5IT", "A.L5IT", "apical", 0.6, 1.0, 0.30)]},
        ("A", "L5PT"): {"apical": [Synaptic("FB B->A L1", "B.L5IT", "A.L5PT", "apical", 0.6, 1.0, 0.30)]},
        ("B", "L4"): {"basal": [Synaptic("FF A->B L4", "A.L23", "B.L4", "basal", 0.10, 8.0, 0.25)]},
        ("B", "L23"): {"basal": [Synaptic("FF A->B L2/3", "A.L23", "B.L23", "basal", 0.15, 3.0, 0.25)]},
    }
    for area in ("A", "B"):
        for g in EXC:
            for comp in LOCAL[g]:
                syn += _with_local(area, g, comp, long_.get((area, g), {}).get(comp, []))
    pooled = _area_inhibition("A", topdown="B.L5IT") + _area_inhibition("B")
    pooled += [Pooled("CT-4 L6->TRN", "A.L6", "TH.TRN", 3.0, sigma=0),
               Pooled("TRN<-relay collaterals", "TH.TC", "TH.TRN", 2.0, sigma=0),
               Pooled("CT-5 TRN->relay", "TH.TRN", "TH.TC", 1.0, channel="inh")]
    return Spec(groups, syn, pooled, inputs=["retina"])


def _cap(ctype, cap):
    return replace(ctype, long_range_max=cap)


def stage2(with_hippocampus=True):
    """Stage 1 + the hippocampal memory loop (HC pathways). DG/CA3/CA1/subiculum share one structure
    label ("HPC"), so only cortex <-> entorhinal <-> hippocampus links count as long-range."""
    spec = stage1()
    if not with_hippocampus:
        return spec
    g = [Group("EC.stellate", _cap(C.EC_STELLATE, 0.8), 2, "EC"),
         Group("EC.L3", _cap(C.EC_L3, 0.8), 2, "EC"),
         Group("EC.L5", _cap(C.EC_L5, 0.8), 2, "EC"),
         Group("EC.PV", C.PV, 1, "EC"),
         Group("DG.gc", _cap(C.DG_GRANULE, 0.9), 6, "HPC", kwta=0.02),
         Group("DG.mossy", C.MOSSY_CELL, 1, "HPC"),
         Group("DG.PV", C.PV, 1, "HPC"), Group("DG.HIPP", C.HIPP, 1, "HPC"),
         Group("CA3.pyr", _cap(C.CA3_PYR, 0.4), 4, "HPC", kwta=0.05), Group("CA3.PV", C.HC_PV_BASKET, 1, "HPC"),
         Group("CA1.pyr", C.CA1_PYR, 4, "HPC", kwta=0.10), Group("CA1.PV", C.HC_PV_BASKET, 1, "HPC"),
         Group("CA1.OLM", C.OLM, 1, "HPC"),
         Group("SUB.pyr", C.SUB_BURST, 2, "HPC"), Group("SUB.PV", C.PV, 1, "HPC")]
    spec.groups += g
    S = Synaptic
    spec.synaptic += [
        S("HC-1 cortex->EC superficial", "B.L23", "EC.stellate", "basal", 0.7, 2.0, 0.3),
        S("local", "EC.stellate", "EC.stellate", "basal", 0.3, spread=0.15, kind="local"),
        S("HC-1 cortex->EC L3", "B.L23", "EC.L3", "basal", 0.6, 2.0, 0.3),
        S("local", "EC.L3", "EC.L3", "basal", 0.4, spread=0.15, kind="local"),
        S("local", "EC.L3", "EC.L3", "apical", 1.0, spread=0.15, kind="local"),
        S("HC-14 subiculum->EC L5b", "SUB.pyr", "EC.L5", "basal", 0.5, 2.0, 0.3),
        S("HC-13 CA1->EC L5", "CA1.pyr", "EC.L5", "basal", 0.2, 1.0, 0.3),
        S("local", "EC.L5", "EC.L5", "basal", 0.3, spread=0.15, kind="local"),
        S("local", "EC.L3", "EC.L5", "apical", 1.0, spread=0.15, kind="local"),
        S("HC-3 perforant path->DG", "EC.stellate", "DG.gc", "basal", 0.85, 2.0, 0.4),
        S("DG mossy cell loop", "DG.mossy", "DG.gc", "basal", 0.15, spread=0.4, kind="local"),
        S("local", "DG.gc", "DG.mossy", "basal", 0.7, spread=0.3, kind="local"),
        S("CA3 back-projection", "CA3.pyr", "DG.mossy", "basal", 0.3, spread=0.3),
        S("HC-7 mossy fibers (detonator)", "DG.gc", "CA3.pyr", "basal", 0.05, 8.0, 0.1),
        S("HC-8 CA3 recurrent", "CA3.pyr", "CA3.pyr", "basal", 0.65, 1.0, 0.5, kind="local", ach_suppress=0.8),
        S("HC-4 perforant path->CA3", "EC.stellate", "CA3.pyr", "basal", 0.30, 1.0, 0.4),
        S("HC-8 CA3 recurrent (apical)", "CA3.pyr", "CA3.pyr", "apical", 0.5, 1.0, 0.5, kind="local",
          ach_suppress=0.8),
        S("HC-4 perforant path->CA3 tuft", "EC.stellate", "CA3.pyr", "apical", 0.5, 1.0, 0.4),
        S("HC-9 Schaffer collaterals", "CA3.pyr", "CA1.pyr", "basal", 0.9, 1.0, 0.4, ach_suppress=0.5),
        S("local", "CA1.pyr", "CA1.pyr", "basal", 0.1, spread=0.2, kind="local"),
        S("HC-5 temporoammonic->CA1 tuft", "EC.L3", "CA1.pyr", "apical", 1.0, 1.0, 0.3),
        S("HC-13 CA1->subiculum", "CA1.pyr", "SUB.pyr", "basal", 0.7, 1.0, 0.3),
        S("local", "SUB.pyr", "SUB.pyr", "basal", 0.3, spread=0.2, kind="local"),
        S("local", "SUB.pyr", "SUB.pyr", "apical", 1.0, spread=0.2, kind="local"),
    ]
    # HC-15: entorhinal L5a -> neocortex, landing on apical tufts of area B layer 2/3
    spec.synaptic = [p for p in spec.synaptic if not (p.dst == "B.L23" and p.comp == "apical")]
    spec.synaptic += [S("HC-15 EC L5->cortex (apical)", "EC.L5", "B.L23", "apical", 0.4, 1.0, 0.3),
                      S("local", "B.L23", "B.L23", "apical", 0.3, spread=0.25, kind="local"),
                      S("local", "B.L5IT", "B.L23", "apical", 0.3, spread=0.25, kind="local")]
    P = Pooled
    ec = ["EC.stellate", "EC.L3", "EC.L5"]
    spec.pooled += [
        P("EC PV<-E", ec, "EC.PV", 6.0), P("EC E<-PV", "EC.PV", ec, 1.5, channel="pv"),
        P("DG PV<-granule", "DG.gc", "DG.PV", 8.0), P("DG granule<-PV (sparse)", "DG.PV", "DG.gc", 3.0, channel="pv"),
        P("HIPP<-granule", "DG.gc", "DG.HIPP", 4.0), P("granule<-HIPP", "DG.HIPP", "DG.gc", 1.0, channel="dend"),
        P("mossy<-PV", "DG.PV", "DG.mossy", 1.0, channel="pv"),
        P("CA3 PV<-pyr", "CA3.pyr", "CA3.PV", 6.0), P("CA3 pyr<-PV", "CA3.PV", "CA3.pyr", 1.5, channel="pv"),
        P("CA1 PV<-pyr", "CA1.pyr", "CA1.PV", 6.0), P("CA1 PV<-CA3 (feedforward)", "CA3.pyr", "CA1.PV", 3.0),
        P("CA1 pyr<-PV", "CA1.PV", "CA1.pyr", 1.5, channel="pv"),
        P("O-LM<-CA1", "CA1.pyr", "CA1.OLM", 4.0), P("CA1 tuft<-O-LM", "CA1.OLM", "CA1.pyr", 1.0, channel="sst"),
        P("SUB PV<-pyr", "SUB.pyr", "SUB.PV", 6.0), P("SUB pyr<-PV", "SUB.PV", "SUB.pyr", 1.5, channel="pv"),
    ]
    return spec


def sized(spec, n_neurons):
    """The same brain at a different size: every group (and the sensory input) scaled to give about
    n_neurons simulated neurons, on the same map with the same connection rules. Testing sizes (notes/24):
    fly ~140,000; 1,000,000."""
    import copy
    spec = copy.deepcopy(spec)
    now = sum(g.per_tile for g in spec.groups if g.key not in spec.inputs) * GRID * GRID
    f = n_neurons / now
    for g in spec.groups:
        g.per_tile = max(1, int(round(g.per_tile * f)))
    return spec


N_ACTIONS = 4


def action_channel(tiles, n_actions=N_ACTIONS):
    """Action channel of each map tile: vertical stripes of the 8x8 map (striatum and pallidum keep parallel
    channels; Alexander et al. 1986)."""
    return (tiles % GRID) * n_actions // GRID


def stage_bg(n_actions=N_ACTIONS):
    """Stage 1 + basal ganglia action loop (notes/29): cortex/thalamus -> striatum (D1 go, D2 no-go, FSI)
    -> GPe / GPi, in parallel action channels (map stripes). MOT is the efference copy of the chosen action
    (motor thalamus/cortex back to striatum), set from outside by the agent."""
    spec = stage1()
    # Corticostriatal plasticity lasts (hours, not seconds) and must show up after a few rewarded tries:
    # slower fast-weight decay and a larger step than cortex (env MSN_LR, MSN_DECAY for sweeps)
    lr, decay = float(os.environ.get("MSN_LR", "0.6")), float(os.environ.get("MSN_DECAY", "0.99995"))
    msn = lambda ct: replace(ct, long_range_max=1.0,   # all MSN excitatory input comes from other structures
                             rules=replace(ct.rules, lr=lr, fast_decay=decay))
    d1, d2 = msn(C.D1_MSN), msn(C.D2_MSN)
    spec.groups += [Group("MOT", None, 1, "MOT"),
                    Group("STR.D1", d1, 4, "STR"), Group("STR.D2", d2, 4, "STR"),
                    Group("STR.FSI", C.STR_FSI, 1, "STR"),
                    Group("GPe", C.GPE_PROTO, 1, "GPe"), Group("GPi", C.GPI, 1, "GPi")]
    ctx = ["A.L5IT", "A.L5PT", "A.L23"]
    for msn in ("STR.D1", "STR.D2"):
        # Corticostriatal input converges broadly (spread 0.5): any cortical cell can reach any channel
        spec.synaptic += [Synaptic("BG-1 cortex->striatum", ctx, msn, "basal", 0.8, 1.0, 0.5),
                          Synaptic("BG-2 thalamus->striatum", "TH.TC", msn, "basal", 0.2, 1.0, 0.5)]
    spec.pooled += [
        Pooled("FSI<-cortex", ["A.L5IT", "A.L5PT"], "STR.FSI", 4.0, sigma=None),
        Pooled("MSN<-FSI", "STR.FSI", ["STR.D1", "STR.D2"], 1.0, channel="pv", sigma=None),
        Pooled("efference", "MOT", ["STR.D1", "STR.D2"], 3.0, channel="exc", sigma=0),
        Pooled("D1->GPi (direct)", "STR.D1", "GPi", -6.0, sigma=0),
        Pooled("D2->GPe (indirect)", "STR.D2", "GPe", -6.0, sigma=0),
        Pooled("GPe->GPi", "GPe", "GPi", -0.25, sigma=0),   # GPi keeps a tonic brake on every action
    ]
    spec.inputs = ["retina", "MOT"]
    return spec


def stage_da(n_actions=N_ACTIONS, so_weight=None):
    """stage_bg + dopamine cells that compute reward prediction error themselves (notes/32).
    US: the reward signal arriving from the body (lateral hypothalamus / brainstem), set by the agent.
    STR.SO: striosome MSNs learn, from cortex, how much reward the situation predicts, and inhibit the dopamine
    cells by that much (striosome -> SNc/VTA; Crittenden et al. 2016). VTA.DA fires above its tonic level when
    reward beats the prediction and dips below it when reward falls short."""
    spec = stage_bg(n_actions)
    so_weight = float(os.environ.get("SO_WEIGHT", "4")) if so_weight is None else so_weight
    lr, decay = float(os.environ.get("MSN_LR", "0.6")), float(os.environ.get("MSN_DECAY", "0.99995"))
    lr = float(os.environ.get("SO_LR", lr))   # striosomes can learn expectations faster than actions
    so = replace(C.STRIOSOME_MSN, long_range_max=1.0, rules=replace(C.STRIOSOME_MSN.rules, lr=lr, fast_decay=decay))
    spec.groups += [Group("US", None, 1, "US"), Group("STR.SO", so, 1, "STR"), Group("VTA.DA", C.DA_NEURON, 1, "VTA")]
    spec.synaptic += [Synaptic("BG-3 cortex->striosome", ["A.L5IT", "A.L5PT", "A.L23"], "STR.SO", "basal", 0.8, 1.0, 0.5),
                      Synaptic("BG-4 thalamus->striosome", "TH.TC", "STR.SO", "basal", 0.2, 1.0, 0.5)]
    spec.pooled += [Pooled("reward -> dopamine", "US", "VTA.DA", 1.0, sigma=None),
                    Pooled("striosome -> dopamine (expectation)", "STR.SO", "VTA.DA", -so_weight, sigma=None),
                    Pooled("MSN<-FSI", "STR.FSI", "STR.SO", 1.0, channel="pv", sigma=None),
                    # the chosen action's copy reaches striosomes too, so their expectation arrives when the outcome
                    # is due and is specific to (situation, action): an action value
                    Pooled("efference -> striosome", "MOT", "STR.SO", float(os.environ.get("SO_MOT", "1.0")),
                           channel="exc", sigma=0)]
    spec.inputs = spec.inputs + ["US"]
    return spec


def stage_amy(n_actions=N_ACTIONS):
    """stage_da + amygdala (notes/32). Basolateral amygdala (BLA) principal cells, positive and negative valence,
    hear the thalamus (fast "low road") and cortex ("high road"); pain or reward arriving from the body teaches them in
    a few tries, gated by arousal (BLA_RULES). Negative cells learn from dopamine dips (punishment), positive from
    bursts. The central amygdala chain BLA(-) -> CeL SST (fear-on) -| CeL PKCd (fear-off) -| CeM output sets arousal
    (via locus coeruleus). BLA also reaches the dopamine cells: a learned danger cue makes dopamine dip before the
    pain arrives, a learned good cue makes it rise."""
    spec = stage_da(n_actions)
    noap = lambda ct, **kw: replace(ct, n_apical=0, long_range_max=1.0, **kw)
    spec.groups += [Group("BLA.pos", noap(C.BLA_POSITIVE), 2, "BLA"),
                    Group("BLA.neg", noap(C.BLA_NEGATIVE, dopamine_sign=-1.0), 2, "BLA"),
                    Group("BLA.PV", C.PV, 1, "BLA"),
                    Group("CeL.SST", replace(C.CEL_SST, long_range_max=1.0), 1, "CeA"),
                    Group("CeL.PKCD", C.CEL_PKCD, 1, "CeA"), Group("CeM", C.CEM_OUTPUT, 1, "CeA")]
    for bla in ("BLA.pos", "BLA.neg"):
        spec.synaptic += [Synaptic("AM-1 thalamus->LA (low road)", "TH.TC", bla, "basal", 0.3, 1.0, 0.5),
                          Synaptic("AM-2 cortex->LA (high road)", ["A.L23", "A.L5IT"], bla, "basal", 0.7, 1.0, 0.5)]
    spec.synaptic += [Synaptic("AM-3 BLA->CeL", "BLA.neg", "CeL.SST", "basal", 1.0, 2.0, 0.5)]
    us = float(os.environ.get("AMY_US", "2.0"))
    spec.pooled += [
        Pooled("reward -> BLA(+)", "US", "BLA.pos", us, channel="exc", sigma=None),
        Pooled("pain -> BLA(-)", "US", "BLA.neg", -us, channel="exc", sigma=None),
        Pooled("BLA PV<-E", ["BLA.pos", "BLA.neg"], "BLA.PV", 6.0, sigma=None),
        Pooled("BLA E<-PV", "BLA.PV", ["BLA.pos", "BLA.neg"], 1.5, channel="pv", sigma=None),
        Pooled("CeL SST -| PKCd", "CeL.SST", "CeL.PKCD", -3.0, sigma=None),
        Pooled("CeL PKCd -| CeM", "CeL.PKCD", "CeM", -1.0, sigma=None),
        Pooled("BLA(-) -> dopamine dip (via CeA / RMTg)", "BLA.neg", "VTA.DA", -float(os.environ.get("AMY_DA", "2.0")), sigma=None),
        Pooled("BLA(+) -> dopamine", "BLA.pos", "VTA.DA", float(os.environ.get("AMY_DA", "2.0")), sigma=None),
    ]
    return spec


def stage_timed_dopamine():
    """Reconstructed phasic VTA stage. Cue and outcome are separate events."""
    spec = stage_amy()
    spec.timed_dopamine = True
    # VTA receives the difference between cue value and the preceding expectation.
    # The timing of the paired afferents is a stand-in (Eshel et al., 2015).
    spec.pooled = [p for p in spec.pooled if p.dst != "VTA.DA"]
    spec.pooled += [Pooled("timed value -> VTA", None, "VTA.DA", 1.0, signal="value_change"),
                    Pooled("reward -> VTA", "US", "VTA.DA", 1.0, sigma=None)]
    return spec


def stage_hypothalamus():
    spec = stage_timed_dopamine()
    spec.groups += [Group("HYP.AgRP", C.AGRP, 1, "HYP"), Group("HYP.POMC", C.POMC, 1, "HYP"),
                    Group("LH.orexin", C.OREXIN, 1, "LH"), Group("LH.MCH", C.MCH, 1, "LH")]
    spec.pooled += [Pooled("hunger -> AgRP", None, "HYP.AgRP", 2., signal="hunger"),
                    Pooled("satiety -> POMC", None, "HYP.POMC", 2., signal="satiety"),
                    Pooled("AgRP disinhibits LH", "HYP.AgRP", "LH.orexin", 1.),
                    Pooled("POMC suppresses LH", "HYP.POMC", "LH.orexin", -1.),
                    Pooled("fed state -> MCH", "HYP.POMC", "LH.MCH", 1.)]
    return spec


def stage_prefrontal():
    spec = stage_hypothalamus()
    spec.groups += [Group("PFC.IT", _cortex(C.L5_IT), 3, "PFC"), Group("PFC.PV", C.PV, 1, "PFC"),
                    Group("MD.relay", C.MD, 1, "MD")]
    spec.synaptic += [Synaptic("association cortex -> PFC", "B.L23", "PFC.IT", "basal", 0.08),
                      Synaptic("PFC recurrent", "PFC.IT", "PFC.IT", "basal", 0.84, kind="local"),
                      Synaptic("MD -> PFC", "MD.relay", "PFC.IT", "basal", 0.08),
                      Synaptic("MD -> PFC tuft", "MD.relay", "PFC.IT", "apical", 0.16),
                      Synaptic("PFC recurrent tuft", "PFC.IT", "PFC.IT", "apical", 0.84, kind="local"),
                      Synaptic("PFC -> MD", "PFC.IT", "MD.relay", "basal", 1.0)]
    spec.pooled += [Pooled("PFC feedforward inhibition", "PFC.IT", "PFC.PV", 4.),
                    Pooled("PFC PV -> IT", "PFC.PV", "PFC.IT", 1.5, channel="pv"),
                    Pooled("PFC -> striatum", "PFC.IT", ["STR.D1", "STR.D2"], 0.3, channel="exc")]
    return spec


def stage_cerebellum():
    spec = stage_prefrontal()
    spec.groups += [Group("PONS.relay", C.PONTINE, 1, "PONS"),
                    Group("CB.granule", _cap(C.CB_GRANULE, 1.0), 8, "CB"),
                    Group("CB.Purkinje", C.PURKINJE, 1, "CB"), Group("CB.Golgi", C.GOLGI, 1, "CB"),
                    Group("IO.olive", C.OLIVE, 1, "IO"), Group("DCN.proj", _cap(C.DCN_PROJ, 1.0), 1, "DCN")]
    spec.synaptic += [Synaptic("corticopontine", "A.L5PT", "PONS.relay", "basal", 1.0),
                      Synaptic("mossy fibers", "PONS.relay", "CB.granule", "basal", 1.0),
                      Synaptic("parallel fibers", "CB.granule", "CB.Purkinje", "basal", 1.0),
                      Synaptic("mossy collaterals", "PONS.relay", "DCN.proj", "basal", 1.0)]
    spec.pooled += [Pooled("granule -> Golgi", "CB.granule", "CB.Golgi", 3.),
                    Pooled("Golgi feedback", "CB.Golgi", "CB.granule", 1., channel="inh"),
                    Pooled("error -> olive", None, "IO.olive", 1., signal="timing_error"),
                    Pooled("climbing fiber", "IO.olive", "CB.Purkinje", 1., channel="cf", sigma=0),
                    Pooled("Purkinje -> DCN", "CB.Purkinje", "DCN.proj", 1., channel="inh"),
                    Pooled("cerebellothalamic", "DCN.proj", "MD.relay", 0.3, channel="exc")]
    return spec


def stage_game_hippocampus():
    """Reuse the verified stage2 loop, including its replacement tuft afferents."""
    spec = stage_cerebellum()
    memory = stage2()
    core = stage1()
    existing = {g.key for g in core.groups}
    spec.groups += [g for g in memory.groups if g.key not in existing]
    spec.synaptic = [p for p in spec.synaptic if p not in core.synaptic] + memory.synaptic
    spec.pooled += [p for p in memory.pooled if p not in core.pooled]
    return spec


def stage_ca2():
    spec = stage_game_hippocampus()
    spec.groups += [Group("CA2.pyr", C.CA2_PYR, 1, "HPC"), Group("CA2.PV", C.HC_PV_BASKET, 1, "HPC")]
    spec.synaptic += [Synaptic("CA3 -> CA2", "CA3.pyr", "CA2.pyr", "basal", 0.9),
                      Synaptic("EC -> CA2", "EC.L3", "CA2.pyr", "basal", 0.1),
                      Synaptic("CA3 -> CA2 proximal apical", "CA3.pyr", "CA2.pyr", "apical", 0.9),
                      Synaptic("EC -> CA2 distal apical", "EC.L3", "CA2.pyr", "apical", 0.1),
                      Synaptic("CA2 -> CA1", "CA2.pyr", "CA1.pyr", "basal", 0.2)]
    spec.pooled += [Pooled("CA2 basket recruitment", "CA2.pyr", "CA2.PV", 5.),
                    Pooled("CA2 basket inhibition", "CA2.PV", "CA2.pyr", 1., channel="pv")]
    return spec

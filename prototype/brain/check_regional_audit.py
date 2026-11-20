"""Audit the reference CPU brain. A checkpoint must be trusted and match the full specification."""
import argparse
import copy
import json
from collections import defaultdict

import numpy as np

from run_signal_game import Brain, S, prepare
from cells import ExcitatoryGroup
import regions as R


# A and B are sensory and association maps. B uses the sensory column as its
# nearest catalog comparison. The catalog groups MD and pulvinar together.
TERRITORIES = {
    "A": R.SENSORY_CORTEX, "B": R.SENSORY_CORTEX, "PFC": R.MOTOR_PREFRONTAL_CORTEX,
    "ACC": R.CINGULATE_INSULA, "INS": R.CINGULATE_INSULA,
    "TH": R.FIRST_ORDER_THALAMUS, "TRN": R.TRN_REGION, "MD": R.HIGHER_ORDER_THALAMUS,
    "PUL": R.HIGHER_ORDER_THALAMUS, "CLA": R.CLAUSTRUM.subregions[0],
    "EC": R.ENTORHINAL, "DG": R.DENTATE_GYRUS, "CA3": R.CA3, "CA2": R.CA2,
    "CA1": R.CA1, "SUB": R.SUBICULUM, "BLA": R.BLA, "ITC": R.ITC_REGION,
    "CeL": R.CEA, "CeM": R.CEA, "STR": R.STRIATUM, "GPe": R.GPE, "GPi": R.GPI,
    "STN": R.STN_REGION, "SNr": R.SNR, "SNc": R.SNC, "CB": R.CEREBELLAR_CORTEX,
    "DCN": R.DCN, "IO": R.INFERIOR_OLIVE, "VTA": R.VTA, "LC": R.LC,
    "DR": R.DORSAL_RAPHE, "MR": R.MEDIAN_RAPHE, "NB": R.NUCLEUS_BASALIS,
    "MS": R.MEDIAL_SEPTUM, "PPN": R.PPN_LDT, "HYP": R.ARCUATE,
    "LH": R.LATERAL_HYPOTHALAMUS, "PVN": R.PVN, "SCN": R.SCN,
    # Additional relays from connections.md with no regional composition table
    "PONS": None, "VA": None, "LHb": None, "RMTg": None, "PAG": None,
}
EPSILON = 1e-6


# One directional prediction per territory. The pulse and quiet control start
# from identical developed states. Each receiving population advances alone,
# so an indirect network response cannot rescue a missing direct projection.
PROJECTIONS = {
    "A": ("A.L23", "B.L23", 1), "B": ("B.L23", "EC.stellate", 1),
    "PFC": ("PFC.IT", "MD.relay", 1), "ACC": ("ACC.IT", "STR.D1", 1),
    "INS": ("INS.IT", "ACC.IT", 1), "TH": ("TH.TC", "A.L4", 1),
    "TRN": ("TH.TRN", "TH.TC", -1), "PUL": ("PUL.matrix", "A.L23", 1),
    "MD": ("MD.relay", "PFC.IT", 1), "VA": ("VA.relay", "PFC.IT", 1),
    "CLA": ("CLA.pyr", "A.PV", 1), "EC": ("EC.stellate", "DG.gc", 1),
    "DG": ("DG.gc", "CA3.pyr", 1), "CA3": ("CA3.pyr", "CA2.pyr", 1),
    "CA2": ("CA2.pyr", "CA1.pyr", 1), "CA1": ("CA1.pyr", "SUB.pyr", 1),
    "SUB": ("SUB.pyr", "EC.L5b", 1), "BLA": ("BLA.neg", "CeL.SST", 1),
    "ITC": ("ITC.gate", "CeM", -1), "CeL": ("CeL.PKCD", "CeM", -1),
    "CeM": ("CeM", "PAG.out", -1), "STR": ("STR.D1", "GPi", -1),
    "GPe": ("GPe", "STN.proj", -1), "GPi": ("GPi", "VA.relay", -1),
    "STN": ("STN.proj", "GPi", 1), "SNr": ("SNr.out", "MD.relay", -1),
    "SNc": ("SNc.DA", "STR.FSI", 1), "PONS": ("PONS.relay", "CB.granule", 1),
    "CB": ("CB.Purkinje", "DCN.proj", -1), "DCN": ("DCN.proj", "VA.relay", 1),
    "IO": ("IO.olive", "CB.Purkinje", 1), "VTA": ("VTA.DA", "phasic dopamine", 1),
    "LC": ("LC.NE", "BLA.neg", 1), "DR": ("DR.5HT", "BLA.PV", 1),
    "MR": ("MR.5HT", "CA1.PV", 1), "NB": ("NB.ACh", "A.VIP", 1),
    "MS": ("MS.theta", "CA1.PV", -1), "PPN": ("PPN.ACh", "TH.TC", 1),
    "LHb": ("LHb.proj", "RMTg.GABA", 1), "RMTg": ("RMTg.GABA", "VTA.DA", -1),
    "PAG": ("PAG.out", "LH.orexin", 1), "HYP": ("HYP.AgRP", "LH.orexin", 1),
    "LH": ("LH.orexin", "MR.5HT", 1), "PVN": ("PVN.vasopressin", "CA2.pyr", 1),
    "SCN": ("SCN.clock", "LH.orexin", 1),
}


def projection(b, source, target, background=None, soma=False, interneuron=None, drive_bias=0.):
    from striatal_projection import amygdala_input, transmit

    rates = []
    for pulse in (0., 1.):
        trial = copy.deepcopy(b)
        fixed = dict(background or {})
        ach = pulse if source == "ach" else .5
        if source != "ach":
            fixed[source] = pulse
        for key, value in fixed.items():
            trial.r[trial.index[key]] = value
            if key in trial.g:
                trial.g[key].rate[:] = value
        if target == "phasic dopamine":
            rates.append(trial._own_dopamine())
            continue
        if interneuron:
            channels = trial._channels({"ach": ach}, False)
            trial.r[trial.index[interneuron]] = trial._update(
                interneuron, trial.g[interneuron], channels[interneuron], 1, False, ach, 0., 0.)
        group = trial.g[target]
        exc = isinstance(group, ExcitatoryGroup)
        channels = trial._channels({"ach": ach}, exc)
        if exc:
            amygdala_input(trial, channels)
            transmit(trial, channels)
        if drive_bias:
            name = "exc" if exc else "drive"
            channels[target][name] = channels[target].get(name, 0.) + drive_bias
        emitted = trial._update(target, group, channels[target], 2 if exc else 1, False, ach, 0., 0.)
        rates.append((group.soma if soma else emitted).copy())
    return float(np.mean(rates[1] - rates[0]))


def followup_probes(b):
    # PV cells need co-drive near threshold. CA3 sparsity selects a few
    # winners, so measure the pre-competition drive as well as the mean rate.
    results = {
        "raphe_with_BLA_drive": projection(b, "DR.5HT", "BLA.PV", {"BLA.pos": .5}),
        "SNc_with_cortical_drive": projection(b, "SNc.DA", "STR.FSI", {"A.L5IT": .5, "A.L5PT": .5}),
        "thalamus_to_striosome_soma": projection(b, "TH.TC", "STR.SO", soma=True),
        "mossy_fiber_soma": projection(b, "DG.gc", "CA3.pyr", soma=True),
        "prefrontal_to_claustrum": projection(b, "PFC.IT", "CLA.pyr"),
        "prefrontal_to_L6IT": projection(b, "PFC.IT", "PFC.L6IT"),
        "prefrontal_to_NP": projection(b, "PFC.IT", "PFC.NP"),
        "sensory_chandelier": projection(b, "A.L23", "A.chandelier"),
        "CA1_axoaxonic": projection(b, "CA1.pyr", "CA1.axoaxonic"),
        "B_VIP_with_ACh": projection(b, "ach", "B.VIP"),
        "pontine_collateral_to_DCN": projection(b, "PONS.relay", "DCN.proj"),
        "EC_island_to_CA1_soma": projection(b, "EC.island", "CA1.pyr", soma=True),
        "EC_island_via_CA1_NGF": projection(b, "EC.island", "CA1.pyr", soma=True, interneuron="CA1.NGF"),
        "EC_island_to_CA1_NGF": projection(b, "EC.island", "CA1.NGF"),
        "corticopontine": projection(b, "A.L5PT", "PONS.relay"),
        "cerebellar_MD": projection(b, "DCN.proj", "MD.relay"),
    }
    # Activate the timing ensemble in exactly the state used on its first
    # training step. Test its instantaneous Purkinje output independently of LTD.
    timing = copy.deepcopy(b)
    timing.parallel_fiber_ltd = np.zeros((8, 8), np.float32)
    results["pontine_collateral_during_timing"] = projection(timing, "PONS.relay", "DCN.proj")
    for source in ("IO.olive", "CB.basket", "CB.stellate"):
        results[source + "_during_timing"] = projection(timing, source, "CB.Purkinje")
    return results


def glutamate_signs(b):
    """Count signed excitatory contacts and isolate their contribution to CA3."""
    negative = contacts = affected = routes = 0
    for group in b.excitatory().values():
        for bank in group.banks():
            weights = bank.weights()
            for i, (source, _, _) in enumerate(bank.paths):
                if source not in b.g or b.g[source].c.transmitter != "glutamate":
                    continue
                slots = (bank.path == i) & (bank.gain > 0)
                if np.any(slots):
                    n = int(np.count_nonzero(weights[slots] < 0))
                    negative += n
                    contacts += int(slots.sum())
                    affected += int(n > 0)
                    routes += 1
    trial = copy.deepcopy(b)
    mossy = trial.g["CA3.pyr"].basal
    slots = np.isin(mossy.src, trial.index["DG.gc"])
    intact_weights = mossy.weights()
    mossy.conductance_floor = None
    signed_soma = projection(trial, "DG.gc", "CA3.pyr", soma=True)
    weights = mossy.weights()
    reversed_slots = slots & (weights < 0.)
    mossy.gain[reversed_slots] = 0.
    return dict(negative_contacts=negative, contacts=contacts, affected_routes=affected, routes=routes,
                mossy_contacts=int(slots.sum()), mossy_negative=int((intact_weights[slots] < 0).sum()),
                mossy_mean_weight=float(intact_weights[slots].mean()),
                signed_mossy_negative=int(reversed_slots.sum()), soma_signed=signed_soma,
                soma_intact=projection(b, "DG.gc", "CA3.pyr", soma=True),
                soma_negative_muted=projection(trial, "DG.gc", "CA3.pyr", soma=True))


def territory(key):
    return "TRN" if key == "TH.TRN" else key.partition(".")[0]


def members(value):
    return value if isinstance(value, list) else [value]


def composition(b):
    """Compare exact local type names, aggregating maps that share a catalog entry."""
    local = defaultdict(lambda: defaultdict(int))
    for key, group in b.g.items():
        subregion = TERRITORIES[territory(key)]
        if subregion:
            local[subregion.name][group.c.name] += group.n
    plan = {}
    for region in R.REGIONS:
        for sub in region.subregions:
            sub.check()
            shares = defaultdict(float)
            for cell, percent, _, _ in sub.cells:
                shares[cell.name] += percent
            realized = local[sub.name]
            plan[sub.name] = {"planned_percent": dict(shares), "actual_neurons": dict(realized),
                              "missing_types": sorted(shares.keys() - realized.keys())}
    return plan


def check_wiring(b):
    assert len(b.index) == len(b.spec.groups)
    expected = defaultdict(set)
    neighbors = defaultdict(set)
    for p in b.spec.synaptic:
        expected[p.dst, p.comp].update(members(p.src))
    for key, group in b.excitatory().items():
        for comp in ("basal", "apical"):
            bank = getattr(group, comp)
            planned = expected.pop((key, comp), set())
            if bank is None:
                assert not planned, (key, comp, planned)
                continue
            assert {s for s, _, _ in bank.paths} == planned, (key, comp, planned)
            for i, (source, _, _) in enumerate(bank.paths):
                slots = bank.src[bank.path == i]
                indices = b.index[source]
                assert slots.size and np.isin(slots, indices).all(), (source, key, comp)
    assert not expected, expected
    for p in [*b.spec.synaptic, *b.spec.pooled]:
        for target in members(p.dst):
            assert target in b.g, (p.id, target)
            if hasattr(p, "channel"):
                channels = {"pv", "sst", "dend", "ais", "volume", "inh", "exc", "cf", "tuft"} if isinstance(
                    b.g[target], ExcitatoryGroup) else {"drive"}
                assert p.channel in channels, (p.id, target, p.channel)
                if p.channel == "tuft":
                    assert b.g[target].apical is not None, (p.id, target)
                assert p.weight != 0
            for source in members(p.src):
                if source is None:
                    assert p.signal
                elif source in b.g:
                    if territory(target) != territory(source):
                        neighbors[territory(source)].add(target)
                else:
                    assert source in b.spec.inputs, (p.id, source)
    return neighbors


def check_pooled_currents(b):
    """Verify each signed terminal and each leg of a multi-source projection."""
    probe = copy.copy(b)
    probe.spec = copy.copy(b.spec)
    probe.r = np.zeros_like(b.r)
    for path in b.spec.pooled:
        probe.spec.pooled = [path]
        sources = members(path.src)
        for source in sources:
            for pulse in (0., 1.):
                signals = {}
                if source is None:
                    signals[path.signal] = pulse
                else:
                    probe.r[b.index[source]] = pulse
                for target in members(path.dst):
                    channels = probe._channels(signals, isinstance(b.g[target], ExcitatoryGroup))
                    current = channels[target][path.channel]
                    np.testing.assert_allclose(current, pulse * path.weight / len(sources),
                                               atol=1e-5, err_msg=f"{source} -> {target} ({path.id})")
            if source is not None:
                probe.r[b.index[source]] = 0.


def response(b, stimulated, silence=()):
    trial = copy.deepcopy(b)
    trial.ablated.update(silence)
    # Fixed external schedule keeps subsequent observations and movements
    # identical even when a lesion would have changed the selected action.
    trace, dopamine = [], []
    for t in range(24):
        light = stimulated and 2 <= t < 12
        outcome = stimulated and t in (8, 18)
        x = {"retina": light * (trial.tile["retina"] % 8 < 4),
             "MOT": outcome * (trial.tile["MOT"] % 8 < 2),
             "US": float(outcome) * (1. if t < 12 else -1.)}
        signals = {"hunger": float(light), "satiety": float(stimulated and t >= 12),
                   "social_contact": float(light), "thirst": float(light),
                   "outcome_due": outcome, "timing_error": float(outcome)}
        if outcome:
            signals["timing_offset"] = 1 if t < 12 else -1
        trial.step(x, signals=signals)
        assert np.isfinite(trial.r).all() and np.isfinite(trial.last_dopamine)
        trace.append(trial.r.copy())
        dopamine.append(trial.last_dopamine)
    return np.stack(trace), np.array(dopamine)


def audit(b):
    assert b.device == "cpu" and not b.ablated
    assert b.spec == S.full_brain(), "use a reference-scale default full brain"
    neighbors = check_wiring(b)
    check_pooled_currents(b)
    groups = defaultdict(list)
    for key in b.g:
        groups[territory(key)].append(key)
    assert groups.keys() == TERRITORIES.keys() == PROJECTIONS.keys()
    assert {id(s) for s in TERRITORIES.values() if s} == {id(s) for r in R.REGIONS for s in r.subregions}
    intact, intact_da = response(b, True)
    quiet, _ = response(b, False)
    rows = {}
    for name, keys in groups.items():
        lesion, lesion_da = response(b, True, keys)
        assert all(np.all(lesion[:, b.index[k]] == 0) for k in keys), name
        evoked = {k: float(np.max(np.abs(intact[:, b.index[k]] - quiet[:, b.index[k]]))) for k in keys}
        external = sorted(neighbors[name])
        downstream = {k: float(np.max(np.abs(intact[:, b.index[k]] - lesion[:, b.index[k]]))) for k in external}
        if name == "VTA":
            downstream["phasic dopamine"] = float(np.max(np.abs(intact_da - lesion_da)))
        source, target, sign = PROJECTIONS[name]
        delta = projection(b, source, target)
        rows[name] = dict(source=source, target=target, delta=delta, expected=sign,
                          confirmed=sign * delta > EPSILON, cells={k: [b.g[k].c.name, b.g[k].n] for k in keys},
                          evoked=evoked, downstream=downstream)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-brain")
    args = parser.parse_args()
    brain = Brain.load(args.load_brain) if args.load_brain else prepare(args.seed, brain="full")
    print(json.dumps({"territories": audit(brain), "composition": composition(brain),
                      "followup": followup_probes(brain), "glutamate": glutamate_signs(brain)}, indent=2))

"""Cumulative reconstruction of additional routes from notes/regions/connections.md.

Each row adds a population with an explicit afferent and an efferent. Gains are
reconstruction parameters. A pooled terminal approximates a population projection.
"""
from dataclasses import dataclass, replace
import celltypes as C
from federation import Group, Pooled, Synaptic


@dataclass(frozen=True)
class Circuit:
    key: str
    cell: object
    source: str
    target: str
    terminal: str
    weight: float
    input_weight: float = 1.0
    copies: int = 1
    apical: str | None = None
    human: str | None = None


CIRCUITS = [
    Circuit("PUL.matrix", C.TC_MATRIX, "B.L5PT", "A.L23", "exc", 0.4),
    Circuit("CLA.pyr", C.CLAUSTRUM_PN, "PFC.IT", "A.PV", "drive", 0.5),
    Circuit("STN.proj", C.STN, "A.L5PT", "GPi", "drive", 0.5),
    Circuit("SNr.out", C.SNR, "STR.D1", "MD.relay", "inh", 0.4, -1.0),
    Circuit("VA.relay", C.VA_VL, "DCN.proj", "PFC.IT", "exc", 0.4),
    Circuit("GPe.arky", C.GPE_ARKY, "STR.D2", "STR.D1", "pv", 0.4, -1.0),
    Circuit("ITC.gate", C.ITC, "BLA.pos", "CeM", "drive", -0.5),
    Circuit("PAG.out", C.PAG, "CeM", "LH.orexin", "drive", 0.3, -1.0),
    Circuit("LC.NE", C.LC_NE, "CeM", "BLA.neg", "exc", 0.3),
    Circuit("NB.ACh", C.BF_ACH, "CeM", "A.VIP", "drive", 0.4),
    Circuit("MS.theta", C.SEPTUM_PACEMAKER, "CA1.pyr", "CA1.PV", "drive", -0.3),
    Circuit("DR.5HT", C.RAPHE_5HT, "PFC.IT", "BLA.PV", "drive", 0.2),
    Circuit("MR.5HT", C.RAPHE_5HT, "LH.orexin", "CA1.PV", "drive", 0.2),
    Circuit("PPN.ACh", C.PPN_ACH, "SNr.out", "TH.TC", "exc", 0.2, -0.4),
    Circuit("SNc.DA", C.DA_NEURON, "STR.SO", "STR.FSI", "drive", 0.3, -0.5),
    Circuit("LHb.proj", C.LHB, "GPi", "VTA.DA", "drive", -0.4),
    Circuit("RMTg.GABA", C.RMTG, "LHb.proj", "VTA.DA", "drive", -0.4),
    Circuit("PVN.CRH", C.CRH, "CeM", "LC.NE", "drive", 0.2),
    Circuit("SCN.clock", C.SCN_CLOCK, "retina", "LH.orexin", "drive", 0.2),
    Circuit("DG.young", C.DG_GRANULE_YOUNG, "EC.stellate", "CA3.pyr", "exc", 0.5),
    Circuit("PFC.L6IT", C.L6_IT, "PFC.IT", "B.L23", "exc", 0.2, copies=3),
    Circuit("B.L6b", C.L6B, "B.L23", "PUL.matrix", "exc", 0.2),
    Circuit("PFC.NP", C.NP, "PFC.IT", "PFC.PV", "drive", 0.3),
    Circuit("ACC.VEN", C.VON_ECONOMO, "ACC.IT", "INS.IT", "exc", 0.2),
    Circuit("A.chandelier", C.CHANDELIER, "A.L23", "A.L5PT", "ais", 0.4),
    Circuit("A.SSTnonMartinotti", C.SST_NONMARTINOTTI, "A.L4", "A.PV", "drive", -0.3),
    Circuit("A.LAMP5", C.LAMP5_NGF, "A.L23", "A.L23", "volume", 0.4),
    Circuit("B.CCK", C.CCK_BASKET, "B.L23", "B.L23", "pv", 0.3),
    Circuit("B.rosehip", C.ROSEHIP, "B.L23", "B.L23", "sst", 0.3),
    Circuit("TH.core", C.TC_CORE, "retina", "A.L4", "exc", 0.4, copies=3),
    Circuit("TH.interneuron", C.TH_INTERNEURON, "retina", "TH.TC", "inh", 0.3),
    Circuit("EC.island", C.EC_ISLAND, "B.L23", "CA1.NGF", "drive", 0.3),
    Circuit("SUB.regular", C.SUB_REGULAR, "CA1.pyr", "EC.L5", "exc", 0.4),
    Circuit("CA1.axoaxonic", C.HC_AXOAXONIC, "CA1.pyr", "CA1.pyr", "ais", 0.4),
    Circuit("CA1.bistratified", C.BISTRATIFIED, "CA3.pyr", "CA1.pyr", "dend", 0.4),
    Circuit("CA1.ivy", C.IVY, "CA1.pyr", "CA1.pyr", "volume", 0.3),
    Circuit("CA1.NGF", C.HC_NGF, "EC.L3", "CA1.pyr", "volume", 0.3),
    Circuit("CA1.IS", C.IS_CELL, "CA1.pyr", "CA1.OLM", "drive", -0.3),
    Circuit("CeL.CRF", C.CEA_CRF, "BLA.neg", "LC.NE", "drive", 0.3),
    Circuit("STR.TAN", C.TAN, "TH.TC", "STR.FSI", "drive", 0.2),
    Circuit("STR.LTS", C.STR_LTS, "PFC.IT", "STR.D1", "dend", 0.4),
    Circuit("STR.CR", C.STR_CR, "B.L23", "STR.D2", "pv", 0.3),
    Circuit("STR.TH", C.STR_TH, "TH.TC", "STR.D1", "pv", 0.3),
    Circuit("CB.basket", C.CB_BASKET, "CB.granule", "CB.Purkinje", "ais", 0.5),
    Circuit("CB.stellate", C.CB_STELLATE, "CB.granule", "CB.Purkinje", "dend", 0.4),
    Circuit("CB.Lugaro", C.LUGARO, "DR.5HT", "CB.Golgi", "drive", -0.3),
    Circuit("CB.UBC", C.UBC, "PONS.relay", "CB.granule", "exc", 0.3),
    Circuit("DCN.NO", C.DCN_NO, "CB.Purkinje", "IO.olive", "drive", -0.4, -0.4),
    Circuit("VTA.GABA", C.VTA_GABA, "PFC.IT", "VTA.DA", "drive", -0.2),
    Circuit("VTA.Glu", C.VTA_GLU, "LH.orexin", "STR.FSI", "drive", 0.2),
    Circuit("PVN.oxytocin", C.OXYTOCIN, "INS.IT", "CeM", "drive", -0.2),
    Circuit("PVN.vasopressin", C.VASOPRESSIN, "INS.IT", "CA2.pyr", "exc", 0.3),
    Circuit("PVN.MC4R", C.PVN_MC4R, "HYP.POMC", "HYP.AgRP", "drive", -0.3),
    Circuit("EC.L5b", C.EC_L5B, "SUB.pyr", "EC.stellate", "exc", 0.3),
    Circuit("EC.L5a", C.EC_L5A, "EC.L5b", "B.L23", "exc", 0.3),
]


def add_circuit(spec, circuit, region):
    cell = circuit.cell
    sampled = cell.family in {"pyramidal", "stellate", "relay", "granule", "msn", "rebound", "learning_inh"}
    if circuit.human:
        cell = replace(cell, human=circuit.human)
    if sampled:
        cell = replace(cell, n_apical=cell.n_apical if circuit.apical else 0, long_range_max=1.0)
    spec.groups.append(Group(circuit.key, cell, circuit.copies, region))
    afferent = f"{circuit.source} -> {circuit.key}"
    if sampled:
        spec.synaptic.append(Synaptic(afferent, circuit.source, circuit.key, "basal", 1., circuit.input_weight))
        if circuit.apical:
            spec.synaptic.append(Synaptic(f"{circuit.apical} -> {circuit.key} tuft", circuit.apical,
                                          circuit.key, "apical", 1.))
    else:
        spec.pooled.append(Pooled(afferent, circuit.source, circuit.key, circuit.input_weight))
    efferent = f"{circuit.key} -> {circuit.target}"
    spec.pooled.append(Pooled(efferent, circuit.key, circuit.target, circuit.weight, channel=circuit.terminal))


def extend(spec, through):
    keys = [c.key for c in CIRCUITS]
    if through not in keys:
        raise ValueError(f"unknown reconstruction stage {through!r}")
    last = keys.index(through)
    circuits = CIRCUITS[:last + 1]
    # Earlier partial reconstructions need the island cell's SLM receiver.
    # Keep full-brain population order stable for seeded wiring and saved maps.
    if keys.index("EC.island") <= last < keys.index("CA1.NGF"):
        circuits.append(CIRCUITS[keys.index("CA1.NGF")])
    for circuit in circuits:
        add_circuit(spec, circuit, circuit.key.partition(".")[0])
        if circuit.key in ("PVN.oxytocin", "PVN.vasopressin"):
            signal = "social_contact" if circuit.key == "PVN.oxytocin" else "thirst"
            spec.pooled.append(Pooled(f"{signal} -> {circuit.key}", None, circuit.key, 2., signal=signal))
        if circuit.key == "STN.proj":
            spec.pooled.append(Pooled("GPe -> STN", "GPe", circuit.key, 0.5, channel="inh"))
        elif circuit.key == "VA.relay":
            spec.pooled.append(Pooled("GPi -> motor thalamus", "GPi", circuit.key, 1., channel="inh"))
        elif circuit.key == "RMTg.GABA":
            # Replace the earlier indirect habenula terminal with its GABA relay.
            spec.pooled = [p for p in spec.pooled if p.id != "LHb.proj -> VTA.DA"]
    return spec

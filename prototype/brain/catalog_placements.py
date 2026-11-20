"""One additional 8×8 map per missing local catalog type at reference scale.

The cohorts follow the regional composition tables. Each row names the actual
afferent, emitting cell, receiver and terminal in the federated game brain.
"""
import celltypes as C
from regional_circuits import Circuit, add_circuit


PLACEMENTS = {
    "sensory cortex": [
        Circuit("A.NP", C.NP, "A.L23", "A.PV", "drive", .3, apical="B.L5IT"),
        Circuit("B.L6IT", C.L6_IT, "B.L23", "CLA.pyr", "exc", .2),
    ],
    "prefrontal cortex": [
        Circuit("PFC.L4", C.L4_STELLATE, "MD.relay", "PFC.PV", "drive", .35),
        Circuit("PFC.L23", C.L23_IT, "PFC.L4", "PFC.IT", "exc", .3, apical="B.L5IT"),
        Circuit("PFC.L5PT", C.L5_PT, "PFC.L23", "STN.proj", "exc", .25, apical="MD.relay"),
        Circuit("PFC.L6CT", C.L6_CT, "PFC.IT", "MD.relay", "exc", .2),
        Circuit("PFC.L6b", C.L6B, "PFC.IT", "PFC.L6CT", "exc", .2),
        Circuit("PFC.CCK", C.CCK_BASKET, "PFC.L23", "PFC.IT", "pv", .3),
        Circuit("PFC.chandelier", C.CHANDELIER, "PFC.L23", "PFC.L5PT", "ais", .3),
        Circuit("PFC.SST", C.SST, "PFC.IT", "PFC.L23", "sst", .3),
        Circuit("PFC.SSTnonMartinotti", C.SST_NONMARTINOTTI, "PFC.L4", "PFC.PV", "drive", -.3),
        Circuit("PFC.VIP", C.VIP, "B.L23", "PFC.SST", "drive", -.3),
        Circuit("PFC.NGF", C.LAMP5_NGF, "MD.relay", "PFC.L23", "volume", .3),
        Circuit("PFC.rosehip", C.ROSEHIP, "PFC.L23", "PFC.L5PT", "sst", .3),
    ],
    "cingulate and insular cortex": [
        Circuit("INS.L4", C.L4_STELLATE, "TH.TC", "INS.IT", "exc", .35),
        Circuit("ACC.L23", C.L23_IT, "INS.IT", "ACC.IT", "exc", .3, apical="PFC.IT"),
        Circuit("ACC.L5PT", C.L5_PT, "ACC.L23", "STN.proj", "exc", .25, apical="MD.relay"),
        Circuit("INS.NP", C.NP, "INS.IT", "INS.PV", "drive", .3, apical="B.L5IT"),
        Circuit("ACC.L6CT", C.L6_CT, "ACC.IT", "MD.relay", "exc", .2),
        Circuit("INS.L6IT", C.L6_IT, "INS.IT", "CLA.pyr", "exc", .2),
        Circuit("INS.L6b", C.L6B, "INS.IT", "INS.L6IT", "exc", .2),
        Circuit("ACC.CCK", C.CCK_BASKET, "ACC.L23", "ACC.IT", "pv", .3),
        Circuit("INS.chandelier", C.CHANDELIER, "INS.IT", "INS.IT", "ais", .3),
        Circuit("ACC.SST", C.SST, "ACC.IT", "ACC.L23", "sst", .3),
        Circuit("INS.SSTnonMartinotti", C.SST_NONMARTINOTTI, "INS.L4", "INS.PV", "drive", -.3),
        Circuit("ACC.VIP", C.VIP, "PFC.IT", "ACC.SST", "drive", -.3),
        Circuit("INS.NGF", C.LAMP5_NGF, "B.L23", "INS.IT", "volume", .3),
        Circuit("ACC.rosehip", C.ROSEHIP, "ACC.L23", "ACC.L5PT", "sst", .3),
    ],
    "first-order thalamus": [
        Circuit("TH.matrix", C.TC_MATRIX, "retina", "A.L23", "tuft", .25),
    ],
    "higher-order thalamus": [
        Circuit("PUL.core", C.TC_CORE, "B.L5PT", "B.L4", "exc", .35),
        Circuit("PUL.interneuron", C.TH_INTERNEURON, "B.L5PT", "PUL.matrix", "inh", .3),
    ],
    "claustrum": [
        Circuit("CLA.PV", C.PV, "CLA.pyr", "CLA.pyr", "pv", .3),
        Circuit("CLA.SST", C.SST, "CLA.pyr", "CLA.pyr", "dend", .3),
        Circuit("CLA.VIP", C.VIP, "PFC.IT", "CLA.SST", "drive", -.3),
    ],
    "entorhinal cortex": [
        Circuit("EC.SST", C.SST, "EC.L3", "EC.L3", "sst", .3),
        Circuit("EC.VIP", C.VIP, "B.L23", "EC.SST", "drive", -.3),
        Circuit("EC.NGF", C.LAMP5_NGF, "B.L23", "EC.stellate", "volume", .3),
    ],
    "dentate gyrus": [
        Circuit("DG.chandelier", C.CHANDELIER, "DG.gc", "DG.gc", "ais", .3),
    ],
    "CA3": [
        Circuit("CA3.axoaxonic", C.HC_AXOAXONIC, "CA3.pyr", "CA3.pyr", "ais", .3),
        Circuit("CA3.bistratified", C.BISTRATIFIED, "CA3.pyr", "CA3.pyr", "dend", .3),
        Circuit("CA3.OLM", C.OLM, "CA3.pyr", "CA3.pyr", "sst", .3),
        Circuit("CA3.CCK", C.CCK_BASKET, "CA3.pyr", "CA3.pyr", "pv", .3),
        Circuit("CA3.ivy", C.IVY, "CA3.pyr", "CA3.pyr", "volume", .3),
        Circuit("CA3.NGF", C.HC_NGF, "EC.stellate", "CA3.pyr", "volume", .3),
        Circuit("CA3.IS", C.IS_CELL, "CA3.pyr", "CA3.OLM", "drive", -.3),
    ],
    "CA1": [
        Circuit("CA1.CCK", C.CCK_BASKET, "CA3.pyr", "CA1.pyr", "pv", .3),
    ],
    "subiculum": [
        Circuit("SUB.OLM", C.OLM, "SUB.pyr", "SUB.pyr", "sst", .3),
        Circuit("SUB.NGF", C.HC_NGF, "EC.L3", "SUB.pyr", "volume", .3),
    ],
    "basolateral amygdala": [
        Circuit("BLA.CCK", C.CCK_BASKET, "BLA.pos", "BLA.pos", "pv", .3),
        Circuit("BLA.SST", C.SST, "BLA.neg", "BLA.neg", "dend", .3),
        Circuit("BLA.IS", C.IS_CELL, "BLA.pos", "BLA.SST", "drive", -.3),
        Circuit("BLA.NGF", C.LAMP5_NGF, "B.L23", "BLA.neg", "volume", .3),
    ],
    "central amygdala": [
        Circuit("CeL.Glu", C.VTA_GLU, "BLA.neg", "CeM", "drive", .25, human="central amygdala glutamatergic minority (VGluT2 cell model)"),
    ],
}


def add_placements(spec):
    structures = {}
    for group in spec.groups:
        structures.setdefault(group.key.partition(".")[0], group.structure)
    for circuits in PLACEMENTS.values():
        for circuit in circuits:
            add_circuit(spec, circuit, structures[circuit.key.partition(".")[0]])

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
}


def add_placements(spec):
    structures = {}
    for group in spec.groups:
        structures.setdefault(group.key.partition(".")[0], group.structure)
    for circuits in PLACEMENTS.values():
        for circuit in circuits:
            add_circuit(spec, circuit, structures[circuit.key.partition(".")[0]])

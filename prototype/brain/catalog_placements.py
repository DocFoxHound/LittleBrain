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
}


def add_placements(spec):
    structures = {}
    for group in spec.groups:
        structures.setdefault(group.key.partition(".")[0], group.structure)
    for circuits in PLACEMENTS.values():
        for circuit in circuits:
            add_circuit(spec, circuit, structures[circuit.key.partition(".")[0]])

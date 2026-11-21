"""
Hearing (plan track L1, note 81): ear -> auditory thalamus (MGB) -> auditory cortex (A1), on a tonotopic map.

  EAR.sus, EAR.on   auditory-nerve fibres (cochlea.py), 16x16 map: frequency along x, fibre threshold along y
  MGB.v             ventral medial geniculate relay (TC relay cells), driven by the nerve; layer-6 feedback from A1;
                    inhibition from the thalamic reticular nucleus, shared with the visual relays
  AUD               primary auditory cortex, all layers (vision.add_area), narrow feedforward reach so the frequency
                    map survives (tonotopy)
"""

from dataclasses import replace

import celltypes as C
from federation import Group, Synaptic

MID = 16


def add_hearing(spec):
    from vision import add_area
    spec.groups += [Group("EAR.sus", None, 1, "ear", grid=MID), Group("EAR.on", None, 1, "ear", grid=MID),
                    Group("MGB.v", C.TC_RELAY, 1, "MGB", grid=MID)]
    spec.inputs = spec.inputs + ["EAR.sus", "EAR.on"]
    add_area(spec, "AUD", ff=["MGB.v"], spread=0.15, ff_spread=0.08)
    spec.synaptic += [Synaptic("auditory nerve -> MGB (driver)", ["EAR.sus", "EAR.on"], "MGB.v", "basal", 0.3, 3.0, 0.05),
                      Synaptic("A1 L6 CT -> MGB (modulator)", "AUD.L6", "MGB.v", "basal", 0.7, 0.15, 0.08)]
    pooled = []
    for p in spec.pooled:
        if p.id == "TRN<-relay collaterals":
            p = replace(p, src=(p.src if isinstance(p.src, list) else [p.src]) + ["MGB.v"])
        elif p.id == "CT-5 TRN->relay":
            p = replace(p, dst=(p.dst if isinstance(p.dst, list) else [p.dst]) + ["MGB.v"])
        pooled.append(p)
    spec.pooled = pooled
    return spec

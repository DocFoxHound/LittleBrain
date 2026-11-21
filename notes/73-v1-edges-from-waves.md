# 73: V1 edge detectors from retinal waves

*2026-11-21. Vision step 3 of 6. Measured with `check_vision.py orientation`: drifting gratings at 8 angles; per
cell the orientation selectivity index (OSI, 0 = untuned, 1 = one angle only), the share of clearly tuned cells
(OSI > 0.3), and how preferred angles spread over 4 bins (a real orientation map covers all angles evenly).
The LGN measures ~0.10-0.12 here, the floor of this measurement. Real V1: most cells tuned, OSI ~0.3-0.5.*

## What was built
- **Retinal waves** (`vision.Wave`): before visual experience, bands of ganglion-cell activity sweep across the retina
  in random directions; ON cells lead OFF cells (Meister et al. 1991; Kerschensteiner & Wong 2008). `VISION_WAVES` =
  days of waves (default 2).
- **Separate ON and OFF parvocellular relays** (`TH.TC` ON, new `TH.OFF`). The first version mixed ON and OFF in each
  relay cell, which throws away the sign that edge detectors are built from (my error, found here).
- **Critical period:** V1 layer 4 learns at the normal cortical rate during development (`VISION_L4_LRF`, default 20,
  undoing the cortex-wide 0.05 of note 28 for this layer only).
- Option `VISION_TC_FRAC`: more, weaker thalamic synapses per L4 cell (same total drive). Not adopted (no gain).

## Seed 0 exploration (A L4 / A L2/3: OSI median, tuned share)
| development | L4 | L2/3 |
|---|---|---|
| none | 0.14, 4% | 0.16, 0% |
| moving shapes (ON/OFF mixed relays) | 0.16, 15% | 0.21, 32% (57% at one angle) |
| waves + shapes (mixed relays) | 0.16, 16% | 0.10, 8% |
| shapes, separate ON/OFF | 0.19, 19% | 0.17, 10% |
| waves + shapes, separate ON/OFF | 0.19, 21% | 0.12, 5% |
| shapes + critical period | 0.22, 23% | 0.24, 20% |
| **waves + shapes + critical period** | **0.22, 29%**, angles even (.32/.28/.18/.22) | 0.23, 32% |
| 4x thalamic synapses: none / shapes+cp / waves+cp / 4 days waves+cp | 0.13 2% / 0.13 7% / 0.20 23% / 0.19 24% | |

## Three seeds (A L4: tuned share / OSI median / largest angle bin)
| seed | moving shapes only | waves + critical period |
|---|---|---|
| 0 | 19% / 0.19 / 0.50 | **29% / 0.22 / 0.32** |
| 1 | 13% / 0.17 / 0.46 | **23% / 0.21 / 0.39** |
| 2 | 14% / 0.18 / 0.45 | **25% / 0.20 / 0.37** |

- **Waves plus a critical period roughly double the edge-tuned V1 input cells and spread their preferred angles more
  evenly, on every seed.** With 4x thalamic synapses the waves are clearly what matters (23% vs 7% without).
- **Still about half of real V1.** Layer 2/3 is inconsistent (32% / 17% / 5%).
- Now the vision brain's default (`VISION_WAVES=2`, `VISION_L4_LRF=20`).

## Likely next improvements
- Hubel-Wiesel-style feedforward structure: L4 cells sampling ON and OFF relay cells from adjacent strips (or
  Ringach's statistical-wiring account of how ON/OFF mosaics seed orientation maps), with learning refining it.
- Recurrent sharpening within V1 (iso-orientation excitation, broad inhibition).

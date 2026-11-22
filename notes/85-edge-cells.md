# 85: V1 edge cells, and what blocks recognising shapes anywhere

*2026-11-22. Plan phase 5, after note 82. The owner chose to build V1's edge detection in as a fixed stage rather than
keep trying to grow it.*

## Why V1 had to change
- V1 layer 4's orientation tuning had dropped to the measurement's floor (OSI median 0.12, 14% tuned; note 73 had
  0.22 / 29%). At lower spatial frequencies it was **less** tuned than the relay cells feeding it (L4 0.09-0.12 vs
  relays 0.18-0.20).
- Each L4 cell had **2 ON + 2 OFF** relay synapses (the thalamic share is split over 8 relay types); real simple
  cells get ~30 LGN inputs (Alonso et al. 2001). Laying 17 + 16 of them out as Hubel-Wiesel ON/OFF strips on a pinwheel
  orientation map did not help after development (OSI 0.10). The relay map (16×16) is too coarse for strips at the
  scale of the test gratings. That attempt is not kept.

## Built: `edges.py`
- `A.E0..A.E3`: V1 complex cells for edges at 0/45/90/135°, 16×16 each. Computed from the retina's ON and OFF midget
  cells on their full 32×32 map: even and odd Gabor filters over ON − OFF (wavelength ~1/8 of the field), energy
  √(e² + o²) (phase-invariant, Movshon et al. 1978), divided by the summed energy of all orientations nearby
  (cross-orientation normalisation, Heeger 1992). Like V1, they work in map coordinates, so they are finer in the
  fovea.
- Computed inside `Retina.__call__`, so every caller (agent, development movies, checks) gets them.
- They feed V2 and V4 (layers 4 and 2/3) alongside V1 layer 2/3, and are added to those areas' like-to-like
  feedforward sources. `VISION_EDGES=0` leaves them out. Learning still happens from V2 upwards.
- `test_edges.py` 4/4: silent on a blank field; each orientation drives its own cells (0.90-0.98 vs ≤ 0.02);
  phase-invariant (0.975-0.977 over three phases); a square drives the 0° and 90° cells, not the diagonals.

## Results (3 seeds, shape at a random place, size and brightness; chance 0.25; 60 trials, ±0.06)
Linear decoder (ridge, 5-fold) on each stage's activity:
| | edge cells | V2 L2/3 | V4 L4 | V4 L2/3 | IT L4 | IT L2/3 |
|---|---|---|---|---|---|---|
| old brain (seed 0) | - | 0.35 | 0.22 | 0.15 | 0.30 | 0.23 |
| with edges, seed 0 / 1 / 2 | **0.68 / 0.72 / 0.70** | 0.22 / 0.12 / 0.25 | 0.27 / 0.55 / 0.35 | 0.25 / 0.50 / 0.32 | 0.20 / 0.30 / 0.27 | 0.22 / 0.28 / 0.22 |
| + trace learning (note 82) | same | 0.28 / 0.42 / 0.32 | 0.50 / 0.55 / 0.45 | 0.37 / 0.38 / 0.35 | 0.32 / 0.32 / 0.23 | 0.27 / 0.35 / 0.23 |

Shape game, 600 turns: 0.28-0.31 for the old brain, 0.27-0.28 with edges, 0.29-0.32 with trace learning. **Still
chance.**

Other games unharmed (seed 0, 300 turns): colour 0.39 (0.41 before), motion 0.38 (0.38).

## Found: what blocks learning (seed 0)
- **The information is there but not position-tolerant above V1.** Each V4/IT synapse set mixes all four orientation
  channels at random positions, so the shape signal fades going up. Feature-specific V2/V4 input (each cell's edge
  synapses from one orientation, set by a column map), wired into a grown brain afterwards: V4 L4 0.27 → 0.43, IT
  unchanged.
- **The readout drowns small signals.** The striatum reads the 64-tile thalamic summary and the cortical areas under
  one joint scale (note 77). The thalamic part mostly says where something is, and is large.
- The test that shows both: the striatum reading **only** the edge cells (`VISION_ROUTE=cortex`):
  | edge code read by the striatum | shape game |
  |---|---|
  | every edge cell (1024, position-specific), with the thalamic summary | 0.33 |
  | 4 global energies (one per orientation), with the thalamic summary | 0.24 |
  | 4 global energies, cortex only | 0.34 |
  | **16: each orientation pooled over the four quadrants, cortex only** | **0.57** |
  The 4 global energies separate the shapes offline (held-out decode 0.72: bars are mostly one orientation, squares
  two, discs all four), but they are tiny (0.003-0.034).

## Next
A visual area whose cells pool one orientation over a large region (feature-specific convergence, as V4 has), giving
the striatum a position-tolerant code; and a readout whose thalamic part does not swamp it.

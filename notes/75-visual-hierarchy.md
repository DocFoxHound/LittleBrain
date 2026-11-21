# 75: Like-to-like wiring, MT and the ventral stream (V2, V4, IT)

*2026-11-22. Vision step 5 of 6. Default vision brain now 30,592 neurons (with the colliculus of note 76).*

## Making layer 2/3 carry what layer 4 sees
Measured as colour difference / trial-to-trial noise (relays 3-6, V1 L4 ~2.2-2.5, L2/3 ~1.1 at the start):
- **Like-to-like wiring after eye opening** (`vision.like_to_like`; Ko et al. 2011, 2013): each receiving cell keeps the
  sending cells within reach whose *preferences* match its own (response per movie, minus the layer's response to that
  movie, correlated across 150 movies), 3 contacts each; 3 rounds. Correlating raw time courses was wrong (it measured
  shared on/off timing). Alone: no gain (1.18).
- **Cells were near on/off switches.** Calibration set sharpness ~19, so tiny input fluctuations flipped cells and
  amplified noise. A cap on visual-cortex sharpness set during calibration (`VISION_SCALE_CAP`, default 8,
  `Brain.scale_caps`): L4 2.5 → 3.75, L2/3 1.18 → 1.43; with like-to-like: L4 4.4, L2/3 1.4-1.9 (this measure varies
  ±0.3 with presentation order; a later default brain measured 1.1). Partial.
- L2/3 is mostly driven by L4 (drive 0.38 vs 0.02 from its own recurrence; L4-drive vs response r = 0.61).

## New areas (`vision.add_area`, all layers, local wiring, PV/SST/VIP like V1)
- **MT**: L4 from the four direction relays and V1 L2/3; like-to-like across the four relays forms direction
  preferences. Feedforward to L4 uses 10% of synapses at gain 8 (as FF A→B), within the L4 long-range cap.
- **V2** from V1 L2/3 + konio LGN; **V4** from V2 + V1; **IT** from V4 + V2 (large fields).
- Like-to-like at every stage in feedforward order (`vision.LIKE_PAIRS`).
- The striatum reads **the higher areas' L2/3** (IT, V4, MT, V2; V1 only if nothing higher), as in primates.

## What each area carries (seed 0; identify one of 4 classes, nearest centroid, chance 0.25)
| area | colour | motion | shape at random place/size |
|---|---|---|---|
| V1 L4 | 0.96 | 0.62 | 0.46 |
| V1 L2/3 | 0.38 | 0.25 | 0.21 |
| V2 L2/3 | 0.50 | 0.21 | 0.33 |
| V4 L2/3 | 0.38 | 0.33 | 0.04 |
| IT L4 / L2/3 | 0.67 / **0.71** | 0.25 / 0.21 | 0.21 / 0.25 |
| MT L2/3 | 0.38 | **0.62** | 0.33 |
MT direction selectivity (separate probe): L4 identifies direction 92%, median DSI 0.22, 36% of cells > 0.3,
preferred directions spread evenly (0.21-0.32 each) like MT's direction columns; V1 17% (none, as designed).

## Behaviour (3 seeds, last 100 of 300 turns, chance 0.25)
| game | learning (seeds 0 / 1 / 2) | learning off |
|---|---|---|
| signal (64 turns) | 0.61 / 0.64 / 0.58 | 0.30 / 0.23 / 0.33 |
| **colour** | **0.39 / 0.46 / 0.41** | 0.25 / 0.23 / 0.23 |
| motion | 0.25 / 0.23 / 0.25 | 0.20 / 0.16 / 0.23 |
| shape | 0.28 / 0.30 / 0.31 | 0.29 / 0.27 / 0.22 |
- **Colour is now learned through visual cortex** (via V2/V4/IT) on every seed. (Before the hierarchy: chance.)
- **Motion**: learned when the striatum reads MT alone (seed 0: 0.49-0.50 vs 0.24), lost when V2/V4/IT are read too.
  The reduced striatal learner is diluted by areas that carry nothing for the task.
- **Shape regardless of position**: not represented anywhere, including IT. Real IT learns it from temporal continuity
  (an object seen moving/shifting is linked across its views: trace learning, Földiák 1991). Not built yet.

## Next
- Temporal-trace learning in V4/IT for position-tolerant shape.
- A striatal readout that isn't diluted by uninformative areas (e.g. parallel cortico-striatal loops per area, or
  learned per-area gain).
- Scale: the 100k human-proportion option (note 70) does not yet include the vision areas.

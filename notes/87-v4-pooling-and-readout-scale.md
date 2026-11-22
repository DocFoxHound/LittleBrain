# 87: V4 orientation pooling and a separate readout scale (phase 5, continued)

*2026-11-22. The two steps the owner chose after note 85.*

## Built
- **Feature-specific V4 wiring** (`edges.feature_wiring`, `VISION_V4_FEATURE=1`, opt-in): when a brain is built,
  each V4 cell's synapses from the edge cells all come from **one orientation**, set by a pinwheel column map; their
  receptive field (where on the map they come from) is unchanged. V4 takes its V1 input from the edge cells instead of
  V1 layer 2/3, over a wider field (spread 0.35). The weights learn as usual. 12 edge synapses per L4 cell and 61 per
  L2/3 cell, all of one orientation; saved with the brain (`spec.feature_wiring`).
- **Separate readout scales** (`VISION_SCALE`, now `separate` by default): the striatum's thalamic and cortical inputs
  are each scaled by their own running size, so one can't swamp the other. `VISION_SCALE=joint` restores note 77.

## Results (3 seeds, brains grown with feature wiring; last 100 of 300 turns, shape 600; chance 0.25)
| game | shared scale | **separate scales** |
|---|---|---|
| shape | 0.21 / 0.31 / 0.26 | 0.33 / 0.28 / 0.28 |
| colour | 0.42 / 0.41 / 0.43 | 0.40 / 0.45 / 0.43 |
| motion | 0.24 / 0.20 / 0.32 | **0.34 / 0.30 / 0.39** |
| signal | 0.46 / 0.45 / 0.48 | 0.49 / 0.45 / 0.48 |
Separate scales: motion better on every seed, the rest unchanged → **now the default**.

Shape identification (ridge decoder, 60 trials, ±0.06), V4 L4 / IT L2/3:
| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| edges only (note 85) | 0.27 / 0.22 | 0.55 / 0.28 | 0.35 / 0.22 |
| + feature wiring | 0.43 / 0.18 | 0.38 / 0.42 | 0.37 / 0.23 |
No clear gain → feature wiring stays **opt-in**. **Shape game still at chance; the bar is 0.6.**

## Why (seed 0)
- Reading **only** V4 (`VISION_SOURCE`): L4 0.29-0.35, L2/3 0.34-0.35, IT 0.22-0.28, against **0.57** for edge energy
  pooled by quadrant (note 85). V4 carries less than ideal pooling would.
- **V4's own wiring drowns its input.** While viewing shapes, V4 L4's drive is 33% from V4 L4, 23% from V4 L6, 20%
  from V4 L2/3, 12% from V2 and only **12% from the edge cells**; V4 L2/3 is 87% from V4 L4. The local recurrent
  synapses, never trained on shapes, scramble the pooled signal.
- Stronger edge synapses in a grown brain help a little: V4 L4 0.43 → 0.50 (4×) / 0.53 (10×); IT unchanged.

## Side finding: the signal game
On these brains the signal game scores 0.45-0.49 (chance 0.25), not note 77's 0.83 / 0.69 / 0.75. **Not caused by the
recent work**: a phase-0 brain run with the phase-0 code (commit deed591) gives the same, 0.47 (300 turns) and 0.56
(64 turns). How note 77's numbers were obtained needs checking. Recorded as an open item.

## Next
The higher visual areas need their feedforward input to dominate, as real layer 4 does (few but strong, reliable
feedforward synapses that local circuits amplify, rather than override): stronger feedforward gain in V2/V4/IT (or
weaker untrained local recurrence there until it has learned), re-checked on colour and motion as well as shape.

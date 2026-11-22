# 88: Visual layer 4 listens to its input (phase 5, continued)

*2026-11-22. Note 87 found that in V4, layer 4's own local inputs made up ~76% of its drive and the edge cells only
12%. Here those local inputs are turned down.*

## Change
`VISION_L4_REC` (default now **0.3**): in V2, V4, IT and MT (not V1), layer 4's local inputs from L4, L6 and L2/3 are
scaled by 0.3, as note 74 already did for L2/3 (`VISION_REC`). Real layer 4 is driven by its few, strong
feedforward synapses, which local circuits amplify rather than override. Together with the feature-specific V4
wiring of note 87 (`VISION_V4_FEATURE`, now also **on by default**), since the gain came from the pair.

## Results (3 seeds; separate readout scales; last 100 of 300 turns, shape 600; chance 0.25)
| | colour | motion | shape |
|---|---|---|---|
| V4 feature wiring only (note 87) | 0.40 / 0.45 / 0.43 | 0.34 / 0.30 / 0.39 | 0.33 / 0.28 / 0.28 |
| **+ layer 4 local inputs × 0.3** | **0.48 / 0.48 / 0.49** | **0.40 / 0.38 / 0.44** | 0.29 / 0.28 / 0.32 |
| layer 4 × 0.3 without feature wiring (seed 0) | 0.41 | 0.36 | 0.26 |
Colour and motion are better on every seed. **Shape is still at chance** (bar 0.6). Shape identification from V4 L4
(ridge, ±0.06): 0.45 / 0.32 / 0.35; IT L2/3 0.20 / 0.23 / 0.28. The edge cells themselves stay at 0.68-0.72.

## Where shape stands
Every step so far (edge cells, separate readout scales, feature-specific V4, feedforward-dominated layer 4) helped
colour and motion or made shape more decodable early on, but the shape signal still fades by IT and the game
doesn't learn it. What has been ruled out and what's left is in note 86.

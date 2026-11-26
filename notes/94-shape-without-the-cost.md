# 94: Shape without the colour/motion cost (phase 5)

*2026-11-26. Closing the colour gap left by note 93.*

## Finding
The "colour gap" was mostly in the brains, not the readout. Note 93's pooled brains, read with the old single readout
and the pooled cells left unread, scored colour 0.45 / 0.46 / 0.32 (average 0.41), not the 0.48 of note 88's brains:
during development IT also heard the pooled cells, which cost colour on one seed. Compared on the same brains, the
territories cost colour little.

## Change
`VISION_POOLED_IT=0`: the pooled edge cells feed only the striatum (V4 → tail of the caudate), not IT.

## Results (3 seeds, new brains grown with `VISION_POOLED=1 VISION_POOLED_IT=0`; ±0.05; chance 0.25)
| same brains | shape | colour | motion |
|---|---|---|---|
| old single readout, pooled cells unread (baseline) | 0.19 / 0.29 / 0.17 | 0.39 / 0.44 / 0.50 | 0.35 / 0.38 / 0.42 |
| **territories** (`VISION_POOLED_READ=map VISION_SCALE=source STRIATUM_TERRITORIES=streams`) | **0.54 / 0.51 / 0.66** | 0.45 / 0.37 / 0.45 | 0.31 / 0.27 / 0.44 |
| change (average) | **+0.36** | −0.02 | −0.04 |
- **Shape is learned on every seed (0.51-0.66), with colour and motion within the noise of the same brains.**
- Shape reaches the 0.6 bar on one seed of three.
- Differences between independently grown brains (colour 0.44 here vs 0.48 in note 88) are as large as the
  remaining colour/motion changes; settling them needs more seeds.

## Status
All opt-in. Making it the vision default would change every vision game's readout, so the other games (signal,
homing, key-door, poison) need a 3-seed check first.

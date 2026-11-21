# 77: A readout the striatum can use

*2026-11-22. After note 75, colour was learned through visual cortex but motion was lost when the striatum read all
higher areas at once (MT alone: 0.49; all areas: ~0.25).*

## Diagnosis (seed 0)
- **Not dilution.** A standalone copy of the learning rule trained on the striatum's recorded input learns best from
  all areas together: motion 0.47 (MT alone 0.34), colour 0.49 (IT alone 0.46, V2 alone 0.49).
- **The input was too small.** After centring (note 74) the input vectors were ~0.05-0.2 in size, so their learned
  drive was weak against the striatum's other currents (a clean code of size 1 learned perfectly, note 74).
  Fix: rescale the centred input by a slow running average of its size (`agent.choose`, `Brain.trace_scale`).
  Seed 0: signal 0.61 → 0.80, colour 0.39 → 0.43, motion 0.25 → 0.28.
- **The higher areas' code drifted during play.** With learning on, the same direction's average input early vs late
  in a game was only weakly alike (cosine: MT 0.43, IT 0.26, V2 -0.11), though still separable. V2, V4, IT and MT
  were learning at the adult cortical rate from every reward. Freezing them after development (adult sensory cortex
  changes slowly and mainly when attention gates it): motion 0.28 → 0.38; slowing them 10x did not help.
  `VISION_ADULT_LRF` (default 0: stable after development; V1 keeps its adult rate).

## Results (3 seeds; last 100 of 300 turns, signal 64 turns; chance 0.25)
| game | before | **now** | learning off |
|---|---|---|---|
| signal | 0.61 / 0.64 / 0.58 | **0.83 / 0.69 / 0.75** | 0.30 / 0.23 / 0.33 |
| colour | 0.39 / 0.46 / 0.41 | **0.41 / 0.45 / 0.51** | 0.25 / 0.23 / 0.23 |
| motion | 0.25 / 0.23 / 0.25 | **0.38 / 0.30 / 0.30** | 0.20 / 0.16 / 0.23 |
Motion is now learned above the learning-off level on every seed. (Brains developed before this change; the adult
setting applied to them afterwards, as `end_critical_period` now does in development.)

## Still open
- Motion remains well below what the input supports offline (~0.5).
- Shape regardless of position (trace learning in V4/IT).

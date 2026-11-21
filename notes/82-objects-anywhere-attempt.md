# 82: Recognising objects anywhere: first attempt (plan phase 5) - not achieved

*2026-11-22.*

## Tried
- **Trace learning** (`vision.trace_learning`, opt-in, not part of development yet): "object movies" in which one
  shape slides a long way across the field (12 frames), then the like-to-like rewiring of note 75 on **whole-movie**
  responses into V4's and IT's input layers and IT L4 → L2/3. Senders that answer the same object at different
  positions should correlate with the same receivers, which would then pool across positions (Földiák 1991;
  Wallis & Rolls 1997).

## Result (seed 0; identify one of 4 shapes at random place, size and brightness; chance 0.25; 24 trials, ±0.09)
| | V1 L4 | V2 L2/3 | V4 L4 | V4 L2/3 | IT L4 | IT L2/3 |
|---|---|---|---|---|---|---|
| before | 0.38 | 0.25 | 0.21 | 0.12 | 0.17 | 0.29 |
| trace learning, varied colour/size | 0.29 | 0.38 | 0.38 | 0.29 | 0.29 | 0.17 |
| trace learning, fixed colour/size | 0.25 | 0.29 | 0.17 | 0.12 | 0.17 | 0.25 |
No reliable improvement. With varied colour and size, whole-movie responses are dominated by brightness and size (the
kept partners correlate 0.77-0.88 by those); with them held fixed, the shape signal is too weak.

## Why, and what next
- V1's orientation tuning is about half of real V1 (note 73), and the four shapes differ mostly by their edges.
- Next: stronger V1 edge tuning (Hubel-Wiesel / Ringach-style feedforward structure, note 73), then trace learning
  with saccade-driven views (note 76), and an easier test first (shape at random position, fixed size and contrast).

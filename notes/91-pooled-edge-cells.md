# 91: Pooled edge cells: shape learned for the first time (phase 5)

*2026-11-26. The owner's choice after note 90: a fixed pooling stage, then learning on top of it.*

## Built (opt-in: `VISION_POOLED=1`)
- **`V4.P0..V4.P3`** (`edges.py`): for each edge orientation, an 8×8 map of large-field cells; each sums that
  orientation's edge cells over a wide region around it (Gaussian, σ = 0.25 of the field), like V4's large-field
  complex cells. Fixed, like the edge cells. They respond by shape: horizontal bars drive P2, vertical bars P0,
  squares both, discs all four a little.
- They feed IT (with V4 L2/3 and V2) and the striatum directly (V4 projects to the tail of the caudate).
  `VISION_POOLED_READ` sets how the striatum reads them: `map` (per tile, 256 inputs), `4` (4×4 regions, 64), or `2`
  (quadrants, 16, the default).
- `it_learning.competitive_it` now takes several source groups, so IT's competitive learning can run on the pooled
  cells.

## Results (3 seeds, brains grown with the pooled cells; last 100 of 600 turns, ±0.05; chance 0.25)
| | shape | colour | motion |
|---|---|---|---|
| without pooled cells (note 88, default) | 0.29 / 0.28 / 0.32 | 0.48 / 0.48 / 0.49 | 0.40 / 0.38 / 0.44 |
| **pooled, read per tile** | **0.51 / 0.42 / 0.58** | 0.37 / 0.35 / 0.44 | 0.25 / 0.24 / 0.32 |
| pooled, read in 4×4 regions | 0.46 / 0.45 / 0.59 | 0.45 / 0.46 / 0.32 | 0.35 / 0.26 / 0.31 |
| pooled, read by quadrant | 0.42 / 0.33 / 0.46 | 0.40 / 0.38 / 0.43 | 0.35 / 0.35 / 0.46 |
| pooled cells read alone (per tile) | 0.50 / 0.48 / 0.56 | | |
- **Shape is learned above chance on every seed for the first time.** Shape decodable from the pooled cells
  0.67-0.75; IT, which now hears them, 0.52-0.62 (IT L4; was ~0.2-0.3).
- **Below the bar** (0.6 on every seed), and **it costs colour and motion**; how coarsely the striatum reads the pooled
  cells trades one against the other. Hence opt-in.
- Learning curves flatten at ~0.5 by turn 300-400 (seed 0: 0.46 → 0.59 → 0.51), so more turns won't reach 0.6.

## Learning on top of the pooled cells (seed 0, `competitive_it` with the pooled cells as IT's input)
| decoded from | shape (linear) | shape (the striatum's rule, one pass of 200 trials) | where (quadrant) | brightness |
|---|---|---|---|---|
| pooled cells | 0.76 | 0.62 | 0.86 | 0.63 |
| IT as learned, exactly | 0.70 | 0.56 | 0.85 | 0.55 |
| IT per tile | 0.68 | 0.55 | 0.83 | 0.55 |
Shape survives (unlike note 90), but IT adds nothing: position still dominates even after pooling, so competition
still sorts by place. Not used.

## Where phase 5 stands
Shape at any position is now learned to ~0.5 (chance 0.25, bar 0.6) with the fixed pooled stage, at a cost to colour
and motion. The learner gets ~0.5 from a code that a linear decoder reads at ~0.7. Notes 82-91 hold all attempts.

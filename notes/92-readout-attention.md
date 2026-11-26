# 92: Keeping colour and motion while adding shape (phase 5)

*2026-11-26. Note 91's pooled edge cells made shape learnable but cost colour and motion. The owner asked to fix the
cost first. All options here are opt-in; none adopted.*

## Where the cost comes from (pooled brains, seeds 0 / 1 / 2)
| | colour | motion |
|---|---|---|
| default brain, no pooled cells (note 88) | 0.48 / 0.48 / 0.49 | 0.40 / 0.38 / 0.44 |
| pooled brain, pooled cells read | 0.37 / 0.35 / 0.44 | 0.25 / 0.24 / 0.32 |
| **pooled brain, pooled cells not read** | 0.45 / 0.46 / 0.32 | 0.41 / 0.35 / 0.34 |
Growing up with the pooled cells barely changed the brain (seed 2's colour aside): the cost is the striatum having to
learn past 256 inputs that don't matter for the current game.

## Tried
| switch | idea | shape | colour | motion |
|---|---|---|---|---|
| `VISION_SCALE=source` (pooled read per tile) | each source area on its own scale (the pooled groups as one) | 0.46 / 0.33 / 0.48 | 0.34 / 0.40 / 0.42 | 0.34 / 0.34 / 0.29 |
| same, pooled read 4×4 | | 0.40 / 0.34 / 0.39 | 0.41 / 0.42 / 0.46 | 0.24 / 0.26 / 0.34 |
| `VISION_ATTEND=input` | learned attention on each source's **input size** | 0.29 / 0.36 / 0.39 | 0.35 / 0.25 / 0.36 | 0.20 / 0.18 / 0.22 |
| `VISION_ATTEND=1` | learned attention on each source's **learning speed** (associability; Mackintosh 1975) | 0.48 / 0.39 / 0.28 | 0.39 / 0.46 / 0.32 | 0.35 / 0.34 / 0.41 |
(last 100 of 600 turns for shape, 300 for colour and motion; each ±0.05; chance 0.25.)

- The attention rule picks the right areas: in the colour game V2 rose to 3.8 and the pooled cells fell to 0.26; in the
  shape game the pooled cells rose to 4.5 and everything else to ~0.3 (seed 0). Scaling the **input** by it made
  everything worse: each change of an input's size invalidates what was learned for it, and boosting an area boosts
  its noise too. Scaling the **learning speed** instead (what Mackintosh's theory describes) recovers most of motion
  but not colour, and loses shape on seed 2.
- No setting keeps colour and motion at their default level while learning shape.

## Where phase 5 stands
Shape is learnable at ~0.5 with the pooled cells (note 91); keeping colour and motion at the same time is not solved.

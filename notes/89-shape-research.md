# 89: Recognising shapes anywhere: where the signal is lost (phase 5 research)

*2026-11-22. After note 88 the owner asked to keep researching shape (bar: shape game ≥ 0.6 on every seed). Five more
experiments and a step-by-step trace of the signal from the eye to the choice. Shape is still not learned, but it is
now clear why.*

## New options (all opt-in; none adopted)
| switch | what | result (3 seeds) |
|---|---|---|
| `VISION_COLUMNS=tiles` | V4's orientation columns tiled to the readout's 8×8 map: one orientation per tile, all four in every 2×2 block (the pinwheel map of note 87 left some regions of the field without some orientations); development's like-to-like rewiring no longer re-mixes them (it had: only 48% of V4 L4 cells stayed single-orientation) | **V4 L4 carries shape 0.60 / 0.57 / 0.63** (was 0.45 / 0.32 / 0.35); with trace learning into IT, V4 L2/3 **0.70 / 0.62 / 0.65**. Shape game 0.34 / 0.31 / 0.23; colour 0.42 / 0.40 / 0.47 and motion 0.33 / 0.36 / 0.39 (slightly below note 88) |
| trace learning V4 → IT only | note 82's rule, restricted so V4's wiring stays | IT L4 0.37 / 0.45 / 0.45 (was ~0.2-0.3); shape game 0.18 / 0.25 / 0.29 |
| `VISION_IT_FEATURE=1` | each IT cell's V4 input from one orientation column | IT 0.18-0.37; shape game 0.27-0.34 (IT alone 0.30-0.39) |
| `STRIATUM_NORM=feature` | each striatal input divided by its own running spread | shape 0.29-0.32; motion worse (0.19-0.26) |
| 2,000 turns | longer training | 0.17-0.31 throughout: no learning |

## The trace: where shape is lost (seed 0, tiled columns + trace learning)
| stage | shape decodable? |
|---|---|
| edge cells | 0.68 (linear decoder, ±0.06) |
| V4 L2/3 | 0.70 |
| **what the striatum actually receives during the game**, cortical part | **0.65** (offline decoder trained on half the turns) |
| the striatum's learned values (which action they favour) | 0.32 |
| the choice | 0.29-0.33 |
- **The choice follows the learned values** (correlation 0.90; the favoured action is chosen 69-76% of the time).
- **The reward signal is right**: dopamine at the outcome +0.84 ± 0.13 when correct, −0.30 ± 0.26 when wrong.
  Credit goes exactly to the chosen action's channels. Weights are far from their limits.
- **The learning rule, replayed offline on the recorded inputs in a single pass** (learn 200 turns, test 200):
  0.30-0.36, i.e. what the brain does. Even told the right answer every turn (a delta rule) it reaches only
  0.46-0.49. (An earlier replay that cycled the same 400 inputs three times gave 0.38-0.54, but that let it memorise.)
- So the striatum is about as good as one-example-at-a-time learning can be on this input. **The input is the
  problem**: shape is in it, but spread over ~200 noisy features. A batch decoder needs ~200 examples to find it.
- Contrast: edge energy pooled by quadrant (16 features) decodes at 0.74-0.78 offline and the game learns it to 0.57
  (note 85). Colour is learned because colour features differ by a lot.

## What would have to be true
The brain needs a **small, position-tolerant shape code**: a few cells that respond to "horizontal bar" wherever it
is. The learned visual areas (V2, V4, IT) do not make one. Neither feature-specific wiring, nor trace learning, nor
both do; V4 gets the information but keeps it spread across positions, and IT pools noise. Real IT is thought to get
there through long experience (millions of fixations) and competitive learning that makes cells specialise, which
these areas lack (their recurrence is untrained and their competition weak).

## Options from here
1. A fixed pooling stage (orientation energy per region), as with the edge cells: known to work (0.57, 1 seed).
2. Competitive learning in IT (winner-take-all with Hebbian learning, so IT cells specialise on recurring feature
   combinations), driven by many object views, possibly with saccades: the research path, uncertain.
3. Move on to phase 3 and revisit.

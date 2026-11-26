# 93: Separate striatal territories (phase 5)

*2026-11-26. The owner's choice after note 92: give the striatum separate input territories per visual stream, as the
real striatum has (e.g. the tail of the caudate for vision), each learning on its own and combined at the choice.
Opt-in: `STRIATUM_TERRITORIES` (needs `VISION_SCALE=source`).*

## Built (`striatal_projection.py`)
- **Territories**: `1` = one per source (thalamus, IT, V4, MT, V2, pooled edge cells, grid cells, context);
  `streams` = the pooled edge cells, the grid cells and MT each alone (`STRIATUM_MT_ALONE=1`, default), everything
  else together (colour was learned best from all of those at once, note 77).
- **Each territory learns on its own** with the readout's usual rule, from its own input only.
- **The choice weights each territory by how far it has proven trustworthy**: how much more often the choice was
  rewarded when that territory's own favourite action was chosen than when it wasn't (running over ~100 trials).

## What it took (seed 0 diagnostics)
1. Each territory predicting the dopamine signal (a delta rule) and trust by low prediction error: the uninformative
   territories looked most reliable (they predict the average), and the colour territory alone fell to 0.35.
2. Trust by correlation of a territory's value with the **dopamine**: fine early, then every correlation faded once
   learning worked (dopamine then signals only surprise), trust flipped at random and colour collapsed 0.47 → 0.22.
3. Trust by correlation with the **reward**: spurious (MT looked useful in the colour game, as each territory's value
   for the chosen action is shaped by the choice itself).
4. Trust by reward when agreeing vs disagreeing (final), over ~30 trials: noisy (the grid cells, whose input doesn't
   change in this game, briefly got the most trust). Over ~100 trials: below.

## Results (3 seeds, pooled brains of note 91, pooled cells read per tile; ±0.05; chance 0.25)
| | shape | colour | motion |
|---|---|---|---|
| default brain, no pooled cells (note 88) | 0.29 / 0.28 / 0.32 | 0.48 / 0.48 / 0.49 | 0.40 / 0.38 / 0.44 |
| pooled cells, one readout (note 91) | 0.51 / 0.42 / 0.58 | 0.37 / 0.35 / 0.44 | 0.25 / 0.24 / 0.32 |
| territories per source | 0.55 / 0.46 / 0.51 | 0.28 / 0.35 / 0.34 | 0.31 / 0.34 / 0.29 |
| streams, trust by dopamine correlation | 0.54 / 0.52 / 0.65 | 0.41 / 0.37 / 0.55 | 0.28 / 0.32 / 0.36 |
| streams + MT alone, trust by dopamine correlation | 0.49 / 0.50 / 0.67 | 0.22 / 0.43 / 0.51 | 0.42 / 0.38 / 0.39 |
| **streams + MT alone, trust by agreement (~100 trials)** | **0.55 / 0.52 / 0.60** | 0.34 / 0.42 / 0.35 | **0.45 / 0.42 / 0.52** |
- **Shape ~doubles over the default and motion is above the default on every seed**, together.
- **Colour is still ~0.1 below the default.** Shape is at the bar on seed 2 only.

## Where MT sits (`STRIATUM_MT_ALONE`, same brains)
| | shape | colour | motion |
|---|---|---|---|
| MT alone (`1`, default) | 0.55 / 0.52 / 0.60 | 0.34 / 0.42 / 0.35 | 0.45 / 0.42 / 0.52 |
| MT with the rest (`0`) | 0.65 / 0.53 / 0.63 | 0.35 / 0.36 / 0.50 | 0.42 / 0.27 / 0.42 |
| MT in both (`both`) | 0.63 / 0.53 / 0.55 | 0.47 / 0.37 / 0.42 | 0.38 / 0.38 / 0.29 |
Averages (shape / colour / motion): alone 0.56 / 0.37 / 0.46; with the rest 0.60 / 0.40 / 0.37; both 0.57 / 0.42 / 0.35;
the default brain 0.30 / 0.48 / 0.41. Within the noise of each other: moving MT trades motion for a little colour.
The colour territory alone picks right only 0.35-0.40 (the single readout: 0.48); its values have grown large
(prediction error 13-21 vs ~0.7 for the others).

## Next
Colour's territory: find why it learns less than the single readout did (the per-source scaling it now gets, or
sharing trust with three other territories early on).

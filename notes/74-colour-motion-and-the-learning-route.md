# 74: Colour and motion games, and routing learning through vision

*2026-11-21/22. Vision step 4 of 6 (partly done). Before this, game learning in the full brain used a 64-number
thalamic summary of brightness (`striatal_projection.visual_activity`), so better vision could not change behaviour.*

## New games (`games/vision_games.py`), built so brightness alone cannot solve them
- **Colour game**: one disc, same place, red / green / blue / yellow at equal brightness → the matching button.
- **Motion game**: 60 random dots drifting right / down / left / up, new positions every turn → the matching button.
  Sends a movie (`Observation.frames`); the agent's eye (`retina.Sight`) steps through it.

## The learning route (`striatal_projection.py`, vision brains only; older brains unchanged and bit-identical)
`VISION_ROUTE` = `thalamus` (the old summary), `cortex` (visual cortex L2/3), or `both` (default: thalamostriatal
plus corticostriatal input, as in the real striatum; one joint scale so an uninformative part stays small).
`VISION_SOURCE` overrides the cortical source for experiments.

## What it took (each step measured; seed 0 unless stated)
1. **Cortex didn't track the picture.** Same game picture twice: thalamic relays r = 0.85, V1 L4 0.56, L2/3 0.21
   (identifies the picture 44%), L5 ~0. Stronger L4 → L2/3 drive (×10) and weaker L2/3 recurrence (×0.3) for visual
   areas (`VISION_FF`, `VISION_REC`): L2/3 r 0.45, 81%. L2/3 → L5 drive did not fix L5 (stays ~0.1): **L5 is open.**
2. **Homeostasis adapted visual cortex away from what it was looking at** (a 20-step picture). Without it L2/3
   identified colour pictures 75% vs 50%. Visual cortex now balances 10x more slowly (`vision.visual_homeostasis`).
3. **Direction relays.** One magnocellular relay averaging all four direction channels cancels direction (the same
   mixing mistake as ON/OFF in note 73). Now four direction-specific relays `TH.D0..3` (as in the LGN shell that receives
   direction-selective cells in mice, Cruz-Martin et al. 2014), and direction cells 10x stronger (they were too weak
   for random dots): each relay now prefers its direction (e.g. left relay 0.33 for leftward vs ~0.13-0.20).
4. **A single step is too noisy to learn from.** The striatum's input is integrated over the look (eligibility
   traces) and the live drive uses the same integrated input. Colour from V1 at decision time: 20% → 53% (NN).
5. **Firing strength carries colour** (red and yellow drive the same red-green cells, red more): per-step
   normalisation to unit length erased it; now a slow running scale.
6. **"A picture is on" dominated learning.** Every picture differs from the blank rest screen the same way; with
   ~78% misses the striatum learned "picture = bad" for all colours. Each turn's input is now centred on a running
   average of recent turns (habituation at the trial timescale).
7. **A perfect colour code learns perfectly** (oracle input: 86% → 100% → 100% per 50 turns): the action machinery
   is fine; the problem was the visual input.
8. **The decisive one: the critical period never closed.** Learning stayed on in the retina-to-relay synapses, and
   V1 L4 kept its 20x critical-period rate during play, so every dopamine burst reshaped the visual input. Colour
   separability halved during a game (0.90 → 0.46) and the same colour's input early vs late was unrelated
   (cosines -0.2..0.3). `vision.end_critical_period` (now run after development): relays stop learning (mature
   retinogeniculate synapses are stable in adults), L4 returns to the adult rate. Separability then stayed 1.7-1.9 and
   same-colour cosines 0.80-0.93.

## Results, default vision brain (3 seeds; colour: last 100 of 200 turns; chance 0.25)
| seed | signal (64 turns): learning / off | colour, default route: learning / off | colour, thalamic colour relays as source |
|---|---|---|---|
| 0 | 0.80 / 0.28 | 0.26 / 0.28 | **0.64** |
| 1 | 0.66 / 0.27 | 0.21 / 0.25 | **0.64** |
| 2 | 0.59 / 0.27 | 0.23 / 0.27 | **0.53** |

- **The brain can learn colour when the colour signal reaches the striatum cleanly** (relay source, every seed).
- **Through V1 L2/3 it can't yet.** Colour difference vs trial-to-trial noise: relays 3-6x, V1 L4 2.2-2.5x, L2/3
  ~1.1x. L2/3 cells sum many L4 cells with different preferences (red-ON, green-ON, blue ...), so colour contrasts
  partly cancel. A critical period for L2/3 did not sort its inputs (1.15 vs 1.18), nor did removing feedback from B,
  L5 input or recurrence; without adaptation 1.31, doubled L4 drive 1.33.
- **Motion:** retina and direction relays now carry direction; V1 does not (real brains compute it mainly in MT).
  Motion game at chance on every route (MT is part of step 5).
- Signal game with the vision brain is a bit below the reference brain (0.59-0.80 vs 0.78-0.95), as the cortical part
  of the input adds noise on a task the brightness summary already solves.

## Next (blocking steps 5-6)
- **Like-to-like wiring from L4 to L2/3** (Ko et al. 2011, 2013: it emerges after eye opening). Candidate: night-time
  structural rewiring that keeps L2/3 partners whose activity correlates with the cell's own (the brain already has
  value-guided seeded choice at night, note 24). Then V2 / V4 / IT can inherit feature-specific input.
- **MT** for motion: direction-specific input from the direction relays and V1, and MT L2/3 as a striatal source.
- **V1 L5 reliability.**

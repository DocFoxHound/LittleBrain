# 76: Eye movements: superior colliculus and saccades

*2026-11-22. Vision step 6 of 6.*

## Built (`vision.add_colliculus`, `vision.saccade_target`, agent)
- **Superior colliculus**: superficial layer = retinotopic salience map (direct retinotectal input from transient and
  ON/OFF contrast cells, plus V1 L5 corticotectal input); intermediate layer = saccade motor map with fast competition
  (kWTA 2%), held back by the basal ganglia output (SNr → SC, the saccade gate; Hikosaka et al. 2000); top-down input
  from frontal L5; superficial layer → pulvinar. Two new cell types (SC visual, SC saccade; Munoz & Wurtz 1995).
- **Saccades**: each look starts fixating the centre; when the motor map bursts (top 2% of cells ≥ 3x the map mean and
  ≥ 0.15) the eye jumps there once (`Retina.look_at`), converting map position to visual field through the fovea warp.
  `VISION_SACCADES=0` holds the eye still (control).
- **Peripheral colour game** (`peripheral`): a small coloured disc away from the centre, with a dark outline.
- Viewer: V2, MT, V4, IT and the colliculus on the map; a circle on "what it sees" marks the gaze.

## Getting it to work (seed 0)
1. Saccades landed at random (target farther after the saccade: 0.315 → 0.404). The colliculus is largely colour-blind
   (its retinal input is brightness/change cells) and the first targets were colour discs at the background's
   brightness: invisible to it. Real objects have brightness edges, so targets got a dark outline.
2. The motor map then peaked at the target (0.355 vs 0.060 elsewhere) 3 steps after onset but had faded by the fixed
   read time (step 6). Saccades are triggered by the burst, so the eye now moves the first time the map bursts.

## Results (seed 0, 40 turns)
- **Saccades land on the target**: distance from gaze 0.315 → **0.086**, closer on **100%** of turns, within 0.1 on 68%.
- **No behavioural gain yet** in the peripheral colour game (300 turns: eyes moving 0.32, held still 0.27, learning off
  0.30): colour learning itself is still weak (~0.4 for a large central disc, note 75), so the benefit of foveating a
  small target can't show. Single seed.

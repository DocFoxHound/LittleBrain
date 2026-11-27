# 98: Finishing phase 3: a third seed, the prefrontal lesion, latent learning, other games

*2026-11-27. The owner asked to finish phase 3 before deciding whether the world model becomes the default.*

## Changes
- **Prefrontal lesion** (`PFC` silenced): the planner keeps the wants it was trained with, ignores the body's
  current ones and skips the replay burst (the brain's PFC group stands in for orbitofrontal / ventromedial
  prefrontal cortex). It can still plan the way through the room.
- **The world model plans only in worlds where the character moves** (once a game has reported self-motion). In a
  game where it never moves the place never changes, and the model would only learn which button pays on average.
- `WM_GATE=1` (off): the planner steers only after something rewarding has been found (see latent learning).

## Revaluation, all seeds (always-on planner)
| seed | before the switch | world model | habit only | hippocampal formation silenced | **PFC silenced** |
|---|---|---|---|---|---|
| 0 | A first | switches by the 3rd test episode | never | never | **never** (10 episodes) |
| 1 | A first | by the 2nd | never | never | **never** (B only after episode 40) |
| 2 | B already (not a valid test) | | | | |
| 3 (new brain) | A first | reaches B from the 3rd, straight there from the 7th | never | never | **never** |
Reached B in the first 5 test episodes: world model 0.8 / 1.0 / 0.6 (seeds 0, 1, 3); **PFC silenced 0.0 / 0.0 / 0.0**.
(PFC-silenced seed 2, not a valid test for the intact brain, switched at its 3rd test episode.)

## Latent learning (Tolman & Honzik 1930): **not met**
The detour room with both gaps open; 15 food-free episodes first vs food from the start. Food episodes 2-6 (steps):
| seed | explored | naive |
|---|---|---|
| 0 | **4.4** | 12.0 |
| 1 | 24 (not found) | **4.0** |
| 2 | 24 (not found) | **4.0** |
Exploring first helps on one seed and stops the brain finding the food on two. After exploring without reward, the
planner knows the room but its tiny value differences (weight 10) become strong preferences that stop it exploring.
Two fixes, both worse overall:
- Scaling the planner by how much its values differ silenced it just when revaluation needed it (all four seeds kept
  going to the worthless food for 10 episodes).
- Letting it steer only after a first reward (`WM_GATE=1`) made the brain find the food less often at all: before
  any reward the planner's systematic wandering had been helping it explore (detour: seed 2 never found the food;
  latent: seeds 0 and 2 found nothing, explored or not).
Latent learning needs a real exploration drive (e.g. curiosity: places seldom visited are worth a visit).

## Other games with the world model on (3 seeds)
| | world model off (note 94) | on |
|---|---|---|
| signal | 0.43 / 0.55 / 0.56 | identical |
| colour | 0.45 / 0.37 / 0.45 | identical |
| key-door | 0.43 / 0.36 / 0.35 | identical |
| homing | 0.89 / 0.83 / 0.86 | **1.00 / 1.00 / 1.00** |
No cost where the character doesn't move; homing becomes perfect.

## Phase 3 against its bar
"Detour and revaluation solved within a few trials on 3 seeds, clearly better than the brain with hippocampus or
OFC/vmPFC silenced": **detour met** (3 seeds); **revaluation met** (3 valid seeds, within 2-7 trials; silenced
hippocampus and silenced PFC never switch); **latent learning not met**.

# 97: Revaluation: a planner changes its mind at once (plan phase 3, part 2)

*2026-11-27. Opt-in with the world model (`WORLD_MODEL=1`).*

## Built
- **Outcome identity** (`world_model.py`): besides plain reward, the model predicts *which* outcome each action
  brings (food A, food B: the orbitofrontal cortex's outcome identity; Schoenbaum et al. 2009) and keeps, for each
  outcome, how much of it lies ahead (successor features; Barreto et al. 2017). At the moment of choice these are
  weighted by what the body wants now (`obs.body["want"]`), so a food that became worthless loses its pull without
  relearning. `WM_OUTCOMES` (default 2).
- **Replanning when wants change**: a burst of replay through the model (`WM_REPLAN`, 400 steps) whenever the body's
  wants change (replay rises after changes of reward and motivation).
- **Time-outs are not endings**: a game's time limit no longer teaches the model "episodes end here"
  (`info["timeout"]`).
- **`RevaluationGame`** (`games/maze_games.py`; Adams & Dickinson 1981): an open room, unmarked foods A (near) and B
  (far). Episodes 0-9 only B; 10-19 only A (a habit of going to A); 20-29 both, ending at the first food eaten (A
  should win); from 30 the character is full of A (want [0, 1]) and the episode ends when B is eaten. The body's want
  for a food drops to 0 once eaten that episode (it knows it has eaten).

## What it took (seed 1)
1. Test design, four times: the habit for A must be built before the test and A must win the choice before the
   switch (first versions: B never found, or B already preferred, so "going to B" showed nothing).
2. After eating A, nothing told the brain A was gone: it kept expecting A there. Now the body's want says so.
3. **Successor features predict only what the current habit leads to** (Russek et al. 2017): after 20 episodes of
   going to A, the B still ahead from the start was −0.007 (next to B: 0.47). A burst of replay when the wants change
   (0.042), and not learning "episodes end here" from the 24-step time-outs, which had cut off the value flowing back
   from B, fixed it: from the second test episode on, straight to B by the shortest route.

## Results (3 seeds; test episodes 30-34: foods eaten / steps)
| seed | **world model** | habit only | hippocampal formation silenced |
|---|---|---|---|
| 0 | A/24, AB/20, then **B/6** ×3 | nothing eaten ×5 | A/24 ×5 |
| 1 | AB/22, then **B/6** ×4 | A/24 ×5 | A ×2, then nothing |
| 2 | B/8 ×5 (chose B already before the switch: **not a valid test**) | A/24 ×5 | A/24 ×5 |
Episodes 40-49: world model reaches B every time (6.0-8.0 steps); habit 0.0-0.1; silenced 0.0-0.6.
- **The planner revalues within one or two trials on both valid seeds**; the habit and the silenced brain keep going
  to the worthless food. Seed 2 needs a re-run (it never built the A preference the test needs).

## Detour re-checked (note 96, after the time-out change)
Food 1.0 before the block on all 3 seeds (4 steps); first 5 after: 1.0 / 1.0 / 1.0; episodes 10-20 after: 1.0 in
10.2-10.5 steps (optimal 10). Habit 0-0.1, silenced 0-0.2. Unchanged.

## Phase 3 against its bar
"Detour and revaluation solved within a few trials, clearly better than the brain with hippocampus or OFC/vmPFC
silenced": detour **met** (3 seeds); revaluation **met on 2 valid seeds** (seed 2 to re-run). Not yet: latent
learning measured on its own, the OFC/vmPFC lesion (the outcome model sits with the world model here), and whether
the world model becomes the default.

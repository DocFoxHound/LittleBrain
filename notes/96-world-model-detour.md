# 96: A world model and look-ahead: the detour (plan phase 3, part 1)

*2026-11-26. Opt-in: `WORLD_MODEL=1` (vision brain with grid cells).*

## Built
- **`world_model.py`** (reduced model): on a place code, for each action, where it leads (T_a, delta rule), what it
  pays (R_a), whether it ends the episode (E_a); a value map v. Before each choice the brain **looks two steps
  ahead** through the model; the result biases the basal ganglia's choice (`agent.choose`), weight `WM_BETA`
  (default 10). After each step it **replays** remembered places through the model, each moving toward its best
  imagined action (Dyna; Sutton 1990), a stand-in for awake hippocampal replay (Pfeiffer & Foster 2013). Silencing
  the hippocampal formation (EC, DG, CA3, CA1, SUB) switches the look-ahead off.
- **Place code** (`SelfMotion.place_code`): 100 localised firing fields tiling the room, read from the position
  estimate the grid cells come from (place cells from grid input; Solstad et al. 2006). The hippocampus's own place
  cells (note 84) are still too noisy to plan on.
- **`games/maze_games.py`, DetourGame** (Tolman 1948): a room split by a wall with a short (middle) and a long (top)
  gap; 10 food-free episodes to explore; then food (unmarked) on the far side; after episode 50 the middle gap
  closes.
- `test_world_model.py` 3/3: in a 5×5 grid on its own, the model takes the short route (4 steps), re-routes around a
  new wall within a few episodes (19, 12, 11, 10, 8 steps, then 8 = optimal) and has learned that walking into the
  wall goes nowhere.

## What it took (seed 1)
1. **Unfair test**: the detour gap first opened only when the short one closed, so it had never been seen; no
   model can plan through unknown territory. Tolman's rats knew the routes: both gaps open from the start, plus an
   exploration period.
2. **Replay memory** kept recent moments, so the many food runs pushed out the explored places: now one memory per
   distinct place.
3. **The grid cells are the wrong basis for a value map**: their fields repeat over the room, so every update
   changed values everywhere and the map stayed flat (≈ −0.06 everywhere, food included). With localised place
   fields it has a peak next to the food (1.05) and rises along the detour.
4. **The planner was outvoted**: its bias (0.3 × its value spread, ~0.015) was tiny next to the basal ganglia's own
   differences (~0.15) and the habit's pull. Weight 3: detour found after ~10 episodes; **10: within the first few**.

## Results (3 seeds, vision brain with TD on; per episode)
| | food, last 10 before | steps | food, first 5 after | food, episodes 10-20 after | steps |
|---|---|---|---|---|---|
| **world model** | **1.0 / 1.0 / 1.0** | **4.0** (optimal 4) | **1.0 / 0.8 / 0.8** | **1.0 / 1.0 / 1.0** | **10.2-10.4** (optimal 10) |
| habit only (world model off) | 0.4 / 0.6 / 0.0 | 14-24 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.1 | 23-24 |
| hippocampal formation silenced | 0.6 / 0.8 / 0.4 | 11-17 | 0.2 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 24 |
**The detour is solved within a few trials on every seed, clearly better than the brain with the hippocampus
silenced, which behaves like a habit learner.** This meets phase 3's detour bar.

## Next (phase 3)
Revaluation (food made worthless when full: a planner changes at once, a habit doesn't), latent learning measured
directly, and whether the world model should become the default.

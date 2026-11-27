# 99: Curiosity, and the world model becomes the default (phase 3 closes)

*2026-11-27. Owner's choice after note 98: make the world model the vision default, adding an exploration drive
first.*

## Built
- **Curiosity** (`world_model.py`): the model counts visits to each remembered place; when looking ahead, each
  imagined place counts for how new it is, `WM_CURIOSITY` (0.2) × 1/√(1 + visits) (count-based novelty; the
  hippocampus's novelty signal drives dopamine, Lisman & Grace 2005). It steers the look-ahead only and is never
  learned as value (replay ignores it). After the first reward it fades to a tenth (`WM_CURIOSITY_FADE`), so a new
  place no longer outbids known food a few steps away.
- **Imagined places are blurry**: the model's prediction of a place it is still learning about matches no memory
  closely, and first counted as brand new, pulling the character toward exactly the moves it was unsure of (the
  unit test then never re-routed). Imagined places now count as visited from a looser match (`WM_NOVEL_SIM` 0.8;
  storing a new memory still needs 0.97).
- **`WORLD_MODEL` defaults to 1** for brains with grid cells (the vision brain). It plans only once a game reports
  self-motion, so games where the character doesn't move are unaffected (note 98).

## Results (3 seeds; curiosity on)
| | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| **revaluation**, test episodes 30-34 | **B/6, B/8, B/6, B/6, B/6** | **B/18, B/6, B/6, B/6, B/6** | (preferred B before: not a test) | AB/16, B/8, B/8, B/8, B/8 |
| detour: food, first 5 after the block / steps, episodes 10-20 | 0.8 / 10.3 | 1.0 / 11.0 | 0.8 / 10.2 | |
| homing (got home) | 1.0 | 1.0 | 1.0 | |
- **Revaluation now from the first test episode** on seeds 0 and 1, the second on seed 3 (without curiosity: 2-7
  trials, note 98).
- Detour and homing hold.

### Latent learning (food episodes 2-6: steps; 15 food-free episodes first vs none)
| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| explored first | 14.8 | **16.2** | **14.2** |
| naive | 11.2 | 24.0 | 20.0 |
Exploring first helps on 2 of 3 seeds (without curiosity: 1 of 3, and it stopped two seeds finding food at all).
Neither finds the food in the very first food episode. **Partly met.**

## Phase 3: closed
Detour met (3 seeds), revaluation met (3 valid seeds, within 1-2 trials), both lesion controls (hippocampal
formation, prefrontal) fail as predicted, latent learning partly met. The world model is the vision default.

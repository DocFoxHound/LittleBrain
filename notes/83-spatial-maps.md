# 83: Knowing where you are (plan phase 2)

*2026-11-22. Head-direction and grid cells in the vision brain (`SPATIAL=0` leaves them out), a homing game, and an
honest negative: the reconstructed hippocampus does not yet form place cells.*

## Built
- **`spatial.py`**: self-motion (the game reports `obs.motion`, a list of (turn, distance) steps) is integrated into
  heading and position (path integration, with drift). **Reduced model**: in the brain, attractor networks do this
  (McNaughton et al. 2006; Burak & Fiete 2009); here a small integrator drives the cells, as the retina and ear are
  filter banks.
  - `ATN.hd` (64): head-direction cells, von Mises tuning (Taube, Muller & Ranck 1990).
  - `MEC.grid` (256, 16×16 map): grid cells, sum of three cosines 60° apart (Solstad et al. 2006), three modules whose
    spacing grows along the map's y axis (2, 3 and 4.5 cells: dorsal small, ventral large; Stensola et al. 2012).
  - A landmark fix: smelling home (`obs.body["home_scent"]`) or a path that starts there (`from_home`) resets the
    estimate.
  - Wired into the hippocampal formation: grid → DG, CA3 (perforant path), CA1 tuft; head direction → EC (via
    postsubiculum).
- **The striatum reads the grid code**: the entorhinal cortex projects to the nucleus accumbens (Totterdell & Meredith
  1997). `SPATIAL_READ=CA1` reads CA1 instead.
- **`games/spatial_games.py`, HomingGame**: a 7×7 room with coloured walls and an unmarked home in the middle (the
  centre's 3×3 cells see no wall: the view alone can't tell them apart). Each episode the character is carried 2-4
  random steps away (it receives the path as self-motion), then must walk back: N/E/S/W, home +1, −0.02 per step,
  at most 8 steps.
- `test_spatial.py` 5/5: head-direction tuning, path integration, the home fix, grid field size, and spacing along
  the dorsoventral axis.
- `check_spatial.py`: place-field analysis (split-half stable, peaked rate maps) along a random walk.

## Results: homing (3 seeds, TD on (γ 0.9), last 100 of 1500 turns; random walk gets home 0.25)
| got home | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **intact** | **0.82** | **0.73** | **0.77** |
| TD off | 0.64 | 0.50 | 0.68 |
| grid + head-direction cells silenced | 0.00 | 0.00 | 0.13 |
With its sense of position the brain learns the way home on every seed. Without it, it does worse than wandering (it
learns a fixed habit that suits only some starts). Credit across steps (note 80) adds 0.09-0.23.

## Negative: no place cells yet
CA1 shows no stable fields on any seed (median split-half stability 0.00-0.02; grid cells themselves 0.72).
- The grid input reaches DG/CA3/CA1, but their activity is dominated by their own dynamics. Under a fixed input,
  cortex, EC, DG and CA3 keep changing (CA3's step-to-step change is about 65% of its spread across cells), and two
  20-step visits to the same place don't correlate.
- Not fixed by slower k-WTA adaptation (`KWTA_ADAPT`), by silencing cortex, septum, EC or the dentate feedback
  interneurons, by longer visits, or by 10× stronger grid input (CA3 between/within-place 1.07 → 1.3-2.3; CA1 stays
  at 1.0).
- The loop was tuned as a memory/replay system (notes 26, 55, 60-61), where change is the signal. Real place cells
  need input to dominate during running (theta, high ACh) and recurrence during rest (sharp waves). Making that
  switch without breaking the memory tests is the next job here.

## Done-when (note 78)
- Return home clearly above the silenced control: **met** (via the entorhinal route).
- Place-like cells on every seed: **not met**.

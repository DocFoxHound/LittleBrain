# 86: Where we are in the plan (status, 2026-11-22)

*A checkpoint against [note 78](78-minecraft-plan.md), the working plan toward a Minecraft-trained template. Work is
paused here at the owner's request. Planned order: 0 → 1 → (2, 5) → 3 → 4 → 6 → 7 → 8 → 9, with tracks L, C and N
alongside. The owner chose to finish phases 2 and 5 before starting phase 3.*

## Summary
| step | status | notes | against its "done when" |
|---|---|---|---|
| **0** groundwork | ✅ done | 79 | design stamps refuse mismatched saves; vision brain 0.86 s/decision (was 1.98 s) |
| **1** credit across steps | ◐ partly | 80 | TD critic and dopamine work; key-door 3-step chain 0.26-0.49 per step, **below the 0.6 bar** (expected to need phase 3); dopamine moving earlier in the chain and the danger-game regression not yet measured |
| **N1** harm across steps | ✅ done | 80 | slow poison avoided on 3 seeds (eat 0.06-0.15 vs 0.27-0.67 without TD); silencing the habenula brings eating back on 2 of 3 seeds |
| **L1** hearing | ◐ mostly | 81 | ear, MGB, A1; text heard as sound; syllables identified 0.87-1.00 (chance 0.17); statistical word learning only a hint (right direction on 3 seeds, small) |
| **2** spatial maps | ✅ done (CA1 open) | 83, 84 | head-direction and grid cells; place cells in the dentate (37-43%) and CA3 (31-35%) on every seed; homing 0.74-0.83 vs 0.14-0.20 silenced; **CA1 has no place cells** (inhibition-dominated; deferred by the owner) |
| **5** objects anywhere | ◐ in progress | 82, 85 | V1 edge cells built (fixed stage, owner's choice): shape decodable from them at 0.68-0.72; **shape game still at chance (0.27-0.32), bar is ≥ 0.6** |
| 3 world model and look-ahead | not started | | next after 5 |
| 4 goals and routines | not started | | |
| 6 learning in sleep | not started | | |
| 7 Minecraft connection | not started | | needs owner decisions (Mojang EULA, account or browser view) |
| 8 training the template | not started | | |
| 9 shipping templates | not started | | |
| L2-L6, C1-C6, N2-N5 | not started | | |

## Where phase 5 stands (the open task)
- Found (note 85): the shape signal is there in the edge cells but nothing above them is position-tolerant, and the
  striatum's readout lets its thalamic part (which mostly says *where*) swamp the small cortical signal. With
  quadrant-pooled edge energy read alone, the shape game reached **0.57** (seed 0, vs 0.25).
- **Next, as decided by the owner:**
  1. a V4-like area whose cells each take edge input from **one orientation over a large receptive field**
     (feature-specific wiring), with **learned** weights; IT learns on top;
  2. **separate scales** for the thalamic and cortical parts of the striatal readout;
  3. re-check colour, motion and signal games on 3 seeds (note 77 chose the shared scale for colour), then the shape
     game on 3 seeds against its ≥ 0.6 bar.
- Earlier tries that did not work (kept in the notes): trace learning alone (note 82), Hubel-Wiesel rewiring of V1
  L4 (note 85), feature-specific rewiring of an already-grown brain (note 85).

## Open items carried forward
- **CA1 place cells** (note 84): excitation ~0.015 vs PV inhibition ~1.05; needs CA1's excitation/inhibition rebuilt
  without breaking the memory tests.
- **Phase 1's chain bar** (note 80): revisit after the world model (phase 3); also measure dopamine moving earlier in
  the chain and re-run the danger game as a regression.
- **TD is opt-in** (`TD_GAMMA`): games must mark episode ends before TD becomes a default.
- **Statistical word learning** (note 81): a stronger test (more exposure, trial statistics, a belt area).

## What the game brain is now
The vision brain (`BRAIN=vision`, 33,152 neurons plus input maps): retina → LGN streams → V1 (+ fixed edge cells) →
V2, V4, IT, MT; superior colliculus and saccades; ear → MGB → auditory cortex; head-direction and grid cells →
hippocampus; TD critic with habenula pauses (opt-in). Games available: signal, colour, motion, shape, peripheral,
key-door, poison, homing, plus the older danger, forage, delayed and timing games.

## Next commands (when work resumes)
```
# grow a vision brain (seed 0) and test the shape game
BRAIN=vision .venv/bin/python prototype/brain/check_full_games.py --seed 0 --prepare-only /path/b.brain
BRAIN=vision .venv/bin/python prototype/brain/check_full_games.py --seed 0 --load-brain /path/b.brain --game shape --turns 600
# all tests
for t in prototype/brain/test_*.py prototype/test_data_archive.py; do .venv/bin/python $t | tail -n 1; done
```

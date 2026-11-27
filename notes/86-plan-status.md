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
| **5** objects anywhere | ◐ nearly met | 82, 85, 87-95 | V1 edge cells built (fixed stage, owner's choice): shape decodable from them at 0.68-0.72; **shape game still at chance (0.27-0.32), bar is ≥ 0.6** |
| **3** world model and look-ahead | ✅ met | 96-99 | **detour** (3 seeds) and **revaluation** (3 valid seeds, 2-7 trials) met, silenced hippocampus and PFC never switch; latent learning partly (2 of 3 seeds, with curiosity); **the vision default** (`WORLD_MODEL=1`) |
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

## Update (note 87)
Separate readout scales done (now default: motion up on 3 seeds). Feature-specific V4 wiring built, opt-in (no
clear gain). Shape still at chance: V4's untrained local recurrence drowns its feedforward input (edges are 12% of
V4 L4's drive). Next: make feedforward input dominate in V2/V4/IT.

## Update (note 88)
Visual layer 4 now listens mostly to its feedforward input (`VISION_L4_REC=0.3`), together with the V4 feature
wiring (both default): colour 0.48-0.49 and motion 0.38-0.44 on 3 seeds (both up). Shape still at chance.
Ruled out for shape so far: trace learning alone (82); Hubel-Wiesel V1 rewiring (85); reading the edge cells
directly, position by position (0.33); feature-specific V4 alone (87); stronger edge synapses in a grown V4 (87);
feedforward-dominated layer 4 (88). What did work: edge energy pooled by quadrant, read alone (0.57, note 85).

## Update (note 89)
Five more shape experiments (tiled V4 columns, trace learning into IT, orientation-specific IT, per-input striatal
normalisation, 2,000 turns), all opt-in, none adopted. Traced the signal: shape reaches the striatum (its game input
decodes at 0.65) but is spread over ~200 noisy features, and one-trial-at-a-time learning can't find it in a few
hundred trials (the rule replayed offline: 0.30-0.36, as in the brain). Needed: a small position-tolerant shape code.

## Update (note 90)
Competitive learning in IT (k-winners-take-all with Hebbian learning, a self-organising map, and the trace rule) made
IT specialise on position (0.77) and brightness (0.66), not shape (0.34): position dominates V4's variance, and
learning position-tolerance needs input already pooled over position. Not adopted.

## Update (note 91)
Fixed pooled edge cells (`VISION_POOLED=1`, opt-in): **shape learned above chance on every seed for the first time**
(0.51 / 0.42 / 0.58; bar 0.6), but colour and motion drop (to 0.35-0.44 and 0.24-0.32 when read per tile); reading
the pooled cells more coarsely trades shape back for motion. Competitive IT on the pooled input adds nothing.

## Update (note 92)
The colour/motion cost of the pooled cells comes from the striatum learning past irrelevant inputs (not reading them
restores colour and motion). Per-source scales and learned attention (on input size, or on learning speed) tried:
attention picks the right areas but no setting keeps colour and motion while learning shape.

## Update (note 93)
Separate striatal territories (pooled cells, grid cells and MT each alone; the rest together; each learning on its
own; the choice weighted by how often following each territory paid off), opt-in: shape 0.55 / 0.52 / 0.60 and
motion 0.45 / 0.42 / 0.52 together (both at or above the default), colour 0.34 / 0.42 / 0.35 (~0.1 below).

## Update (note 94)
With the pooled cells feeding only the striatum (`VISION_POOLED_IT=0`) plus the territories: shape 0.54 / 0.51 / 0.66
(+0.36 over the same brains' old readout), colour and motion within noise (−0.02, −0.04). Opt-in; becoming the
default needs a 3-seed check of the other vision games.

## Update (note 95)
The shape setup is now the vision default (pooled cells read by the striatum, per-source scaling, striatal
territories). Phase 5: shape 0.51-0.66 on 3 seeds (bar 0.6 on every seed: met on one), recorded as nearly met.
Next: phase 3 (world model and look-ahead).

## Update (note 99)
Curiosity (count-based novelty in the look-ahead) added; revaluation now from the first test episode; latent
learning on 2 of 3 seeds; the world model is the vision default. Phase 3 closed. Next: phase 4 (goals and routines).

## Open items carried forward
- **CA1 place cells** (note 84): excitation ~0.015 vs PV inhibition ~1.05; needs CA1's excitation/inhibition rebuilt
  without breaking the memory tests.
- **Phase 1's chain bar** (note 80): revisit after the world model (phase 3); also measure dopamine moving earlier in
  the chain and re-run the danger game as a regression.
- **TD is opt-in** (`TD_GAMMA`): games must mark episode ends before TD becomes a default.
- **Signal-game score** (note 87): 0.45-0.49 now vs note 77's 0.83; phase-0 code gives the same, so not a recent
  regression; how note 77 measured it needs checking.
- **Signal game under the new readout** (note 94): −0.07 on average (seed 0 −0.14).
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

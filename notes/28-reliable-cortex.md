# 28: Making the cortex listen to its input

*2026-11-17. Project resumed with a new end goal: a tool that plugs this brain into small simulated games
to see how characters learn under different inputs and environments, with a desktop visualizer and analysis
tools. First step, agreed: make the brain actually learn, measured honestly (note 26's open problem).*

## New honest tests (`prototype/brain/diagnostics.py`)
- **Reliability (test-retest):** show 6 patterns, then the same 6 again later in a different order. Per group:
  match with its own earlier response minus match with the other patterns. 1 = reproducible and specific, 0 = unrelated.
- **Memory with a control:** learn 10 patterns, sleep, then a short **wake period** (unrelated patterns, no
  learning) so the brain recovers from the night. Then all full patterns, then all half-cues, each set shown
  twice in fresh random orders and averaged. Scored on the cortex half the cue leaves out. Reported as
  match and **identification accuracy** (chance 0.10), always next to an **untrained copy** of the same brain.
- The old single-pass score swung between −0.19 and +0.30 for *untrained* brains, so single-seed differences
  of that size meant nothing.

## What was wrong, layer by layer (stage-1 brain, 4,928 neurons, seed 0)
Before any tuning, the cortex is reliable: A L4 retest 1.00, A L2/3 0.80. Calibration and development break it.

1. **L2/3 barely hears L4.** Per L2/3 cell during a pattern:

   | input | synapses | active inputs | drive |
   |---|---|---|---|
   | from L4 (the senses) | 90 | 2.4 | 0.03 |
   | from other L2/3 cells | 128 | 12.5 | 0.14 |
   | feedback from area B | 20 | 4.6 | 0.05 |

   L4 fires at ~3%, L2/3 at ~10%, so L2/3's own chatter is ~5x louder than the senses. Even with every input
   except L4 cut, L2/3 retest was only 0.11. In real cortex L4 → L2/3 is a strong feedforward drive.
2. **The balancing rule can't recover from silence.** Both threshold rules (the cell's own `threshold_lr` and
   the brain's quantile tracker) and inhibitory plasticity step up by (1 − target) when a cell fires and down by
   target when it doesn't. At an 8% target, **recovering from silence is ~12x slower than reacting to over-firing.**
3. **One day of learning is huge.** Fast weights reach 0.28 on synapses whose long-term strength is ~0.01–0.05
   (25x; real LTP is ~1.5–3x). The learning rate was tuned on single neurons (note 16) and never rescaled when
   cells got hundreds of synapses each. During the day L2/3's drive spread grows 4–6x, its threshold climbs
   (0.05 → 1.5) and its PV inhibition grows (1.3 → 3.3). At night the fast weights are wiped, the drive
   collapses, and the raised threshold leaves L2/3 **silent** (rate 0.000). The memory test ran right then.
4. Development also briefly **shut the thalamus off** (L6 at 0.64 → TRN → relay rate 0) when balancing was off.

## Changes (all opt-in; defaults reproduce earlier results)
- `HOMEO_RULE=log`: the brain's always-on homeostasis pushes each threshold by log(recent rate / target),
  so 10x too quiet and 10x too loud are corrected equally fast. It replaces the cell's own asymmetric drift,
  and inhibitory plasticity uses the same error.
- `L4_L23_GAIN=3`: stronger L4 → L2/3 synapses (brainspec).
- `MAX_SCALE`: optional cap on intrinsic sharpness (tested: no help).
- Learning-rate factor `LRF` and fast-weight cap `FR`: in `run_cortex_fix.py` only, for now.

## Results (3 seeds each, 4,928 neurons; `run_cortex_fix.py`)
| config | A L2/3 retest | memory: identified, learned / untrained (chance 0.10) |
|---|---|---|
| original | +0.10, −0.04, −0.06 | 0.30/0.10, 0.20/0.10, 0.10/0.10 |
| `HOMEO_RULE=log` | +0.30, +0.59, +0.55 | 0.10/0.00, 0.10/0.00, 0.20/0.20 |
| + `L4_L23_GAIN=3`, `LRF=0.2` | +0.43, +0.63, +0.70 | 0.20/0.00, 0.20/0.10, 0.10/0.00 |
| + `L4_L23_GAIN=3`, `LRF=0.05` | +0.43, +0.52, +0.54 | 0.00/0.00, 0.10/0.10, 0.20/0.20 |

- **Area A now listens to its input:** L2/3 reliability goes from ~0 to 0.4–0.7 on every seed. L4 stays ~0.9.
- With `LRF=0.2`, A L2/3 is very quiet after development (rate ≤ 0.01), so its retest rests on few cells.
  `LRF=0.05` keeps it at 0.04–0.05.
- **Area B is still unreliable** (~0); it gets A's L2/3 output, which is still sparse.
- **No memory yet in any configuration.** Fill-in is a hard task for this circuit: L2/3's local connections
  reach ~1 tile, so the left half of a pattern can only recruit the right half through area B and its feedback.

## Next (open decision)
- Either keep going on cortical fill-in (area B reliability, then the A ↔ B loop), or
- switch the learning target to what game characters need: **reward-driven action learning** (cortex → striatum
  → action, dopamine as teacher), tested as "does a trained character beat an untrained one" in a tiny game.

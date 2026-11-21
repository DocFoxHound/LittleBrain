# 70: A 100,000-neuron brain with human proportions

*2026-11-20. The owner asked for a brain closer to 100,000 neurons, grown with human proportions
(`docs/scaling.md` steps 1–3), selectable rather than the default.*

## How it's built (`prototype/brain/human_scale.py`, `brainspec.full_brain_human`, `BRAIN=human100k`)
1. **Budget per catalog subregion = its share of human neurons.** Catalog totals from `regions.py` where they exist.
   Where the catalog has none, `ASSUMED_TOTALS` records the value used and its basis: cerebral cortex 16.34 billion
   (Azevedo et al. 2009), split 86% sensory/association (areas A and B), 8% prefrontal (Gabi et al. 2016),
   6% cingulate/insula (assumed); globus pallidus from Thörner et al. 1975; the thalamus, claustrum and small
   nuclei are order-of-magnitude assumptions. At this size only the cortex values matter: every other subregion's
   human share is far below one 64-cell map.
2. **Split by cell type** using the catalog percentages; maps sharing one catalog entry (A and B are both "sensory
   cortex") split it evenly.
3. **No population shrinks below its verified reference size.** At ~100k, human proportions would give the striatum,
   hippocampus, amygdala and every nucleus far less than one map, which would remove working circuits. So the
   human budget only *adds* cells where it exceeds the reference brain. The human scale is raised until the total
   reaches the requested size (`BRAIN_NEURONS`, default 100,000).

**Result: 100,032 neurons, 187 populations (same as the reference), 8.2 M sampled synapses.**
| population | reference | 100k |
|---|---:|---:|
| cerebellar granule cells | 1,280 | 71,424 |
| A and B L2/3 IT (each) | 640 | 1,920 |
| A and B L4 (each) | 384 | 1,152 |
| B L6 IT | 64 | 1,024 |
| A and B PV (each) | 256 | 896 |
| A and B L5 IT (each) | 192 | 640 |
| prefrontal L2/3 | 64 | 448 |
| cerebellar stellate | 64 | 384 |
| 18 more cortical/cerebellar types | | grown by their catalog shares |
| everything else | | unchanged |

71% of the brain is cerebellar granule cells and ~19% cortex, as in humans (~80% / ~19%).

## Cost (CPU, this machine, one seed)
| brain | neurons | synapses | build | step | memory |
|---|---:|---:|---:|---:|---:|
| reference (`full`) | 18,944 | 1.8 M | 1 s | 76 ms | 0.3 GB |
| everything ×5.2 (not used) | 97,664 | 22.2 M | 12 s | 421 ms | 1.1 GB |
| **`human100k`** | **100,032** | **8.2 M** | 3 s | **103 ms** | |
| `human100k` on the GPU (CuPy) | | | 5 s | 204 ms | |

Granule cells have few synapses, so the human-proportion brain is ~4x cheaper per step than scaling everything.
**The GPU is slower** at this size: 187 small populations mean many tiny GPU launches. Use the CPU.
Growing and developing one brain (`--prepare-only`) took **6.7–7.0 min** per seed (single-threaded BLAS, 3 seeds in
parallel); the viewer's first build 6.5 min, then cached.

## Checks (3 seeds; checkpoints prepared once per seed)
- `check_full_runtime.py`: **passed on all three** (anatomy, slow terminals, isolated and regional responses,
  frontal maintenance, scheduled night, external bridge, controls). A night rewired ~225,000 sampled partners
  (reference: ~45,000). The anatomy check required exactly 18,944 neurons; it now requires the reference brain's
  187 populations, none smaller than in the reference, and still exactly 18,944 when the sizes match the reference
  (re-run on the reference brain, seed 0: passed).
- 36/36 cell and learning tests and the data-archive test pass.
- **Signal game, 64 turns, from the same checkpoint (chance 0.25):**
  | seed | learning | learning off | reference brain (note 66) |
  |---:|---:|---:|---:|
  | 0 | 0.78 | 0.23 | 0.80 |
  | 1 | 0.95 | 0.27 | 0.92 |
  | 2 | 0.88 | 0.33 | 0.94 |
  The 100k brain learns on every seed, about as well as the reference.
- **Not yet run at 100k:** the regional audit (`check_regional_audit.py`), placements check, and the other four games
  (danger, forage, delayed, timing) with their controls. Those are needed before making it the default.

## Reproduce
```sh
export OPENBLAS_NUM_THREADS=1 BRAIN=human100k
python prototype/brain/check_full_games.py --seed N --prepare-only /tmp/h100-N.brain
python prototype/brain/check_full_runtime.py --seed N --load-brain /tmp/h100-N.brain
python prototype/brain/check_full_games.py --seed N --load-brain /tmp/h100-N.brain --game signal --turns 64 --condition learning
python prototype/brain/check_full_games.py --seed N --load-brain /tmp/h100-N.brain --game signal --turns 64 --condition off
python prototype/viz/brainview.py --brain human100k            # viewer
```

## Viewer
Works unchanged (level of detail caps each group's dots). The legend now gives the range of neurons per dot
(1–357 at 100k: one granule-cell dot stands for ~357 cells).

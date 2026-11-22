# 84: Place cells (fixing phase 2)

*2026-11-22. Note 83 found no place cells: the hippocampus did not hold a steady code for a steady place. Now the
dentate gyrus and CA3 have place cells on every seed; CA1 still does not.*

## What was wrong (seed 0, traced stage by stage)
The grid cells were stable (split-half 0.72), the hippocampus was not:
1. **The grid input was too weak to steer it** (~0.013 per dentate cell). The k-winners-take-all competition ranks
   cells on how unusual their drive is, so small fluctuations chose the winners.
2. **Learning on the grid synapses rewrote the fields on every visit.** The dentate's winners change from step to
   step, and each learned whichever place it was in, so fields smeared away. With learning off, CA3 had 38%
   place-like cells; with learning on, 0%. With only the grid synapses frozen, 27%.
3. **Homeostasis ran at full speed in the sparse principal cells**: it pushed every recent winner back down.
Not the cause (tested): the competition's adaptation speed, cortex/septum/EC/interneuron inputs, a population-wide
competition scale, visit length, the view.

## Fix (spatial brains only, in `spatial.add_spatial`)
- Grid → dentate gain 10 (was 2), grid → CA3 gain 5 (was 1).
- **The perforant path from grid cells learns at 1/20 the speed** (`spec.plasticity`, a new per-pathway learning
  speed, applied as the synapses' learning mask). Frozen synapses did about as well (DG 30%, CA3 29%) as slow ones
  (31%, 22%); slow is closer to the real, plastic perforant path.
- **Homeostasis 50× slower** in DG, CA2, CA3 and CA1 principal cells (`spec.homeo_speed`), as real homeostatic
  plasticity takes hours, not seconds.

## Results (`check_spatial.py`, 3 seeds, 800-step random walk, learning on)
| place-like cells (stable, peaked fields) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| dentate gyrus | **38%** | **43%** | **37%** |
| CA3 | **31%** | **31%** | **35%** |
| CA1 | 0% | 0% | 0% |
| shuffled positions (chance) | DG 2%, CA3 0% | 1%, 0% | 3%, 1% |
| grid and head-direction cells silent | 0% | 0% | 0% |

Homing (note 83) still works: home 0.74 / 0.83 / 0.83 vs 0.14 / 0.14 / 0.20 with grid and head-direction cells
silenced. Reading CA3 instead of the grid cells into the striatum (`SPATIAL_READ=CA3`) is worse: 0.63 / 0.31 / 0.39.

## Still open: CA1
CA1's soma is set by inhibition, not input: excitation averages 0.015, PV inhibition 1.05 and swings by 0.46 from
step to step (its learned inhibition weight is ~4.2). Tried without success (≤ 1 place-like cell): CA3 → CA1 30-60×,
grid → CA1 tuft 60×, CA1 inhibition weight lowered (0.5, 0) or frozen, CA1 learning frozen, and these combined.
Fixing CA1 means rebalancing its excitation and inhibition, which the memory tests depend on: a separate job.

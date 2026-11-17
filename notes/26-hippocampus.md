# 26 — Fixing the hippocampus

Until now the hippocampal loop contributed nothing: after one presentation the memory score was ≈ 0 with or
without it, and silencing it at recall changed nothing (notes/23, 25). Diagnostic: `prototype/brain/run_hippocampus.py`
measures every station of the loop (cortex B → EC → DG → CA3 → CA1 → subiculum → EC L5 → cortex) for
activity, pattern overlap (separation) and cue-to-pattern completion, in a learned brain and an untrained copy.
One run on a 7k-neuron brain takes ~45 s. From now on: single test runs by default (user, 2026-10-28).

## What was wrong (found in order)
1. **One-shot (BTSP) learning in CA1 wrote into every cell.** Every CA1 cell potentiated every recently active
   input at the dopamine moment, so CA1 activity quadrupled and all selectivity vanished (completion 0.24 → −0.01).
   In biology only the few cells with a dendritic plateau do this.
2. **"Winners take all" let everyone win.** The rule cut on the cell's *output*; CA3's outputs were saturated at
   1.0, so every saturated cell tied with the winners (CA3 ran at 27% active under a "5%" rule).
3. **The dentate gyrus was silent.** After fixing (2), the winners were picked correctly but still sat below their
   own threshold, so DG (the pattern separator) output nothing.
4. **Every pattern picked the same cells.** The entorhinal input is nearly identical for every pattern (overlap
   0.99–1.00); the pattern lives in small deviations. Cells that respond strongly to *anything* won every time.
5. **CA3's recurrent connections dominated while learning**, so it settled into one shared state.
6. **The cortex flip-flops every step** (an old bug, also in the cortex-only brain): area B's and the entorhinal
   cortex's pattern-specific activity reversed sign every step (step-to-step correlation −0.99 in B L5). PV cells
   reacted instantly to the previous step's excitation and overshot. The 4-step readout averaged it away.
7. **Timing.** Each relay adds about a step. The learning signal (dopamine at step 2) came before the pattern
   reached the hippocampus (step ~5–7), and the readout (steps 4–7) ended before the hippocampus's answer could
   travel back to area A (~12 steps). The hippocampus learned the gap between patterns and was never heard.

## Fixes
| # | fix | where | biology |
|---|---|---|---|
| 1 | BTSP only in cells with a plateau (stand-in: the cell's own rate, kept sparse by winners-take-all) | `cells.SynapseBank.modulate(plateau=)`, `ExcitatoryGroup.plateau()` | Bittner 2017: plateau cells only |
| 2 | winners ranked by membrane drive, exactly k of them | `federation._update` | fast feedback inhibition |
| 3 | winners fire by how far they clear the bar set by the k-th cell | same | inhibition sets the bar, not a fixed threshold |
| 4 | cells compete on how unusual their drive is (z-score against their own running mean and swing) | same | fast adaptation, gain control |
| 5 | acetylcholine mutes CA3 recurrents (0.8 per unit ACh) and Schaffer collaterals (0.5) in transmission only; the synapses still learn | `Synaptic(ach_suppress=)`, `SynapseBank.transmit` | Hasselmo: encoding vs recall modes |
| 6 | PV response speed 1.0 → 0.5 | `celltypes.PV` | GABA-A inhibition decays over ~5–10 ms |
| 7a | DG, CA3, CA1 learn whatever arrives in encoding mode (ACh > 0.5), not only at dopamine moments | `rules.encoding_learning` | NMDA plasticity needs no reward |
| 7b | stage-2 tests show each pattern 12 steps and let a probe stand 16 steps (cortex-only tests unchanged) | `memory_task(hold=, recall=)` | hippocampal recall takes ~100s of ms |

The PV change was checked on 8 cortex-only brains: memory 0.41 ± 0.09 (old) vs 0.29 ± 0.14 (new), not a
significant difference. The cell test for PV fading was relaxed (gone within 10 steps instead of 5). All 36 cell
tests pass.

## Results, 7k-neuron brain, seed 0, one presentation
| station | overlap between patterns | cue → own pattern (learned / untrained) |
|---|---|---|
| DG | 0.44 | 0.19 / 0.08 |
| CA3 | 0.19 | 0.37 / 0.36 |
| CA1 | 0.28 | 0.14 / 0.11 |
| EC L5 (output) | 0.44 | 0.71 / 0.89 |
| area A (memory score) | | **0.55** / 0.13 |

Memory score after one presentation: **intact 0.55; entorhinal + hippocampus silenced at recall 0.37;
hippocampus only (DG, CA3, CA1, subiculum) silenced 0.48.** Before: ≈ 0 in every condition.
- Pattern separation now works (DG, CA3, CA1 overlaps 0.2–0.4 instead of ~1).
- **CA3 learns proper memories:** after one presentation and sleep, the connections among a pattern's CA3 cells
  are +0.3 to +0.8 stronger, others −0.1; patterns share 10% of their CA3 cells.
- **But CA3 doesn't yet complete them at this size.** Each memory is only 12 CA3 cells, each hearing from ~4 of the
  others; too few to pull a fragment back into the whole. Real CA3 cells receive thousands of recurrent synapses.
- Heterosynaptic balancing erodes earlier patterns while later ones are learned (pattern 0's synapses lose most of
  their day gain by evening). Watch this.
- The subiculum carries almost nothing (completion ≈ 0); EC L5's output is mostly the cortex relayed through EC L3.

## Results, fly-sized brain (139,712 neurons, GPU, seed 0, 7.4 min)
| station | overlap | cue → own pattern (learned / untrained) |
|---|---|---|
| DG | 0.62 | 0.12 / 0.35 |
| CA3 | 0.25 | 0.48 / 0.74 |
| CA1 | 0.31 | 0.19 / −0.03 |
| subiculum | 0.07 | 0.06 / −0.02 |
| EC L5 (output) | 0.31 | 0.17 / 0.13 |
| area A | | 0.47 / 0.17 |

Memory score after one presentation: **intact 0.56** (the 16-seed fly run before these fixes: 0.04 ± 0.02);
entorhinal + hippocampus silenced at recall 0.46; **hippocampus only silenced 0.63**.
- One-shot memory is much better, but the credit goes to the cortex–entorhinal loop and the longer
  show/recall, not to the hippocampus proper: silencing DG–subiculum doesn't hurt (one seed).
- At fly size, learning made CA3's completion *worse* (0.74 → 0.48) and DG's too. The bigger CA3 didn't rescue it.
- The output route is the bottleneck: CA1 learned something (−0.03 → 0.19), but the subiculum passes almost
  nothing on, and EC L5 carries little memory (0.17).

## Next
1. Why learning hurts CA3 completion at fly size: heterosynaptic erosion, or the z-score competition adapting
   away the learned pattern during the 16-step recall.
2. The output route CA1 → subiculum → EC L5 → cortex: test CA1 → EC L5 directly (HC-13 is only 0.2 of EC L5's
   basal input), and let EC L5 learn the CA1 → cortex mapping during encoding (it doesn't learn in encoding mode).
3. Then confirm with a few seeds.

## ⚠️ The memory test was measuring timing, not memory (2026-10-28)
Following the hippocampus down to its input showed that responses beyond the thalamus are not reproducible.
**Test-retest** (same pattern shown twice, ~130 steps apart; own − other match; 1 = identical, 0 = unrelated):
| area | small brain (7k): rate / retest | fly brain (140k): rate / retest |
|---|---|---|
| thalamus relay | 0.22 / **1.17** | 0.18 / **1.15** |
| A L4 (cortex input layer) | **0.009** / 0.52 | **0.003** / 0.20 |
| A L2/3 (where memory is scored) | 0.25 / **−0.02** | 0.15 / **0.05** |
| B L2/3 | 0.003 / 0.25 | 0.06 / −0.15 |
| EC stellate | 0.06 / −0.25 | 0.07 / 0.22 |
| DG | 0.02 / −0.10 | 0.04 / −0.02 |
| CA3 | 0.07 / −0.01 | 0.07 / 0.14 |
Freezing thresholds and inhibitory plasticity during the test doesn't fix it. The cortex's input layer is nearly
silent, and area A's response to a pattern is not reproducible at all.

**The memory score is a timing artifact.** memory_task shows each full pattern and then its cue right after it,
and calls a match "memory". A brain whose state drifts slowly matches things shown close together in time,
whatever they are. Control (`scratchpad order.py`, 8 cortex-only brains each, 7k-neuron size, 6 presentations):
| | interleaved (as memory_task) | full patterns first, cues later in reversed order |
|---|---|---|
| learned | 0.29 ± 0.14 | −0.07 ± 0.05 |
| **untrained** | **0.35 ± 0.06** | 0.02 ± 0.02 |
An untrained brain scores as well as a trained one, and scrambling the order removes the score for both.
**Every brain-level memory result so far (notes 19, 20, 23, 25, and the one-shot scores above) is mostly this
artifact.** Single-neuron and cell-type results (notes 13–18, 22, 24) are not affected: they don't use this test.
The huge brain-to-brain variance (−0.15..1.04) also fits: it measures how each brain drifts, not what it learned.

## Next (replaces the list above)
1. Fix the memory test: cues in a scrambled order after all full patterns, and always an untrained control.
2. Fix the cortex input: why L4 is nearly silent and A L2/3 unreliable (calibration targets, thalamic drive at
   the official synapse scale, the always-on homeostasis). Target: retest ≥ 0.5 in A and B.
3. Then the hippocampus again, with the diagnostic in run_hippocampus.py (its fixes above stay; they were real bugs).

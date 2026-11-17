# 20: Which pathway carries recall?

Follow-up to note 19, where removing the apical feedback *from birth* didn't hurt recall. Here, lesions are
applied **after normal development**, like acute lesions in experiments, either during learning and recall
or at recall only. Waves are off: they're dropped from the design (note 19: no benefit, possibly an effect
we're reading backwards).

Code: `prototype/brain/run_recall_routes.py` · Results: `prototype/brain/results_recall_routes.txt` (6 seeds, ~3 min)

| Lesion | Fill-in of the unseen half (± s.e.) |
|---|---|
| intact | **0.42 ± 0.08** |
| no apical feedback (B → A apical tufts) | 0.27 ± 0.12 |
| no top-down drive to VIP (B → A disinhibition) | 0.35 ± 0.06 |
| no B → A at all | 0.23 ± 0.10 |
| area B silenced during learning and recall | 0.30 ± 0.09 |
| area B silenced at recall only | **0.15 ± 0.12** |
| area A's local synapses can't learn | 0.15 ± 0.05 |
| long-range synapses can't learn | **0.09 ± 0.05** |

## What it means
1. **Recall is distributed.** About half comes back from the higher area (B → A) and the rest from area A's
   own learned local connections. Neither route alone gives full recall.
2. **Both top-down routes contribute.** Apical feedback (−0.15 when removed) is a bit stronger than the
   disinhibitory route through VIP cells (−0.07). Together −0.19.
3. **Learning has to happen in both places, and the long-range synapses matter most.** Freezing long-range
   plasticity nearly abolishes recall (0.09). The *learned* long-range connections (A → B and back) are what
   bind the two halves of a pattern.
4. ~~The brain compensates for lesions it learns with.~~ **Not supported on rerun (see below).** The first
   run showed silencing B only at recall hurting more (0.15) than silencing it all along (0.30). A rerun after
   the cell-type changes of note 21 gave 0.29 vs 0.32, so the gap was within noise. What still stands: removing
   feedback *from birth* (note 19) hid its role, while removing it after development (here) shows it. So
   lesion timing matters at least for development, which matches real neuroscience.

## Consequences for the design
- Keep long-range connections plastic, with the same v2 rules as local ones.
- Both feedback routes (apical and VIP disinhibition) stay in the regional model.
- When we test what a region does, use **acute** lesions; chronic ones hide its role.

## Rerun (after note 21's cell changes; `prototype/brain/results_recall_routes_rerun.txt`)
| Lesion | First run | Rerun |
|---|---|---|
| intact | 0.42 ± 0.08 | 0.48 ± 0.12 |
| no apical feedback | 0.27 ± 0.12 | 0.17 ± 0.12 |
| no VIP top-down | 0.35 ± 0.06 | 0.30 ± 0.10 |
| no B → A at all | 0.23 ± 0.10 | 0.25 ± 0.06 |
| B silenced throughout | 0.30 ± 0.09 | 0.32 ± 0.10 |
| B silenced at recall only | 0.15 ± 0.12 | **0.29 ± 0.10** |
| A local can't learn | 0.15 ± 0.05 | 0.10 ± 0.06 |
| long-range can't learn | 0.09 ± 0.05 | 0.08 ± 0.05 |

Points 1–3 hold in both runs. Point 4 doesn't. With 6 seeds, effects smaller than ~0.15 aren't reliable;
more seeds are needed before drawing finer conclusions.

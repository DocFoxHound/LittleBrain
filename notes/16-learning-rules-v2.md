# 16: Learning rules v2

The six experiments (note 15) found the learning rules, not the storage, were the real limit.
v2 is in `prototype/neuron.py` as `WORKHORSE_V2` (v1 stays as `WORKHORSE` for comparison).
Tests: `python3 prototype/test_learning_rules.py` (about 6 s).

## What changed

| Problem (found by) | v1 | v2 |
|---|---|---|
| **Dose blindness**: 7 bad trials and 40 consolidated about the same (rolling-local-sleep) | fast weight capped at ±1 | fast weight may accumulate to ±2 ("how much was learned"); the *effective* weight is still capped at ±1; the night writes the full amount (`consolidation_rate` 1.0) |
| **Learning without firing**: every synapse with active input learned, fired or not (genome-diff) | eligibility = input × (0.5 + rate) | eligibility = input × (0.1 + rate): a cell learns mostly when it fires; a small floor allows subthreshold learning |
| **Rewiring too eager**: about 30% of slots rewired in 3 days (genome-diff) | any weak, unused, untagged synapse rewired every night | only an **under-active** neuron rewires (need = how far below its target rate), at most 3% of slots per night, scaled by its plasticity (fewer as it ages) |
| **Tags clutter storage** (genome-diff) | tags fade by half each night | tags last **until sleep only** (biological tags last hours), so they can live in a nightly-cleared side list |
| **Fading untagged memory erased the old self** (genome-diff) | — | kept as a rule: **untagged ≠ unimportant.** Nothing fades just for being untagged. |

## Results (16 seeds)

| Test | v1 | v2 |
|---|---|---|
| Story: day-6 recovery / relearn trials | 0.13 / 2 | **0.13 / 3** (same old-self behavior) |
| Dose: surface weight after 5 → 80 bad trials in one day (starts ≈ +1) | 0.15 → 0.00 (flat after ~10) | **0.25 → −0.38** (keeps growing with experience) |
| Spacing: recovery, 80 bad trials over 3 nights vs 2 | 0.23× | **0.65×** |
| Sparsity: synapses changed by experience (2% inputs active, top 10% of neurons fire) | 9.8%, the same in neurons that never fired | **1.9%**: 3.4% in neurons that fired, 1.7% in those that didn't |
| Rewired slots over 3 days | 29.8% | **5.2%** |

## Things learned along the way
- **The spacing effect is real, so it shouldn't be "fixed."** The same experience spread over more sleeps
  consolidates more deeply in humans. v1's problem was that the effect was *overwhelming* (0.23×), because of
  dose blindness. v2 keeps a moderate spacing effect (0.65×).
- **Firing-gated learning only creates sparsity with competition.** If every neuron sees the same input, every
  neuron fires. Sparsity comes from **population inhibition** (only the strongest few fire), so inhibitory
  neurons are essential, not optional. The test uses a simple "top 10% fire" rule as a stand-in.
- **A fully firing-gated rule (floor 0) is sparser (0.3% changed, 0% in silent neurons) but too lenient**:
  once the cell stops responding during a hard day, it stops learning, so hard times barely consolidate and
  the old self returns too strongly (0.78). The 0.1 floor is the compromise. Worth revisiting once real
  inhibitory neurons exist.
- **Measure learning against a twin.** Sleep alone moves every synapse's value a little (the genome-diff
  finding), so "changed since birth" overcounts. The test compares each neuron with a twin that slept the same
  nights with no experience.

## Packed version updated (2026-10-28)
`packed_neuron.py` now runs v2 too: the fast field covers ±2, tags clear at sleep, each neuron keeps an
average firing rate, and rewiring is need-gated.
- **The eligibility history had to grow from 4 to 8 steps.** Under v2, learning depends more on when the
  neuron fired, and 4 steps made the old self recover too strongly (0.21 vs 0.12, 32 seeds).
- With 8 steps: day-6 recovery 0.146 ± 0.016 (packed) vs 0.123 ± 0.008 (reference), relearn 3 vs 3.
  **1,051 bytes per neuron.**
- The experiment kernels in `prototype/experiments/` still use v1; they're research snapshots, and the
  prototype will be rebuilt from them.

# 25: The brain on lean neurons (port, roadmap step 7)

## What changed
- **`cells.LeanBank`:** the brain's synapse banks stored lean (notes/24).
  - Long-term state: the seed default evolved by one shared matrix, plus a small integer difference per level,
    at each type's precision (16 or 32 levels).
  - Day buffer: fast weight, eligibility, input trace.
  - Tags per branch; gentle surface-only sleep reset.
  - Wiring is regenerable from the brain's seed and pathway table, so it isn't stored.
- **How to switch it on:** `LEAN=1`, or `SUITE=low|standard|biological` (sizes each type from biology; implies lean).
  `SCALE=1` caps synapses at ~4 contacts per partner within reach.
- **Size normalization:** with more (or fewer) synapses than a type's reference layout, each synapse counts
  proportionally less (more), so thresholds mean the same at every size.
- **All 36 cell-type tests pass** with float and lean storage, and under the standard and low suites. There was one
  test-harness fix: test wiring now keeps each branch's own run of inputs at any size.
  - The exception is "clustered plasticity" at large sizes: with 128 random inputs per branch, every branch looks
    statistically the same, so cooperativity has nothing to amplify. It needs seeded choice in the brain (not yet ported).
- **Intercalated, CeL SST (on) and DCN projection cells now learn** (user decision, biology). ITC and CeL SST are a
  new family, `learning_inh`: GABA output, learnable inputs.

## Results (`prototype/brain/results_lean_port.txt`, `results_lean_ablation.txt`)
**Does lean storage change behavior?** No.
| stage 1, 6 presentations, 16 seeds | fill-in ± s.e. | range |
|---|---|---|
| float (original) | 0.53 ± 0.08 | −0.15 .. 1.04 |
| lean, all features | 0.57 ± 0.06 | 0.02 .. 0.96 |
| lean, storage only (no tags/reset change, 255 levels) | 0.57 ± 0.06 | 0.11 .. 0.97 |
A first run with 4 seeds suggested lean lost recall (0.70 → 0.44). With 16 seeds the difference is gone: it was
noise.

**Important finding: the memory score varies hugely from brain to brain.** Different random wiring of the *same*
design scores anywhere from −0.15 to 1.04. Earlier conclusions drawn from 4–8 seeds (notes 20, 23) may be shakier
than they looked. From now on, use ≥16 seeds for brain-level comparisons, and look into *why* some brains fail (E/I
balance near a tipping point, see note 23).

**Size and cost** (4 seeds, fixed reference wiring, before the scaling rule):
| brain | stage1 ×6 | stage2 ×1 | stage2 ×6 | synapses | float MB | packed MB | lean MB | min/run |
|---|---|---|---|---|---|---|---|---|
| float | 0.70 ± 0.06 | 0.01 | 0.73 | 0.67M | 25 | 2.7 | — | 1.6 |
| lean | 0.44 ± 0.15 | 0.07 | 0.54 | 0.67M | 25* | 2.7 | 0.63 | 2.0 |
| low suite | 0.47 ± 0.06 | 0.08 | 0.41 | 0.90M | 33* | 3.6 | 0.96 | 2.1 |
| standard suite | 0.64 ± 0.13 | 0.10 | 0.66 | 14.4M | 534* | 58 | 11.4 | 33.6 |
\* the working copy in RAM is still float; "lean MB" is the long-term storage.
- **Lean long-term storage is ~40× smaller than float and ~4× smaller than packed**: about 0.6–0.8 bytes per synapse.
- The 4-seed differences between suites are within the noise found above.
- The hippocampus still doesn't help one-shot recall (stage 2 ×1 ≈ 0 for every brain).
- The standard suite's 14.4M synapses were mostly duplicate contacts. That is why the Biological Synapse Count
  Scale (notes/24) is now the official testing scale.

## Next
1. Move the brain engine to the GPU (fly-scale tests: ~140k neurons, ~75M synapses).
2. Rerun at the official scale with ≥16 seeds, plus the single-neuron tests with no duplicate contacts.
3. Then the neuron fixes and the hippocampus.

## Fixes before fly-level testing (2026-10-28)
1. **The official scale is the default.** `Brain()` uses the Biological Synapse Count Scale (sized by the
   brain's own neuron count) plus the partner cap. `SUITE=none` gives the old reference layout.
2. **Seeded-choice turnover is in the brain** (`cells.LeanBank.sleep/rewire`, `federation.Brain._seeded_choice`).
   - Each night every learning cell replaces its weakest 3% of slots (adult spine turnover).
   - Exempt: synapses that changed meaningfully today (|fast| ≥ 0.05 on a tagged branch) and newborn
     synapses in their 3-night grace.
   - Candidates: 64 offered by the seed, drawn from the synapse's own pathway with the wiring's distance
     rule. The winner is the one that fired with the branch's strongest input when it mattered (a day log of
     dopamine moments, max 512).
   - Storage: 10 bits per rewired synapse.
   - Bug found on the way: heterosynaptic plasticity nudges *every* synapse, so "changed today" needed a
     threshold, or nothing was ever eligible for turnover.
3. **The 9 remaining cell types are defined** (98 total): pontine relay, lateral habenula, RMTg GABA, PAG freezing,
   PVN MC4R, VA/VL motor thalamus, MD thalamus, EC L5a and L5b. All follow the same scale.
4. **Brains can be built at any size** (`brainspec.sized(spec, n)`); wiring is computed in chunks.
5. **Regression:** all cell tests pass in float, lean and biological modes. The known exception is
   clustered plasticity in a *standalone* cell at full human size, which has no brain-level seeded choice.

## Fly-sized brain on this machine (stage-2 layout, `sized(stage2(), 140_000)`)
- 139,712 neurons, 72M synapses (L2/3: 1,156 per cell; CA3: 576). About 4 GB RAM.
- CPU: ~1.1 steps/s; one night (sleep + rewiring) takes ~6 min. That is ~1.5 h per brain per test, or ~8 h for
  16 brains × 3 conditions using 12 processes.

## Single-neuron tests with no duplicate contacts (fly and 1M synapse counts)
Capacity and precision (`results_fly_capacity.txt`; 32,768 inputs; AUC after / 6 days later; bytes at P=128):
| synapses | levels | P=32 | P=128 | P=512 | bytes |
|---|---|---|---|---|---|
| 1,152 (fly L2/3) | 16 | 1.00/0.97 | 0.80/0.76 | 0.60/0.59 | 454 |
| 1,152 | 32 | 1.00/1.00 | 0.92/0.89 | 0.66/0.64 | 1,149 |
| 2,304 (1M L2/3) | 16 | 1.00/0.99 | 0.87/0.82 | 0.63/0.62 | 884 |
| 2,304 | 32 | 1.00/1.00 | 0.97/0.95 | 0.71/0.70 | 2,292 |
| 4,096 | 16 | 1.00/1.00 | 0.94/0.90 | 0.67/0.64 | 1,582 |
| 4,096 | 32 | 1.00/1.00 | 0.99/0.98 | 0.76/0.74 | 4,011 |
- **Duplicates did not skew the earlier conclusions.** 4096 × 16 at P=128: 0.94 without duplicates vs 0.93 with.
- **Precision matters more in smaller cells.** At fly size, 32 levels beats 16 by +0.12 at P=128 (0.92 vs 0.80).
  The high-memory types already get 32 levels.

Dendrites and seeded choice at fly size (`results_fly_dendrites.txt`; 1,152 synapses, 8,192 inputs, parts world):
- **Same pattern as at 4096, but smaller gains.**
  - Co-firing clustering still hurts (P=128: 0.66 vs 0.75 for random rewiring).
  - Value-guided choice matches random rewiring at P=128 (0.74–0.76) and helps a little at P=512 (0.62–0.65 vs 0.61);
    in the random world, 0.66 vs 0.62.
  - Branch spikes are rare and selective (13% good vs 4% bad).
- Storage cost: +50–80%.

## GPU port with a CPU/GPU switch (`prototype/brain/backend.py`)
- **`DEVICE=cpu|gpu`.** GPU runs need the venv Python (`.venv/bin/python`, which has CuPy).
- **How it works.** A brain is built and wired on the CPU, then `Brain.to("gpu")` moves every array. The code stays
  numpy; CuPy arrays dispatch numpy functions to the GPU. Explicit changes were needed only for run-time array
  creation, random numbers (stochastic rounding), host-side small matrices (cascade flow, M), and row-wise sampling.
  Row-wise sampling is now one vectorized `searchsorted` (`_rowwise_search`), which also sped up wiring on the CPU.
- **`Brain.save(path)` / `Brain.load(path, device)`.** The whole brain is saved as a compressed host copy and can be
  loaded on either device. Tested both ways: saved on the CPU, run on the GPU, and the reverse.
- **Same brain, either device, statistically the same; not bit-identical** (summation order, random streams).
- **Speed:**
| | CPU | GPU |
|---|---|---|
| 7k-neuron brain, prepare (calibrate + develop) | 31 s | 87 s (slower: thousands of tiny GPU launches) |
| fly brain (140k neurons, 72M synapses), steps/s | 1.1 | **17** |
| fly brain, one night (sleep + rewiring) | 343 s | **14 s** |
| fly brain memory | 4 GB RAM | 2.7 GB VRAM (5.2 GB peak) |
- **Rule of thumb:** small brains on the CPU (many in parallel), fly-sized and larger on the GPU (one at a time).

## Fly-level brains at the official scale (2026-10-28)
`prototype/brain/run_fly.py`; Biological Synapse Count Scale + partner cap, lean neurons, seeded-choice turnover,
16 seeds per size. Results: `results_fly_140000.jsonl` (GPU), `results_fly_7104.jsonl` (CPU).
| brain | neurons | synapses | per neuron | stage1 ×6 | stage2 ×1 | stage2 ×6 | lean MB | min/seed |
|---|---|---|---|---|---|---|---|---|
| small | 6,848 | 0.9M | 135 | 0.36 ± 0.06 (−0.02..0.74) | −0.02 ± 0.03 | 0.43 ± 0.06 (0.02..0.85) | 0.8 | 1.7 (CPU, 12 parallel) |
| fly | 139,712 | 72.1M | 516 | **0.63 ± 0.07** (0.05..1.10) | 0.04 ± 0.02 | **0.59 ± 0.06** (0.07..1.00) | 58 | 8.0 (GPU) |
- **The bigger brain remembers better.** Stage 1: +0.28 (permutation p = 0.006). Stage 2 ×6: +0.16 (p = 0.07).
- **The scale fix cost the small brain.** The same 7k-neuron brain with the old fixed sizes scored 0.57; at the
  official scale (135 synapses per neuron) it scores 0.36. At fly size it is back to 0.63.
- **Failures are rarer but not gone.** Brains below 0.2: 2 of 16 at fly size vs 5 of 16 small (stage 1).
  The same seed's stage-1 and stage-2 brains don't fail together (r = 0.09), so it is the particular wiring, not the
  seed's "luck" in general.
- **The hippocampus still doesn't help one-shot recall** (stage 2 ×1 ≈ 0 at both sizes).
- Storage: 0.81 bytes per synapse long-term (lean).

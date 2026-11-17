# 24: Lean neuron plan (tests to run before wiring the rest of the brain)

Goal: make each neuron as cheap as physically possible *so it can be more complex*: realistic
synapse counts and dendrites, while still fitting a respectably sized brain on 8 GB VRAM / 62 GB RAM.

## Decisions so far
- **Seed + changes storage.** Default synapses are regenerated from a seed (0 bytes). Only synapses
  experience changed are stored: a 1-bit "changed?" map plus their state.
- **Day buffer.** The day's new changes go to a side buffer that the neuron *reads live while awake*
  (it is part of today's behavior, not just storage). During sleep the buffer is sorted: important
  changes merge into compact long-term storage; the rest fade back to default.
- A trained neuron can't be collapsed back into a new seed. Most trained states have no short seed
  (counting argument), so the learned part is the one thing that can't be compressed away.

## Tests to run (user's list)
1. **Synapse precision:** 64 (baseline), 32 (5 bits, about biology's ~26 sizes), 16 (4 bits), and 3 levels (ternary).
   Note: 4 bits = 16 levels; 24 or 26 levels need 5 bits. Rerun the existing single-neuron and cell-type tests
   and check whether behavior holds.
2. **Synapse count:** 256 (now), 4096, 8192. Powers of two fit index bits (13 bits address up to 8192),
   GPU warps and 64-bit bitmap words. Human cortical average is about 7,000; it varies by type
   (cerebellar granule ~4, Purkinje ~200,000).
3. **Smart dendrites:** branch-level local spikes (branch ≈ soft AND, soma ≈ OR of branches). Measure
   patterns stored per neuron with vs without branch nonlinearity, at each synapse count.
4. **Seed + changes with day buffer:** measure the fraction of synapses changed over days, size per
   neuron, speed cost of regenerating from the seed, and sleep-time merge.

## Estimates (bits per synapse: 64 levels = 32, 32 levels = 28, ternary = 16)
| 7,000 synapses | in full | seed + changes 10% | 30% |
|---|---|---|---|
| 64 levels | 27 KB | 3.6 KB | 9.1 KB |
| 32 levels | 24 KB | 3.2 KB | 8.0 KB |
| ternary | 14 KB | 2.2 KB | 5.0 KB |

## The "bad day" effect (today's events colouring today, fading overnight)
Already partly built:
- **Fast weights:** decay unless consolidated.
- **Tags and eligibility.**
- **Hippocampal fast learning.**
- **Slow endocrine cells.**

Still missing:
- **Tonic mood:** slow neuromodulator and stress levels that rise from one event and stay up all day.
  Comes in stage 3 (neuromodulators) and stage 5 (amygdala).
- **Sleep partial reset:** sleep should partly reset mood and the emotional charge of memories, the
  "overnight therapy" idea about REM.

## Decisions (2026-10-28)
- Test 16 and 32 levels; lean toward 16 if the loss is small.
- Test 4096 and 8192 synapses; lean toward 4096 (much cheaper, still 16× today).
- Build the lean neuron (seed + changes, day buffer) *before* running the 8192 tests. Don't bolt them onto the current model.

## Shrinking it further (ideas to test)
Biggest space user: long-term learned synapse state (the 3 cascade levels + the "changed?" map).
The day buffer is small (sparse activity), and running awake neurons in VRAM costs speed, not storage.
1. Fast weight, tag and in-use flag are day-only: they live in the day buffer, not long-term storage.
2. "Settled" synapses (all cascade levels agree) store one value; only synapses still in flux store all 3.
3. Deeper cascade levels coarser, or shared per branch (late-phase LTP is partly branch-level, which fits the two-eras-on-two-branches result).
4. Rewiring generation stored only for rewired synapses (separate small list).
5. Cold storage: gap-coded index lists instead of a bitmap, plus zstd/lz4.
6. The master lever: keep learning sparse (firing-gated learning, sleep resets small changes).

## Test priority for the shrinking ideas (user, 2026-10-28)
- **Must test:** 1 (day-only parts out of long-term storage), 2 (settled synapses store one value),
  5 (gap-coded change lists + zstd/lz4 in cold storage).
- **Test, then decide on cost:** 3 (coarser or branch-shared deep levels), 4 (rewiring tag only for
  rewired synapses), 6 (sleep resets tiny changes to the seed). Report each one's space saving vs its
  effect on learning, recall and era restoration, so the user can choose.

## More "cake and eat it" ideas (brainstorm, not yet prioritized)
A. **"Chemical" state:** move per-synapse bookkeeping into shared, slow fields (per branch, per neuron,
   or per tile region) like neuromodulator volume transmission. Candidates: tags (per branch, as in
   synaptic tagging & capture), eligibility (recomputed from per-neuron spike histories at reward time).
   Slow fields also update rarely, which saves compute.
B. **Learned templates ("seeds that learn"):** during sleep, find what many neurons of one type have
   learned in common, fold it into a shared template, and store each neuron as template + small
   difference. Sleep as compression; resembles schema formation. Commit only if behavior is unchanged.
C. **Level of detail (as in video games):** full detail for regions in use; idle regions run as a few-number
   population stand-in or go dormant to RAM/SSD. Risk: learning in coarse mode.
D. **Multi-rate clocks:** slow regions (hypothalamus, endocrine, mood) update every N steps.
E. **Precision fades with age:** new memories high precision, old ones coarser (see experiments/fading-precision).

## Decisions on the brainstorm (user, 2026-10-28)
- **A (chemical / shared state): build now** as part of the lean neuron: tags per branch, eligibility
  recomputed from per-neuron spike histories at reward time. Test against per-synapse versions.
- **B (learned templates in sleep): yes, try it.** Probably lands with step 8 (sleep), but the
  storage format should allow template + difference from the start.
- **C (level of detail): try it.**
- **D (multi-rate clocks): do it regardless** (user prefers it on biological grounds).
- **E (fading by age): replaced** by the user's idea, *fade by importance and redundancy*:
  - drop what is unimportant (no emotional tag, never reused);
  - merge repeated similar memories into one gist, tracking how many times it happened, when, and
    the small differences between occurrences (episodic → semantic memory, "semanticization");
  - keep distinctive or emotionally tagged episodes in full detail.
  Core personality memories recur often, so this protects them rather than fading them.
  Open design question: the repetition count and timestamps could live implicitly in synaptic
  strength (pure neural) or in a compact hippocampal "index" of episodes (hippocampal indexing
  theory, Teyler & DiScenna 1986). **User leans toward the hippocampal index.**

## Results, round 1 (256 synapses; `prototype/lean_neuron.py`, `prototype/test_lean.py`, `prototype/results_lean.txt`)
The lean neuron stores each synapse as a difference from its seed default. One shared 3 × 3 matrix tracks
every global operation (cascade flow, sleep downscaling), so untouched synapses stay exactly zero.
At 64 levels it behaves like the packed neuron (story, dose).

**Precision** (8 seeds, v2 rules):
| | day-6 recovery | relearn | dose N=20 (lower = learned) |
|---|---|---|---|
| packed reference | 0.15 | 4 | 0.05 |
| 64 levels | 0.13 | 3 | 0.05 |
| 32 levels | 0.12 | 3 | 0.05 |
| 16 levels | 0.13 | 3 | 0.05 |
| 3 levels | 0.42 | 2 | 0.39 ✗ |
- 16 and 32 levels: no measurable loss. **16 levels looks fine** (at 256 synapses; recheck at 4096).
- Ternary: moderate experiences (20 hard trials) barely stick. Rejected as expected.

**Storage** (128 neurons, 12 days, sparse world; bytes per neuron; packed today = 1,033):
| | changed | +ideas 1,2,4,5 | buffer/day |
|---|---|---|---|
| 64 lv | 31% | 255 | 63 |
| 32 lv | 29% | 215 | 63 |
| 16 lv | 27% | 177 | 63 |
| 16 lv, chem tags | 12% | 71 | 49 |
| 16 lv, chem, +6 reset ≤ 1 step | 0.8% | 14 | 48 |

**Caveats found:**
1. **Rewiring dominates "changed" without chem:** 21.6% of synapses rewired in 12 days, about 80% of all
   changes. A rewired synapse is stored as a full difference because its fresh value doesn't match the
   evolved default.
   - Fix to try: a newborn synapse takes the seed's evolved default for its new generation, so it costs
     only a gen-list entry.
2. **Chem's storage win is mostly a side effect.** A per-branch "used" flag protects whole branches from
   pruning, so rewiring dropped to 0%.
   - Fix to try: a pruning rule that doesn't need a per-synapse used flag, or keep "used" per synapse
     (1 bit, day buffer only).
3. **Idea 6 (reset ≤ 1 step at 16 levels)** wipes 99% of differences. The story still works (relearn 4.5 vs 3),
   but day-6 recovery of the old self drops from 0.18 to 0.11. It costs some era restoration; the user decides.
4. Idea 2 (exact "settled") helps little: 3–26% of changed synapses. Allowing ±1 step would settle 13–85%,
   but that is lossy.
5. zlib is worse than plain gap coding at these sizes.

## Precision sweep (16 seeds, `prototype/results_lean_precision.txt`)
| levels (bits) | day-6 comeback [range] | dose N=20 (lower = stuck) |
|---|---|---|
| 64 (6) | 0.14 [0.09–0.23] | 0.05 |
| 32 (5) | 0.13 [0.08–0.21] | 0.05 |
| 16 (4) | 0.16 [0.08–0.30] | 0.05 |
| 8 (3) | 0.24 [0.06–0.77] | 0.08 |
| 5 (3) | 0.32 [0.07–0.72] | 0.14 |
| 3 (2) | 0.53 [0.20–0.99] | 0.34 |
Breaking point: between 16 and 8 levels. Below 16, the "comeback" *rises*, but for the wrong reason: the hard
times were never fully recorded (dose column), and results become erratic (wide ranges). 16 levels is the
lowest precision that matches full precision. (Signed values: "16 levels" is -7..7 in 4 bits.)

## Round 2 (fixes, idea 6, big neurons, dendrites, clocks, level of detail)
**Fixes.**
- **Rewired synapses:** a newborn synapse now takes the seed's evolved default for its new generation, so
  it costs only a gen-list entry.
- **Chem mode:** only *tags* are per branch. The used flag is per synapse again, but day-only. (Per-branch
  used had switched rewiring off entirely.)
- **Storage accounting:** rewired synapses are no longer charged for values. Bytes per neuron at 256
  synapses / 16 levels: 99 (chem off), 124 (chem on; branch tags consolidate a bit more), 1,033 packed.
  The day buffer is about 110–125 B.

**Idea 6, significance** (`results_lean_round2.txt`, 32 seeds, paired):
- **Reset ≤ 1 step everywhere:** comeback 0.18 → 0.12 (−0.066 ± 0.017, p < 0.001); relearn 3.2 → 4.3 trials
  (p < 0.001). Significant: it costs about a third of the old self. **Rejected.**
- **Gentle version, adopted:** reset only *surface-level* differences of ≤ 1 step where the deep levels hold
  nothing. Comeback 0.20 (+0.02, slightly better); relearn unchanged (p = 0.22); storage 124 → 99 B.

**Capacity** (`results_lean_capacity.txt`): P patterns, half good and half bad, 4 days of practice, then
6 days of rest. Scores are AUC (0.5 = chance, 1 = perfect), after practice / after rest.
| | P=32 | P=128 | P=512 | B/neuron at P=128 |
|---|---|---|---|---|
| 256 syn, 16 lv | 0.87/0.79 | 0.63/0.59 | 0.54/0.53 | 109 |
| 256 syn, 32 lv | 0.95/0.90 | 0.74/0.72 | 0.57/0.56 | 267 |
| 4096 syn, 16 lv | 1.00/1.00 | 0.93/0.89 | 0.67/0.65 | 1,574 |
| 4096 syn, 32 lv | 1.00/1.00 | 0.98/0.98 | 0.75/0.74 | 3,981 |
| 8192 syn, 16 lv | 1.00/1.00 | 0.97/0.96 | 0.71/0.69 | 3,127 |
| 8192 syn, 32 lv | 1.00/1.00 | 1.00/0.99 | 0.78/0.78 | 7,946 |
- **Size:** 256 → 4096 is a huge gain; 4096 → 8192 is a smaller one.
- **Precision matters under heavy load:** 32 levels beats 16 when a neuron is near capacity, which the
  one-memory story test couldn't show. But 32 levels changes about 2.5× more synapses, so it costs about
  2.5× the storage. Per byte the two are similar: 8192 × 16 lv (3.1 KB) ≈ 4096 × 32 lv (4.0 KB).
  The user chose 16 levels.
- **Smart dendrites did nothing:** linear and spiking dendrites gave identical results, and branch-gated
  learning hurt. Cause: random wiring spreads each pattern's inputs across all branches, so no branch
  ever reaches its spike threshold (max branch drive 1.85 vs threshold 3.2 at 4096). Dendrites need
  *clustering*: inputs that fire together must land on the same branch.
  - Next idea, "seeded choice": a rewiring synapse picks the best of 8 seed-generated candidate inputs
    (the one that fires with the branch's existing inputs) and stores only its 3-bit choice.

**Clocks** (`brain/results_clocks_lod.txt`): a slow hormone group updated every 10 steps follows the
every-step course (r = 0.987, 2.3% mean difference) with 10× fewer updates. **Adopted:** `Group(every=N)`.

**Level of detail** (stage 2, 3 seeds): regions go dormant when their incoming activity drops.
| | memory score | updates skipped |
|---|---|---|
| off | 0.76 ± 0.12 | 0% |
| frozen when dormant, 50% | 0.36 | 11% |
| silent when dormant, 80% | 0.00 | 48% (cascade: the whole brain went dark) |
- In a tightly coupled loop (thalamus ↔ cortex ↔ hippocampus) every region is in use. Dormancy by input
  level either replays stale activity or cascades into total silence.
- **Mechanism kept, off by default.** Revisit when regions with real idle periods exist (cerebellum,
  amygdala, hypothalamus), with a smarter wake rule or a cheap stand-in instead of silence.

## Capacity detail: storage at every load (`results_lean_capacity_storage.txt`)
Long-term bytes per neuron (all ideas). Packed format for comparison: 256 syn = 1,033 B, 4096 = 16.4 KB,
8192 = 32.8 KB. AUC after practice / 6 days later.
| synapses | levels | P=8 | P=32 | P=128 | P=512 |
|---|---|---|---|---|---|
| 256 | 16 | 0.95/0.85, 78 B | 0.87/0.79, 119 B | 0.63/0.59, 109 B | 0.54/0.53, 96 B |
| 256 | 32 | 1.00/0.96, 160 B | 0.95/0.90, 275 B | 0.74/0.72, 267 B | 0.57/0.56, 241 B |
| 4096 | 16 | 1.00/1.00, 1.1 KB | 1.00/1.00, 1.7 KB | 0.93/0.89, 1.6 KB | 0.67/0.65, 1.4 KB |
| 4096 | 32 | 1.00/1.00, 2.5 KB | 1.00/1.00, 4.2 KB | 0.98/0.98, 4.0 KB | 0.75/0.74, 3.6 KB |
| 8192 | 16 | 1.00/1.00, 2.2 KB | 1.00/1.00, 3.4 KB | 0.97/0.96, 3.1 KB | 0.71/0.69, 2.8 KB |
| 8192 | 32 | 1.00/1.00, 4.9 KB | 1.00/1.00, 8.4 KB | 1.00/0.99, 7.9 KB | 0.78/0.78, 7.3 KB |
Storage peaks around P=32 and falls slightly at heavier load: competing memories cancel each other
on shared synapses, leaving more of them at zero difference.

## Seeded-choice rewiring, attempt 1 (`results_dendrites.txt`): did not work yet
- At first no rewiring happened at all: need-gated rewiring never fires when homeostasis keeps every
  neuron at its target rate. Added `turnover=True`: constant spine turnover, where each night the weakest
  `rewire_budget` of slots are replaced, regardless of need (adult cortex replaces a few % of spines per day).
- Seeded choice (best of 8 seed-offered candidates by co-firing with the branch's strongest inputs):
  clustering rose only slightly. Most active synapses one branch gets from a memory: 12 → 15
  (4096 syn, 10%/night, 12 days). Branch spikes: 0%. AUC unchanged. Storage +20–25%.
- Adding a 3-night grace period for newborn synapses and anchoring the theme on the branch's single
  strongest input changed nothing.
- Likely causes:
  1. **Too few candidates.** A memory uses 5% of inputs, so on average only 0.4 of the 8 candidates
     belong to the branch's memory. There is usually nothing good to choose. (Try 64 candidates = 6 bits.)
  2. **The task has no structure to cluster on:** 128–512 unrelated random patterns. Real experience is
     correlated (objects, places).
  3. **New synapses start weak.** 15 weak inputs don't reach the branch threshold (3.2 at 4096 syn), so no
     spikes, so dendrites never matter, whatever the clustering.

## Seeded choice, attempt 2: a structured world and value-guided choice (`results_dendrites2.txt`)
- **New world** (`prototype/world.py`): objects built from 32 shared parts (3 parts + 30 details of their
  own each; features drop out 10% of the time, plus a little noise). Like animals built from ears, coats
  and calls. Half the objects are good, half bad, so good and bad things *share* parts.
- Setup: 4096 synapses, 16 levels, 64 candidates (6 bits), turnover 10%/night, 3-night grace, 12 days.

| (parts world) | AUC P=128 | AUC P=512 | cluster | branch spikes good/bad | bytes |
|---|---|---|---|---|---|
| random rewiring | 0.82 | 0.62 | 4.9 | 0% / 0% | 4.1 KB |
| seeded, all co-firing | 0.73 | 0.61 | 13–17 | 66% / 41% | 8.8 KB |
| **seeded, value-guided** | **0.81** | **0.66** | 6.1 | 29% / 8% | 7.4 KB |
| value-guided, strong newborn | 0.81 | **0.71** | 6.3 | 34% / 10% | 8.2 KB |
In the random world at P=512: value-guided 0.80 vs random rewiring 0.74.

**Findings:**
1. **Clustering works once the world has structure.** Seeded choice gathers a part's inputs onto one branch
   (about 14 vs 5 by chance), and branches finally spike.
2. **Clustering on raw co-firing hurts.** Parts are shared between good and bad things, so a branch learns
   "pointed ears" and spikes for friend and foe alike. Linear dendrites score the same, so the loss comes
   from *what* got clustered, not from the spikes.
3. **Value-guided choice wins.** Co-firing is counted only during plasticity, signed by dopamine. This
   matches biology: new spines grow near synapses that were just potentiated. Branch spikes become selective
   (29% of good objects vs 8% of bad), and capacity under heavy load improves: P=512, 0.62 → 0.66, or 0.71
   with stronger newborn synapses.
4. **Dendritic spikes themselves still add nothing measurable:** spiking and linear dendrites score the same.
   The gain comes from better *wiring choices*, not from branch nonlinearity.
5. **Cost:** +40–80% storage, because more synapses carry learned values, plus the 6-bit choice per rewired synapse.

**Adopted** for both suites: value-guided seeded choice, 64 candidates, turnover, 3-night grace.
Newborn strength stays at the default (0.1) until spikes vs linear is tested at strong newborn strength.

## Two suites (`prototype/brain/suites.py`; data `prototype/brain/synapse_budget.py`; research `notes/regions/synapse-budget.md`; table `notes/regions/suites-report.txt`)
- **Research:** biological synapse counts and memory role for all 89 types plus the 9 still to build.
  Confidence: 6 measured (●), 13 from reviews (◐), 79 estimates (○). Most counts are rodent.
- **Standard suite:** L2/3 IT = 4096 (16 levels is the baseline). Every other type scales with its biological
  count. The agent anchored 4096 to a 7–10k pyramid, which put human L2/3 (~30k) at 8192, so `suites.py`
  halves everything to keep L2/3 at 4096.
  - Sizes: Purkinje 32,768; CA1 8192; L2/3, L5 PT, CA3 and others 4096; most interneurons 512–2048;
    granule 4.
  - Precision: 32 levels for 10 high-memory types (L2/3 IT, young DG granule, CA3, CA1, BLA ±, D1/D2/striosome MSN,
    Purkinje); 16 levels for 29; fixed for 59 that don't learn.
- **Low-synapse suite (experimental):** the same types at 1/16 the synapses (L2/3 IT = 256, minimum 4).
- Both use the adopted lean settings: chem tags, gentle reset, turnover with value-guided seeded choice
  (64 candidates), 3-night grace.
- **Open questions:**
  - Three types are marked non-learning in the catalog, but biology says they learn: intercalated cells,
    CeL SST (on), DCN projection.
  - The two-brain comparison needs the brain engine (`federation.py`) ported from float synapse banks to lean
    populations. That is the next big engineering step.

## Third suite: the Biological Synapse Count Scale (optional)
- Each type gets the power of two nearest its biological upper synapse count, capped at 32,768.
  L2/3 IT ~30,000 → 32,768; CA1 → 32,768; L4 stellate → 8192.
- Purkinje is exempt from the cap: ~300,000–500,000 parallel-fiber synapses in humans → 524,288 (2¹⁹).
- `suites.synapses(type, "biological")`; full table in `notes/regions/suites-report.txt`.

## Scaling synapses to the size of the simulated brain (user's correction)
**Problem.**
- A real L2/3 cell's 30,000 synapses come from thousands of distinct partners out of 86 billion neurons.
- Our stage-2 brain has about 7,000 cells, so a 4096-synapse cell must contact each partner within reach
  dozens of times. The duplicate contacts add almost nothing, but they cost memory and compute.
- They also probably skewed earlier single-neuron results (only 2048 inputs for 4096–8192 synapses): duplicates
  make 8192 look barely better than 4096, and average away rounding noise, making 16 levels look better.

**Rule** (`federation.Brain(scale=True)` or `SCALE=1`):
- Synapses per cell = min(the suite's size for the type, CONTACTS × effective partners within reach), with
  CONTACTS = 4 (cortical pairs connect by ~1–8 contacts, typically ~4–5).
- Effective partners: the participation ratio (Σw)²/Σw² of the cell's sampling weights for each pathway, summed
  over the pathways into that compartment.
- One factor scales both compartments, so a cell keeps its basal/apical proportions. Capping them separately
  shifted cells toward apical feedback and broke the 20% long-range cap.

In stage 2 the reach allows about 120 (thalamus) to 4,000 (area B) synapses per cell. Totals: low 0.9M,
standard 7.8M (14.4M unscaled), biological 9.6M.

## The scale: synapses per cell vs. size of the simulated brain (Biological Synapse Count Scale)
Rule: synapses = min(type ceiling, 4 contacts × partners within reach). Partners within reach grow with the
number of simulated neurons N, if the map and connection spreads stay the same as cells are added.
- Typical reach in our wiring is ~5% of the brain, so synapses ≈ N / 5 for a typical cell.
- Exceptions: thalamus 0.4% (its input is a fixed 256-cell retina); area B up to 20%.

Generic ladder (typical cell, ~5% reach):
| simulated neurons | typical cell gets | milestone |
|---|---|---|
| ≤ ~20 | 4 | everything is 4 |
| ~1,300 | 256 | low-suite L2/3 size |
| ~20,000 | 4,096 | standard-suite L2/3 size |
| ~40,000 | 8,192 | L4, L6, thalamic relay at full biological size |
| ~80,000 | 16,384 | L5 IT, hippocampus, entorhinal at full size |
| ~160,000 | 32,768 | L2/3, L5 PT, CA1 at full size |
| ~13 million | 524,288 | Purkinje at full size* |
\* Purkinje: one contact per parallel fiber (not 4), drawn from granule cells (~80% of neurons in human
proportions), at ~5% reach: synapses ≈ 0.04 N.

Our stage-2 layout, per group (biological suite; from `_partner_caps`, extrapolated linearly from 7,104 cells):
| group | ceiling | 1k | 10k | 100k | 1M |
|---|---|---|---|---|---|
| TH.TC relay | 8,192 | 16 | 166 | 1,668 | 8,192 |
| A.L2/3 | 32,768 | 312 | 3,124 | 31,242 | 32,768 |
| B.L2/3 | 32,768 | 816 | 8,169 | 32,768 | 32,768 |
| CA3 | 16,384 | 362 | 3,625 | 16,384 | 16,384 |
| CA1 | 32,768 | 204 | 2,047 | 20,475 | 32,768 |
| DG granule | 16,384 | 79 | 798 | 7,980 | 16,384 |

Caveat: in real brains reach *shrinks* as brains grow (connections stay mostly local), which is why a human
L2/3 cell has ~30,000 synapses, not billions. Our linear extrapolation assumes the same map gets denser. A
larger simulated brain would also get more tiles or areas, so reach would grow more slowly than this.

## Correction: the Biological Synapse Count Scale follows brain size, calibrated on real animals (official testing scale)
The ladder above saturated far too early: it assumed a cell reaches ~5% of the brain, but a real neuron reaches a
minuscule fraction of 86 billion. Synapses per neuron grow with brain size much more slowly than neuron count:
| animal | neurons | synapses/neuron (typical) | fraction of human |
|---|---|---|---|
| worm | 302 | ~25 | 0.25% |
| fly | ~140,000 | ~360 | 3.6% |
| mouse | ~70 million | ~10,000 (cortical pyramidal) | 33% |
| human | ~86 billion | ~30,000 (L2/3) | 100% |
- **Rule (all types alike):** synapses = ceiling × human_fraction(N), interpolated log-log between these animals.
  Rounded to multiples of 32 (powers of two below 64), minimum 4. The partner cap stays as a second limit.
- **Check:** at mouse size the scale predicts Purkinje ~173,000; real mouse Purkinje cells have ~175,000.
- **Examples:**
| simulated neurons | L2/3 | CA3 | thalamic relay | Purkinje |
|---|---|---|---|---|
| 7,104 (stage 2 now) | 320 | 160 | 96 | 5,184 |
| 140,000 (fly) | 1,184 | 576 | 288 | 18,880 |
| 1,000,000 | 2,368 | 1,184 | 608 | 38,048 |
| 70 million (mouse) | 10,816 | 5,408 | 2,688 | 173,024 |
| 86 billion (human) | 32,768 | 16,384 | 8,192 | 524,288 |
- **Consequence:** our current brain is biologically worm-to-fly sized, and the low suite (L2/3 = 256) is close to
  what's biologically proportional for it.

**Hardware at the testing sizes** (stage-2 composition scaled up):
| | learnable synapses | CPU (numpy) today | lean working copy | packed (4 B) |
|---|---|---|---|---|
| fly, 140k | ~75M | ~1 step/s | ~1.8 GB | ~0.3 GB |
| 1M | ~1.1 billion | ~0.07 step/s | ~26 GB | ~4.4 GB |
The fly level needs the brain engine on the GPU, where the working copy fits in 8 GB. 1M needs packed GPU kernels
(prototyped in `experiments/`) plus RAM tiering.

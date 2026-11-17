# Fading precision: memories lose resolution, not existence

**Verdict: mixed, with one clear winner.** Freezing cold neurons *losslessly* into compressed blocks,
letting time pass for them as a single 3x3 matrix, and re-freezing them now and then during sleep
stores a long-idle neuron in **~200 bytes instead of 1,041 (5.2x)**, gives a cold neuron **zero
work per night**, and changes behavior **not at all** (population test identical to 3 decimals).
*Lossy* precision fading works too: ternary (2-bit) levels are behavior-safe if done right. But on
top of good lossless coding it saves only 0-15%, and every lossy variant nudges the balance between
the old self and the recent self. Recommendation: lossless + lazy + re-freeze; skip lossy fading.

Run everything: `.venv/bin/python prototype/experiments/fading-precision/run.py` (~13 min, ≤3 busy
threads, peak 0.3 GB RAM per process, CPU only; deletes its ~35 MB of snapshots at the end; `--keep`
keeps them).

## The idea

Active neurons need full detail. A neuron unused for many nights could be kept at lower
resolution, or compressed, and "sharpened" again only when needed, the way old memories get
vaguer but are still there. The danger is the deep cascade level (`c2`), which holds the
"old self" that notes/10 says must stay recoverable.

Biology: memory traces are not erased but become less accessible (silent engrams, Ryan et al.
2015); remote memories lose episodic detail but keep their gist (systems consolidation /
trace transformation); a real synapse holds only ~4.7 bits (Bartol et al. 2015).

## Scenario (stated on purpose: this needs locality)

With the prototype's wiring (every synapse picks from the same 64 inputs) every neuron gets input
every step and nothing is ever cold. So the simulated brain uses **local wiring**: 20 columns x
1,024 workhorse neurons (20,480 `PackedPopulation` neurons). Each column owns 64 inputs of a
1,280-input space, and its synapses' hashed targets stay inside that region. One column is
therefore exactly the demo neuron's world (8 of 64 inputs per pattern), and an idle column really
is idle (3-6% of all inputs active at a time). Life story over 16 days, demo-style days (40 trials,
or a 200-step rest when idle, then sleep):

| group | columns | story |
|---|---|---|
| A | 0-5 | pattern rewarded days 1-6, punished days 7-8 (hard times), idle 9-16 (cold, holds an old self) |
| B | 6-9 | one pattern rewarded, another punished, days 1-8; idle 9-16 (cold, stable memories) |
| C | 10-14 | idle (never used) days 1-8, a new life days 9-16 (hot) |
| D | 15-17 | idle until day 12, rewarded days 13-16 (hot, recent) |
| E | 18-19 | never used (spare reserve) |

Group A reproduces the demo at population scale: response 0.03 after hard times, then 0.06 / 0.14 /
0.17 on idle days 3 / 6 / 8 (demo: 0.07 / 0.15). **Design finding: cold storage only pays if
neurons that go quiet together are wired and stored together (columns -> blocks).**

## What I built

| file | what |
|---|---|
| `brainsim.py` | the 16-day population simulation (`LocalPopulation` = `PackedPopulation` with column-local hashed wiring); snapshots at day 8 and day 16 |
| `lossless.py` | 8 layouts x 9 codecs x 5 block sizes on the realistic cold words; entropy bounds |
| `fade.py` | the fading quantizer (per-level bits, per-branch scale, stochastic rounding, idempotent on its grid), fading policies, and `FadingNeuron`, a drop-in adapter for `run(...)` with **eager** (every write goes through the coarse grid) or **lazy** (freeze once, apply elapsed time exactly at thaw) timing |
| `behavior.py` | the demo story via `run(label, ntype, seed, make=FadingNeuron...)`, 32 seeds, plus controls (plain neuron under the same fading; a never-trained control pattern; per-level "contrast" = mean level value on A's synapses minus the rest) |
| `coldstore.py` | the real cold store: freeze (fade -> "symbols" layout -> zstd-3, one blob per 64 neurons), lazy nights (one 3x3 matrix per block), sleep-time re-freeze, thaw; speed/memory benchmarks; population behavior test (groups A+B through idle days 9-16, then A relearns) |
| `fringe.py` | synapse-state codebooks (VQ), cross-neuron dedup / prototype + residual, holographic superposition |
| `run.py` | runs it all; logs and JSON in `results/` |

## Results

### 1. What realistic cold neurons look like

After a few unused nights a cold synapse has `fast = 0`, `used = 0`, `tag = 0` (all cleared or decayed
by sleep), so only `gen` and the three cascade levels carry anything. And because the cascade flow
keeps pulling the levels towards each other, **long-cold synapses are simple**: across all 3.1
million cold synapses (12,288 neurons) only **549 distinct (c0, c1, c2) states occur** (of 250,047
possible). Joint entropy 5.6 bits for the triple, against 11.6 bits if the three levels are coded
separately. Order-0 entropy of a whole cold synapse: ~7.0 bits (vs 32 stored). Hot neurons: ~12 bits.

### 2. Lossless compression of cold blocks (cold set A+B+E, 64-neuron blocks, one thread)

Bytes per neuron include the 17 bytes of per-neuron state. Speeds are GB/s of uncompressed
neuron data, *including* the numpy layout transform (pure codec speed is higher), single thread,
noisy (other agents were running; the ratios are exact).

| layout | codec | B/neuron | ratio | compress GB/s | decompress GB/s |
|---|---|---|---|---|---|
| raw words | lz4 | 511 | 2.04x | 0.46 | 2.35 |
| raw words | zstd-3 | 316 | 3.29x | 0.26 | 1.03 |
| raw words | zstd-19 | 260 | 4.00x | 0.003 | 1.18 |
| byte planes | zstd-3 | 394 | 2.64x | 0.37 | 0.70 |
| bit planes (bitshuffle) | zstd-3 | 378 | 2.75x | 0.10 | 0.24 |
| field planes | zstd-3 | 338 | 3.08x | 0.21 | 0.56 |
| field planes, levels delta-coded (c2, c1-c2, c0-c1) | zstd-3 | 287 | 3.63x | 0.22 | 0.38 |
| field planes, neurons sorted (+permutation) | zstd-3 | 289 | 3.60x | 0.23 | 0.45 |
| **symbols** (per-block table of distinct (c0,c1,c2) + index per synapse) | **zstd-3** | **219** | **4.75x** | 0.12 | 0.57 |
| symbols | zstd-19 | 210 | 4.95x | 0.006 | 0.60 |
| symbols | lz4 | 350 | 2.97x | 0.15 | 0.82 |

- **Symbols + zstd-3 wins**: 4.75x at usable speed, even below the brain-wide order-0 entropy
  (7.0 bits per synapse = 241 B with the per-neuron state), because each block's table adapts to
  its own neurons. Delta coding
  the levels is second best. Byte- and bit-shuffling *hurt* (they split the correlated fields).
  Sorting neurons does nothing (hash wiring leaves no row-to-row structure).
- Raw lz4 decompresses 4x faster than symbols/zstd-3 but stores 2.3x more.
- Block size (symbols + zstd-3): 1 neuron 469 B, 4 neurons 310 B, 16 neurons 245 B, **64 neurons 219 B**, 1,024 neurons 223 B. 64 neurons (64 KB) is the sweet spot.
- Dictionaries (16 KB, trained on cold blocks) only help small blocks: 1 neuron 469 -> 351 B, 4 neurons 310 -> 254 B, 16 neurons 245 -> 223 B (= the 64-neuron size without one). Use one if you need fine-grained random access. A dictionary trained on *hot* neurons made small cold blocks *worse* (first run): the cold and hot distributions differ.
- By group (symbols + zstd-3): A (old self) 190 B (5.5x), B 270 B (3.9x), E (never used) 206 B (5.1x); hot groups C 507 B, D 565 B (about 2x). **Cold neurons compress 2.4x better than hot ones.**
- **A 90%-cold brain** (hot neurons uncompressed at 1,041 B, + 8 B block index per 64 neurons): symbols/zstd-3 **301 B per neuron** (3.5x), symbols/zstd-19 293 B, raw/zstd-3 389 B, raw/lz4 564 B. With the sleep-time re-freeze of section 4 (201 B per cold neuron) it is **285 B** (3.7x): 1 million neurons in 0.27 GiB instead of 0.97 GiB.

### 3. Lossy fading: behavior (the demo story, 32 seeds, fading from the first unused night)

All 6 "time passes" days happen at reduced precision. Bits are (c0 surface, c1 middle, c2 deep);
block scale and stochastic rounding unless noted. "rewired" = synapses rewired on unused nights.

| variant | day 3 | day 6 mean [range] | relearn trials, median | rewired | deep / surface contrast, day 6 |
|---|---|---|---|---|---|
| packed baseline (no fading) | 0.07 | 0.16 [0.06-0.41] | 2 | 0.2 | -0.098 / -0.133 |
| plain neuron (no deep levels) | 0.03 | 0.03 | 400 (never) | 8.8 | |
| **lazy, 2/2/2 (ternary everywhere)** | 0.07 | 0.16 [0.05-0.39] | 2 | 0 | -0.102 / -0.138 |
| lazy, 2/2/4 | 0.07 | 0.16 [0.06-0.52] | 2 | 0 | -0.101 / -0.138 |
| lazy, 3/3/3 | 0.06 | 0.16 [0.06-0.44] | 2 | 0 | -0.098 / -0.134 |
| eager, 2/2/2 (re-rounded at every write) | 0.11 | 0.28 [0.07-0.74] | 2 | 0 | -0.103 / -0.138 |
| eager, 2/2/2, pruning while cold (current sleep rule) | 0.12 | 0.35 [0.05-0.85] | 2 | **203** | -0.066 / -0.108 |
| plain neuron, eager 2-bit, pruning while cold | 0.03 | 0.04 | **8** | **380** | |
| lazy, surface erased (0/6/6) | 0.18 | 0.34 [0.09-0.92] | **1** | 0 | -0.044 / -0.058 |
| lazy, middle erased (6/0/6) | 0.05 | 0.10 [0.06-0.18] | 2 | 0 | -0.040 / -0.062 |
| **lazy, deep erased (6/6/0)** | 0.03 | **0.03** | **4** | 0 | -0.115 / -0.147 |
| lazy, everything erased (0/0/0) = a fresh neuron | 0.03 | 0.03 | 3 | 0 | 0 / 0 |
| eager, 2/2/2, *nearest* rounding | 0.03 | 0.03 | 3 | 0 | +0.055 / -0.359 |
| eager, 4/4/4, *nearest* rounding | 0.28 | 0.76 | 0 | 0 | |
| lazy, 2/2/2, *global* scale (no per-branch scale) | 0.12 | 0.31 [0.07-0.75] | 2 | 0 | |

How far can precision fade? (Full grids of 125 bit combinations x eager/lazy x pruning in
`results/behavior_grid_*.txt`.)

- **With lazy timing, down to ternary on every level** (2 bits + a 5-bit scale per branch, 10.5
  bits per synapse fixed-width instead of 32) **behavior is indistinguishable** from no fading.
- **Behavior breaks the moment any level is erased** (0 bits). Erasing the **deep** level kills the
  old self: no spontaneous recovery (0.03, plain-like) and no savings (relearning 4 trials, slower
  even than with every level erased (3), because the hard times stay while the old self is gone). Erasing the **surface** erases the *hard times* instead: the old self
  comes back at once (day 6 0.34, relearn 1). That is "forgetting the recent layer", a real
  change of personality, not a storage detail.
- Three rules make fading safe, each found by breaking it:
  1. **Stochastic rounding, never nearest.** Nearest rounding loses every change smaller than half a
     coarse step, so the slow cascade flow stops: the neuron is either stuck in the hard times
     (2-bit: 0.03, surface contrast frozen at -0.36) or flips wildly (4-bit: 0.76).
  2. **Quantize once (lazy), don't re-round every night (eager).** Eager re-rounding compounds the
     noise; through the sigmoid this inflates and spreads the recovery (0.28 [0.07-0.74]).
  3. **Freeze the wiring with the neuron (no pruning while cold).** Fading rounds small weights to
     exactly 0 on every level, which the sleep rule reads as "weak, unused, untagged": prune and
     rewire. Under eager 2-bit fading a workhorse saw 203 rewirings for its 256 synapses in 6 nights
     (existence lost, not just resolution), and the plain neuron "relearned" in 8 trials instead of never because
     its punished synapses were wiped and rewired. With frozen wiring both effects vanish.
- **Deep level: last to fade, never erased.** All named policies (deep-last, deep-protected,
  uniform, deep-first) pass in lazy mode, so the single-neuron story cannot tell them apart above
  2 bits. But the deep level is the only one whose erasure destroys the old self, it is the slowest
  to change, and it carries what keeps a cold synapse non-zero (and so un-pruned). My policy is
  **deep-last with a 4-bit floor**: surface 4 -> 3 -> 2 bits, middle a night later, deep 6 -> 5 -> 4.
  It costs ~1 bit per synapse over all-ternary and buys margin for the thing the project depends on.

### 4. Population test: groups A and B through idle days 9-16, then A relearns

6,144 A neurons (old self + hard times) and 4,096 B neurons (stable likes and dislikes) are frozen
in the real cold store from the first idle night. Thousands of neurons make the means tight:
baseline vs lossless freeze agree to 3 decimals, so differences below are real.

| policy (bits c0/c1/c2) | A day 9 | day 11 | day 14 | day 16 | B liked / disliked, day 16 | A relearn: median trials (share by trial 2) | bytes per cold neuron |
|---|---|---|---|---|---|---|---|
| baseline (hot, nightly sleep) | 0.03 | 0.06 | 0.14 | 0.17 | 1.00 / 0.04 | 2 (0.92) | 1,041 |
| lossless freeze, once | 0.03 | 0.06 | 0.14 | 0.17 | 1.00 / 0.04 | 2 (0.93) | 389 |
| **lossless freeze + re-freeze at nights 4 and 8** | 0.03 | 0.06 | 0.14 | 0.17 | 1.00 / 0.04 | 2 (0.93) | **201** |
| deep-last (4/6/6 -> ... -> 2/2/4) | 0.03 | 0.07 | 0.20 | 0.20 | 1.00 / 0.04 | 2 (0.92) | 255 |
| ternary 2/2/2 | 0.03 | 0.07 | 0.16 | 0.20 | 1.00 / 0.04 | 2 (0.92) | 235 |
| lossless, then ternary from night 4 | 0.03 | 0.06 | 0.17 | 0.20 | 1.00 / 0.04 | 2 (0.93) | 218 |
| codebook, 64 states (fringe) | 0.03 | **0.16** | 0.23 | 0.24 | 1.00 / 0.04 | 1 (0.96) | 198 |
| codebook, 256 states | 0.03 | 0.08 | 0.13 | 0.14 | 1.00 / 0.04 | 2 (0.90) | 233 |
| re-freeze, then codebook-64 from night 4 | 0.03 | 0.06 | 0.21 | 0.21 | 1.00 / 0.04 | 2 (0.95) | 184 |
| surface + middle erased (0/0/4) | 0.03 | 0.07 | 0.14 | 0.17 | **0.67** / 0.04 | 2 (0.94) | 170 |
| only a 2-bit deep level kept (0/0/2) | 0.03 | 0.07 | 0.16 | 0.21 | **0.66** / 0.04 | 2 (0.93) | 117 |
| deep erased (4/4/0) | 0.03 | **0.03** | **0.03** | **0.03** | 0.97 / 0.04 | **3 (0.00)** | 229 |

- **Lossless + re-freeze is the best policy**: 5.2x smaller than hot neurons with *no* behavior change.
  A block frozen once stays as complex as on the night it froze (389 B). Re-freezing it at sleep
  after 4 and 8 nights (thaw, apply the elapsed time exactly, freeze again) lets the cascade's own
  convergence show up in the coder: 201 B. Old memories really do get simpler; lossless compression
  harvests that for free.
- **Every lossy variant shifts the old-self / recent-self balance measurably** (by 0.02-0.10 on
  A's recovery; mostly towards the old self, e.g. ternary 0.20 vs 0.17 on day 16, codebook-64 0.16 vs
  0.06 on day 11; codebook-256 slightly the other way). The sharp, rare, recent states (the
  hard-times suppression) suffer more from rounding than the broad old ones. Ternary is 40% smaller
  than a one-time lossless freeze (235 vs 389 B) but *larger* than lossless re-freeze (201 B).
- **Erasing levels** saves real space (117-170 B) but loses memories: B's liked pattern drops from
  1.00 to 0.67 when the surface goes (B's recent learning lives there), and with the deep level
  gone A's recovery and savings disappear (relearning median 3, nobody by trial 2).
- Fixed-width packing alone (no entropy coding) would need 352 B for ternary and 721 B for
  lossless 22-bit synapses: entropy coding matters more than the bit budget.

### 5. Cost table (CPU, NumPy, one thread)

Sizes: hot = the C group (5,120 neurons, `PackedPopulation`, unchanged code); cold = 4,096 neurons of
A+B+E in 64-neuron blocks. Median of 3, one thread, NumPy, other agents running.

| | resident bytes per neuron (RAM) | peak temporaries | throughput, neurons/s |
|---|---|---|---|
| hot: forward | 1,041 | ~7 KB per neuron (baseline: 684 MiB / 100k) | 165k |
| hot: modulate | | ~12 KB per neuron (baseline) | 106k |
| hot: sleep | | ~12 KB per neuron (baseline) | 107k |
| cold: freeze, lossless (symbols + zstd-3) | **192** (day-16 cold; 201 in the population test) | 9.5 KB per neuron *of the block* (0.6 MB per 64-neuron block) | 166k (0.17 GB/s) |
| cold: thaw (decompress) | | 12.6 KB per neuron of the block | 853k (0.89 GB/s) |
| cold: thaw + 8 nights of exact time evolution | | same | 175k |
| cold: one night of sleep | 0 extra | 0 | no work (one 3x3 matrix per block) |
| cold: freeze with fading 4/4/6 / 2/2/4 | 226 / 244 | 22 KB per neuron of the block | 84k / 65k |
| VRAM | 0 (CPU-only brief) | | |

(Synapse-updates per second = neurons/s x 256.)

- Hot neurons are untouched: same words, same code, same speed as the baseline.
- **A cold neuron's night costs nothing**: flow and downscaling are linear, so N nights are one
  3x3 matrix per block; tags, age and plasticity are caught up at thaw. In a 90%-cold brain a
  night costs the hot 10% plus the re-freezes (two per cold neuron in its first 8 nights, then
  rarer), so sleep gets about 2.6x cheaper at first (hot sleep 9.3 us per neuron; a re-freeze
  11.7 us, twice in 8 nights) and approaches 10x.
- Waking a cold block of 64 neurons with 8 nights of exact evolution takes ~0.4 ms; that is the
  latency a thalamic "attention" signal would need to hide, and why blocks should follow columns.
- Temporaries are bounded by the block (64 neurons -> ~1 MB), not by the brain size.
- VRAM: not used (CPU-only brief). The natural split is hot words in VRAM at 1,041 B and the cold
  store in RAM at ~200 B; the cold store's nightly cost is zero either way.

### 6. Fringe

**a) Synapse-state codebook (vector quantization).** One brain-wide table of K prototype
(c0, c1, c2) states; a cold synapse stores one index for all three levels. Capacity-weighted k-means
(weights 1, 2, 4: what the surface eventually drifts to is the levels' capacity-weighted mean).
Error of that mean, in 6-bit code steps, on long-cold synapses:

| | bits for the three levels | error of the weighted mean (code steps) |
|---|---|---|
| lossless (549 states) | 5.6 (entropy) | 0 |
| VQ, K = 16 | 3.5 (entropy) | 0.41 |
| VQ, K = 64 | 4.8 (entropy) | 0.10 |
| VQ, K = 256 | 5.6 (entropy) | 0.01 |
| per-level fading 4/4/6 | 14.3 (fixed width, with scales) | 0.26 |
| per-level fading 3/3/3 | 9.5 | 1.20 |
| per-level fading 2/2/2 | 6.5 | 3.49 |

A codebook is a far better lossy knob than per-level bits, because it keeps the levels'
correlation; but on long-cold neurons lossless already costs only 5.6 bits, so there is little to win. Sizes with VQ-64: long-cold A 142 B (lossless 193 B), freshly cold A 224 B (lossless
462 B). **But** a codebook built only from long-cold neurons rounded away the *conflict* states of a
freshly cold neuron (surface punished, deep still fond): the demo's old self returned early (day 3
0.14-0.20 vs 0.07). Built from long-cold *and* freshly cold states (2,945 distinct states in all),
VQ-64/256 pass the single-neuron story (day 3 0.07, day 6 0.14/0.15, relearn 2). In the more
sensitive population test, though, VQ-64 makes A's old self return early (day 11 0.16 vs 0.06,
relearn 1), VQ-256 is close to baseline but needs 233 B (more than lossless re-freeze), and the full
2,945-entry table matches baseline at 338 B. Promising as a *lossless* brain-wide symbol table;
lossy VQ is not worth it here.

**b) Cross-neuron deduplication / prototype + residual.** No duplicate neurons or branches at all.
Neurons of one column all learned the same pattern and their weights *by input* correlate 0.64, but
*by slot* (how they are stored) 0.0001: hash wiring scrambles which slot sees which input. Residual
against a column prototype: 3.09 bits vs 3.06 raw, no gain. **Dead end under hash wiring**; it would
need input-aligned storage or wiring shared within a column.

**c) Holographic superposition** (K cold neurons' level vectors bound with random +-1 keys and
permutations, summed into one vector; read back by unbinding):

| K neurons per trace | recall correlation | codes exactly right | demo (lazy): day 3 / day 6 / relearn |
|---|---|---|---|
| 1 | 1.00 | 100% | 0.07 / 0.15 / 2 |
| 2 | 0.49 | 10% | 0.08 / 0.20 / 2 |
| 4 | 0.28 | 3% | 0.14 / 0.35 / 1 |
| 8 | 0.20 | 3% | 0.26 / 0.42 / 1 |
| 16 | 0.13 | 2% | 0.28 / 0.44 / 1 |

Crosstalk grows like sqrt(K), and the stored neurons are dense, so recall degrades from K = 2. The
crosstalk washes out the sharp hard-times suppression first, so the old self comes back too early.
It does not even save space: a float16 trace shared by 2 neurons is 768 B each, against ~200 B
lossless. **Dead end.**

## Behavior check (summary)

- Packed baseline through my adapter with no fading: **bit-identical** to `PackedNeuron`
  (same seeds, same RNG stream: fading uses its own RNG).
- Lazy lossless freeze: day 6 0.15 [0.06-0.40], relearn median 2 (baseline 0.16 / 2; seeds 1-32).
  The only change on purpose: **no pruning while a neuron is cold** (baseline prunes 0.2 synapses
  per workhorse and 8.8 per plain neuron over those 6 nights).
- Lazy ternary 2/2/2 and the deep-last policy: day 6 0.16 / 0.19, relearn 2, no rewiring.
- Population (6,144 + 4,096 neurons): lossless freeze and lossless re-freeze match the baseline to
  3 decimals at every probe and in the relearning curve.

## Verdict

**Mixed, with a clear winner.**

- Promising: **lossless, lazy, re-frozen cold blocks** (symbols + zstd-3). ~200 B per long-cold
  neuron (5.2x), zero nightly work for cold neurons, exact behavior, simple code. A 90%-cold brain
  drops from 1,041 to under 300 B per neuron.
- Works but not worth it here: **lossy precision fading**. Ternary levels are behavior-safe only
  with stochastic rounding + lazy timing + frozen wiring, and then save 0-15% over lossless
  re-freeze while measurably tilting the old/recent balance. Its one niche: neurons that *just*
  went cold (levels still disagree), where ternary halves the lossless size.
- Dead ends: holographic superposition; cross-neuron dedup under hash wiring; per-level bits at
  4 bits (worse than lossless: independent rounding breaks the levels' correlation).

## Surprises

1. **The cascade compresses itself.** Over cold nights the flow pulls c0, c1 and c2 together, so a
   long-cold synapse is one of a few hundred states. "Old memories get vaguer" is already in the
   neuron's dynamics; we only need to re-freeze to cash it in.
2. **Lossy can be bigger than lossless.** Stochastic rounding each level on its own grid breaks the
   correlation between levels: 4/4/6 fading made long-cold blocks 17% *larger*.
3. **Fading x pruning.** Rounding small weights to zero makes healthy synapses look dead to the
   sleep's prune rule. That turned "losing resolution" into "losing existence" and even faked
   *better* relearning for the plain neuron (8 trials instead of never).
4. **The old self is tiny.** After hard times the deep level's preference for pattern A is +0.043
   (about 1.3 steps of a 6-bit code), and by day 6 the levels have mixed (deep contrast -0.098). The
   recovery is the surface being pulled back to the levels' capacity-weighted mean. Yet unbiased
   ternary rounding keeps it, because the neuron sums ~32 synapses and the errors average out.
5. **Codebooks forget the rare states first.** A synapse-state codebook trained on long-cold
   neurons is near-lossless on them, but it rounded away the *conflict* states of a freshly cold
   neuron (surface punished, deep still fond), and the old self came back too early. Rare states
   are the meaningful ones.

## What I'd try next

- Re-freeze on an exponential schedule (nights 2, 4, 8, 16, ...) and measure the size curve over
  months of simulated idleness; a never-used neuron (group E, ~200 B of pure initial noise and
  prune churn) could be stored as one "pristine" bit and re-initialized on demand.
- A brain-wide synapse-state table (lossless, ~3,000 entries) shared by all blocks instead of a
  table per block; and coding `gen` per column.
- Put the hot set in VRAM and the cold store in pinned RAM / NVMe; thaw blocks on a predictive
  "this column is about to be needed" signal, overlapped with compute.
- A harder behavior test for the deep level (the demo's relearning is easy: a fresh neuron needs
  3 trials), e.g. weak reminders, renewal by context, or two competing old selves.

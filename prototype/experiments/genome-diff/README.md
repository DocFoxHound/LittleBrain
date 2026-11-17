# genome-diff: "a neuron that has learned nothing costs nothing"

Run everything (about 9 minutes, each part in its own process of 2 minutes or less):

```
.venv/bin/python prototype/experiments/genome-diff/run.py            # all parts
.venv/bin/python prototype/experiments/genome-diff/run.py behavior   # or: world [name], formats, bench, regen
```

The raw output of the run these numbers come from is in `results/*.txt`, and the per-day curves are in `results/world_*.csv`.
Machine was shared with other agents. Timings below are from the full run. Compare them as ratios.

## The idea

**DNA plus epigenetics.** A cell does not store its body plan. It regenerates it from the genome
and stores only the *marks* that life put on top. In the same way, every synapse's birth state (wiring and
initial weight) can be regenerated from `(neuron id, slot, generation)`, so only what experience
changed needs to be stored, as a sparse diff against the "genome". If experience is sparse, a huge
brain could fit in very little memory. Sleep could then act as garbage collection for the diff:
untagged changes fade back to the genome, while tagged ones ("this mattered") persist like epigenetic marks.

## What I built

| file | what |
|---|---|
| `core.py` | `GenomePop`, the packed workhorse (same 32-bit word, same rules) as numba kernels (≤3 threads). Birth state is fully procedural: wiring **and** innate weight come from the *same* 64-bit hash the prototype already computes, so no extra hashing is needed. Stochastic-rounding noise comes from a counter-based hash, so there is no RNG state. Forward and modulate are event-driven (they visit only cells that can see input, which is exact). There are two dynamics modes (below) and the sleep "diff GC" knobs. It also includes `GenomeNeuron`, an adapter for `demo_single_neuron.run`. |
| `world.py` | A sparse, structured world: 4,096 inputs in 64 regions, one column per region, 16,384 neurons, local wiring. It has things, valence, rest and sleep. It records diff-vs-genome statistics after every night. |
| `formats.py` | Six diff formats plus zstd: encoders, row "materialize" (genome + diff → dense row) kernels, and exact byte counts. |
| `bench.py`, `regen.py` | Benchmarks: forward, modulate and sleep throughput and temporaries; forward straight from a CSR diff; the cost of regenerating the genome on CPU and GPU (CUDA C via cupy). |
| `behavior.py` | The childhood → hard times → time passes → relearn story for every variant, seeds 1–16. |

### Two dynamics modes

* **BASE**: exactly the baseline rules. The word stores `c0 = innate + learned`, so the genome is the birth state.
* **ANCHORED + genome clock**: the word stores only the *learned deviation* `l0, l1, l2` (starting at 0). The innate weight is
  never stored. Its evolution under awake flow, sleep flow and downscaling is linear and **identical for the whole cohort**,
  so it is tracked by one shared 3×3 matrix `M(t)`, the "genome clock" (think of an epigenetic clock).
  The effective weight is `fast + g·(M e₀)₀ + l0`. Zero is an exact fixed point of flow, downscaling and stochastic rounding, so a
  synapse that experience never touched stays **bit-identical to a constant word (`PRISTINE`) forever**.
  Its diff entry does not exist, and sleep skips it. A regrown synapse stores its new birth weight, like BASE does.

## Result 1: the genome has to be a trajectory, not a snapshot

Under the baseline rules, **sleep itself moves every synapse**, touched or not. Each night it sinks c0 into c1 and c2, downscales
c0 and re-rounds every value. After the first night in the sparse world, **97.8% of synapses differ from their
birth state** (BASE row below), and that includes synapses whose input was never active. Compressing the BASE words gives 139 B/neuron
at birth (the random innate weights) and 467 B/neuron by day 40.

With the genome clock, an untouched synapse costs exactly nothing: a newborn brain compresses to 0.1 B/neuron, and sleep on
a never-used brain takes 0.4 ms instead of 115 ms. This is a 285× speedup (65,536 neurons). Behavior is unchanged (next table).

## Behavior check (16 seeds, `run(..., make=GenomeNeuron adapter)`)

| variant | hard times | day 3 | day 6 mean [range] | relearn median [range] |
|---|---|---|---|---|
| baseline `PackedNeuron` (prototype) | 0.03 | 0.07 | 0.15 [0.09–0.38] | 2 [1–2] |
| BASE: procedural birth, same rules | 0.03 | 0.07 | 0.17 [0.10–0.41] | 2 [1–2] |
| ANCHORED, *static* innate weight (no clock) | 0.03 | 0.12 | **0.36** [0.14–0.84] | 1 [0–2] |
| **ANCHORED + genome clock** | 0.03 | 0.07 | 0.18 [0.09–0.41] | 2 [1–2] |
| + prune only cells active today | 0.03 | 0.07 | 0.18 [0.09–0.41] | 2 [1–2] |
| + turnover-limited pruning (5%/night) and GC (consolidate only tag ≥ 2/7, snap untagged ≤1-step residue to genome) = **sparse-world default** | 0.03 | 0.07 | 0.17 [0.09–0.41] | 2 [1–2] |
| + Hebbian gate (learn only if rate ≥ 0.1) | **0.19** | 0.90 | 0.91 | 0 |
| + untagged learning decays toward genome 10%/night | 0.03 | 0.07 | **0.10** [0.07–0.16] | 2 [1–3] |
| + same at 30%/night | 0.03 | 0.06 | **0.06** [0.05–0.09] | 3 [2–3] |

* The genome clock matters. Without it, the innate weights never fade, so the old self comes back **twice as strongly** (0.36).
* All the storage-motivated changes I would actually use (clock, activity-gated pruning, turnover cap, tag-threshold
  consolidation, snapping 1-step residues) keep the story: the old self recovers slowly and relearning takes 2 trials.
* **The fringe idea partly fails.** Making *untagged* changes decay back toward the genome during sleep erodes the old self.
  Childhood's memory lives in the deep cascade levels after its tags have faded, so "untagged" ≠ "unimportant".
  Recovery drops from 0.18 to 0.10 (10%/night) and 0.06 (30%/night), and relearning slows to 3 trials. Epigenetic-style
  *erasure* of untagged marks is incompatible with the cascade's way of storing the old self. The gentle version (only snap
  1-step untagged residues) is harmless but buys little.
* Hebbian gating makes learning sparse but breaks the story: once the cell stops responding to A, punishment can no longer
  overwrite A, so "hard times" never sink in. This is a design question, not a storage one.
* The plain neuron (1 level) relearns in 7/16 seeds (baseline), 9/16 (BASE) and 8/16 (sparse-world default). It depends on lucky rewiring, and these differences are within noise.

## Result 2 (the key number): how sparse is experience, really?

**Scenario:** 4,096 inputs in 64 regions of 64. There is one column of 256 neurons per region (16,384 neurons, 4.2 M synapses).
Synapses draw from their own region, or from a neighbouring region with probability 10%. There are 2,048 "things", each lighting up 12 inputs of its region,
encountered by a Zipf law. 13% of things are good and 7% bad, and the 10 most familiar good things turn bad on day 21. A day has 50 episodes of 4 steps,
each showing 7 things (84 inputs, **2.05% of inputs active**). At the end of an episode, dopamine is ±1 if a valenced thing is present
(ACh 0.8, arousal 0.8, about 48 events/day). The day has two rests of 100 steps and ends with sleep. Each scenario runs 40 days. Event-driven stepping visits 25% of neuron-steps
(the columns whose region, or a neighbouring region, has input).

Fraction of **synapses** that differ from the genome after the night, split into *rewired* (pruned and regrown), *learned*
(any cascade level ≠ genome; ≥2 steps in brackets), and *tag-only*. The last column gives the fraction of **neurons** with any diff.

| scenario | day 1 | day 7 | day 20 | day 40 | neurons (day 1 → 40) |
|---|---|---|---|---|---|
| BASE rules (diff vs birth state) | **97.8%** (rw 28, drifted 67) | 99.8% | 100% | 100% | 100% → 100% |
| ANCHORED + clock, baseline pruning | 87.7% (rw **51**, ln 28, tag 9) | 98.5% | 99.8% | 99.9% | 100% → 100% |
| sparse-world default (activity-gated 5%/night pruning, GC) | 39.2% (rw 3, ln 28 [8], tag 9) | 71.2% | 80.6% | **86.3%** (rw 37, ln 38 [28], tag 12) | 100% → 100% |
| + 1% random background input (20 days) | 81.9% | 98.9% | 99.2% | – | 100% |
| + dopamine *targeted* to the valenced thing's column | 10.7% (rw 3, ln 7 [3], tag 1) | 36.7% | 55.1% | 74.3% (rw **60**, ln 8 [5], tag 6) | 100% → 100% |
| targeted, **no pruning** (pure learning diff) | **8.1%** (ln 7 [3], tag 1) | 20.6% | 18.7% | **21.6%** (ln 11 [5], tag 11) | 33% → 90% |
| targeted, *growing world* (4 regions on day 1, +1.5/day) | 4.1% | 14.2% | 26.0% | 51.3% | 9% → 100% (= regions seen) |

What sets these numbers:

1. **The baseline prune rule assumes dense input.** "Weak and unused today and untagged" is the *normal* state of a synapse under
   sparse coding. On the first night it rewires **51% of all synapses** (131 per neuron), because under the clock a birth weight of 0–1 steps
   is already below `prune_eps`. Even capped at 5% per night of eligible synapses, rewiring adds about 1–1.5% of all synapses per night and
   becomes the dominant long-run diff (37–60% by day 40). A regrown synapse is a genuine diff (new target).
2. **Learning does not need the cell to fire.** Eligibility is `input × (0.5 + rate)`, so a silent cell's synapses learn at half
   strength, and dopamine is broadcast. Every input that is active near any reward changes about 1,000 synapses. 2% of inputs × 200
   steps a day covers the 4,096-input space about 4× per day, so 28% of synapses learn on day 1 and about 40% by day 7, then saturate.
   With **targeted dopamine** and **no pruning**, the persistent learning diff plateaus at about **11% of synapses (5% by ≥2 steps)**. It plateaus
   because downscaling plus snap-GC erase old residue about as fast as new learning arrives. Tags add another 6–12%.
3. **Background noise is fatal to sparsity**: 1% random input, combined with the presynaptic-only eligibility, touches 82% of synapses on day 1.
4. **Neurons are not sparse** in this world, even when synapses are. Every region is used every day, so 33–100% of neurons
   carry *some* diff within days. In the **growing world**, the dirty fraction tracks exactly the part of the brain that life
   has explored. That is the "storage proportional to experience" picture working as intended, but only while life is narrower than the brain.

## Result 3: diff formats (bytes per neuron, averaged over *all* neurons)

The value of each entry is the 32-bit word (only 25 bits persist). For each neuron, add **about 5 B of dense state that stays**: threshold (f32, which homeostasis
changes for every cell every step) and the gate byte. Age and plasticity are cohort-wide. Fast-weight time, post history and "active today" are
daytime scratch for the working set.

| snapshot | % syn differ | dense | block32 | block8 | **bitmap** | **csr** | coo | hash | zstd(csr) | zstd(dense) |
|---|---|---|---|---|---|---|---|---|---|---|
| growing world d1 | 4.1 | 1024 | 88 | 76 | **46** | 57 | 85 | 128 | 15 | 15 |
| growing world d20 | 26.0 | 1024 | 538 | 501 | **285** | 336 | 532 | 1024 | 117 | 118 |
| growing world d40 | 51.3 | 1024 | 1021 | 989 | **562** | 661 | 1052 | 2048 | 208 | 212 |
| targeted, no pruning d40 | 21.6 | 1024 | 832 | 687 | **254** | 281 | 443 | 1024 | 98 | 106 |
| targeted d1 | 10.7 | 1024 | 744 | 444 | 146 | **141** | 220 | 512 | 50 | 59 |
| targeted d40 | 74.3 | 1024 | 1025 | 1028 | **798** | 956 | 1523 | 4096 | 271 | 260 |
| sparse-world default d40 | 86.3 | 1024 | 1025 | 1028 | **920** | 1109 | 1767 | 4096 | 389 | 368 |
| synthetic: 1% of neurons × 16 entries | 0.06 | 1024 | 2.5 | 5.5 | **1.2** | 4.8 | 1.3 | 2.0 | 0.4 | 0.7 |
| synthetic: 10% × 32 | 1.25 | 1024 | 27 | 30 | **17** | 20 | 26 | 64 | 6.4 | 8.3 |
| synthetic: 100% × 64 | 25 | 1024 | 385 | 388 | **292** | 324 | 512 | 1024 | 103 | 116 |

Access cost: materialize a neuron's 256-word row from genome + diff (25% of neurons, ns per neuron, numba 3 threads, from the full run).
Dense copy takes 45–67 ns. **csr 32–88 ns**, block32 43–113, coo 38–257, bitmap 79–513, block8 109–255,
**hash 1,559–3,121 ns** (256 probes per row), and zstd decompression 0.07–1.8 µs per neuron. Rebuilding (encoding) after sleep costs
0.25–1.6 µs per neuron for every format except hash (up to 21 µs) and zstd (up to 12 µs).

* **bitmap** (dirty-neuron bit + rank directory, 256-bit map per dirty neuron) is the most compact usable format at every density. Its break-even
  against dense is about 96% of synapses differing (36 B + 4 B/entry per dirty neuron). **csr** costs slightly more bytes but is the fastest to read (same speed as a dense copy).
  Its break-even is about 80% (4 B + 5 B/entry). Dirty blocks help only when changes cluster on a few branches, which they don't: a column sees many patterns.
  Hash maps cost 16 B per entry and are slow for row access, so they are out.
* **Forward can run straight from the CSR diff**: regenerate every synapse from the hash and merge-walk the neuron's sorted entries.
  This gives the same rates as dense (checked) in **1.01× the time while using 9.8% of the memory** (65,536 neurons, 30% with 64 entries).
  Modulate needs a writable daytime working set: materialize the day's active cells once (cheap, as above), then merge back at night.
* **Control: generic compression is as good as any diff format** *once the genome is subtracted*. zstd on the ANCHORED dense
  words ≈ zstd(csr) in every snapshot. The genome subtraction is what does the work. Without it (BASE words), zstd gives 139 B/neuron at birth
  versus 0.1 B. For the SSD tier, "subtract the genome, then zstd" gives 15–390 B/neuron in these worlds.

## Throughput and memory (full run; 65,536 neurons × 256 synapses)

**A. Dense activity** (the prototype's 64-input wiring; every cell has input every step):

| implementation | forward | modulate | sleep | peak temporaries fwd / mod / sleep |
|---|---|---|---|---|
| prototype NumPy `PackedPopulation` | 853 ms (20 M syn/s) | 1,763 ms | 1,738 ms | 449 / 770 / 784 MiB |
| numba BASE | 46 ms (363 M syn/s) | 123 ms | 163 ms | about 1 MiB |
| numba ANCHORED + clock | 27 ms (622 M syn/s, 2.4 M neuron-steps/s) | 70 ms | 181 ms (diff 89%) | about 1 MiB |
| ANCHORED sleep, never-used brain | | | **0.3 ms** | |

In a quieter earlier run, BASE and ANCHORED forward were 17 and 15 ms, so they are equal within noise. Regenerating the innate weight costs nothing extra
because it comes from spare bits of the wiring hash. The numba kernels are 10–37× faster than NumPy, but that comes from the kernels, not the genome.
Temporaries are Python-side index and rate arrays (about 16 B/neuron) plus a 512 B scratch per 64-neuron chunk. No float unpacking of the population happens.

**B. Sparse activity** (local wiring, 2% of inputs active): event-driven forward of the 23% of cells that *can* see input takes 10.6 ms,
versus 37.5 ms for all cells (3.5×, exact). ANCHORED sleep cost scales with the diff:

| dirty neurons (64 entries each) | 0% | 1% | 10% | 50% | 100% | BASE (always full sweep) |
|---|---|---|---|---|---|---|
| sleep | 0.4 ms | 1.1 ms | 6.0 ms | 18.3 ms | 35.3 ms | 115 ms |

**Bytes per neuron.** The baseline is 1,041 B in RAM. This experiment keeps no VRAM state (CPU design). With ANCHORED + bitmap it is about 5 B for a never-used neuron, and
46–920 B + 5 B in the worlds above. My simulation keeps 37 B/neuron of per-neuron arrays (f32 scratch; this could be 17 B as in the baseline).

## Regenerating the genome on the fly (`regen.py`)

Forward-style sweep, drive per branch, 4,096 inputs:

| how a synapse gets (target, weight) | CPU, 65k neurons, 3 threads | GPU RTX 4070, 262k neurons |
|---|---|---|
| stored target + weight (3 B/syn) | 0.18 ns/syn (1.00×) | 14.9 ps/syn (1.00×) |
| prototype: 32-bit word + wiring hash (4 B/syn) | 0.41 ns/syn (2.3×) | 19.2 ps/syn (1.28×) |
| **procedural: one hash, nothing read (0 B/syn)** | 0.55 ns/syn (3.1×) | **13.1 ps/syn (0.88×, faster than reading)** |

On the GPU the procedural genome is **free**: reading memory is the bottleneck and a hash is cheaper than a load. GPU and CPU agree
bit-for-bit. VRAM used was 456 MiB of the 1 GB cap. On the CPU the hash is compute-bound: about 3× the cost of reading stored state, but
only 1.35× the prototype's word-plus-hash path.

## Verdict: **mixed**

* **Promising, and worth keeping: the genome clock.** It makes "untouched" an exact constant without changing behavior. That makes
  a never-used neuron cost about 5 B instead of 1,041 B and makes sleep cost proportional to the diff (285× faster on an unused brain). It also makes generic
  compression work (0.1 B/neuron newborn, versus 139 B without it). Regenerating from the hash is free on the GPU and cheap on the CPU,
  and forward can run straight from a sparse diff at dense speed.
* **But under the current learning rules, experience is not sparse** in any world that uses the whole brain every day. With baseline rules,
  98% of synapses differ after night 1. With the best behavior-preserving settings, 39% differ on day 1 and 86% by day 40. Even with targeted dopamine and no pruning,
  21% of synapses and 90% of neurons carry a diff. For the RAM tier, the saving after a few weeks is 1.1–4× (bitmap format), not the 100× hoped for. Large
  savings happen only where life hasn't been yet (growing world: 22× on day 1, 3.6× on day 20, 1.8× on day 40) and in cold storage
  (3–70× with zstd after genome subtraction).
* Making the diff truly sparse needs changes to the *learning* design, not the storage design: postsynaptic gating, local or targeted
  modulators, and pruning that does not churn unused synapses. Of those I tried, the Hebbian gate broke the behavior story.

## Surprises

1. **Sleep is the biggest source of "change"**: housekeeping alone moves every synapse every night. The genome has to age with the cohort.
2. **The prune rule rewires half the brain on night one** under sparse input. "Use it or lose it" with a 1-day memory assumes dense input.
   In a genome-diff brain an unused synapse is *free*, so there is no storage reason to prune it.
3. Eligibility does not depend on firing (`0.5 + rate`), so learning sparsity equals input sparsity, and 1% background noise touches everything.
4. **Tag-only entries make up 8–50% of the diff.** Tags are short-lived (halved nightly) and belong in a separate list that is cleared nightly, not in the persistent diff.
5. zstd on genome-subtracted words matches every hand-made format. On the GPU, regenerating is faster than reading.
6. Decaying untagged learning toward the genome erodes the old self, because old memories are untagged by design.

## What I'd try next

* **Prune on demand** (only when a neuron needs a new input), plus a *procedural default rewiring schedule* for never-used synapses,
  so that structural turnover stops dominating the diff.
* Tags in a separate nightly list and 3-byte diff values. Make the threshold lazy with a precomputed homeostasis trajectory for silent cells,
  so the per-neuron floor becomes about 1 B.
* Population-level Hebbian gating (competition picks which cells learn) that still lets "hard times" overwrite. Topographic dopamine.
* A GPU forward kernel that merges a CSR or bitmap diff with hash regeneration (the CPU version already runs at dense speed),
  plus SSD cold storage of `zstd(diff)` for dormant columns.

*Notes on exactness.* Event-driven forward and modulate are exact: a cell skipped at a step has zero input on all synapses, so it adds
no eligibility, and its threshold gets the same zero-input homeostatic update as before. Fast-weight decay is applied lazily per neuron, which gives the same
expectation with fewer re-roundings. Eligibility history is kept in f32 instead of f16. The behavior table is the check.

# cpu-event-driven: making the CPU a real home for neurons

**Question.** Brains are event-driven: a synapse costs (almost) nothing unless a spike arrives at
it, and most of the brain's energy goes into exactly those arrivals (Attwell & Laughlin 2001). Our
NumPy code instead touches every synapse every step and builds huge temporary arrays. How fast and
how lean can a 3-thread CPU be if it only does work where events happen?

**Answer in one line.** Store synapses by their *presynaptic input* ("axon-major") and give each
input its own lazy clock. Then both acting (forward) and learning (modulate) cost scale with the
spikes, not with the synapses. At 2% input activity that makes forward **~270x** and modulate
**~125x** faster than the NumPy baseline on one thread (~640x / ~270x on 3 threads), with
temporaries of a few MiB instead of 0.7-1.2 GiB, and the same behavior. The catch is 50% more bytes per synapse (6 instead of 4), unless connectivity is local.

Run everything (about 4 minutes, 3 threads, under 2.5 GB RAM):

    .venv/bin/python prototype/experiments/cpu-event-driven/run.py

## What I built

| file | strategy | layout | stored bytes / synapse |
|---|---|---|---|
| `dm_fused.py` | **1. Fused kernel** (Numba, `parallel`, `fastmath`) on the *current* layout | dendrite-major `words[n, branch, slot]`, input = hash | 4 |
| `axon_major.py` | **2. Event-driven, axon-major** (Numba) | buckets of synapses per (block of neurons, input) + 16-bit post id | 6 |
| `bitslice.c`, `bitslice.py` | **3. Bit-sliced branches** (C, AVX2 + BMI2, via ctypes) | a branch = 32x32 bit matrix; wiring bitmap in the 4 gen planes | 4 |
| `bench_cache.py` | **4. Cache awareness** | size sweep across the 24 MB L3, axon-major tile size | |
| `wiring_bits.py` | can axon-major wiring fit in the 4 freed gen bits? | delta-coded post ids | |
| `validate.py`, `behavior.py` | correctness and the behavior story for all of them | | |

All three implementations have the same interface as `PackedPopulation` (forward / modulate /
tick / rest / sleep) plus a single-neuron adapter for `demo_single_neuron.run`.

### 1. Fused dendrite-major kernel
One kernel per step walks the packed words, unpacks each into registers, computes and writes it
back. Same storage, same hash wiring, same math as `PackedPopulation`. The only difference is the
random numbers for stochastic rounding: a stateless counter hash (splitmix64) instead of NumPy's
generator, so no random arrays are allocated. In "exact" mode the kernel consumes NumPy's uniforms in
the baseline's order, and then **modulate is bit-for-bit identical to the baseline** (checked).

### 2. Axon-major, event-driven
```
words[j]  uint32   the baseline synapse word; bits 0-3 (gen) are now FREE
post[j]   uint16   which (neuron in block, branch) the synapse sits on
off[k,i]  int64    bucket of input i in neuron-block k = words[off[k,i] : off[k,i+1]]
stamp[k,i] int32   lazy fast-weight clock of that bucket
```
* **forward**: for each *active* input, walk its bucket and scatter `weight x input` into a per-block
  dendritic accumulator (8 floats per neuron, kept in L2); then one pass over the somas; then set the
  "used" bit on the same active buckets (branch-free).
* **modulate**: eligibility is non-zero only for inputs that spiked in the last 4 steps, so only those
  buckets are visited. The baseline folds a *global* fast-weight decay clock into every synapse at
  every modulate. Here each bucket keeps its own clock (`stamp`) and is rescaled only when it is
  touched. That per-axon lazy clock is what makes learning event-driven, not just acting.
* **cascade catch-up** (every 512 ticks) and **sleep** still touch every synapse, as in the baseline.
* **Rewiring** (worked out and implemented): a synapse keeps its postsynaptic side and moves to
  another input's bucket. Rewiring only happens during sleep, and sleep visits every synapse anyway,
  so each block's buckets are rebuilt with one stable counting sort. That needs no slack, no
  generation counter and no 16-rewiring wrap-around, and costs about +30% sleep time. *Online*
  rewiring while awake would need fixed-capacity buckets: with about 256 synapses per bucket and
  Poisson-like counts, roughly 25% slack avoids overflow; a move is then an O(1) swap-remove from the
  old bucket plus an append to the new one. I did not build that, because sleep-time rebuild is
  cheaper and matches the design (structure changes at night).
* **"The storage position is the wiring"** is only half true. The *pre* side becomes free, but the
  *post* side must now be stored, and 16 bits cost more than the 4 gen bits freed. See
  `wiring_bits.py` below: with local connectivity the post id fits in the 4 free bits as a gap code.

### 3. Bit-sliced branches (the fringe one, and it works)
A branch's 32 synapses x 32 bits are stored transposed: `rec[k]` = bit k of all 32 synapse words.
Planes 4-31 are exactly the baseline fields. The 4 gen planes (4 x 32 = 128 bits) hold the
**wiring as a bitmap over a window of 128 consecutive inputs** (window start = hash of neuron and
branch), with exactly 32 bits set. So the record is still **4 bytes per synapse**, with no hash
per synapse.
* **forward** with binary spikes: `mask = PEXT(input bits of the window, wiring bitmap)` gives the
  32 active slots in one instruction pair; then `drive = sum_k 2^k * popcount(plane_k & mask)` per
  field. That is 24 AND+POPCNT per 32 synapses, with no gather and no per-synapse arithmetic. The
  sum is exact integer arithmetic.
* **modulate / catch-up / sleep**: an AVX2 32x32 bit transpose (byte shuffle + 32 `movemask`)
  turns the branch back into 32 baseline words; the baseline per-synapse update runs (clang
  vectorizes it 8-wide); transpose back. Rewiring moves a synapse to a random **free** input inside
  its window (PDEP picks the r-th free bit) and re-sorts the 32 words by position.
* Same bytes per neuron as the baseline: **1,041**.

### 4. Cache awareness
A population-size sweep from 2 MiB to 195 MiB of synapses (L3 is 24 MB), and a sweep of the
axon-major block size (the tile whose dendritic accumulator must stay in cache).

## Scenario

100,000 workhorse neurons x 256 synapses = **25.6 M synapses**; **4,096 inputs**; binary spikes;
a **new random pattern every step** at activity 1-100% (worst case for the 4-step eligibility union,
since nothing repeats). Dendrite-major and axon-major use the baseline's hash wiring (global random,
so every neuron gets input every step, which is the brief's warning case). Bit-sliced uses local
128-input windows. The modulate protocol calls `tick()` first, as the demo loop does, so the
baseline's global fast clock has to be folded into every synapse. The machine was shared with
other agents (load average 2-9 of 20 threads). Implementations were timed interleaved, and **each
cell is the best of 2 repeats of 5-rep medians**; earlier runs under heavier load were up to 2x
slower in absolute terms but never changed the ranking. The NumPy baseline run below was hit by load
(modulate 1,954 ms); on a quiet machine earlier in the session it measured **forward 640 ms,
modulate 1,200 ms**. I quote speedups against those quiet numbers, which is the conservative choice.

## Results

**forward: milliseconds per step, 100,000 neurons (25.6 M synapses), 4,096 inputs**

| input activity | elig. union | dendrite 1T | axon 1T | bitslice 1T | dendrite 3T | axon 3T | bitslice 3T |
|---|---|---|---|---|---|---|---|
| 1% | 4% | 34.5 | 2.1 | 8.6 | 12.6 | 0.9 | 3.6 |
| 2% | 8% | 34.6 | 2.4 | 11.3 | 12.5 | 1.0 | 5.1 |
| 5% | 19% | 34.1 | 4.4 | 12.6 | 12.4 | 1.7 | 4.8 |
| 10% | 35% | 34.2 | 6.4 | 12.6 | 12.3 | 2.6 | 5.5 |
| 20% | 58% | 34.3 | 10.8 | 12.7 | 12.4 | 4.4 | 5.5 |
| 50% | 94% | 33.3 | 21.7 | 12.6 | 12.5 | 8.6 | 4.9 |
| 100% | 100% | 32.7 | 36.8 | 12.3 | 12.4 | 13.8 | 4.7 |

NumPy baseline, same scenario: 844 ms (1 thread).

**modulate (tick first): milliseconds per step, 100,000 neurons (25.6 M synapses), 4,096 inputs**

| input activity | elig. union | dendrite 1T | axon 1T | bitslice 1T | dendrite 3T | axon 3T | bitslice 3T |
|---|---|---|---|---|---|---|---|
| 1% | 4% | 94.2 | 5.9 | 53.0 | 33.2 | 3.0 | 19.1 |
| 2% | 8% | 93.9 | 9.5 | 52.6 | 33.1 | 4.4 | 19.2 |
| 5% | 19% | 93.0 | 19.8 | 51.8 | 32.7 | 8.3 | 19.0 |
| 10% | 35% | 93.2 | 35.6 | 52.0 | 33.0 | 13.4 | 19.1 |
| 20% | 58% | 92.3 | 56.5 | 51.6 | 32.8 | 21.0 | 19.0 |
| 50% | 94% | 90.8 | 87.4 | 50.8 | 32.4 | 32.3 | 19.0 |
| 100% | 100% | 90.0 | 92.0 | 50.7 | 32.1 | 33.9 | 19.4 |

NumPy baseline, same scenario: 1954 ms (1 thread).

**Throughput, 2% activity (synapses of the whole population per second; events/s for axon-major)**

| layout | step | 1 thread | 3 threads | speedup vs NumPy (1T / 3T) |
|---|---|---|---|---|
| dendrite | forward | 0.74 G/s | 2.05 G/s | 24x / 67x |
| axon | forward | 10.71 G/s | 24.91 G/s (0.21 / 0.50 G events/s) | 353x / 821x |
| bitslice | forward | 2.27 G/s | 5.04 G/s | 75x / 166x |
| dendrite | modulate | 0.27 G/s | 0.77 G/s | 21x / 59x |
| axon | modulate | 2.70 G/s | 5.85 G/s (0.21 / 0.45 G events/s) | 206x / 446x |
| bitslice | modulate | 0.49 G/s | 1.33 G/s | 37x / 102x |

**Steps that touch every synapse (ms)**

| layout | catch-up 1T | catch-up 3T | sleep 1T | sleep 3T |
|---|---|---|---|---|
| dendrite | 57 | 21 | 132 | 47 |
| axon | 57 | 21 | 163 | 60 |
| bitslice | 69 | 26 | 110 | 40 |

NumPy baseline sleep: 2551 ms (catch-up is part of it).

Axon-major forward with ZERO active inputs (the per-neuron floor): 1.23 ms (1T), 0.49 ms (3T).

**Memory**

| layout | bytes/neuron (stored) | peak temp. forward | modulate | sleep |
|---|---|---|---|---|
| dendrite | 1041 | 0.6 MiB | 2.2 MiB | 0.4 MiB |
| axon | 1565 | 0.5 MiB | 2.8 MiB | 12.1 MiB |
| bitslice | 1041 | 0.4 MiB | 0.6 MiB | 0.4 MiB |
| NumPy baseline | 1041 | 684 MiB | 1175 MiB | 1294 MiB |

**Cache: ns per synapse vs population size (1 thread, 10% activity)**

| neurons | synapse MiB | stream_read | dendrite_fwd | axon_fwd | bitslice_fwd | dendrite_cu | bitslice_cu |
|---|---|---|---|---|---|---|---|
| 2,000 | 2 | 0.07 | 1.28 | 0.27 | 0.52 | 2.21 | 2.74 |
| 8,000 | 8 | 0.13 | 1.26 | 0.27 | 0.52 | 2.24 | 2.74 |
| 23,000 | 22 | 0.21 | 1.37 | 0.28 | 0.55 | 2.45 | 3.00 |
| 64,000 | 62 | 0.19 | 1.42 | 0.26 | 0.55 | 2.44 | 2.97 |
| 200,000 | 195 | 0.19 | 1.38 | 0.26 | 0.52 | 2.34 | 2.86 |

**Axon-major block (tile) size, 100,000 neurons, 1 thread (ms)**

| block (neurons) | accumulator | index bytes/neuron | fwd 2% | mod 2% | fwd 20% | mod 20% |
|---|---|---|---|---|---|---|
| 256 | 10 KiB | 192 | 6.47 | 16.1 | 17.13 | 81.2 |
| 1024 | 40 KiB | 48 | 3.50 | 11.0 | 13.42 | 63.2 |
| 4096 | 160 KiB | 12 | 2.78 | 9.1 | 10.87 | 59.2 |
| 8192 | 320 KiB | 6 | 2.65 | 8.5 | 10.75 | 59.5 |

(`results_tables.md` is regenerated by `summarize.py`; raw numbers are in `results_*.json`.)

### Reading the numbers
* **Fused kernel, same layout (strategy 1):** forward 34 / 12.5 ms, modulate 93 / 33 ms, sleep
  132 / 47 ms on 1 / 3 threads, against NumPy's quiet 640 / 1,200 / ~1,500 ms. That is **19x / 51x**
  (forward), **13x / 36x** (modulate) and **~11x / ~30x** (sleep), with 0.4-2 MiB of temporaries
  instead of 684-1,294 MiB. Time is flat in activity, because every synapse is hashed and visited.
  The hash and the 4 eligibility gathers per synapse are the cost (1.3 ns/synapse forward,
  3.6 ns modulate per thread); it is compute-bound, not memory-bound.
* **Axon-major (strategy 2), scaling with sparsity:** forward time is roughly *floor + events x 3 ns*:
  2.4 ms at 2% (14x faster than the fused kernel on the same data), 10.8 ms at 20%, and slower
  than dendrite-major only at 100%. Modulate scales with the 4-step eligibility union: 9.5 ms at 2%
  (10x faster than fused), equal to it at ~50%. Per thread that is **~0.3 G synapse-events/s
  forward** (beyond the floor) and **~0.2 G events/s modulate**, about 0.45-0.5 G events/s on
  3 threads. **The new bottleneck at low activity is the per-neuron pass** (soma, homeostatic
  threshold, histories): 1.2 ms for 100k neurons even with zero input, which is half the forward
  time at 2%.
* **Bit-sliced (strategy 3):** activity-independent and the fastest *dense* option. Forward
  12 / 5 ms (2.8x faster than the fused kernel, 0.5 ns per synapse), modulate 52 / 19 ms (1.8x),
  sleep 110 / 40 ms (the best of the three), at the baseline's 1,041 B/neuron. It wins forward
  above ~20% activity and modulate above ~20%. Catch-up is 20% slower than fused (the two transposes).
* **Cache (strategy 4): tiling to L3 does not help, because nothing here is memory-bound at ≤ 3 threads.** A plain
  streaming read gets 3x slower once the data leaves L3 (0.07 → 0.19 ns per 4 bytes), but every
  kernel stays flat from 2 MiB to 195 MiB, since its compute per synapse (0.26-2.9 ns) is far
  above the memory cost. What *does* matter is the axon-major accumulator tile: blocks of
  4,096-8,192 neurons (160-320 KiB accumulator, in L2) are best. Small blocks pay a per-bucket
  overhead and a bucket index of 12 B x inputs per block (192 B/neuron at 256-neuron blocks).
* **Bytes:** axon-major is 1,565 B/neuron (6 B/synapse + 12 B/neuron of bucket index), +50%.
  `wiring_bits.py` shows when the 4 freed gen bits can hold the post id as a gap code inside
  post-sorted buckets:

  | inputs each block draws from | 64 | 128 | 256 | 512 | 4,096 (global) |
  |---|---|---|---|---|---|
  | bytes / synapse (4-bit gap code + spacer words) | 4.00 | **4.08** | 4.68 | 6.49 | 35.9 |

  So **axon-major at baseline size needs locality**: each block of neurons must draw from at most
  about 4 x (slots per branch) = 128 inputs. With global random wiring, store explicit 16-bit
  ids (6 B).

## Behavior check (seeds 1-8, `behavior.py`)

| | hard times | day 3 | day 6 (mean, range) | relearn trials (median) |
|---|---|---|---|---|
| workhorse / NumPy baseline | 0.03 | 0.07 | 0.15 [0.09-0.38] | 2 [1-2] |
| workhorse / fused kernel | 0.03 | 0.07 | 0.13 [0.11-0.19] | 2 [1-2] |
| workhorse / axon-major | 0.03 | 0.07 | 0.14 [0.08-0.24] | 2 [1-2] |
| workhorse / bit-sliced | 0.03 | 0.07 | 0.17 [0.10-0.43] | 2 [1-2] |
| plain / all four | 0.03 | 0.03 | 0.03 | relearned in 2-4 of 8 seeds (baseline 3/8) |

All four implementations show the same story: slow spontaneous recovery and fast relearning. The plain
neuron's relearning is the known all-or-nothing artifact (about 15 trials or never), so its
median flips between 16 and 400 with the random stream.

### Correctness (`validate.py`, all pass, at 64 and 4,096 inputs)
* Fused: forward rates within 3e-7 of `PackedPopulation`, used flags identical; **modulate
  bit-identical** given the same uniforms.
* Axon-major: built from the same `PackedPopulation`, same inputs; forward rates within 2e-7, used
  flags and post-history identical; **modulate bit-identical** while the fast clock is fresh;
  after 50 ticks, the lazy per-bucket decay has the same mean fast weight (difference < 0.02
  quantization steps); sleep prune count within 0.2%; buckets consistent after rebuild.
* Bit-sliced: against the baseline NumPy code reading the same bitmap wiring explicitly: forward
  within 4e-8, **modulate bit-identical**, sleep within 0.1%, rewired inputs stay inside the window.
* Production kernels (own RNG, float32 tag math) vs the exact paths: every synapse within one
  quantization step, mean bias < 0.002 steps.

### How each layout changes behavior (honestly)
* **Fused:** nothing, except where the stochastic-rounding random numbers come from.
* **Axon-major:** (1) fast weights decay lazily per input bucket. The baseline re-rounds every
  fast weight at every modulate, so it carries *more* rounding noise; the expected value is the
  same. (2) Rewiring draws the new input uniformly, like the hash, but without the baseline's
  wrap-around after 16 rewirings of a slot. (3) Production modulate uses float32 tag math (the baseline uses float64).
* **Bit-sliced:** (1) wiring is **local**: each branch takes 32 *distinct* inputs from one window
  of 128, and rewiring stays inside that window. (2) **Binary spikes only**; graded rates would need
  one bit-plane pass per input bit. Day-6 recovery came out slightly higher (0.17 vs 0.15) but
  within the seed range.

## Verdict: **promising** (axon-major for awake steps), **mixed** (bit-sliced), **solid but bounded** (fused)

* Fused kernels on the current layout are a free, exact **13-50x** win with zero temporaries.
  They should simply replace the NumPy step on CPU.
* **Axon-major plus per-axon lazy clocks is the real event-driven win.** Cost follows spikes:
  **14x (forward) and 10x (modulate) faster than the already-fused kernel at 2% activity**, still
  better up to about 20-50%, and the behavior is unchanged. It does **not** need neuron sparsity or
  locality to be fast, only *input* sparsity. That answers the brief's warning: every neuron can get
  input every step and it still wins. Locality is needed only to bring it back to 4 B/synapse.
  On this laptop's CPU, 3 threads run 100k neurons with learning every step at 2% activity in
  ~5.5 ms (≈180 steps/s), or about a million neurons at ~15-20 steps/s, in 1.5 GB of RAM.
* **Bit-sliced** is a genuine surprise. It fits wiring into the baseline's 4 bytes, has the
  fastest dense forward and sleep, and its rewiring is *local by construction*, which may be the
  more biological choice. But it requires binary spikes, and modulate still has to transpose.
  It is the right layout for **dense phases (high activity, sleep sweeps)**.

## Surprises
1. **A silent 64-bit promotion broke learning.** Numba promotes `uint32 * uint32` to 64 bits, so my
   32-bit hash stopped wrapping and returned "uniforms" up to 4e9. Every bit-exact check still
   passed, because those use NumPy's uniforms; only the behavior test caught it (day-6 recovery
   fell to 0.05). `validate.py` now also checks production kernels against exact ones.
2. **The textbook bit transpose was 20x too slow** (store-forwarding chains, about 210 ns per 32x32
   block). An AVX2 byte-shuffle + `movemask` version takes about 11 ns.
3. **The compiler mattered more than the algorithm.** gcc 15 did not vectorize the per-synapse loops;
   clang did (8-wide), which made bit-sliced modulate 12x faster. In Numba, indexing `a[lo + j]`
   keeps negative-index checks that block vectorization; looping over a slice `a[lo:hi]` removes them (1.6x).
4. **L3 size is irrelevant at 3 threads**: everything is compute-bound.
5. With event-driven synapses, the **soma/homeostasis pass becomes the floor** (about 12 ns per neuron per step).

## What I'd try next
1. **Shrink the per-neuron floor**: vectorized soma pass (structure-of-arrays, approximate exp), and
   lazy homeostasis for neurons that got no input (their threshold drift has a closed form between
   events). That would roughly halve axon-major forward at 1-2%.
2. **Column-local wiring with gap-coded post ids** in the freed bits: axon-major at 4.08 B/synapse.
3. **Hybrid by phase**: axon-major while awake (sparse) and a bit-sliced or dendrite-major sweep for
   sleep. Converting between them is one counting sort, about the cost of a sleep.
4. Axon-major is also the "push" layout that GPU spiking simulators use. The per-axon lazy clock
   should carry over directly to a GPU kernel.

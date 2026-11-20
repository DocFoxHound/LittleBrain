# VRAM–RAM paging: virtual memory for a brain

Managed-memory hints use NVIDIA’s CUDA 13 Python driver package (`cuda-bindings>=13,<14`).

**Verdict: promising, with two conditions.** An 8.4-million-neuron population (8.6 GB of packed
synapses, which does not fit the 2.5 GB VRAM cap) runs at 3.5 ms per step with 5% of neurons
active. That is about 120 M active neuron-steps/s, as fast per active neuron as a fully resident
population, while the GPU holds only 19% of the synapse words and PCIe carries about 10 MB per step.
It only works if (1) **neurons that are active together are stored together** (structured
connectivity plus a block-aligned layout) and (2) **the active set changes slowly** (attention
dwells; under ~5% of the working set changes per step). When either condition fails, paging degrades
gracefully to the speed of streaming over PCIe (about 11 M active neuron-steps/s), at worst ~8%
slower than zero-copy. The CUDA kernel alone, with no paging, is **~1,600–1,900× faster than the NumPy
baseline per neuron and needs no temporary memory**.

## The idea

Only a few percent of a brain's neurons are active at a time. So keep only those (plus the ones
likely to be needed next) in fast memory, and leave the rest in pinned system RAM. The gate
deciding what gets "paged in" plays the role of attention or the thalamus. This is the operating
system's virtual memory, with "page in" meaning "attend to".

Two biological echoes turned out to matter:
- **Attention dwells.** An attended object stays attended for many steps. That temporal locality is
  what makes a cache work.
- **Fire together, page together.** Neurons that are co-active must sit in the same storage block,
  or a 5% active set touches 34% of the blocks. Sleep, which already sweeps every synapse, is the
  natural moment to reorganize storage by co-activation (tested below).

## What I built

| file | what |
|---|---|
| `kernels.cu` | CUDA kernels for `forward`, `modulate`, `sleep` on the 32-bit packed words. **One warp = one neuron, one lane = one synapse slot of a branch** (32 slots = 32 lanes); each lane keeps its 8 words in registers. **No temporaries.** Every kernel takes a *pointer table* (one address per block), so the same kernel runs on words in VRAM, in a VRAM cache, in mapped host RAM (zero-copy) or in CUDA managed memory. Stochastic rounding uses a counter-based hash, so no RNG state is stored. |
| `gpu_population.py` | `GPUPopulation`: the `PackedPopulation` semantics on the GPU, with five storage modes: `resident`, `paged` (LRU VRAM cache of K blocks of 1,024 neurons, async copies on two copy streams overlapping compute in 8-block chunks, per-slot dirty flags so only changed blocks are written back), `zerocopy`, `managed` (prefetch hints), `managed_lazy` (page faults). Small per-neuron state (threshold, firing history, plasticity, gate, id: 25 B/neuron) always stays in VRAM. |
| `scenario.py` | Structured sparsity (see below). |
| `verify_kernel.py` | Kernels vs `PackedPopulation` from identical states, and speed against NumPy. |
| `test_paging.py` | All storage modes must give **bit-identical** words, thresholds and rates to `resident` (cache of 8 blocks, smaller than the working set, so eviction and thrashing paths run). They do. |
| `behavior.py` | The demo story with the neuron on the GPU (`GPUNeuron` adapter, `PackedNeuron` interface). |
| `bench.py` | Throughput benchmark. Modes run interleaved in rounds on one population and the same scenario sequence, so load from other agents hits all modes alike. |
| `pcie_micro.py`, `cpu_forward.py`, `reorder.py`, `summarize.py`, `run.py` | PCIe ceilings; CPU (numba) alternative; co-activation reordering; tables; run everything. |

**One semantic change, needed for paging:** `PackedPopulation` keeps two population-wide lazy
clocks (pending cascade flow, fast-weight decay), so `modulate` and the periodic catch-up touch
*every* synapse. I keep these clocks **per block**. A block nobody attends to is never touched, not
even by decay. The math is identical; only the timing of stochastic rounding changes (it happens
less often). Inactive neurons still get their per-step threshold and history update, from a cheap
"quiet" kernel that reads no synapses (their drive is exactly 0).

### Scenario: sparsity needs locality

As the brief warns, with the prototype's wiring (every neuron samples 64 shared inputs) every
neuron gets input every step, and no paging can help. So:

- **Columns** of 128 neurons each draw all their synapses from their own 64 inputs (same hash
  wiring: `target = column_base + hash(id, slot, gen) % 64`). The input space has N/2 inputs (4.2
  million at 8M neurons).
- **Assemblies** of 8 columns (1,024 neurons = one storage block) switch on together.
- **Attention** keeps a fraction `frac` of assemblies on; each step, each attended assembly is
  swapped for a random one with probability `churn` (0.02 means it dwells ~50 steps). Inputs of
  active columns are on with p = 0.2; all other inputs are exactly 0.
- **Layout:** *clustered* (an assembly fills one block) for the main benchmarks; *scattered*
  (columns placed at random) for the reordering experiment.

"Active fraction" means the fraction of neurons in active columns. Every step learns:
`forward` + `modulate(0.5, 0.5, 0.5)` + `tick`. `modulate` touches the blocks active in the last 4
steps, because those are the only ones with eligibility.

Machine: RTX 4070 Laptop 8 GB, PCIe 4.0 **x8**; caps of 3 CPU threads, VRAM ≤ 2.5 GB (the process
peaked at 2.04 GB), RAM ≤ 10 GB (peak RSS 8.7 GB at 8M neurons). The GPU was shared with other
agents. Times are the median of the **least-disturbed round** (time-slicing only ever adds time),
and all rounds are in `results/bench.jsonl`. The tables below come from the development runs. A final
clean `run.py` run (its output is what `results/` holds) reproduces them within ~3%.

## Results

### 1. The kernel against the NumPy baseline (dense prototype wiring, all neurons active)

Correctness, from identical states, against `PackedPopulation` (20,000 neurons):
- `forward`: max |rate difference| 3e-7 over 5 steps, 0 neurons off by >1e-4; used bits identical;
  thresholds within 1 ulp; firing history equal (99.99–100%).
- `modulate`: every new fast weight and tag lies on one of the two quantization steps next to the
  exact value, and rounding is unbiased (mean error ≤ 0.0002 steps), for NumPy and GPU alike. In the
  one-shot path, NumPy's `c0` is only 93.7% within one step (it rounds twice); the GPU's is 100%.
- `sleep`: pruned synapses −0.04% vs NumPy, mean levels equal to 4 decimals, rewired fraction
  0.2002 vs 0.2001.

| | forward | modulate | sleep | peak temporaries |
|---|---|---|---|---|
| NumPy, 100k neurons | 729 ms | 1,426 ms | 1,709 ms | 684 / 1,175 / 1,196 MiB |
| GPU, 1M neurons (resident) | 4.4 ms | 7.5 ms | 8.9 ms | **0 MB** (a few KB of block lists) |
| speed-up per neuron | **1,641×** | **1,897×** | **1,917×** | |

`forward` reaches 225 M neuron-steps/s = 58 G synapses/s, about 230 GB/s of words, which is the
card's memory bandwidth. The kernel is bandwidth-bound, as a 4-byte-per-synapse design should be.

### 2. Behavior check (16 seeds, `behavior.py`)

| response to A | hard times | day 3 | day 6 | relearn trials |
|---|---|---|---|---|
| workhorse, packed NumPy | 0.03 | 0.07 [0.05–0.10] | 0.15 [0.09–0.38] | median 2 [1–2], 16/16 |
| workhorse, **packed GPU** | 0.03 | 0.07 [0.05–0.10] | 0.15 [0.09–0.26] | median 2 [1–3], 16/16 |
| plain, packed NumPy | 0.03 | 0.03 | 0.03 | 7/16 relearned |
| plain, packed GPU | 0.03 | 0.03 | 0.03 | 10/16 relearned |

The behavior matches. (Note 14 already showed that the plain neuron's relearning count is noise:
5/16 vs 7/16 there.)

### 3. 1M neurons (fits): is paging nearly free when attention dwells?

Cache = 256 blocks (25% of the population); per-step times. Full table: `results/bench_A_1M.txt`.

| active | churn | resident fwd / mod | **paged** fwd / mod | zero-copy fwd / mod | paged PCIe in per step |
|---|---|---|---|---|---|
| 1% | 0.02 | 0.18 / 0.14 ms | **0.21 / 0.17** | 1.00 / 1.17 | 0.2 MB |
| 5% | 0.02 | 0.44 / 0.55 | **0.53 / 0.60** | 4.51 / 5.55 | 1.0 MB |
| 20% | 0.02 | 1.16 / 2.01 | **1.32 / 2.04** | 17.7 / 22.0 | 3.9 MB |
| 5% | 1.0 | 0.52 / 1.75 | 4.61 / 1.83 | 4.80 / 18.8 | 41 MB in + 41 out |
| 20% | 1.0 | 1.25 / 5.76 | 15.6 / 40.5 | 17.7 / 63.2 | 155 + 155 MB |
| 100% | – | 5.0 / 9.2 | 77 / 78 | 87 / 101 | 805 + 805 MB |

With dwelling attention, **paging costs 10–20% over resident while holding a quarter of the words.**
With random attention (churn 1.0), every step is PCIe-bound: paging and zero-copy both land at
about 11–13 M active neuron-steps/s, 9–12× slower than resident.

### 4. 8M neurons (8.6 GB, does not fit the cap)

Cache = 1,536 blocks (1.5 GiB, 19% of the words). Throughput counts all 8.4M neurons as stepped
(inactive ones get the quiet update).

| active | churn | **paged** fwd / mod | M neuron-steps/s (all) | M active/s | PCIe per step (fwd) | zero-copy fwd / mod |
|---|---|---|---|---|---|---|
| 0.1% | 0.02 | 0.88 / 0.21 ms | 9,561 | 9 | 0.2 MB | 1.49 / 0.98 |
| 1% | 0.02 | **1.41 / 0.93** | **5,953** | 60 | 1.9 MB | 7.85 / 8.95 |
| 5% | 0.02 | **3.55 / 4.14** | **2,361** | **118** | 9.3 MB | 35.6 / 44.0 |
| 20% | 0.02 | 33.0 / 34.6 | 254 | 51 | 244 + 244 MB | 141 / 175 |
| 5% | 1.0 | 35.2 / 38.6 | 238 | 12 | 359 + 359 MB | 37.8 / 151 |
| 20% | 1.0 | 131 / 365 | 64 | 13 | 1.37 + 1.37 GB | 144 / 504 |

For scale: NumPy would need about 61 s and ~57 GiB of temporaries to forward 8.4M neurons once (dense
wiring). Paged, at 5% active, a step takes 3.5 ms. At 20% active the working set (1,638 blocks)
exceeds the cache (1,536 blocks). Paging still runs 4.3× faster than zero-copy. It degrades
gracefully rather than falling off a cliff, because cache hits are processed first and evictions
then take this step's already-computed blocks (MRU-like behavior inside a step).

### 5. Where paging stops paying off (8M, 5% active; `D_churn`, `E_cache`)

**Churn** (fraction of the working set replaced per step), cache 1,536 blocks:

| churn | 0 | 0.005 | 0.05 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| paged forward | 3.5 ms | 2.8 | 4.2 | 9.2 | 18.5 | 35.2 |
| zero-copy forward | 35.8 | 35.8 | 36.1 | 37.1 | 38.3 | 37.8 |
| **paged advantage** | **10×** | **13×** | **8.7×** | **4.0×** | **2.1×** | **1.07×** |

**Cache size** (working set W = 410 blocks; modulate needs the last 4 steps, ≈ 1.06 W at churn 0.02):

| cache (blocks) | 1,536 | 1,024 | 512 | 448 | **384** | 256 | 128 | (zero-copy) |
|---|---|---|---|---|---|---|---|---|
| forward, churn 0.02 | 4.4 ms | 3.7 | 3.6 | 3.6 | **8.5** | 20.4 | 31.7 | 35.6 |
| forward, churn 0.2 | 10.2 | 9.9 | 17.7 | 24.9 | 30.9 | 37.4 | 40.1 | 37.1 |

The crossover rules, all measured:
- **Cache ≥ ~1.1 × the working set over the eligibility window** (4 steps): `W × (1 + 3 × churn)`.
  Below that, speed falls smoothly to the zero-copy level (at W/3 and churn 0.02 it is still 12% faster than zero-copy; at W/3 and churn 0.2 it is 8% slower).
- **Per step, paging costs** ≈ compute + `churn × W × 1 MB / ~10 GB/s` (pages come in and dirty pages
  go out at the same time on the two copy engines). **Zero-copy costs** `W × 1 MB / 12 GB/s` whatever
  the churn. So paging is better at any churn below 1, and equal at churn 1 (every step a new world).
- **Against resident** (when the population fits): paging adds 10–20% at churn ≤ 0.02 and is ~10×
  slower at churn 1. Resident is the right choice whenever it fits *and* attention jumps around.
- **Fixed cost per step:** 0.88 ms at 8M neurons even with almost nothing active. 0.68 ms of it is
  the O(N) quiet update. At 1% active this is half the step time (see "next").

### 6. The other strategies

| strategy (8M unless noted) | 5% active, churn 0.02 | 5% active, churn 1.0 | notes |
|---|---|---|---|
| **explicit paging** (DMA, LRU) | **3.55 ms** | **35.2 ms** | best everywhere |
| zero-copy (kernel reads mapped RAM) | 35.6 | 37.8 | 0 VRAM for words; reads the working set every step |
| managed + prefetch hints, residency budget 1,536 | 3.70 | 94 | driver migration ~2.7× slower than DMA at high churn; sleep 1.77 s vs 0.85 s |
| managed, preferred location = host | 37.8 | 41.2 | = zero-copy |
| managed, no hints, page faults (1M, 256-block budget) | 0.75 | 9.4 | vs paged 0.53 / 4.6 ms: faults ≈ 2× slower than explicit copies |
| CPU, numba, 3 threads, words stay in RAM | 397 ms (forward) | – | 1.06 M active neuron-steps/s: GPU zero-copy is 10× faster, warm paging ~110× |

PCIe ceilings on this machine (`pcie_micro.py`): DMA 12.2 / 12.8 GB/s at 1 MB copies (H2D / D2H),
21.6 GB/s both ways at once, 9 GB/s at 64 KB copies, 1.8 GB/s at 4 KB. Zero-copy kernel reads reach
12.3 GB/s, as fast as DMA, when each warp keeps 8 loads in flight.

### 7. Sleep (consolidation over every synapse)

| | 1M neurons | 8M neurons |
|---|---|---|
| resident | 0.010 s (109 M neurons/s) | – |
| paged (stream all blocks through the cache) | 0.118 s | 0.85 s (7.8 GB in + 8.6 GB out) |
| zero-copy (kernel works in host RAM in place) | 0.104 s | 0.86 s |
| managed + prefetch | – | 1.77 s |
| NumPy (extrapolated from 1.7 s / 100k) | 17 s | 143 s and ~100 GB of temporaries |

Sleep is a single sweep, so it is PCIe-bound (~10 GB/s each way), and zero-copy is as good as paging
here: it needs no cache. An 8M-neuron night takes under a second.

### 8. Fringe: "neurons that fire together page together" (`reorder.py`, 1M neurons)

Columns start **scattered**. During a "day" of 3,000 steps we log which columns are active. At
sleep we compute co-activation (Jaccard similarity from a GPU matmul of the activity log), merge the
strongest pairs with a cap of 8 columns (one block), and bin-pack the clusters into blocks. Columns
never seen active keep their current order. Then the columns are **moved**: words plus per-neuron
state. On the next day (a new attention sequence over the same assemblies), the blocks touched per
step are:

| scenario | active neurons | ideal | scattered | **learned** | learned / ideal | gain |
|---|---|---|---|---|---|---|
| 1% active (only 47% of columns seen in the day) | 10,240 | 10 | 76 | 50 | 5.0 | 1.5× |
| 5% active | 52,224 | 51 | 342 | **65** | 1.28 | **5.3×** |
| 20% active | 209,920 | 205 | 852 | **205** | 1.00 | 4.2× |
| 5% active + 10% random noise columns | 57,208 | 89 | 374 | **99** | 1.11 | 3.8× |
| 5%, churn 0.2, 10% noise | 57,198 | 89 | 371 | **89** | 1.00 | 4.2× |

Paging with the same 256-block cache (5% active, 10% noise): **forward 21.6 → 3.7 ms (5.8×),
modulate 22.2 → 1.4 ms (16×), PCIe 441 → 31 MB per step.** (An earlier run, under heavier GPU
contention, gave 47 → 10 ms.) The move costs one sleep sweep (0.10 s) plus a host-side permutation
(0.18 s per GiB). It must happen **right after sleep**, because sleep is when every
block's lazy clocks are caught up and equal, so whole columns can change blocks. After moving,
every neuron gives a bit-identical rate. Clustering takes ~1 s for 8,192 columns. Assemblies the
day never saw stay scattered, so coverage grows night by night.

### Bytes per neuron

| | VRAM | RAM | temporaries per step |
|---|---|---|---|
| NumPy baseline | – | 1,041 | ~7,200 (forward) to ~12,500 (sleep) |
| resident | 1,024 words + 25 state + 6 input history = **1,055** | 0 | < 100 KB in total |
| paged, 8M, cache 1,536 blocks | 192 words + 31 = **223** | 1,024 (pinned) | < 100 KB |
| zero-copy | **31** | 1,024 (pinned) | < 100 KB |

At 25 B/neuron, the state that never pages is what limits the scale. Under a 2.5 GB cap it allows
~60M neurons, so making that state pageable or lazy is the next lever.

## Surprises

1. **Partial-line writes to host RAM over PCIe are 10–40× slower than full lines.** With learning on,
   `forward` sets scattered "used" bits. Zero-copy `forward` ran at 65 ms instead of 5 ms, and
   `modulate` at 160–190 ms instead of 19 ms, until the kernels wrote whole 128-byte lines whenever
   any lane changed (`__any_sync`). The cost in VRAM is negligible.
2. **`forward` dirties pages.** Because the `used` flag lives inside the synapse word, a block that
   was only read still has to be written back on eviction. Moving `used` to a separate 32 B/neuron
   bitmap would make `forward` read-only on the 1 KB.
3. **The eligibility window enlarges the working set:** `modulate` needs the last 4 steps' blocks,
   so at churn 1 its working set is 4× that of `forward`, and that is where high churn hurts most.
4. **Silent neurons are not free.** Homeostasis and firing history change even with zero input. The
   per-neuron "quiet" update is O(N) (0.68 ms at 8M) and is half the step time at 1% activity.
5. **Overrunning the cache did not cause a thrashing cliff** (it degraded smoothly), and **zero-copy
   reads were as fast as DMA**.
6. The CPU is not a serious alternative for this neuron: hashing 256 targets per neuron costs
   ~11 ns per synapse per core, so 3 threads give ~1 M active neuron-steps/s.

## What I'd try next

- **Lazy quiet neurons:** a per-block "quiet since" clock and a jump table for the threshold
  recurrence (it is a 1-D map), so a step costs O(active), not O(N). Expected ~2× at 1% active.
- **Move `used` out of the word** so forward-only blocks are clean and evict for free.
- **Prefetch with look-ahead:** let attention announce the next target one step early (as saccade
  planning does) and copy it in during the current step's compute.
- **Smaller pages** (128 KB, one column): DMA still runs at ~11 GB/s, and the scattered-layout
  penalty would shrink.
- **Fuse reordering into the sleep sweep on the GPU** (write each block back to its new place), and
  run it nightly so coverage grows.
- **Lazy sleep** for blocks untouched all day (their night is linear except pruning), and an **SSD
  tier** below RAM for blocks dormant for days. The pointer-table kernels already allow any tier.

## Run

```
.venv/bin/python prototype/experiments/vram-ram-paging/run.py           # everything, ~15-20 min
.venv/bin/python prototype/experiments/vram-ram-paging/run.py --quick   # checks + 1M bench + reorder
.venv/bin/python prototype/experiments/vram-ram-paging/summarize.py     # tables from results/bench.jsonl
```

Each step's output is in `results/<step>.txt`; benchmark rows are in `results/bench.jsonl`. No large
files are left behind (the 8.6 GB populations live only in pinned RAM while a benchmark runs).

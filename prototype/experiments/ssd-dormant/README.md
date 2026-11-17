# ssd-dormant: dormant neurons live on the NVMe SSD

**Verdict: mixed.** The SSD works well as the home of the *sleeping and dormant* brain. It is a
poor home for the *waking* brain unless a RAM cache holds the hot working set. Two design rules
came out of this, and they matter more than any single speed number:

1. **Never write to the SSD while awake.** The day's learning stays in RAM and goes to the SSD
   only at night, as one sequential write.
2. **Dormant neurons skip their nights.** Because the cascade is linear, any number of skipped
   nights collapses into one 3×3 matrix that is applied when the neuron is recalled. A night then
   only costs I/O for what was touched that day.

## The idea

Some neurons aren't always firing, so they can rest somewhere slower. Hypothesis: **waking access
is sparse and scattered**, which is bad for an SSD. **Sleep is a sequential sweep over the whole
brain**, which is what an SSD is best at. So the SSD may be a natural home for the sleeping brain
and a poor one for the waking brain.

The biology lines up with the design that resulted. The hippocampus holds today's changes (RAM,
dirty blocks). Sleep consolidates them into cortex (the SSD, one write per night). Old memories are
not erased, they just become slow to reach (dormant on the SSD). Recalling a memory makes it plastic
again (reconsolidation).

## What I built (all in this folder)

| file | what |
|---|---|
| `aio.c` → `_aio.so` | Linux native AIO through raw syscalls (no libaio). One CPU thread keeps up to *QD* O_DIRECT requests in flight |
| `common.py` | thread caps, AIO wrapper, pre-faulted aligned buffers, page-cache eviction, SSD-write ledger |
| `kernels.py` | Numba kernels that work on the packed 32-bit words: `k_forward`, `k_modulate`, `k_sleep`. Same bit layout, hash wiring, stochastic rounding and maths as `packed_neuron.py`, with counter-based RNG so results don't depend on thread scheduling |
| `gpu_night.py` | the same night kernel in CUDA (cupy). Its output is bit-identical to the Numba kernel |
| `brain.py` | the on-disk brain: 25,165,824 neurons × 256 synapses = **24 GiB** (6.4 G synapses, 3× my 8 GB RAM cap), neuron *i* at offset *i*·1 KiB, columns of 1,024 neurons (1 MiB) |
| `drive.py`, `span.py` | drive characterization |
| `wake.py` | cold waking throughput: active fraction × locality × method |
| `day.py` | one day with a 4 GiB RAM cache of hot 64 KiB blocks, then a lazy night |
| `night.py` | sleep as a streaming sweep, with read-ahead and write-behind threads (CPU or GPU compute) |
| `dormant.py` | zstd dormancy and recall latency |
| `behavior.py` | behavior check: `SSDNeuron` adapter (eager, and lazy + dormant on SSD) run through `demo_single_neuron.run` |
| `run.py` | runs everything: `.venv/bin/python prototype/experiments/ssd-dormant/run.py` (about 25 min, writes about 115 GB, deletes the brain at the end; `--keep` keeps it) |

**Making sure we measure the SSD, not RAM.**
- Everything uses O_DIRECT with page-aligned buffers.
- The memmap runs call `madvise(DONTNEED)` and `posix_fadvise(DONTNEED)` after every batch.
- The data folder is `chattr +C`: no btrfs copy-on-write, no transparent zstd. The filesystem is btrfs with `compress=zstd:1`, which would otherwise flatter the numbers.
- Pitfall check: a buffered read of the same 2 GiB runs at 4.1 GB/s cold and 9.1 GB/s warm (that second number is RAM, not the SSD). `mincore` confirms it is 0 GiB cached after `DONTNEED`.

**Workload scenario** (per the brief's "needs locality" point):
- Each 1,024-neuron column reads its own 4,096-input region, with 3% of inputs active.
- A step evaluates only the **active** neurons (event-driven).
- One step = forward (with traces) + modulate (three-factor update) on each active neuron.
- Locality comes in three kinds:
  - **scattered**: uniform random neurons;
  - **columnar**: 25% of the neurons in 4× as many columns, a sparse code inside active columns;
  - **clustered**: whole columns active.

## 1. The drive (WD_BLACK SN770 2 TB, PCIe 4, DRAM-less)

Files: the 24 GiB brain (reads) and a 2 GiB scratch file (random writes). Median of 3 runs.

| test | result |
|---|---|
| sequential write (1 GiB chunks of 4 MiB requests, QD 8, whole 24 GiB) | **4.29 GB/s** median, 4.10 min, no slowdown across 24 GiB |
| sequential read, 1 MiB requests, QD 1 / 4 / 16 | 4.58 / **5.06** / 5.06 GB/s |
| random read 4 KiB, QD 1 / 4 / 16 / 64 / 256 | 12.5k / 47k / 161k / 366k / **435k IOPS** (1.8 GB/s max). QD1 latency p50 64 µs, p99 175 µs |
| random read 64 KiB, QD 1 / 4 / 16 / 64 | 8.7k / 32k / 65k / 71k IOPS = 0.57 / 2.1 / 4.3 / **4.6 GB/s** |
| random read 1 MiB, QD 1 / 4 / 16 | 2.4k / 4.4k / 4.5k IOPS = 2.5 / **4.6** / 4.7 GB/s |
| 4 KiB reads from 3 blocking Python threads (QD 3) | 41k IOPS |
| random write 4 KiB QD 1 / QD 32; 64 KiB QD 32; 1 MiB QD 16 | 44k / 108k IOPS; 1.07 GB/s; 1.63 GB/s (into a freshly allocated file, which on btrfs means unwritten-extent conversion, so these are pessimistic) |
| **mixed**: read and write back the same data, sequential | **1.71 GB/s each way (3.4 GB/s total)** |
| QD1 4 KiB latency vs address span: 64 MiB → 24 GiB | p50 53–59 µs → 62–71 µs (small effect) |

Takeaways:
- 4 KiB random reads deliver only 35% of sequential bandwidth, even at QD 256.
- 64 KiB reads already reach 92% of sequential.
- QD1 is 30× slower than high QD, so you must keep many requests in flight.

## 2. Waking from the SSD, cold (every byte comes from the drive)

Neuron-steps/s for forward + modulate, Numba, 3 compute threads, 1 I/O thread.
- `aio-*` = explicit O_DIRECT block reads at QD 64, double-buffered so reads overlap compute.
- `scan` = stream the whole brain each step.
- `mmap-*` = `np.memmap` (copy-on-write).
- `*` = time-capped and extrapolated.

| active | locality | RAM | aio-4K | aio-64K | aio-1M | scan | mmap-random | mmap-normal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | scattered | 892k | **297k** | 68k | 36k | 36k | 33k | 34k |
| 1% | columnar | 1,022k | 386k | 649k | **709k** | 37k | 56k | 195k |
| 1% | clustered | 1,529k | 519k | 714k | **775k** | 36k | 207k | 481k |
| 5% | scattered | 1,445k | **380k** | 174k | 182k | 188k | 31k* | 123k |
| 5% | columnar | 1,334k | 485k | 684k | **749k** | 180k | 51k* | 204k |
| 5% | clustered | 1,301k | 936k | 804k | **1,089k** | 182k | 170k | 448k |
| 20% | scattered | 1,291k | 455k | 497k | 565k | **712k** | 42k* | 145k* |
| 20% | columnar | 551k | 444k | **686k** | 587k | 635k | 45k* | 296k* |
| 20% | clustered | 863k | 900k | **1,024k** | 732k | 701k | 147k* | 344k* |

The RAM column is the same kernel on neurons already in RAM. Other agents were loading the CPU
(load average 6–8), so it swings between 0.55 and 1.5 M. The more robust measure is the **share
of the SSD run spent waiting for I/O**, for the best method in each row:

| active | scattered | columnar | clustered |
|---|---|---|---|
| 1% | 57% I/O wait (2.3× vs own compute), aio-4K at 292k IOPS | 28% (1.4×) | 22% (1.3×) |
| 5% | 52% (2.1×), 352k IOPS | 7% (1.08×) | 7% (1.08×) |
| 20% | 29% (1.4×), whole-brain scan at 3.6 GB/s | 8% (1.09×) | 2% (1.02×) |

- **With 3 CPU threads, the CPU is the bottleneck, not the SSD.** Waking costs only 1.0–1.4× when activity has locality, and 2–3× when it is scattered.
  - Scattered waking is IOPS-bound at about one neuron per 4 KiB read. **Ceiling ≈ 0.35–0.43 M neuron-steps/s, however fast the compute.**
  - Clustered waking is bandwidth-bound. **Ceiling ≈ 4.9 M neuron-steps/s** (5 GB/s ÷ 1 KiB).
  - A GPU computing from VRAM (~256 GB/s, so roughly 100 M+ neuron-steps/s) would make the SSD **20× (sequential) to 250× (scattered) slower** than VRAM. That is the honest size of the slowdown for a fast brain.
- **Block size should follow locality.** 4 KiB blocks for scattered access (4× read amplification, since one neuron is 1 KiB). 64 KiB–1 MiB blocks for columnar or clustered access (1× amplification, near full bandwidth).
- **Crossover.** Somewhere between 5% and 20% *scattered* activity (estimated 8–10% from the IOPS and bandwidth figures), reading the whole brain sequentially (7 s per step) beats random reads.
- **np.memmap is 2–10× worse than explicit blocks, for two reasons:**
  - With `MADV_RANDOM`, every page fault is a synchronous 4 KiB read, and the queue depth equals the number of faulting threads (3). Result: 30–56k neuron-steps/s, about 10× below AIO.
  - With default readahead (4 MiB on this btrfs), scattered access reads the **whole brain** (100× amplification). It is fine for clustered access, but still below AIO-1M.
  - memmap also uses page cache outside your RAM budget.
- **Pitfall found:** AIO into a never-touched buffer ran at **14k IOPS instead of 344k**, because `io_submit` faults the destination pages in one at a time. DMA buffers must be pre-faulted (`MAP_POPULATE`) and reused.

## 3. A day with a RAM cache of hot blocks (`day.py`)

- Cache: 4 GiB of 64 KiB blocks, least-recently-used eviction, clean blocks evicted first.
- **Blocks touched today stay in RAM, dirty, until the night. Their share of the cache is the sleep pressure** (note 09: sleep pressure builds as the short-term buffer fills).
- A day is 10 episodes of 20 steps. Each context is 1,024 columns: 896 from a 2 GiB habitual pool plus 128 novel. In each step, 25% of the neurons in the context are active (≈1% of the brain).

| | result |
|---|---|
| block hit rate over the day | **98.4%** (3.35 GB read from the SSD in 200 steps) |
| overall | **864k neuron-steps/s**, essentially RAM speed |
| steps within an episode | 100% hits, 240–350 ms per step |
| first step of an episode (context switch) | 0.9 s cold, then 0.3–0.5 s as the habitual pool warms (reads 1,067 MB → 126 MB) |
| sleep pressure at bedtime | 3.12 GiB dirty = 78% of the cache, no spills |
| **lazy night** (only the dirty blocks, already in RAM) | 2.2 s compute + 0.8 s write-back (4.0 GB/s) = **3.0 s**, writing 3.3 GB. The other 21.9 M neurons skip the night |

So with temporal locality, the SSD costs almost nothing awake. **It shows up as a pause at context
switches**: familiar contexts are instant, and new or long-unused ones take a moment to load.

## 4. Sleep as a streaming sweep (`night.py`)

Setup: a ring of 4 × 128 MiB buffers. A reader thread reads ahead with 16 MiB requests, the main
thread computes, and a writer thread writes behind. All three overlap.

| variant | data | wall | throughput | busy time read / compute / write |
|---|---|---:|---|---|
| read only | 4 GiB | 1.6 s | 2.6 GB/s | 1.6 / – / – |
| I/O only (read + write back) | 4 GiB | 2.5 s | 1.71 GB/s each way | 1.5 / – / 2.5 |
| serial, CPU | 4 GiB | 8.6 s | 0.49 M neurons/s | 1.5 / 5.2 / 1.9 |
| pipelined, CPU | 4 GiB | 5.6 s | 0.74 M neurons/s | 1.4 / 5.5 / 1.5 |
| pipelined, GPU compute | 4 GiB | 2.5 s | 1.67 M neurons/s | 1.5 / 2.2 / 2.4 |
| **FULL NIGHT, CPU (3 threads)** | **24 GiB** | **39.6 s** | 0.64 M neurons/s = 0.16 G synapse-updates/s | 8.4 / **39.5** / 9.9 |
| **FULL NIGHT, GPU compute** | **24 GiB** | **15.2 s** | 1.65 M neurons/s = 0.42 G synapse-updates/s | 9.0 / 11.3 / **15.1** |
| lazy night, 5% of columns touched | 1.2 GiB | 2.9 s | | |
| lazy night, 20% of columns touched | 4.8 GiB | 7.6 s | | |

**Correctness.** The night on the SSD is **bit-exact** with the same night computed in RAM
(checked on 2,000 random neurons, for both the CPU and GPU runs). The CUDA kernel's output is 100%
word-identical to the Numba kernel's.

**Is it I/O- or compute-bound?**
- **On 3 CPU threads it is compute-bound.**
  - The night kernel did 0.4–1.5 M neurons/s depending on machine load: 1.5 M during the day test, 0.39–0.64 M when other agents were busy.
  - So the night takes 17–65 s of compute, against about 15 s of I/O.
- **With the GPU doing the maths, it is I/O-bound.**
  - 15.2 s per night, at the drive's mixed read+write ceiling of 3.4 GB/s.
  - The writer is the busy stage (15.1 of 15.2 s).

**The hypothesis holds.** A sequential sweep is the SSD's best case: 3.4 GB/s combined, about 1.7
M neurons/s. Scattered 4 KiB access gets 1.4 GB/s at best and only one useful neuron per read.
- A whole night for a 24 GiB brain takes **15 s** (GPU) to **40 s** (CPU).
- Pipelining beats serial by 1.5× on CPU. On GPU it brings the night down to the I/O-only time.

**But the bigger win is not sweeping.** The lazy night's cost scales with what was touched: 3.0 s
in the day test, and 2.9 s for 5% of columns, against 39.6 s for everything.
- Dormant neurons don't need the night: their maintenance is linear and can be applied on recall.
- This only helps with locality. With 5% *scattered* activity, almost every 64 KiB block gets touched.

**SSD wear.** A full sweep writes 25.8 GB per night. The drive is rated for 1,200 TBW, which is
about 46,000 full nights. A lazy night after the test day wrote 3.3 GB, 8× less.
- Writing waking changes through to the SSD would be ruinous. At 1% scattered activity with 4 KiB pages, that is about 1 GB per *step*, which would wear the drive out in about 1.2 M steps.
- Hence rule 1: write only at night.

**Fresh-brain caveat:** that first night regrew 98% of synapses. In a never-used brain, every
synapse is weak after 1,100 cascade-flow steps and none are tagged or used. This changes nothing
about the I/O.

## 5. Dormancy and recall (`dormant.py`)

**Compression.** A 64-column population lived for 10 days first (sparse activity, learning, nights), so its words look like used neurons. 1 KiB per neuron raw. Speeds are single-thread.

| | ratio | bytes/neuron | compress | decompress |
|---|---:|---:|---:|---:|
| newborn neurons, per-column frame, zstd-3 | 7.4× | 139 | 230–500 MB/s | 0.4–0.9 GB/s |
| aged, per-column frame, zstd-1 / 3 / 9 / 19 | 3.2 / **3.4** / 3.5 / 4.0× | 320 / **304** / 297 / 254 | 180–356 / 178–337 / 35–52 / 2–3 MB/s | 0.8–1.7 GB/s |
| aged, byte-shuffled + zstd-3 | 3.5× | 292 | | |
| aged, lz4 | 1.9× | 544 | | 1.2–2.6 GB/s |
| aged, **one frame per neuron**, zstd-3, no dictionary / **16 KiB trained dictionary** | 2.35× / **3.1×** | 435 / **331** | | 0.15–0.31 / 0.34–0.69 GB/s |

**Recall latency, cold from the SSD.**
- Each recall includes **30 skipped nights applied at once**: one 3×3 matrix, composed in about 0.6 ms, then pruning and tag fade. Then one forward pass.
- Medians of 200 recalls. Measured while the machine was heavily loaded, so the compute parts are about 2× slower than on a quiet machine.

| recall | p50 total | read | unzip | catch-up + fire |
|---|---:|---:|---:|---:|
| awake neuron already in RAM, fire only | 10 µs | | | 10 µs |
| neuron in RAM, catch up 30 nights + fire | 35 µs | | | 35 µs |
| **1 dormant neuron**, raw, anywhere in 24 GiB | **0.23 ms** (p99 1.5 ms) | 155 µs | | 68 µs |
| 1 dormant neuron, raw, within a 0.35 GB span | 0.15 ms | 74 µs | | 69 µs |
| 1 dormant neuron, its own zstd + dictionary frame | 0.17 ms | 91 µs | 8 µs | 69 µs |
| 1 dormant neuron, through its column's zstd frame | 2.4 ms | 340 µs | 2.0 ms | 109 µs |
| **1 dormant column** (1,024 neurons), raw 1 MiB | **4.3 ms** | 0.6 ms | | 3.7 ms |
| 1 dormant column, zstd frame (~0.3 MiB) | 6.2 ms | 0.4 ms | 1.6 ms | 3.8 ms |
| **batch: 256 random columns** (256 MiB, 262k neurons), raw, QD 16 | **0.79 s** | 78 ms | | 0.71 s |
| same batch, zstd, 3 decompression threads | 1.13 s | 41 ms | 355 ms | 0.74 s |

**What this means for latency:**
- **The SSD is not what makes recall slow.** Reading is 10–20% of the time. Most of it is catching up the skipped nights (CPU).
- **zstd makes recall slower**, because decompressing takes longer than reading the uncompressed data. It saves 3.4× in space and SSD writes.
- So: keep recently dormant neurons raw, and compress only deep-dormant ones.
  - For single-neuron recall, use per-neuron frames with a trained dictionary.
  - For whole columns, use per-column frames.
- RAM cost of a dormant neuron: **0 bytes of synapses**, plus 25 bytes of soma state (that could move to the SSD too).

**What it suggests for behavior (fringe):** say a waking step is about 10 ms.
- **A single old detail comes back within a step** (about 0.2 ms).
- **An old skill or column takes about half a step** (4–6 ms).
- **A whole long-unused region takes about a second** (256 MiB in 0.8–1.1 s, so about 100 steps). That is a "tip of the tongue" pause that grows with how much has to come back.
- **Recalled memories come back aged.** All the skipped nights are applied at once (cascade flow, downscaling, pruning of synapses that stayed unused), so the memory returns as it would have been after those nights.
- **Recall is reconsolidation.** A recalled neuron is in RAM and plastic again. If it learns, it becomes dirty and gets a real night.
- **Dormancy is forgetting without erasing.** Nothing is deleted; old memories just get slower to reach. This matches "new learning suppresses old learning rather than erasing it".
- **Familiar contexts are instant** (cached); new or long-unused ones bring a moment of hesitation (section 3).

## Behavior check

`SSDNeuron` runs the Numba kernels on raw packed words. The "lazy + dormant" variant:
- skips every night with no activity;
- after the first skipped night, zstd-writes its words to the SSD and drops them from RAM;
- on the next probe, recalls them (bit-exact round trip, checked with a checksum) and applies all skipped nights as one matrix.

The demo's 6 quiet days are therefore spent dormant on the SSD.

| seeds 1–8: response to A | hard times | day 3 | day 6 | relearn trials (median) |
|---|---|---|---|---|
| workhorse, packed (baseline) | 0.03 | 0.07 [0.06–0.10] | 0.15 [0.09–0.38] | 2 [1–2] |
| workhorse, our kernels, eager nights | 0.03 | 0.07 [0.06–0.09] | 0.14 [0.09–0.23] | 2 [1–3] |
| workhorse, our kernels, **lazy nights + dormant on SSD** | 0.03 | 0.07 [0.05–0.08] | 0.12 [0.09–0.17] | 2 [2–3] |

- **Seeds 1–32, day-6 mean ± SEM:** packed 0.156 ± 0.014, eager 0.152 ± 0.012, lazy + dormant 0.137 ± 0.009. Relearning median is 2 for all. The behavior matches.
- **Dormant neurons recover their old self slightly less** (−0.02, not significant), because weak synapses are pruned and regrown once on recall instead of every night.
- **Known difference, plain neuron (no deep memory):**
  - It relies on lucky rewiring to relearn.
  - Seeds 1–8, relearn trials: packed 400 [15–400], eager 18 [14–400], lazy 400 [106–400]. All very noisy.
  - Fewer pruning opportunities while dormant means fewer lucky rewires. This is deliberate: no maintenance happens while dormant.

## Resources and numbers in the brief's format

| | value |
|---|---|
| bytes per neuron on the SSD | 1,024 (raw), 304 (zstd-3 column frames, aged), 331 (per-neuron frames with dictionary) |
| RAM per neuron, cold or dormant | 25 B of soma state (gate, threshold, plasticity, 4-step firing history in float32; 17 B with float16) |
| RAM per neuron, hot | 1,024 + 25 B, in a cache of fixed size (4 GiB holds 4.2 M hot neurons) |
| VRAM | GPU night only: one 128 MiB chunk buffer plus the CUDA context (cupy pool capped at 768 MiB) |
| peak temporaries | waking staging 2 × 256 MiB; night ring 4 × 128 MiB; **constant, whatever the brain size** |
| peak RSS | wake 3.75 GB (includes a 2 GiB in-RAM reference sample), day 4.9 GB (4 GiB cache), night 1.6 GB |
| forward + modulate, cold from SSD | 0.30 M (scattered) – 1.09 M (clustered) neuron-steps/s |
| forward only / forward + modulate, RAM, 3 threads | 2.5–4.0 M / 0.76–0.94 M neuron-steps/s |
| sleep, whole 24 GiB brain from SSD | 0.64 M neurons/s on CPU (39.6 s per night); 1.65 M on GPU (15.2 s, I/O-bound) |
| NumPy baseline (100k neurons, RAM) for comparison | forward 0.14 M/s, modulate 0.09 M/s, sleep 0.07 M/s |
| SSD writes by this experiment | 133.5 GB in total (ledger), including one aborted night run. A clean full run is about 115 GB |
| CPU | 3 Numba threads; I/O threads mostly blocked; measured average ≤ 3.0 cores |

The SSD tier runs **250× more neurons than the NumPy baseline** (25 M vs 100k) in less than 5 GB
of RAM. Because the kernels are Numba instead of NumPy, it is still 5–20× faster per neuron than
the RAM baseline.

## Surprises

1. **On 3 CPU threads, compute is the bottleneck, not the SSD.** The drive can feed about 5 M neurons/s; the kernels manage about 1 M (waking) and 0.4–1.5 M (sleeping). The SSD only really hurts for scattered access or with a GPU.
2. **AIO into never-touched pages is 25× slower** (14k vs 344k IOPS).
3. **Reading and writing at once roughly halves each direction:** 1.7 + 1.7 GB/s, against 5.0 for reads alone and 4.3 for writes alone. That sets the floor for a night.
4. **memmap's 4 MiB readahead** turns 1% scattered activity into reading the whole brain.
5. **Recall is limited by compute, not the SSD**, and compression slows recall down.
6. Newborn neurons compress 7.4×; aged ones only 3.4×. Generations get randomized by regrowth and cascades fill in.

## What I'd try next

- **A faster CPU night kernel:** one 64-bit hash per synapse split into four 16-bit uniforms, float32 throughout. That should make the CPU night I/O-bound (about 15 s).
- **GPU waking with three tiers:** VRAM hot set → RAM cache → SSD. The scattered-access IOPS ceiling (about 0.4 M neuron-steps/s) is the number to design around.
- **Prefetch on context cues:** start reading a context's columns before its first step, to hide the 0.3–0.9 s pause.
- **Per-neuron clocks** (last-touched step for fast-weight fade and cascade catch-up), so sparse operation needs no global catch-up. Also move dormant neurons' soma state to the SSD (25 B → 0).
- **Tie the dirty-block fullness ("sleep pressure") to the hypothalamus's sleep drive**, and test sleep deprivation: what happens when the RAM overlay fills?
- **io_uring with registered buffers**, and a test on the second, idle NVMe drive (nvme0n1 is unmounted and needs root).

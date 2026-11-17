# Rolling local sleep: the GPU stays awake while the CPU sleeps

**Verdict: mixed.**
- **Fast CPU sleep kernel:** a clear win. It is 30–85× faster than the NumPy prototype and needs no temporaries.
- **Rolling sleep for throughput:** a dead end. The night is under 1% of the GPU's day.
- **Rolling sleep for VRAM capacity:** promising. Sleeping blocks need no VRAM.
- **Behavior:** the "old self comes back" mechanism survives, but it depends strongly on *when* a block's night falls. That turned out to be a property of the neuron model, not of the schedule.

Run everything (about 6 minutes, each step under 2 minutes):

    .venv/bin/python prototype/experiments/rolling-local-sleep/run.py            # all experiments + tables
    .venv/bin/python prototype/experiments/rolling-local-sleep/run.py --summary  # tables from results/*.json

## Idea and inspiration
Dolphins and many birds sleep one hemisphere at a time (unihemispheric sleep). Humans show
*local sleep*: small patches of cortex go "offline" while the rest is awake (Vyazovskiy et al.
2011, *Nature*). In NREM sleep the thalamus gates out the senses.

The engineering version: split the population into M blocks. At any moment S of them are asleep.
Their words sit in pinned RAM while CPU cores run the night on them. The other M−S blocks stay
awake on the GPU, running forward and modulate. The blocks rotate, so each one gets one night per day, and
the brain as a whole never goes offline.

## What I built
| file | what |
|---|---|
| `csleep.c` + `cpu_sleep.py` | The CPU night, run in place on the packed uint32 words, one synapse at a time in registers. It does catch-up of the lazy clock → consolidate → 100 flow steps (one 3×3 matrix) → downscale → prune/regrow → clear → fade tags. C/OpenMP via ctypes (the GIL is released). Built with **clang** `-O3 -march=native`, which processes 8 synapses per AVX2 instruction (gcc 15 refuses to vectorize this loop). A Numba `@njit(parallel=True)` version with the same code is also included. |
| `gpu_kernels.py` | CUDA kernels via CuPy: `forward` (one block per neuron, one warp per branch), `modulate`, `catchup`, and a GPU `sleep`. No temporaries. |
| `population.py` | `Block`: a group of neurons with **its own lazy clock** that can move between VRAM and pinned RAM (async copies). `KernelPopulation` / `KernelNeuron`: the prototype's interfaces on top of the kernels. |
| `bench_sleep.py` | Experiment 1: kernel correctness and speed, PCIe, GPU awake kernels. |
| `bench_schedule.py` | Experiment 2: sync-gpu, sync-cpu, micro-sleep and rolling (several M and S) at 1,008,000 neurons. |
| `behavior.py`, `phase_check.py` | Experiments 3 and 4: the demo story, single neuron and population, under each schedule. |

All three sleep kernels (C, Numba, GPU) use the same counter-based RNG (`lowbias32(seed + 4·index + k)`, 16-bit
uniforms) with FMA contraction off. For a given seed they are **bit-identical**, so a block can sleep on
either processor and give the same result.

## Results

Scenario: the prototype's workhorse neuron (8×32 synapses, 1,041 B/neuron) with 64 inputs (dense; every neuron
gets input every step). A "day" is D = 84 awake steps per neuron (forward every step, modulate every other
step, as in the demo), followed by one night. Thread caps: 3 OpenMP threads for sync-cpu; 2 OpenMP threads plus the
GPU-driving main thread for rolling. VRAM was capped at 2.5 GB.

Another agent was using the GPU during some runs, and up to about 6 CPU cores were busy with other agents. I report
medians of 3 runs, and I give a range where two full runs disagreed.

### 1. The CPU sleep kernel
| one night | 100k neurons | 1M neurons | peak temporaries | rate |
|---|---|---|---|---|
| NumPy prototype (`PackedPopulation.sleep`) | 2.41–2.47 s (brief: 1.5 s on a quieter machine) | (would need ~12 GB) | **1,294 MiB** | 0.04 M neurons/s |
| C kernel, 1 thread | 0.082 s (**30×**) | 0.80–0.85 s | 0.4 MiB (4 B/neuron prune counter) | 1.2 M neurons/s |
| C kernel, 2 threads | 0.042–0.045 s | 0.43–0.51 s | same | 2.3 M/s |
| C kernel, 3 threads | 0.028–0.053 s (**46–87×**) | 0.31–0.43 s | same | 2.3–3.2 M/s (0.6–0.8 G synapses/s) |
| Numba, 3 threads (not vectorized) | 0.21–0.23 s (11×) | 2.2 s | same | 0.45 M/s |
| GPU kernel (for comparison) | 0.9 ms | 8.5 ms (120 GB/s, near the bandwidth limit) | 0 | 117 M/s |
| PCIe round trip of the words (13 GB/s) | 16 ms | 160 ms | – | – |

**Correctness.**
- C, Numba and GPU are bit-identical, both with and without a pending clock.
- Against NumPy, from the same caught-up state (20k lived-in neurons): the prune and regrow decisions are identical. Every level is within ±1 quantization step (only the random rounding differs). The bias against the exact pre-rounding value is 1e-4 steps for both versions.
- With a pending clock (an extra rounding step), the means and standard deviations of c0, c1 and c2 match to 5–6 digits.

**The awake GPU kernels** run while the CPU sleeps:

| kernel | time per 1M neurons | neuron-steps/s | synapse updates/s | temporaries |
|---|---|---|---|---|
| forward | 11.0–11.5 ms | 8.7e7 | 2.2e10 | 0 (plus 4 B/neuron rate output) |
| modulate | 8.9 ms | 1.1e8 | 2.9e10 | 0 |

### 2. Schedules (1,008,000 neurons; GPU timeline measured with CUDA events)
| schedule | asleep at a time | VRAM B/neuron | pinned RAM B/neuron | wall per day | vs sync-gpu | GPU awake / sleep / idle | CPU sleep-worker busy | night latency (asleep → slept) |
|---|---|---|---|---|---|---|---|---|
| **sync-gpu** (all wake, then all sleep on the GPU) | 0 | 1,041 | 0 | 1.323–1.325 s | 1.00 | 99.3 / 0.7 / 0 % | – | – |
| sync-cpu (all wake, then RAM, CPU night, back) | 0 | 1,041 | 1,041 | 1.93–2.29 s | **0.58–0.69** | 68–72 / 0 / **28–32 %** | 24–27 % | – |
| micro-gpu (28 slices, a quick GPU night every 3 steps) | 0 | 1,041 | 0 | 1.33 s (one run 1.42) | ≈1.00 | 99.4 / 0.6 / 0 % | – | – |
| rolling M=8 S=1 +spare | 1/8 | 1,041 | 1,041 | 1.32 → 1.98 s (unstable) | 0.69–0.87 | 95–100 / 0 / 0–5 % | 56–65 % | 0.17–0.20 s (max 0.38) |
| **rolling M=8 S=2 +spare** | 1/4 | **911** | 1,041 | 1.33–1.36 s | **0.97–1.00** | 100 / 0 / 0 % | 76–81 % | 0.33 s |
| **rolling M=8 S=4 +spare** | 1/2 | **651** | 1,041 | 1.32–1.33 s | **1.00** | 100 / 0 / 0 % | 80–82 % | 0.65 s |
| rolling M=8 S=4, no spare slot | 1/2 | **521** | 1,041 | 1.48–1.54 s | 0.86–0.90 | 89 / 0 / 11 % | 83–87 % | 0.89 s |
| rolling M=2 S=1 +spare ("two hemispheres") | 1/2 | 1,041 | 1,041 | 1.32–1.53 s | 0.86–1.00 | 99–100 / 0 / 0–1 % | 57–71 % | 0.64–0.86 s |
| 3× heavier night: sync-gpu | 0 | 1,041 | 0 | 1.35 s | 1.00 | 98 / 2 / 0 % | – | – |
| 3× heavier night: rolling S=2 +spare | 1/4 | 911 | 1,041 | 1.44–2.2 s | 0.74 | 83 / 0 / **17 %** | 92 % | 0.40 s |
| 3× heavier night: rolling S=4 +spare | 1/2 | 651 | 1,041 | 1.33–1.45 s | 0.97 | 96 / 0 / 4 % | 95 % | 0.73 s |

The second full run's heavy-night rows were taken while another process held the GPU (sync-gpu read 2.65 s/day), so I
don't use them.

**Sleep coverage:** in every schedule each block got exactly one night per day, and every awake run was exactly 84 steps.
In rolling, the night *latency* (from falling asleep until the CPU has finished the night) grows with S. With S=1 the night must fit inside a
single shift (~190 ms), which a contended CPU sometimes misses, so the GPU stalls (S=1 is fragile). With S≥2 there is slack.

What this shows:
- **There is almost nothing to hide.** Thanks to the lazy-clock trick the night is a single streaming pass: 8.5 ms of
  GPU time per 1M neurons, or 0.7% of a day. Rolling can win at most that 0.7%, and the measured difference is within noise.
  The GPU sleeps **30–50× faster than 3 CPU threads**, and just moving a block over PCIe and back costs **18×** what the GPU
  night costs.
- **sync-cpu is the worst option:** the GPU idles for 28–32% of the day while the CPU sleeps.
- **The real gain is VRAM.** Sleeping blocks don't need to be resident. With half the brain asleep and one spare slot
  for prefetching, VRAM drops to **651 B/neuron at full speed**, so 1.6× more neurons fit per GB. Without the spare slot
  it drops to 521 B/neuron (2×) at 86–90% speed, because the swaps are synchronous: 8 per day × 2 copies of 126 MB. The whole population still lives in
  pinned RAM (1,041 B/neuron).
- **The limit is the CPU keeping up.** The CPU must finish one population-night per GPU day. Uncontended, 2 threads take
  ~0.44 µs per neuron-night against 1.32 µs per neuron-day on the GPU (33%). In practice, with other agents loading the
  machine, the sleep worker was busy **57–87%** of the time, and the in-situ night for a 126k-neuron block took 55–170 ms against 55 ms
  in isolation. A 3× heavier night tips rolling into GPU stalls (0.74×), while sync-gpu only pays 2%. **If the awake
  kernels get faster** (forward is at ~35% of the bandwidth limit), **the CPU becomes the bottleneck.**

### 3. Behavior: does the old self still come back?
**Kernel check** (the brief's test, `demo_single_neuron.run`, seeds 1–16): GPU awake kernels with the C night on the CPU,
compared with the NumPy `PackedNeuron`.

| | hard times | day 3 | day 6 | relearn (median) |
|---|---|---|---|---|
| workhorse, NumPy packed | 0.03 | 0.072 | 0.154 [0.09–0.38] | 2 [1–2] |
| workhorse, GPU awake + CPU sleep | 0.03 | 0.070 | **0.154 [0.09–0.26]** | **2** [1–2] |
| plain, NumPy / kernels | 0.03 / 0.03 | 0.03 / 0.03 | 0.03 / 0.03 | 400 / 400 (same seeds hit the cap) |

The kernels match the baseline (brief: 0.14–0.15, range 0.09–0.38, median 2).

**Population on a world clock** (`behavior.py`): 5 blocks × 200 neurons hear the same world; seeds 1–8. A world day is
50 slots, and every block is awake for 40 of them (40 demo trials) and asleep for 10. Every schedule therefore gives the same experience
and one night per block per day; only the **phase** of the night differs.

| schedule | day 3 | day 6 (mean) | day 6 per block (night at slot 0/10/20/30/40) | neurons > 0.06 at day 6 | relearn median (mean) | world events seen by ≥1 block |
|---|---|---|---|---|---|---|
| sync (all nights at slots 40–49) | 0.065 | **0.143** | 0.14 everywhere | 100% | 2 (1.9) | 80% |
| rolling, sleeping blocks ignore input | 0.043 | **0.083** | **0.14 / 0.03 / 0.04 / 0.06 / 0.14** | 48% | 3 (2.6) | **100%** |
| rolling, night input deferred and replayed on waking | 0.034 | 0.070 | 0.17 / 0.03 / 0.03 / 0.04 / 0.07 | 36% | 3 (2.7) | 100% |
| micro (instant nights spread through the day) | 0.036 | 0.064 | 0.04 / 0.03 / 0.04 / 0.07 / 0.14 (slots 8/16/24/32/0) | 35% | 3 (2.9) | 100% |
| **control:** sync, but life changes at mid-day (slot 20) | 0.029 | **0.034** | 0.03 everywhere | 0% | 3 (3.4) | 81% |

**Recovery is all-or-nothing, and it depends on the night's phase relative to the life change.** It is not caused by the
rolling schedule itself.
- The control (synchronous sleep, events shifted by 20 slots) loses the recovery completely.
- `phase_check.py` shows the identical effect in the **NumPy prototype**, so it is a property of the neuron model and not a kernel bug.

The mechanism, from the cascade trace in `phase_check.py` (A-synapses, after each night):
- Fast weights saturate within about 7 trials, so a night consolidates "fast ≈ ±1 × tag" no matter whether 10 or 40 trials
  came before it. Consolidation counts **nights touched by an episode**, not how much happened.
- If the 2-day "hard times" episode straddles 3 nights instead of 2, the third night (fast −0.6 at onset) pushes the middle
  level c1 negative (−0.08 instead of +0.13). From then on the deep level can't pull the surface back up.

With rolling sleep, some blocks are always misaligned. The brain ends up **heterogeneous**: half of it keeps the old self and half
of it is overwritten. Relearning stays fast everywhere (median 3 trials against 2, and against 400 for the plain neuron), so the
"savings" survive.

### 4. Fringe: input that arrives while a block sleeps
**Choice: gate it (ignore it), like the thalamus.** Reasons:
1. **Physically:** the block's words are in RAM or in flight, so processing input would need them back on the GPU.
2. **Consolidation reads a snapshot** of the fast weights and tags. Input arriving mid-night would race the night, or be
   erased by the fast-weight clear.
3. **Biology:** NREM sensory gating; local-sleep OFF periods are unresponsive.
4. **Measured:** deferring the night's input and replaying it on waking gives each block 25% more experience, but no
   better recovery (0.070 against 0.083). It smears an episode over even more nights, by the mechanism above.

What rolling buys here is that the *brain* never misses an event: 100% of world events are seen by some block, against 80%
when everyone sleeps at once.

**Open question:** should the awake half's activity *drive* the sleeping half's consolidation, like the hippocampus
"teaching" the cortex during sleep? Our night has no input path at all, so adding one would make the night input-dependent
and force it back onto the GPU. Note also that in this test the blocks don't talk to each other (they share only the
world), so "neighbors awake" only matters through that shared input. Recurrent wiring between blocks is untested.

## Bytes per neuron (summary)
| | RAM | VRAM | temporaries |
|---|---|---|---|
| stored state | 1,041 (the full population, pinned, in rolling and sync-cpu) | 1,041 × (M−S+spare)/M: 1,041 / 911 / 651 / 521 | GPU: 0 (+4 B rate buffer); CPU night: 4 B/neuron; NumPy night: ~13,000 B/neuron |

## Surprises
- **The night isn't heavy.** The premise was that sleep is heavy sequential per-neuron work. After the lazy-clock trick it
  is one memory-bound pass, and the GPU does it at 120 GB/s.
- **Compilers:** gcc 15 doesn't vectorize the night loop, but clang 21 does, which makes it 8× faster per thread. Numba
  doesn't vectorize it either (unsigned integers get widened to 64 bits).
- **CuPy's `ndarray.get(out=pinned, stream=s)` is still blocking by default.** It serialized the pipeline (5.8% GPU idle)
  until I passed `blocking=False`.
- **Consolidation depends on the night's phase**, an all-or-nothing effect that the synchronous demo hid because its day
  boundaries happened to coincide with the life changes.

## What I'd try next
1. **Lazy sleep.** The night's flow, downscale and tag fade are all linear. Keep a per-block 3×3 *product* matrix
   instead of a step count, fold the night into it, and touch only synapses with a nonzero fast weight or tag
   (1–10% under sparse activity), doing prune checks lazily on the next touch. The night would become O(changed synapses), which removes the
   CPU-keeping-up limit and makes CPU (or SSD-tier) nights cheap.
2. **Consolidate the amount learned, not the saturated fast weight**, for example a tag that integrates dopamine × eligibility.
   This should remove the phase sensitivity. A second option is **sleep pressure**: a block sleeps when its fast weights saturate,
   which naturally aligns its nights to episodes (note 09).
3. **Pair with the SSD tier.** Sleeping blocks are offline for a known time, so they are the natural thing to page out.
   Rolling plus a spare slot is exactly a prefetch pipeline.

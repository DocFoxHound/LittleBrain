# 79: Phase 0: groundwork (design stamps, game brain size, speed)

*2026-11-22. Plan note 78, phase 0.*

## 1. Design stamps on saved brains (`design.py`)
Brains made by `prepare()` carry `brain.design`: kind (`full`, `vision`, ...), a fingerprint of the whole design
(groups with cell-type parameters, connections, inputs), `RUNTIME_VERSION`, the code version (git) and the date.
`Brain.load(path, kind=...)` checks it **before** repairs of older designs:
- **refused**: a different runtime version (learned state would be misread); the wrong kind when one is expected; a
  design that no longer matches its own fingerprint (damaged or edited file).
- **warning**: saves from before stamps; a design that differs from what the current code builds for that kind
  (it still runs with its own design).
The game bridge checks an explicitly chosen kind (env `BRAIN`); the viewer rebuilds a cached brain whose stamp doesn't
match the current design (this removes the stale-cache trap of AGENTS.md §8). Bump `RUNTIME_VERSION` when step,
learning or sleep semantics change incompatibly. `test_design.py` 6/6.

## 2. A size option for the vision brain
`BRAIN=vision100k` (`brainspec.full_brain_vision_human`): human proportions (note 70) with V2, V4, IT and MT sharing the
sensory-cortex budget. **At 100,000 neurons human proportions add only cerebellar granule cells** (1,280 → 67,520):
the cortex's human share split over six visual/association maps is less per map than they already have. Visual maps
would start to grow at roughly 250,000 neurons in total.
Defaults unchanged for now: audits use `full`; the viewer and bridge keep their defaults until templates exist
(phase 9), because a vision brain takes ~4 minutes to grow at first launch.

## 3. Speed (CPU, 8 BLAS threads, one decision = 27 brain steps)
Profiling showed ~40% of the time rebuilding every synapse weight every step, and much of the rest in whole-array
learning updates. `fastkernels.py` compiles the lean synapse bank's drive, eligibility traces and learning update
(numba, all cores, one pass per cell, no temporary arrays).

| brain | neurons | synapses | before | **after** |
|---|---:|---:|---:|---:|
| full | 18,944 | 1.8 M | 0.69 s | **0.48 s** |
| vision | 30,592 | 5.5 M | 1.98 s | **0.86 s** |
| human100k | 100,032 | 8.2 M | 2.14 s | **0.80 s** |
| vision100k | 100,032 | 9.6 M | 2.75 s | **1.13 s** |
Growing a brain: reference 44 s, vision ~4 min (was 6-7).
**The vision brain meets the ≤ 1 s per decision target.** Minecraft's tick rate can be lowered for the rest.

### Is it the same brain?
- Compiled vs numpy on every lean synapse bank: drive within 1.6e-7 (relative), eligibility, input traces and tags
  exactly equal, fast weights within 1.3e-6 (summation order). `test_fast_kernels.py` 2/2 (also: a 40-step learning run
  correlates 1.000000 with the numpy run).
- `LITTLEBRAIN_FAST=0` runs the original numpy code: **bit-identical** to before (60-step fixed-seed run).
- Behaviour, reference brain, signal game, 3 seeds: 64 turns 0.63 / 0.95 / 0.77 compiled vs 0.73 / 0.94 / 0.80 exact
  (early learning is chaotic: tiny rounding differences send runs down different paths); **200 turns: 0.97 / 0.98 / 0.98
  compiled vs 0.98 / 0.98 / 0.98 exact**: the same.

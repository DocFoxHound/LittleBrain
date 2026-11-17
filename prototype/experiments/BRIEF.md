# Shared brief: holding neurons in RAM + CPU + VRAM

You are one of several agents, each trying a **different experimental strategy** for storing and
running our neurons efficiently across **system RAM + CPU** and **GPU VRAM**, for both
*functioning* (acting: forward / modulate) and *training* (sleep: consolidation, pruning).
Fringe ideas are welcome. Negative results are fine; report them honestly.

## What the user wants
**Fast, cheap methods for both performance (acting) and training (sleep/consolidation).** All three
tiers are fair game: **VRAM, RAM, and the NVMe SSD**. The SSD fits the philosophy: some neurons
aren't always on or firing, so dormant ones can live somewhere slower. The user expects the SSD to
be slow and wants to know how slow, and whether clever use (batched, sequential, sleep-time
sweeps) makes it practical.

## The project (read these first)
- `~/Documents/MemoryPhilosophy/README.md`: the overall goal (a brain-like, emotional, continually learning system).
- `notes/13-our-neuron.md`: the workhorse neuron design (branches, cascade synapses, three-factor learning, sleep).
- `notes/14-neuron-memory.md`: the packed version: **32 bits per synapse, about 1,041 bytes per neuron**, and its measurements.
- `prototype/neuron.py`: the readable float reference implementation.
- `prototype/packed_neuron.py`: the packed `PackedPopulation` (bit layout, hash wiring, stochastic rounding, lazy cascade flow, 4-step eligibility history). **This is the baseline to beat.**
- `prototype/demo_single_neuron.py` and `prototype/compare_packed.py`: the behavior test ("childhood → hard times → time passes → relearn").

## Baseline numbers (NumPy, CPU, 100,000 neurons × 256 synapses, 99 MiB stored)
| step | peak temporary RAM | time |
|---|---|---|
| forward | 684 MiB | 0.7 s |
| modulate | 1,175 MiB | 1.1 s |
| sleep | 1,196 MiB | 1.5 s |

Behavior baseline (workhorse, seeds 1–8): day-6 recovery mean ≈ 0.14–0.15 (range 0.09–0.38),
relearn median 2 trials. **If your strategy changes how the neuron computes or stores values,
you must show behavior still matches.** Use `run(label, ntype, seed, make=YourAdapter, verbose=False)`
from `demo_single_neuron.py`, with an adapter that has the same interface as `PackedNeuron`
(forward / modulate / tick / rest / sleep / memory_bytes). If you change behavior on purpose, say so.

Realistic activity: in the brain, only about **1–10% of neurons are active at a time** (sparse coding),
and only a small fraction of synapses change on a given day. Exploit that if it helps, but measure it.

**Watch out:** the current prototype wires each synapse to a random input out of a tiny input space
(64 inputs), so *every* neuron gets some input *every* step. Under that wiring, skipping inactive
neurons or paging them out can't help. If your strategy depends on sparsity, you'll need **structured
or local connectivity** (for example, blocks or "columns" of neurons that draw from their own region
of a larger input space, such as 4,096+ inputs with 2–5% active). Choose a scenario, state it clearly,
and treat "this needs locality" as a design finding in its own right.

## Machine
- CPU: Intel i9-13900H, 6 performance + 8 efficiency cores, 20 threads, AVX2 + FMA (no AVX-512). L3 cache 24 MB.
- RAM: 62 GB (about 56 GB free). NVMe SSD with 2.4 TB free.
- GPU: RTX 4070 Laptop, **8 GB VRAM**, CUDA 13, driver 580. PCIe link to RAM.
- **Python environment: `~/Documents/MemoryPhilosophy/.venv/bin/python`** (Python 3.14) with numpy, torch 2.14 (CUDA works, pinned memory works), **triton 3.8** (works; `@triton.jit` kernels must live in a real .py file), **cupy-cuda13x** (`cp.RawKernel` compiles CUDA C at runtime), numba 0.68 (CPU `@njit(parallel=True)` works), zstandard, lz4. torch and cupy share GPU memory without copying via DLPack.
- gcc, clang, rustc and cargo are available. C via `ctypes` works (no Python headers needed).

## Rules (several agents share this machine at the same time)
- Work **only inside your own folder**: `prototype/experiments/<your-slug>/`. Don't edit other files; import from `prototype/` instead (add it to `sys.path`).
- **Don't install anything** (no pip, no system packages). If something essential is missing, say so in your report.
- Resource caps: **CPU threads ≤ 3** (set `NUMBA_NUM_THREADS=3`, `OMP_NUM_THREADS=3`, `torch.set_num_threads(3)`). **RAM ≤ 6 GB** unless your brief says otherwise. **VRAM ≤ 2.5 GB** if you use the GPU (`cp.get_default_memory_pool().set_limit(...)`). Delete any big data files you create before finishing.
- Timings are noisy because other agents run at the same time. Compare strategies **within your own run**, repeat measurements (median of 3+), and report ratios as well as absolute numbers.
- Keep individual benchmark runs short (under ~2 minutes each).
- Measure memory properly: resident bytes per neuron (RAM and VRAM separately), plus **peak temporary** memory during each step.

## Deliverables
1. Code in your folder, runnable with one command (`.venv/bin/python prototype/experiments/<slug>/run.py` or similar).
2. `README.md` in your folder:
   - the idea, and the biological or philosophical inspiration if any;
   - what you built;
   - a results table: bytes per neuron (RAM / VRAM / temporaries), throughput (neuron-steps/s or synapse-updates/s) for forward / modulate / sleep, at sizes you chose;
   - the behavior check;
   - a verdict (promising / mixed / dead end) and why;
   - surprises, and what you'd try next.
3. Final message to the coordinator: **under 400 words**. The key numbers, the verdict, and the one idea worth carrying forward.

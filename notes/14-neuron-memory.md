# 14: Shrinking the neuron in memory

Goal: the same workhorse neuron (note 13), in the smallest possible memory.
Code: [`prototype/packed_neuron.py`](../prototype/packed_neuron.py).
Comparison: `python3 prototype/compare_packed.py`.

## Result (measured 2026-10-27)

| | Original (`neuron.py`) | Packed (`packed_neuron.py`) |
|---|---|---|
| Per synapse | 36 bytes | **4 bytes** (one 32-bit word) |
| Per neuron, stored data | 9,248 bytes | **1,041 bytes** |
| Per neuron, measured as separate Python objects | 12,256 bytes | 6,026 bytes |
| Per neuron, measured in a population array | — | **1,041 bytes** |
| 1 million neurons × 256 synapses | ~8.6 GiB (doesn't fit an 8 GB card) | **~0.97 GiB** |

About **9–12× smaller.** An 8 GB card can hold roughly 6–7 million of these neurons
(about 1.7 billion synapses) with room to spare. A fruit-fly-sized brain (~50M synapses) is about 200 MB.

## The six tricks

1. **One 32-bit word per synapse.** All of a synapse's state is packed into 32 bits:

   | bits | field | size | meaning |
   |---|---|---|---|
   | 0–3 | gen | 4 | rewiring generation (see trick 2) |
   | 4–9 | fast | 6 | fast weight, −31..31 steps |
   | 10–15 | c0 | 6 | cascade surface |
   | 16–21 | c1 | 6 | cascade middle |
   | 22–27 | c2 | 6 | cascade deep (the "old self") |
   | 28–30 | tag | 3 | "this mattered," 0..7 |
   | 31 | used | 1 | used since last sleep |

   Real synapses are low-precision too: measurements suggest a real synapse stores about **4.7 bits** of strength (Bartol et al. 2015, *eLife*).

2. **Wiring is computed, not stored.** Which input a synapse listens to = hash(neuron id, slot, generation). Rewiring means adding 1 to the generation, which gives a new pseudo-random target. That's 4 bits instead of 64. The idea comes from "procedural connectivity" (Knight & Nowotny 2021, *Nature Computational Science*), which used it to simulate very large brain models on one graphics card.
   - **Limit:** a regrown synapse can only go to a *random* new target, not one chosen deliberately. The generation wraps after 16 rewirings.

3. **Stochastic rounding.** A change smaller than one step is applied *with probability* equal to its size: 0.3 of a step becomes 1 step 30% of the time. On average nothing is lost, so 6 bits behave much like full precision.

4. **One shared clock instead of per-synapse decay.** Fast weights all fade at the same rate, so the population keeps one number for "how much they've faded." The cascade flow is linear, so any number of steps can be applied at once with a precomputed 3×3 matrix, and only when needed. **A synapse is only touched when something happens to it.**

5. **Eligibility rebuilt instead of stored.** "Was this synapse just involved?" is recomputed from the last 4 steps of each input's activity (shared by the whole population) and each neuron's firing (8 bytes per neuron).
   - **Lesson learned:** a first, cheaper version paired each input's activity with the neuron's *current* firing. That changed the behavior noticeably: off-pattern synapses were punished too little. The firing *at the time of the input* matters. Keeping 4 steps of history fixed it.

6. **Neurons are rows in one big array, not separate objects.** As separate Python objects, the packed neuron still costs 6 KB, mostly Python overhead. In a population array it costs exactly its data.

## Does it behave the same? (8 seeds)

| Response to A | hard times | day 3 | day 6 | relearn trials (median) |
|---|---|---|---|---|
| Workhorse, original | 0.03 | 0.06 | 0.14 (0.09–0.29) | 2 |
| Workhorse, packed | 0.03 | 0.07 | 0.15 (0.09–0.38) | 2 |

Same behavior: the old self recovers slowly, and relearning is fast.

**Known difference:** the packed neuron rewires about **half as often** (7.5 vs 14.8 synapses per neuron
over the test, 16 seeds). That comes from 6-bit "weak" checks and the 1-bit "used" flag. The
plain comparison neuron (which relies on lucky rewiring to relearn) relearned in 7/16 seeds packed
vs 5/16 original, which is within noise.

## The remaining problem: working memory
The packed numbers are small **while stored**. While **computing**, NumPy unpacks everything into
full-size temporary float arrays. For 100,000 neurons (99 MiB stored):

| Step | Peak temporary memory | Time (CPU) |
|---|---|---|
| forward | 684 MiB | 0.7 s |
| modulate | 1,175 MiB | 1.1 s |
| sleep | 1,196 MiB | 1.5 s |

Two fixes:
- **Quick:** process neurons in chunks (e.g. 4,096 at a time) to cap the temporary memory.
- **Real:** a **GPU kernel** (Triton or CUDA). Each GPU thread unpacks its synapse's 32-bit word into
  registers, computes, and writes 32 bits back. **No temporaries at all, and much faster.**

## Which language?
- The *stored* numbers are raw memory either way: a NumPy or PyTorch array is the same bytes as a C array. **Python doesn't make the data bigger.**
- Python costs memory in two places: per-object overhead (solved by population arrays) and temporaries (solved by a kernel).
- **Recommendation:** keep Python as the conductor (the world, the experiments, the logging). Write the inner neuron step as a **GPU kernel**:
  - **Triton** is written in Python-like syntax and runs on the RTX 4070. Easiest path.
  - **CUDA C++** gives the most control.
  - Rust or C++ only matter if we want a fast CPU version.

## Going smaller still (ideas, untested)
- **16 bits per synapse** (gen 2, fast 4, c0 4, c1 3, c2 3). Tags move to a sparse list (only tagged synapses, cleared at sleep), and the used flag is dropped. That halves memory again; quality unknown.
- **Sparse fast weights:** only synapses touched today have a nonzero fast weight. Store them as a "today" list (index + value), emptied at sleep. That is literally the short-term memory buffer from note 09.
- **Fewer, smarter synapse slots:** rewiring toward active inputs instead of random ones would need fewer slots for the same learning, but needs stored targets (back to 16 bits for wiring).

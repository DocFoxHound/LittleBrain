# 19: First wiring test: a thalamus, two cortical areas, long-range connections and waves

Roadmap step 4. Code: `prototype/brain/network.py` (the circuit) and `prototype/brain/run_wiring_test.py`
(the experiment). Results: `prototype/brain/results_wiring_test.txt` (6 seeds, about 7 minutes).

## The circuit
- **retina** (256 inputs) → **thalamus** (256 relay cells + 64 TRN) → **area A** (lower, 2,048 cells) → **area B** (higher, 2,048 cells)
- **Feedback** from B's deep layers (L5 IT) to A's apical tufts; **A's layer 6** back to the thalamus.
- Every area is an 8 × 8 sheet of tiles. Per tile: 6 L4, 10 L2/3, 3 L5 IT, 3 L5 PT, 5 L6, 4 PV, 3 SST, 2 VIP.
- **5,184 cells and 491,520 synapses**, about 3 ms per step on the CPU (readable float version).
- Wiring follows note 17:
  - local connections fall off with distance;
  - long-range connections land at the matching map location;
  - feedforward lands on basal branches, feedback on apical tufts;
  - higher areas pool wider, and feedback spreads wider than feedforward.
- Long-range fractions came out at: L4 10%, L2/3 7–13%, L5 7%, L6 10%, all under the caps.

## The task: memory completion
- The network **develops** for 4 "days" of ordinary sights (random blobs) with learning on and sleep each night.
- It then **learns 6 object-like patterns** (6 presentations each, phasic dopamine and high ACh), and sleeps.
- At **recall**, it sees only the **left half** of each pattern, with low ACh.

Measures:
- **Fill-in:** does area A's layer 2/3 bring back the pattern-specific activity of the **right half**? (Correlation with the full-pattern response over right-half cells, own pattern minus other patterns, after removing each cell's baseline.)
- **Identify:** can the cue's response be matched to the right pattern? (Chance = 0.17.)

## Results (6 seeds; ± = spread across seeds, so the standard error is about 0.4× that)

| Condition | Fill-in (right half) | Identify (A) | Identify (B) |
|---|---|---|---|
| untrained (developed, never saw these patterns) | 0.08 ± 0.11 | 0.42 | 0.28 |
| **learned, no waves** | **0.42 ± 0.19** | 0.39 | 0.33 |
| learned, forward wave always | 0.28 ± 0.22 | 0.39 | 0.39 |
| learned, wave as mode (forward to learn, reverse to recall) | 0.39 ± 0.14 | 0.39 | 0.36 |
| wave-scheduled updates (only cells near the crest) | **−0.14 ± 0.08** | 0.08 | 0.06 |
| long-range × 0 | **0.12 ± 0.26** | 0.31 | 0.22 |
| long-range × 0.5 | 0.27 ± 0.17 | 0.53 | 0.31 |
| long-range × 1 (= wave as mode) | 0.39 ± 0.14 | 0.39 | 0.36 |
| wave as mode, apical feedback removed | 0.45 ± 0.22 | 0.56 | 0.31 |

**Paging** (cortex tiles as memory blocks, cache = 25%): event-driven 16 stalls/step; wave-scheduled 51;
wave-scheduled with prefetch 48 (and twice the loads).

## What it shows
1. **The circuit learns and recalls.** After learning, the missing half comes back (fill-in 0.08 → 0.42).
2. **Long-range connections matter, and the more of them, the better recall works:** 0.12 → 0.27 → 0.39
   as the long-range fraction goes 0 → ½ → full. That's the clearest result, and it supports keeping up to 20%.
3. **But not through the route I expected.** Removing the apical feedback didn't hurt (0.45). The missing
   half comes back some other way: most likely the top-down drive to VIP cells (which open dendrites; that
   pathway wasn't removed), or recurrent loops through area B. A cleaner ablation is needed.
4. **Waves gave no benefit here.**
   - "Wave as mode" ≈ no waves (0.39 vs 0.42).
   - A constant forward wave may hurt a little (0.28), but that's within noise.
   - **Wave-scheduled updates broke recall** (−0.14). Updating only part of the network desynchronizes it.
   - **Prefetch didn't help paging**: the wave's crest covered about 64% of the brain at once, more than the cache.

   This is what the skeptics predict. But our waves were *imposed* (a fixed sine wave), had one wavelength,
   and their crest was very wide. Real waves are *produced* by the circuit itself.
5. **Things the circuit forced us to add**, all biological (details in note 18): heterosynaptic plasticity,
   fast-spiking inhibition, inhibitory plasticity, quiet interneurons, phasic dopamine, tonic vs burst
   thalamus, apical drive, strong-but-few thalamic synapses, and **development**.

## Verdict
- **Keep:** the long-range rules (up to 20%, by class and layer), the development phase, and the biology from point 5.
- **Waves:** off by default. The idea isn't dead, but it needs either *emergent* waves (from the E/I circuit
  itself) or much narrower waves before it's worth testing again. Revisit with spirals and with the
  dream/sleep work, where waves (slow oscillations, spindles) are best established.
- **Open:** find the real route for recall (clean ablations: feedback only, VIP top-down only, area B
  recurrence only); run more seeds; harder tasks (more patterns, overlapping patterns).

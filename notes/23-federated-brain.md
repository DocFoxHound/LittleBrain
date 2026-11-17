# 23: The federated brain (roadmap step 7): in progress

Pathways between regions were researched into `notes/regions/connections.md` (about 80 pathways,
cell type → cell type, with IDs, compartments, strengths and sources), including a build order
("minimal federated brain"). This note tracks the build.

## The engine
- `prototype/brain/federation.py`: a brain described as **data** (groups of cells on a shared 8 × 8
  tile map + pathways) instead of hand-written code.
  - **Synaptic pathways**: learnable, sampled local or map-matched, flagged long-range by structure, with the caps of note 17 checked automatically.
  - **Pooled pathways**: inhibition, inhibitory projections and global signals, arriving on a named channel: cell body, apical, dendrite, axon initial segment, volume, generic inhibition, excitation.
- `prototype/brain/brainspec.py`: `stage1()` (thalamus + two cortical areas), `stage2()` (+ hippocampal loop).
- Every pathway carries its ID from connections.md (e.g. "HC-7 mossy fibers (detonator)").

## Stage 1: does the data-built brain match the hand-built one?
| | Fill-in (memory completion) |
|---|---|
| hand-built (notes 19–20) | 0.42–0.48 |
| data-built, first try | 0.05 ✗ |
| + inhibition updated before excitation in each step | 0.22 |
| + PV plasticity before the cell's update (as hand-built) | **0.68 ± 0.03** (8 seeds) |

**Lessons:**
- **Update order matters a lot.** A one-step lag in feedback inhibition nearly destroyed recall. The
  circuit must update inhibitory cells first, then excitatory cells under *this* step's inhibition.
  Real inhibition is fast (feedforward PV cells fire within a millisecond or two), so this matches biology.
- **Fragility:** moving the PV plasticity update relative to the cell update changed recall from 0.22 to
  0.68. Results this sensitive to small ordering choices need watching. They suggest the E/I balance sits
  near a tipping point.
- The data-built version isn't an exact replica (its development input differs slightly), but it performs
  at least as well, so it replaces `network.py` going forward.

## Stage 2: hippocampal loop
Wired per connections.md: HC-1 (cortex → EC), HC-3/HC-4 (perforant path → DG, CA3), HC-7 (mossy fibers:
0.4% of CA3 synapses but strong, the "detonator"), HC-8 (CA3 recurrent), HC-9 (Schaffer collaterals), HC-5
(EC L3 → CA1 tuft, where one-shot learning happens), HC-13/14 (CA1 → subiculum → EC L5), HC-15 (EC L5 →
cortex, apical). 7,104 cells, ~670,000 synapses, ~160 steps/s on the CPU.

**Stability work:**
1. At first the hippocampus ran hot (dentate 0.33, mossy cells 0.80, CA3 0.37). That's realistic in a
   way: CA3 is the brain's most seizure-prone region, and the dentate–mossy cell loop is a known
   excitatory loop.
2. Stronger static inhibition didn't settle it.
3. A "winners take all" rule (dentate 2%, CA3 5%, CA1 10%) stands in for very fast feedback inhibition.
4. Always-on homeostasis (intrinsic plasticity) **made it worse**: groups went silent and couldn't recover.
   Left off.
5. **After the update-order fix**, the hippocampus is **stable across days**, but **not sparse enough**:
   mossy cells and CA3 are broadly active. Next: tune toward realistic sparseness.

Functional test (does the hippocampus help memory, especially after one presentation; does a lesion at
recall hurt?): results below.

### Stage 2 functional test (`prototype/brain/results_federation_stage2.txt`, 6 seeds)
| Condition | Fill-in (± s.e.) |
|---|---|
| cortex only, 1 presentation | 0.09 ± 0.14 |
| + hippocampus, 1 presentation | 0.00 ± 0.02 |
| + hippocampus, lesioned at recall, 1 | −0.04 ± 0.02 |
| cortex only, 6 presentations | 0.69 ± 0.04 |
| + hippocampus, 6 presentations | 0.59 ± 0.13 |
| + hippocampus, lesioned at recall, 6 | 0.70 ± 0.09 |

**Verdict: the hippocampus doesn't contribute yet.** It doesn't rescue one-shot learning, and lesioning it
at recall changes nothing. It's wired and stable, but not doing its job.

Likely reasons, to check *inside* the hippocampus before expecting effects in the cortex:
1. **No pattern separation:** CA3 and the mossy cells are broadly active, so different patterns may share most of their CA3 code. Measure overlap at DG and CA3 for different patterns.
2. **No completion in CA3:** does a half cue reactivate the full CA3 pattern after learning? (Test CA3 directly.)
3. **The route back is long and weak:** EC L5 → area B apical → feedback to area A. Its contribution may be drowned out. Measure EC L5 and B's response to cues.
4. **Encoding vs recall settings:** acetylcholine suppresses apical input while encoding, which is right biologically, but it should also suppress CA3 recurrents during encoding and release them at recall (Hasselmo). That isn't implemented.

Next: hippocampal internal tests (separation, completion), then fix, then repeat this test.

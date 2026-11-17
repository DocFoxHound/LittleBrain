# 18: Neuron types, built (roadmap step 3)

Each type is modeled on a real human cell and has to show its **signature behavior** in a test.
Code: `prototype/brain/celltypes.py` (parameters), `prototype/brain/cells.py` (dynamics).
Test: `PYTHONPATH=prototype python3 prototype/brain/test_cell_types.py`: **12/12 pass.**

This is a rate model (each cell's state is a firing rate from 0 to 1, updated each step), not a spiking
model. Excitatory synapses use learning rules v2 (note 16) with cascade memory. Inhibitory synapses are
fixed, except PV inhibition, which adapts.

## The types and their signature behaviors

| Type | Real counterpart | Layer / class | What makes it that cell | Test result |
|---|---|---|---|---|
| **L2/3 IT** | layer 2/3 pyramidal | 2/3, IT (cortex → cortex) | basal + apical dendrites; **adapts** to constant input; long-range ≤ 20% | peak 0.88 → settles 0.68 |
| **L4 stellate** | spiny stellate | 4, local | no apical tuft; receives the **thalamus** (≈10% of synapses, strong) | used in network |
| **L5 IT** | layer 5A pyramidal | 5, IT | source of **feedback** to lower areas | used in network |
| **L5 PT** | layer 5B thick-tufted pyramidal | 5, PT (output) | **BAC firing**: basal and apical input together trigger a burst | basal 0.11, apical 0.07, **both 0.46** |
| **L6 CT** | layer 6 corticothalamic | 6, CT | slow; feeds back to the thalamus (relay and TRN) | used in network |
| **PV basket** | parvalbumin fast-spiking | local | targets the cell body: **divisive gain control**; fast-spiking, so its rate keeps rising with drive (inhibition tracks excitation) | responses squashed [0.14, 0.78, 0.92] → [0.02, 0.11, 0.20] |
| **SST Martinotti** | somatostatin | local | **facilitating** input (responds to *sustained* activity); **gates apical tufts** | ramps 0.40 → 0.99; apical-driven cell 0.73 → 0.30 |
| **VIP** | vasoactive intestinal peptide | local | **inhibits SST**, so dendrites open (disinhibition); driven by top-down input and acetylcholine | SST 0.84 → 0.20; cell 0.33 → 0.55 |
| **TC relay** | thalamocortical relay | thalamus, TC | **tonic** mode when awake; **burst** mode after hyperpolarization (T-type calcium), only when ACh is low (drowsy, asleep) | weak input: 0.19 from rest, **0.80 after being held down** |
| **TRN** | thalamic reticular nucleus | thalamus | inhibitory **gate** around the thalamus, driven by layer 6 and relay collaterals | relay 0.19 → 0.02 |
| **DG granule** | dentate gyrus granule cell | hippocampus | extremely sparse (~2%); **pattern separation** | input overlap 0.76 → output 0.25 |
| **DG granule, young** | adult-born granule cell (4–6 weeks) | hippocampus | more excitable, more plastic | threshold 2.5 vs 4.0; learning 2× |

**Neuromodulatory nuclei** are functional models of what those cell groups compute:

| Nucleus | Signal | Test result |
|---|---|---|
| VTA (dopamine) | reward prediction error (reward − expectation, learned per context) | first reward 1.00 → 15th 0.04; omitted −0.96 |
| Basal forebrain (ACh) | high for novelty/mismatch (encode), low when familiar (recall) | novel 0.97, familiar 0.15 |
| Locus coeruleus (NE) | phasic burst on surprise | calm 0.18 → surprise 0.57 |
| Raphe (serotonin) | slow average of reward (patience) | (no test yet) |

## Biology the network forced us to add
Building a real circuit (note 19) exposed missing pieces. Each was solved with a known biological mechanism:

1. **Heterosynaptic plasticity.** When some of a cell's synapses strengthen, the rest weaken slightly. Without it, a recurrent network learns its way into a seizure (every cell at maximum within one epoch).
2. **Fast-spiking inhibition.** PV cells can't saturate at the same rate as pyramidal cells, or inhibition can't keep up with excitation (an "inhibition-stabilized network").
3. **Inhibitory plasticity** (Vogels-style). PV inhibition onto a cell grows when that cell fires above its target.
4. **Interneurons quiet at rest.** They fire when driven, not tonically.
5. **Phasic dopamine.** A brief burst when something novel or rewarding appears. Dopamine on every step drove runaway learning.
6. **Tonic vs burst thalamus.** Bursting is allowed only when acetylcholine is low (drowsy, asleep). While awake the relay stays tonic.
7. **Apical input drives the cell directly, not just multiplicatively.** Otherwise feedback can never recall something the senses aren't currently showing.
8. **Thalamic synapses are few but effectively strong.** About 10% of layer 4's synapses, but synchronous (Bruno & Sakmann 2006). With weak settings, layer 4 just echoed itself.
9. **Development.** A randomly wired cortex isn't selective. A few "days" of ordinary experience with sleep made it selective (layer 2/3: same pattern 0.97, different patterns −0.05).

## Not yet built
- Hippocampal types beyond the dentate gyrus (CA3 recurrent cells, CA1, entorhinal grid cells).
- Basal ganglia (striatal medium spiny neurons, which are inhibitory *projection* neurons), cerebellar Purkinje and granule cells, the amygdala.
- Learning in interneuron input synapses; short-term synaptic depression.
- Packed/GPU versions of the new types (the network uses the readable float version).

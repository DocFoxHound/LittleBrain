# 27. Physical neurons: building a neuron instead of simulating one

*2026-11-11. The brain project is paused because current hardware can't run it at the scale needed. This note starts a new line of thinking: could neurons be built **physically**, from materials that behave like neurons on their own, instead of being simulated on a GPU?*

## Why the brain is so efficient (what hardware would have to copy)

1. **Memory and processing are in the same place.** A synapse holds the connection strength *and* does the work. In computers, data shuttles between separate memory and processor, and that traffic is the main cost.
2. **It mostly sits idle.** Neurons fire rarely, and silence costs almost nothing.
3. **Physics does the math.** The membrane collects charge like a small battery until it hits a threshold. Nothing is calculated.
4. **Low voltage.** About 0.1 V, against roughly 1 V in chips. Energy rises with the *square* of voltage.

## The brain is less 3D than it looks

The cortex is a thin sheet (2–3 mm, six layers) that is crumpled up. The truly 3D part is the **white matter underneath**: long-range cables that take up close to half the brain's volume. So a brain-like chip may not need a solid block of neurons. It may need **a few stacked layers plus dense 3D long-range wiring**. This matches the project's own design: L2/3 feeds forward, L5/6 feeds back (notes 17–18).

## Example 1: Plastic neurons (Linköping University, Sweden, 2022)

- **Who:** Simone Fabiano's group. Paper: Harikesh, Yang, Fabiano et al., *Nature Communications* 2022, doi 10.1038/s41467-022-28483-6.
- **Material:** conductive plastics (polymers) in **organic electrochemical transistors**. The plastic sits in a gel or liquid and soaks ions in and out, and the ions change how much current it passes. It speaks both **ions** (the language of nerves) and **electrons** (the language of electronics).
- **What they showed:**
  - A circuit of these transistors **spikes**: it charges, fires and resets, like a real neuron.
  - **How fast it fires depends on the ions around it**, as in real neurons.
  - An artificial synapse showed **Hebbian learning**: use makes the connection stronger.
  - Wired to a **Venus flytrap**, its pulses made the trap snap shut, showing an artificial neuron and a living organism can share a signal language.
  - The circuits are **printed** on flexible material and run at low voltage.
- **Weaknesses:** large compared with silicon parts, slow (though real neurons are slow too, in milliseconds), and long-term stability in liquid is unsolved. Likely first uses: prosthetics, nerve implants, soft robots.

## Example 2: Silver nanowire "brain" (University of Sydney + UCLA, 2023)

- **Who:** Zhu, Lilak, Loeffler, Lizier, Stieg, Gimzewski, Kuncic. *Nature Communications*, Nov 2023, doi 10.1038/s41467-023-42470-5. Title: "Online dynamical learning and sequence memory with neuromorphic nanowire networks."
- **Material:** silver wires thousands of times thinner than a hair, each with a thin plastic coat. Scattered on a surface, they form a **random tangled web** that looks like brain tissue under a microscope.
- **The synapse:** where two wires cross, current makes **silver atoms creep across the coating and build a tiny bridge**, so the junction conducts better. Left unused, the bridge dissolves and the junction **forgets**. Use strengthens and disuse weakens, with no design and nothing but physics.
- **What they showed:**
  - Handwritten digits, turned into electrical signals and **streamed in one at a time**, were learned **while the data flowed**, with no separate training phase: **93.4% accuracy**.
  - The web **remembered sequences of up to 8 digits**, showing short-term memory built into the material.
- **Caveat (as I understand it):** the web does the brain-like part (mixing signals, holding memory), and a small, simple learning step outside it reads the web's patterns. It isn't yet "the material learns everything by itself."

## Other physical neurons and synapses (from memory, not yet checked)

- **Mott neurons** (vanadium dioxide, niobium dioxide): flip between insulator and conductor when heated, which gives a natural spike. HP Labs "neuristor" (~2012). A 2018 study showed about 23 real neuron firing behaviors.
- **Memristors** (RRAM, phase-change memory): resistance remembers the current's history, so the device is a synapse. Some types learn the way real synapses do, getting stronger when two neurons fire close together in time.
- **Diffusive memristors** (Joshua Yang's group): drifting silver atoms imitate the calcium dynamics behind short-term synaptic memory.
- **Nanofluidic memristors** (Paris, ~2023): salt-water channels with memory, the same physics as real neurons.
- **Superconducting neurons with optical connections** (NIST, Jeff Shainline): light solves the "7,000 connections" problem. Needs cooling to near absolute zero.
- **Spintronic** (electron magnetism) and **photonic** (light-based) neurons.
- **Neuromorphic chips** (Intel Loihi 2 / Hala Point, SpiNNaker 2, BrainScaleS): brain-like, but still flat silicon.
- **Organoids** (Cortical Labs CL1, FinalSpark): real human neurons on chips. The project prefers to avoid this route.

## Connections to the project

- The nanowire junction is close to the **synapses that strengthen and fade** in note 11, but physics does it for free.
- Plastic neurons run on **ions in liquid at low voltage**, the way the brain saves energy.
- **Learning must be local** in physical hardware: each synapse learns from only what's next to it, plus a broad reward signal. The project's rules (BTSP, three-factor rules, dopamine) are this kind, so the neuron work carries over.
- **Device-to-device variation** is a problem engineers fight, but brains tolerate it, and note 25 already studied brain-to-brain variance.
- **Gap:** none of these has brain-wide chemical signals like dopamine or serotonin (the basis of emotion in this design). Plastic neurons already respond to the ions around them, which may be a way in.

## The sci-fi picture

A **3D gel or mesh** of self-assembled nanowires or ion channels, with materials that spike on their own as neurons and memristor-like junctions as synapses. It runs on ions at low voltage and is taught through its inputs, using local learning rules plus a chemical-style reward signal. Pieces exist in labs. Nobody has assembled them.

## Sources

- [Linköping: Building artificial nerve cells (Phys.org)](https://phys.org/news/2022-02-artificial-nerve-cells.html)
- [Artificial neuron makes a Venus flytrap snap (Cosmos)](https://cosmosmagazine.com/technology/ai/artificial-neuron-venus-flytrap)
- [ScienceDaily release](https://sciencedaily.com/releases/2022/02/220222121302.htm)
- [arXiv preprint of the organic neuron paper](https://pdf.arxiv.org/pdf/2403.00005)
- [University of Sydney: Nanowire neural network learns and remembers on the fly](https://sydney.edu.au/news-opinion/news/2023/11/01/nanowire-neural-network-learns-remembers-on-the-fly.html)
- [The Conversation: We built a brain from tiny silver wires](https://theconversation.com/we-built-a-brain-from-tiny-silver-wires-it-learns-in-real-time-more-efficiently-than-computer-based-ai-216730)
- [Phys.org: Nanowire brain network learns on the fly](https://phys.org/news/2023-10-nanowire-brain-network-fly.html)

# 17: Columns, long-range connections, and brain waves

Research for the next design step (2026-10-28). Confidence: ● well established · ◐ supported, details
argued · ○ recent, preprint, or unconfirmed.

## Do neurons work in columns?
**Partly, and not as boxes.**
- **Cortical columns** (Mountcastle, 1957) and Hubel & Wiesel's orientation and ocular-dominance columns are real
  in some areas: neurons stacked vertically through the cortex's layers share what they respond to. ●
- **Minicolumns** (roughly 80–120 neurons, ~50 µm across) are a common anatomical motif. ◐
- But columns aren't universal or sharply bounded. Horton & Adams (2005) called the cortical column "a structure
  without a function." Many species and areas lack clear columns, and one study found ~92% of excitatory synapses
  at a column's center come from neurons *outside* it. ◐
- **A better model:** a continuous sheet where the chance of two neurons connecting **falls off with distance**,
  plus patchy long-range lateral connections and projections to other areas. ●

**Design consequence:** "columns" in our system should be **storage tiles of a continuous sheet**, used for
paging and caching (note 15), not hard walls in the wiring.

## The 80/20 figure
- The best-known 80/20 split is about **neuron types**: roughly **80% excitatory** (mostly pyramidal) and
  **20% inhibitory interneurons** (estimates range from 15–25%). ●
- **Inhibitory interneurons connect almost entirely locally.** ●
- **Pyramidal neurons do both.** A single pyramidal neuron has **local axon branches *and* long-range
  projections**: your guess was right. ●
- Connection-wise, most of the inputs a neuron receives come from nearby. I recall a tracer study (Markov,
  Kennedy et al., macaque) reporting roughly 80% of the neurons projecting into an area are from within that
  area and about 20% from other areas, but I couldn't confirm the exact figure in this search. ○

**Design (proposed):**
- each neuron's synapse slots split into **local** (distance-weighted, same neighborhood) and **long-range**
  (other regions);
- long-range at most 20%, set by neuron type:
  - pyramidal / workhorse ~10–20%
  - hub ~20%
  - inhibitory 0%
  - transient ~0–5%
- this is a **small-world network** (Watts & Strogatz, 1998): dense local clusters with short paths everywhere.

**Storage note from the experiments:** local slots can use compact wiring (about 4 bits within a tile window,
cpu-event-driven), while long-range slots need a full address. So use two slot formats, and expect long-range
slots to widen the paging working set unless "fire together, store together" reordering is used.

## Where do long-range connections come from? (not scattered to the wind)
Long-range wiring is highly organized. Four rules cover most of it:

**1. Which cell sends it depends on its layer and its class.** The cortex has 6 layers. Pyramidal neurons
fall into three projection classes, named for where their long axons go (Harris & Shepherd, 2015) ◐:

| Class | Where it sits | Where its long axon goes | Role |
|---|---|---|---|
| **IT** (intratelencephalic) | mostly **layers 2/3** and 5A | other cortical areas (both hemispheres, via the corpus callosum) and the striatum | cortex-to-cortex conversation |
| **PT / ET** (pyramidal tract, extratelencephalic) | **layer 5B** (big "thick-tufted" cells) | brainstem, spinal cord, higher-order thalamus | output: action, commands to the body |
| **CT** (corticothalamic) | **layer 6** | thalamus | feedback that tunes what the thalamus relays |

- Layer 4 (spiny stellate cells) mostly *receives* thalamic input and connects locally. ●
- Inhibitory interneurons in the cortex are local. ●
  - **Outside the cortex, inhibitory *projection* neurons are common**: striatum (medium spiny neurons), globus pallidus, Purkinje cells of the cerebellum, the thalamic reticular nucleus. "Inhibitory = local" is a rule for the cortex only. ●
- Information flows one way between classes: IT neurons feed PT neurons, but not the reverse. ◐

**2. Direction depends on depth: feedforward from the surface, feedback from deep.** ◐
- **Feedforward** (lower → higher area, e.g. vision → object recognition): mostly from **superficial layers
  (2/3)**, landing in **layer 4** of the target, i.e. on the cell bodies and basal dendrites.
- **Feedback** (higher → lower, e.g. expectations → vision): mostly from **deep layers (5/6)**, landing in
  **layer 1**, i.e. on the **apical tufts** of pyramidal neurons (Felleman & Van Essen, 1991; Markov et al., 2014).
- The fraction of a connection that comes from superficial layers (called "SLN") even *measures* how far up the
  hierarchy the source sits.
- **This maps directly onto our neuron:** basal branches = feedforward input; the apical "context" branch =
  feedback. Biology had already given our neuron the right shape.

**3. Who connects to whom: specific areas, falling off with distance.** ◐
- Between cortical areas, the chance of connecting **falls exponentially with distance** (the "exponential
  distance rule," Ercsey-Ravasz, Markov … Kennedy, *Neuron*, 2013).
- Connection strengths span **five orders of magnitude**. The network is dense (about 66% of area pairs
  connect in the macaque), yet highly specific: each area has its own particular "fingerprint" of partners.
- Most strength is in a few strong, mostly nearby links; the long ones are few and weak, but crucial for
  short paths across the brain.

**4. Where in the target it lands: same spot on the map (topography).** ●
- Projections preserve maps: the "left visual field" part of one area connects to the "left visual field"
  part of the next. Within an area, long lateral connections link patches with **similar preferences**
  (like-to-like).

**Design rule for our model:**
- A long-range synapse is defined by (source region, source layer and class) → (target region, target layer
  and compartment), with probability falling with distance, strength drawn from a heavy-tailed spread, and a
  landing point at the matching map location.
- The **long-range fraction is a property of the neuron class:**
  - layer 2/3 IT about 15–20% (cortex to cortex)
  - layer 5 PT about 20% (to subcortical targets)
  - layer 6 CT about 10–20% (to the thalamus)
  - layer 4 about 0–5%
  - cortical interneurons 0%
  - subcortical projection neurons: mostly long-range
- **Feedforward** lands on basal branches; **feedback** lands on the apical branch.

## Brain waves: the "vortex" study
Your memory is **mostly right**:

| Claim | Status |
|---|---|
| Brain activity moves in **spiral / vortex** patterns | ✅ Spiral waves in human cortex (Xu, Long, Feng & Gong, *Nature Human Behaviour*, 2023, fMRI, 100 people). First direct observation of rotating spiral waves in awake humans with implanted electrodes (Das, Jacobs et al., *Nature Communications*, April 2026). In mice, rotating waves mirrored across the two hemispheres, with axons in somatosensory cortex wired in a spiraling arrangement that may produce them (Ye & Steinmetz, *Science*, June 2026). ◐ |
| They **switch direction** | ✅ Spiral rotation direction changes with the task, and can be used to tell tasks apart (2023). Traveling waves run **back→front while forming memories and front→back while recalling** (Mohan, Zhang, Ermentrout & Jacobs, *Nature Human Behaviour*, 2024; 93 patients). ◐ |
| They **help recall** | ◐ People remembered better when waves ran the "right" direction for storing vs recalling (2024). Rotating waves appeared more during spatial navigation memory than verbal memory (2026). A preprint reports spiral centers moving to mark encoding, holding and retrieval (bioRxiv, 2025). ○ But **causation is unproven**: waves may drive memory or just accompany it. |
| They switch **strength** | ❓ Not specifically found. |
| They **guide long-range connections** | ❓ Not shown. Spirals do **coordinate activity across distant networks** and often sit at the boundaries between networks (2023). The mouse study suggests the reverse: **wiring shapes the waves**. The established case of waves shaping wiring is **before birth: retinal waves** (our parked topic). |

**Skepticism:** there's no consensus on what waves *do*. Buzsáki argues they just reflect underlying synaptic
activity. Keep that in mind: design waves so they can be **switched off and compared**.

## Philosophy we can adapt (even if the biology is unsettled)
1. **Waves as a schedule.** A wave sweeping across the sheet decides *when* each region updates. For our
   tiered memory that's very practical: **the wave's path is a predictable access order**, so blocks can be
   loaded from RAM or the SSD *just ahead of the wave front*. That's perfect prefetch, and it fixes the
   paging agent's weak spot (attention jumping around).
2. **Direction as mode.**
   - forward (senses → executive) = **encoding**: write, learn;
   - reverse (executive → senses) = **recall**: read, reconstruct.

   One global "direction" signal sets whether the brain is storing or remembering. This pairs with the
   acetylcholine idea (Hasselmo's theory: high acetylcholine favors encoding, low favors retrieval).
   **It also lines up with the anatomy above:** back→front waves ride the *feedforward* pathway
   (superficial → layer 4 → basal branches), and front→back waves ride the *feedback* pathway
   (deep → layer 1 → apical branches). In our model, wave direction can simply set whether basal or apical
   input dominates.
3. **Phase as a gate for long-range traffic.** Long-range connections only transmit when sender and
   receiver are in phase ("communication through coherence," Fries). A rotating spiral then lets regions
   take turns talking. Long-range traffic becomes **sparse and scheduled**, which is cheap.
4. **Spirals as coordinators.** A spiral center at the boundary of several regions sweeps them in order:
   round-robin attention, with the rotation direction choosing the order.
5. **Waves shape wiring.** During development (retinal waves) and maybe sleep, neurons that the wave
   activates together get wired together, including long-range. Long connections grow along wave paths.
6. **The night sweep is already a wave.** The SSD agent's sequential sleep sweep is a traveling wave over
   storage. Ordering it by the brain's layout makes sleep both biologically flavored and fast.

## Suggested experiments (later)
- Long-range fraction 0% / 10% / 20%: effect on learning, recall and paging cost.
- Wave-scheduled updates versus all-at-once: speed (prefetch) and behavior.
- Wave direction as an encode/recall switch: does reversing it improve recall of stored patterns?
- Waves on vs off: if behavior doesn't change, the skeptics win, and we keep waves only as a scheduler.

## Sources
- Mohan, Zhang, Ermentrout & Jacobs (2024), *Nature Human Behaviour* 8: travelling-wave direction and memory. [Columbia news](https://www.engineering.columbia.edu/about/news/brain-waves-travel-one-direction-when-memories-are-made-and-opposite-when-recalled) · [ScienceDaily](https://www.sciencedaily.com/releases/2024/03/240308123309.htm)
- Xu, Long, Feng & Gong (2023), *Nature Human Behaviour* 7: interacting spiral waves. [summary](https://ideas.repec.org/a/nat/nathum/v7y2023i7d10.1038_s41562-023-01626-5.html) · [news](https://www.technologynetworks.com/neuroscience/news/spiral-shaped-signals-for-organizing-brain-activity-discovered-374860)
- Das, Jacobs et al. (2026), *Nature Communications*; Ye & Steinmetz (2026), *Science*. Covered in [Quanta, Sept 2026](https://www.quantamagazine.org/surprisingly-complex-waves-reveal-the-brains-inner-workings-20260930/)
- "Hidden Spirals" preprint (2025). [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.03.686225.full.pdf)
- Excitatory/inhibitory proportions: [O'Reilly & Munakata, Computational Cognitive Neuroscience](https://med.libretexts.org/Bookshelves/Neuroscience/Computational_Cognitive_Neuroscience_3e_(O'Reilly_and_Munakata)/03%3A_Networks/3.02%3A_Biology_of_the_Neocortex)
- Local vs long-range synapses in cat visual cortex: [PMC2651285](https://pmc.ncbi.nlm.nih.gov/articles/PMC2651285)
- Horton & Adams (2005), *The cortical column: a structure without a function*; Watts & Strogatz (1998); Fries (2005/2015), communication through coherence; Hasselmo (2006), acetylcholine and encoding vs retrieval (from background knowledge, not re-checked today).
- Feedforward and feedback by layer: [Frontiers in Neuroscience 2014](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2014.00072/pdf) · [PMC4255240](https://pmc.ncbi.nlm.nih.gov/articles/PMC4255240) · Felleman & Van Essen (1991); Markov et al. (2014)
- Projection classes (IT / PT / CT): [Layer 5 subclasses review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9650089/) · Harris & Shepherd (2015), *The neocortical circuit: themes and variations*
- Exponential distance rule: [Ercsey-Ravasz et al. 2013, PubMed](https://www.pubmed.ncbi.nlm.nih.gov/24094111/) · [Horvát et al. 2016, PLOS Biology](https://journals.plos.org/plosbiology/article/info:doi/10.1371/journal.pbio.1002512)

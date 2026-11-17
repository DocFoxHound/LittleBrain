# Synapse budget per cell type

How many input synapses each simulated neuron type should get, and how finely its weights should be stored. It covers the 89 types in the catalog plus 9 types still to build (last 9 rows). The same numbers are in `prototype/brain/synapse_budget.py` as `BUDGET`.

**How to read it**

- **Bio synapses**: excitatory plus inhibitory input synapses on one neuron, as a range. These are biological numbers, not scaled. Most are **rodent** numbers because no human count exists. Human cells are usually larger: about 2–3× for cortical pyramids and about 2–3× for Purkinje cells.
- **Branches**: terminal dendritic segments where known, otherwise `n/k` (not known).
- **Memory role**: how much long-term information the cell's own synapses hold. High = it is a main storage site; Medium = it learns, but it is not the main store; Low = some plasticity, mostly fixed; None = it does not learn.
- **Standard synapses**: the anchor is a human cortical pyramid with 7,000–10,000 synapses = **4096**. Each type is scaled by the geometric mean of its range (`std = gm(lo, hi) × 4096 / 8,367`). The result is rounded to the nearest power of two in log space and clamped to [4, 65536].
- **Low synapses**: standard ÷ 16, minimum 4.
- **Precision**: **32** levels for High; **16** for Medium and Low when the synapses learn; **fixed** when they don't.
- **Confidence**: ● measured in this cell type (in the species shown) · ◐ extrapolated, from a review or a model, or a different subregion · ○ rough estimate by analogy. **Most rows are ○**: for most interneurons, brainstem and hypothalamic cells there is no count of synapses per cell. The order of magnitude is the useful part.

| type name | bio synapses | branches | species | memory role | standard synapses | low synapses | precision | confidence | source |
|---|---|---|---|---|---|---|---|---|---|
| L2/3 IT | 15,000–30,000 | ~60-100 terminal (basal ~40-60) | human | **High**: associative cortical store; spines turn over with learning | 8,192 | 512 | 32 | ◐ | 1,2,3 |
| L4 stellate | 3,000–6,000 | ~30-40 terminal | cat/rat | **Medium**: feature tuning set mostly in development; adult plasticity gated by ACh/NE | 2,048 | 128 | 16 | ◐ | 5,32 |
| L5 IT | 8,000–16,000 | ~60-90 terminal | rat (human larger) | **Medium**: cortico-cortical and cortico-striatal associations | 4,096 | 256 | 16 | ○ | 4,32 |
| L5 PT | 10,000–30,000 | ~80-120 terminal | rat; human extrapolated | **Medium**: output cell; tuft-basal coincidence learning, but mostly a readout | 8,192 | 512 | 16 | ◐ | 4,29 |
| L6 CT | 3,000–6,000 | ~30-50 terminal | rat | **Low**: gain/feedback to thalamus; little stored content | 2,048 | 128 | 16 | ○ | 32 |
| PV basket | 5,000–16,000 | ~20-40 terminal | rat (hippocampal analog) | **Low**: mostly fixed; some anti-Hebbian LTP | 4,096 | 256 | fixed | ◐ | 7,32 |
| SST Martinotti | 2,000–5,000 | n/k | rat | **Low**: fixed dendritic inhibition; facilitating inputs | 2,048 | 128 | fixed | ○ | 32 |
| VIP | 1,000–3,000 | ~10-20 (bipolar) | rat | **Low**: disinhibitory switch; no stored content | 1,024 | 64 | fixed | ○ | 7,32 |
| TC relay | 4,000–8,000 | ~30-60 terminal | cat (LGN) | **Low**: relay; modest adult plasticity | 2,048 | 128 | 16 | ◐ | 22 |
| TRN | 3,000–8,000 | n/k | rat/cat (total never measured) | **None**: gate; ~60% of its synapses are cortical | 2,048 | 128 | fixed | ○ | 25 |
| DG granule | 5,000–12,000 | ~10-20 terminal | rat | **Medium**: pattern separation; classic PP-LTP site, but storage is mainly CA3 | 4,096 | 256 | 16 | ◐ | 9,10 |
| DG granule (young) | 2,500–7,000 | ~8-15 terminal | rat/mouse | **High**: 4-6 wk cells: enhanced LTP, preferentially recruited into new engrams | 2,048 | 128 | 32 | ○ | 9,10 |
| L6 IT | 4,000–8,000 | n/k | rat | **Medium**: cortico-cortical associations | 2,048 | 128 | 16 | ○ | 32 |
| L6b | 2,000–6,000 | n/k | rat | **Low**: modulatory, arousal-related output | 2,048 | 128 | 16 | ○ | 32 |
| L5/6 NP | 3,000–8,000 | n/k | mouse | **Low**: local, sparse spines; little stored content | 2,048 | 128 | 16 | ○ | 32 |
| von Economo | 5,000–15,000 | 1 basal + 1 apical trunk, few branches | human (no count exists) | **Low**: fast relay of interoceptive salience | 4,096 | 256 | 16 | ○ | 3 |
| PV chandelier | 3,000–6,000 | n/k | rat | **None**: fixed axon-initial-segment control | 2,048 | 128 | fixed | ○ | 32 |
| SST non-Martinotti | 2,000–5,000 | n/k | mouse | **Low**: fixed L4 disinhibition | 2,048 | 128 | fixed | ○ | 32 |
| LAMP5 neurogliaform | 1,000–3,000 | ~20-40 short | rat | **Low**: volume transmission; no specific memory | 1,024 | 64 | fixed | ○ | 32 |
| CCK basket | 3,000–8,000 | n/k | rat | **Low**: endocannabinoid short-term suppression only | 2,048 | 128 | fixed | ○ | 7 |
| rosehip | 1,000–3,000 | compact bushy, n/k | human (no count exists) | **Low**: fixed apical-shaft inhibition | 1,024 | 64 | fixed | ○ | 3 |
| TC core | 4,000–8,000 | ~30-60 terminal | cat/primate | **Low**: topographic relay | 2,048 | 128 | 16 | ◐ | 22 |
| TC matrix | 3,000–8,000 | n/k | primate | **Low**: diffuse relay to L1 | 2,048 | 128 | 16 | ○ | 22 |
| thalamic interneuron | 2,000–5,000 | ~20-40 (dendrites also presynaptic) | cat | **Low**: fixed triadic inhibition | 2,048 | 128 | fixed | ○ | 22 |
| EC L2 stellate | 7,500–15,000 | ~40-60 terminal | rat | **Medium**: grid code largely fixed; inputs plastic | 4,096 | 256 | 16 | ○ | 11 |
| EC L2 island | 5,000–12,000 | n/k | rat | **Medium**: temporal-association (trace) signals to CA1 | 4,096 | 256 | 16 | ○ | 11 |
| EC L3 pyramidal | 6,000–12,000 | n/k | rat | **Medium**: temporoammonic instructive input; persistent firing | 4,096 | 256 | 16 | ○ | 11 |
| EC L5/6 pyramidal | 6,000–15,000 | n/k | rat | **Medium**: hippocampal output to cortex; consolidation relay | 4,096 | 256 | 16 | ○ | 11 |
| DG mossy cell | 8,000–20,000 | n/k | rat | **Medium**: associational loop; plastic MF and commissural inputs | 8,192 | 512 | 16 | ○ | 8 |
| HIPP | 3,000–8,000 | n/k | rat | **Low**: fixed feedback inhibition of PP | 2,048 | 128 | fixed | ○ | 7 |
| CA3 pyramidal | 15,000–20,000 | ~70-100 terminal | rat | **High**: recurrent autoassociative storage (~12k RC + ~3.7k PP + ~46 MF) | 8,192 | 512 | 32 | ◐ | 8 |
| CA2 pyramidal | 10,000–20,000 | n/k | rat/mouse | **Medium**: social memory, but Schaffer input resists LTP | 8,192 | 512 | 16 | ○ | 8 |
| CA1 pyramidal | 30,000–32,000 | ~100-200 terminal | rat (human larger) | **High**: BTSP one-shot place/episode learning | 16,384 | 1,024 | 32 | ● | 6,28,30 |
| subiculum bursting | 10,000–20,000 | n/k | rat | **Medium**: output; spatial/boundary codes, plastic | 8,192 | 512 | 16 | ○ | 8 |
| subiculum regular | 10,000–20,000 | n/k | rat | **Medium**: output; plastic | 8,192 | 512 | 16 | ○ | 8 |
| CA PV basket | 15,000–17,000 | ~30-40 terminal | rat | **Low**: mostly fixed; some anti-Hebbian LTP | 8,192 | 512 | fixed | ● | 7 |
| CA axo-axonic | 5,000–10,000 | n/k | rat | **None**: fixed axon-initial-segment control | 4,096 | 256 | fixed | ○ | 7 |
| bistratified | 10,000–16,000 | n/k | rat | **Low**: fixed feedforward/feedback dendritic inhibition | 8,192 | 512 | fixed | ○ | 7 |
| O-LM | 3,000–8,000 | n/k | rat | **Low**: mostly fixed (mGluR1a anti-Hebbian LTP) | 2,048 | 128 | fixed | ○ | 7 |
| ivy | 2,000–5,000 | n/k | rat | **Low**: fixed slow tonic inhibition | 2,048 | 128 | fixed | ○ | 7 |
| hippocampal neurogliaform | 1,000–3,000 | n/k | rat | **Low**: volume GABA-B; no specific memory | 1,024 | 64 | fixed | ○ | 7 |
| interneuron-specific | 1,500–3,000 | ~10-20 | rat (CR cells) | **Low**: fixed disinhibition | 1,024 | 64 | fixed | ◐ | 7 |
| BLA principal (+) | 4,000–12,000 | ~30-60 terminal | rat | **High**: stores CS-reward associations (LTP at sensory inputs) | 4,096 | 256 | 32 | ○ | 33 |
| BLA principal (-) | 4,000–12,000 | ~30-60 terminal | rat | **High**: stores CS-threat associations (LA LTP) | 4,096 | 256 | 32 | ○ | 33 |
| intercalated | 1,000–3,000 | n/k | rat/mouse | **Medium**: extinction memory: BLA->ITC LTP | 1,024 | 64 | 16 | ○ | 27 |
| CeL PKCδ (off) | 2,000–6,000 | n/k (MSN-like spiny) | mouse | **Low**: inputs depress with conditioning; small capacity | 2,048 | 128 | fixed | ○ | 27 |
| CeL SST (on) | 2,000–6,000 | n/k (MSN-like spiny) | mouse | **Medium**: LA->CeL SST synapses potentiate with fear learning | 2,048 | 128 | 16 | ○ | 27 |
| CeA CRF | 2,000–6,000 | n/k | rat | **Low**: slow anxiety state; peptide-driven | 2,048 | 128 | fixed | ○ | 27 |
| CeM output | 2,000–6,000 | n/k | rat | **None**: output gate | 2,048 | 128 | fixed | ○ | 27 |
| D1 MSN | 5,000–15,000 | ~25-35 terminal | rat | **High**: DA-gated three-factor learning of actions/habits | 4,096 | 256 | 32 | ◐ | 20,21 |
| D2 MSN | 5,000–10,000 | ~20-30 terminal | rat | **High**: DA-gated LTP/LTD of NoGo associations | 4,096 | 256 | 32 | ◐ | 20,21 |
| striosome MSN | 5,000–15,000 | ~25-35 terminal | rat | **High**: stores limbic state values sent to DA cells | 4,096 | 256 | 32 | ○ | 20,21 |
| cholinergic TAN | 2,000–6,000 | ~5-10 long aspiny | rat | **Low**: learned pause, but mostly via inputs' plasticity | 2,048 | 128 | fixed | ○ | 20 |
| striatal PV FSI | 3,000–8,000 | n/k | rat | **Low**: fixed feedforward inhibition | 2,048 | 128 | fixed | ○ | 20 |
| striatal SST/NPY/NOS | 1,000–4,000 | n/k | rat | **Low**: fixed | 1,024 | 64 | fixed | ○ | 20 |
| striatal calretinin | 1,000–3,000 | n/k | primate | **Low**: fixed | 1,024 | 64 | fixed | ○ | 20 |
| striatal TH | 1,000–3,000 | n/k | mouse | **None**: fixed | 1,024 | 64 | fixed | ○ | 20 |
| GPe prototypic | 5,000–15,000 | ~10-20 long, sparse | monkey/rat | **None**: tonic, does not learn | 4,096 | 256 | fixed | ○ | 23,24 |
| GPe arkypallidal | 3,000–10,000 | n/k | rat | **None**: stop signal, fixed | 2,048 | 128 | fixed | ○ | 23 |
| GPi | 5,000–20,000 | ~10-20 long (disc-shaped arbor) | primate | **None**: tonic gate, fixed | 4,096 | 256 | fixed | ○ | 23,24 |
| SNr | 5,000–15,000 | ~10-20 long | rat | **None**: tonic gate, fixed | 4,096 | 256 | fixed | ○ | 23 |
| STN | 2,000–6,000 | ~10-20 | rat | **Low**: fixed (some LTP reported) | 2,048 | 128 | fixed | ○ | 34 |
| cerebellar granule | 8–25 | 4 claws (3-5) | rat/cat | **Low**: expansion recoding; MF-GrC plasticity minor | 8 | 4 | fixed | ● | 17 |
| Purkinje | 300,000–500,000 | rat ~500-1,000 branchlets; human several thousand | human (rat ~175k) | **High**: ~10^5 PF synapses learn motor timing (CF-gated LTD) | 65,536 | 4,096 | 32 | ◐ | 15,16 |
| Golgi | 500–2,000 | n/k | rat/mouse | **Low**: fixed feedback inhibition (~400 GrC contacts) | 512 | 32 | fixed | ◐ | 18 |
| cerebellar basket | 1,000–3,000 | planar, n/k | rat | **Low**: PF-MLI plasticity exists but small | 1,024 | 64 | fixed | ○ | 16 |
| cerebellar stellate | 500–2,000 | planar, n/k | rat | **Low**: as basket | 512 | 32 | fixed | ○ | 16 |
| Lugaro | 500–2,000 | n/k | rat | **None**: fixed | 512 | 32 | fixed | ○ | 16 |
| unipolar brush | 2–10 | 1 brush (one MF rosette) | rat | **None**: one mossy fiber input; fixed amplifier | 4 | 4 | fixed | ● | 19 |
| DCN projection | 14,000–28,000 | ~10-20 | cat | **Medium**: MF->DCN LTP stores consolidated motor memories | 8,192 | 512 | 16 | ● | 13,14 |
| nucleo-olivary | 3,000–10,000 | n/k | cat/rat | **Low**: fixed | 2,048 | 128 | fixed | ○ | 13 |
| inferior olive | 1,000–3,000 | ~5-10 (spines in glomeruli) | rat/cat | **None**: error/teaching source, fixed | 1,024 | 64 | fixed | ○ | 35 |
| dopamine | 6,000–10,000 | ~5-8 dendrites | rat (SNc) | **Low**: computes RPE; expectation learned upstream | 4,096 | 256 | fixed | ● | 12 |
| VTA GABA | 2,000–6,000 | n/k | rat | **Low**: subtracts learned expectation coming from upstream | 2,048 | 128 | fixed | ○ | 12 |
| VTA glutamate | 2,000–6,000 | n/k | mouse | **Low**: fixed | 2,048 | 128 | fixed | ○ | 12 |
| LC noradrenaline | 2,000–8,000 | ~3-5 long (pericoerulear) | rat | **None**: broadcast arousal; fixed | 2,048 | 128 | fixed | ○ | 36 |
| raphe serotonin | 3,000–8,000 | n/k | rat | **None**: slow broadcast; fixed | 2,048 | 128 | fixed | ○ | 36 |
| basal forebrain ACh | 3,000–10,000 | n/k | rat/primate | **None**: broadcast mode switch; fixed | 2,048 | 128 | fixed | ○ | 36 |
| septal PV pacemaker | 3,000–8,000 | n/k | rat | **None**: theta pacemaker; fixed | 2,048 | 128 | fixed | ○ | 36 |
| PPN/LDT ACh | 2,000–6,000 | n/k | rat | **None**: arousal/REM; fixed | 2,048 | 128 | fixed | ○ | 36 |
| AgRP/NPY | 300–1,500 | 1-3 dendrites | mouse | **Low**: fasting/leptin rewires synapses (state, not memory) | 256 | 16 | fixed | ○ | 26 |
| POMC | 300–1,500 | 1-3 dendrites | mouse | **Low**: leptin-dependent synaptic remodelling | 256 | 16 | fixed | ○ | 26 |
| orexin | 1,000–4,000 | n/k | mouse | **Low**: LTP after sleep loss/fasting is homeostatic | 1,024 | 64 | fixed | ○ | 26,31 |
| MCH | 1,000–4,000 | n/k | mouse | **Low**: fixed | 1,024 | 64 | fixed | ○ | 26 |
| PVN CRH | 500–2,000 | 1-3 dendrites | rat | **Low**: stress habituation; small | 512 | 32 | fixed | ○ | 37 |
| PVN oxytocin | 1,000–4,000 | 1-3 dendrites | rat | **Low**: lactation/parturition remodelling (state) | 1,024 | 64 | fixed | ○ | 37 |
| PVN/SON vasopressin | 1,000–4,000 | 1-3 dendrites | rat | **Low**: fixed | 1,024 | 64 | fixed | ○ | 37 |
| SCN clock | 300–1,000 | 1-3 simple | rat/mouse | **None**: phase entrainment, not memory | 256 | 16 | fixed | ○ | 37 |
| claustrum projection | 5,000–15,000 | n/k (spiny) | rat/mouse | **Low**: salience broadcast; little stored content | 4,096 | 256 | 16 | ○ | 32 |
| pontine relay | 2,000–6,000 | n/k | rat | **Low**: relays cortex to mossy fibers; fixed | 2,048 | 128 | fixed | ○ | 38 |
| lateral habenula | 1,000–5,000 | ~5-10 (spiny/fusiform) | rat | **Medium**: learned aversive predictions; input potentiation (depression models) | 1,024 | 64 | 16 | ○ | 39 |
| RMTg GABA | 1,000–4,000 | n/k | rat | **Low**: relays LHb to DA cells; fixed | 1,024 | 64 | fixed | ○ | 39 |
| PAG freezing | 1,000–4,000 | n/k | rat | **Low**: fixed defensive output | 1,024 | 64 | fixed | ○ | 27 |
| PVN MC4R | 500–2,000 | 1-3 dendrites | mouse | **Low**: satiety integration; fixed | 512 | 32 | fixed | ○ | 26 |
| VA/VL motor thalamus | 4,000–10,000 | ~30-60 terminal | primate | **Low**: relay of cerebellar/pallidal output | 4,096 | 256 | 16 | ○ | 22 |
| MD thalamus | 4,000–10,000 | ~30-60 terminal | primate | **Medium**: sustains PFC working memory; task-rule plasticity | 4,096 | 256 | 16 | ○ | 22 |
| EC L5a pyramidal | 6,000–15,000 | n/k | rat/mouse | **Medium**: hippocampal output to neocortex; consolidation | 4,096 | 256 | 16 | ○ | 11,40 |
| EC L5b pyramidal | 6,000–15,000 | n/k | rat/mouse | **Medium**: receives CA1/subiculum; persistent firing; feeds back to superficial EC | 4,096 | 256 | 16 | ○ | 11,40 |

## Notes and caveats

- **The anchor is low for humans.** The rule maps 7,000–10,000 synapses to 4096. That is roughly a rodent or average cortical pyramid. Human L2/3 pyramids have about 30,000 excitatory synapses [1], so `L2/3 IT` comes out at 8192, not 4096. If L2/3 should be the 4096 reference, halve every standard number. The ordering stays the same.
- **Mixed species.** Rodent counts are used as measured, not scaled up to human size. This makes rodent-only types (CA1, CA3, MSN, DCN, …) slightly small next to `L2/3 IT`, which uses the human count.
- **Clamp.** Purkinje's scaled value (~190,000) is clamped to 65536. It is the only type that hits the ceiling. The human number itself is uncertain. A 2025 recount gives 300–500 k (mouse 30–50 k) [15]; older estimates go up to ~10⁶. Rat is ~175 k [16].
- **Floor.** `unipolar brush` (one mossy-fiber rosette) and `cerebellar granule` (4 mossy fibers + Golgi contacts) are the only types near the 4-synapse floor. Their tiny size is real biology, not a gap in the data.
- **Catalog vs. biology on learning.** Precision follows the memory rating. Three types rated Medium have `learns False` in the catalog but show real plasticity: `intercalated` (extinction LTP), `CeL SST (on)` (fear-conditioning LTP) and `DCN projection` (MF→DCN LTP). They get **16 levels** here; set them to `None` if the catalog flag should win. The new type `lateral habenula` (learned aversive prediction) also gets 16. The reverse also happens: `L6 CT`, `L6b`, `L5/6 NP`, `von Economo`, `claustrum projection`, `TC relay`/`TC core`/`TC matrix` and the new `VA/VL motor thalamus` are rated Low but learn in the catalog, so they get 16, not fixed.
- **Thalamic relay totals are disputed.** The ~4,000–5,000 figure for cat LGN relay cells is a textbook number [22]. A reanalysis of corticogeniculate input suggests the total may be higher.
- **Dopamine cells**: ~8,000 synapses, of which ≥30% are glutamatergic and 40–70% GABAergic (rat SNc) [12]. That is about a cortical pyramid's worth.
- **Gaps (no per-cell count found):** TRN, GPe/GPi/SNr, STN, BLA, CeA, all hypothalamic types, LC/raphe/basal forebrain/septum/PPN, inferior olive, pontine nuclei, LHb, RMTg, PAG, cerebellar MLIs and Lugaro. For these, the numbers come from dendrite size and analogy with similar cells.

## Sources

1. Eyal G et al. (2018) Human cortical pyramidal neurons: from spines to spikes via models. *Front Cell Neurosci* 12:181. https://doi.org/10.3389/fncel.2018.00181 (human L2/3: ~30,000 excitatory synapses)
2. Elston GN, Benavides-Piccione R, DeFelipe J (2001) The pyramidal cell in cognition: a comparative study in human and monkey. *J Neurosci* 21:RC163 (spine counts by area; >30-fold range across areas)
3. DeFelipe J, Alonso-Nanclares L, Arellano JI (2002) Microstructure of the neocortex: comparative aspects. *J Neurocytol* 31:299–316
4. Ramaswamy S, Markram H (2015) Anatomy and physiology of the thick-tufted layer 5 pyramidal neuron. *Front Cell Neurosci* 9:233
5. Ahmed B, Anderson JC, Douglas RJ, Martin KAC, Nelson JC (1994) Polyneuronal innervation of spiny stellate neurons in cat visual cortex. *J Comp Neurol* 341:39–49
6. Megías M, Emri Z, Freund TF, Gulyás AI (2001) Total number and distribution of inhibitory and excitatory synapses on hippocampal CA1 pyramidal cells. *Neuroscience* 102:527–540 (rat: ~30,000 E + ~1,700 I)
7. Gulyás AI, Megías M, Emri Z, Freund TF (1999) Total number and ratio of excitatory and inhibitory synapses converging onto single interneurons of different types in the CA1 area of the rat hippocampus. *J Neurosci* 19:10082–10097. https://pmc.ncbi.nlm.nih.gov/articles/PMC6782984 (PV ~16,300; CB ~3,800; CR ~2,200)
8. Amaral DG, Ishizuka N, Claiborne B (1990) Neurons, numbers and the hippocampal network. *Prog Brain Res* 83:1–11; Treves A, Rolls ET (1992) *Hippocampus* 2:189–199 (CA3: ~12,000 RC, ~3,750 PP, ~46 MF)
9. Claiborne BJ, Amaral DG, Cowan WM (1990) Quantitative, three-dimensional analysis of granule cell dendrites in the rat dentate gyrus. *J Comp Neurol* 302:206–219 (mean dendrite length ~3,200 µm)
10. Thind KK, Ribak CE, Buckmaster PS (2008) Synaptic input to dentate granule cell basal dendrites in a rat model of temporal lobe epilepsy. *J Comp Neurol* 509:190–202 (implies ~11,000 E per control granule cell)
11. A survey of entorhinal grid cell properties (2018), arXiv:1810.07429. https://arxiv.org/abs/1810.07429 (stellate: 12.8–18.1 mm dendrite → ~7,500–15,000 spines, an estimate)
12. Henny P et al. (2012) Structural correlates of heterogeneous in vivo activity of midbrain dopaminergic neurons. *Nat Neurosci* 15:613–619 (~8,000 synapses per SNc DA neuron)
13. Palkovits M, Mezey E, Hámori J, Szentágothai J (1977) Quantitative histological analysis of the cerebellar nuclei in the cat. *Exp Brain Res* 28:189–209 (~20,000 boutons per nuclear cell; 13,900–27,500 by nucleus)
14. Person AL, Raman IM (2012) Purkinje neuron synchrony elicits time-locked spiking in the cerebellar nuclei. *Nature* 481:502–505 (~20–50 Purkinje cells per DCN cell)
15. Masoli S et al. (2024) Human Purkinje cells outperform mouse Purkinje cells in dendritic complexity and computational capacity. *Commun Biol* 7:5; Masoli S et al. (2025) *Front Physiol*, https://doi.org/10.3389/fphys.2025.1671271 (human ~300–500 k vs mouse ~30–50 k PF synapses)
16. Napper RMA, Harvey RJ (1988) Number of parallel fiber synapses on an individual Purkinje cell in the cerebellum of the rat. *J Comp Neurol* 274:168–177 (~175,000); Palkovits M et al. (1971) cat cerebellar cell ratios (Purkinje:basket 1:6, Purkinje:stellate 1:16)
17. Jakab RL, Hámori J (1988) Quantitative morphology and synaptology of cerebellar glomeruli in the rat. *Anat Embryol* 179:81–88
18. Cesana E et al. (2013) Granule cell ascending axon excitatory synapses onto Golgi cells implement a potent feedback circuit in the cerebellar granular layer. *J Neurosci* 33:12430–12446 (~400 GrC contacts per Golgi cell)
19. Mugnaini E, Floris A (1994) The unipolar brush cell: a neglected neuron of the mammalian cerebellar cortex. *J Comp Neurol* 339:174–180 (one mossy-fiber rosette per brush)
20. Wilson CJ (1995) in *Models of Information Processing in the Basal Ganglia* (Houk, Davis, Beiser, eds.), MIT Press: 5,000–15,000 corticostriatal synapses per MSN (as cited in arXiv:1801.01822); Kincaid AE, Zheng T, Wilson CJ (1998) *J Neurosci* 18:4722–4731
21. Gertler TS, Chan CS, Surmeier DJ (2008) Dichotomous anatomical properties of adult striatal medium spiny neurons. *J Neurosci* 28:10814–10824 (D1 MSNs have more dendrites than D2)
22. Van Horn SC, Erişir A, Sherman SM (2000) Relative distribution of synapses in the A-laminae of the lateral geniculate nucleus of the cat. *J Comp Neurol* 416:509–520 (retinal 12%, cortical 58%, GABA 24%); Sherman SM, Guillery RW, in Shepherd (ed.) *The Synaptic Organization of the Brain*, ch. 8 (~4,000–5,000 synapses per relay cell)
23. DiFiglia M, Pasik P, Pasik T (1982) A Golgi and ultrastructural study of the monkey globus pallidus. *J Comp Neurol* 212:53–75 (dendrites "totally covered" by synaptic terminals)
24. Yelnik J, Percheron G, François C (1984) A Golgi analysis of the primate globus pallidus. II. *J Comp Neurol* 227:200–213
25. Liu XB, Jones EG (1999) Predominance of corticothalamic synaptic inputs to thalamic reticular nucleus neurons in the rat. *J Comp Neurol* 414:67–79 (~60% cortical)
26. Pinto S et al. (2004) Rapid rewiring of arcuate nucleus feeding circuits by leptin. *Science* 304:110–115; Horvath TL, Gao XB (2005) Input organization and plasticity of hypocretin neurons. *Cell Metab* 1:279–286
27. Li H et al. (2013) Experience-dependent modification of a central amygdala fear circuit. *Nat Neurosci* 16:332–339; Amano T, Unal CT, Paré D (2010) Synaptic correlates of fear extinction in the amygdala. *Nat Neurosci* 13:489–494
28. Bittner KC et al. (2017) Behavioral time scale synaptic plasticity underlies CA1 place fields. *Science* 357:1033–1036
29. Beaulieu-Laroche L et al. (2018) Enhanced dendritic compartmentalization in human cortical neurons. *Cell* 175:643–651 (human L5 dendrites much longer than rat)
30. Benavides-Piccione R et al. (2020) Differential structure of hippocampal CA1 pyramidal neurons in the human and mouse. *Cereb Cortex* 30:730–752 (human CA1 larger, not a scaled mouse cell)
31. Rao Y et al. (2007) Prolonged wakefulness induces experience-dependent synaptic plasticity in mouse hypocretin/orexin neurons. *J Clin Invest* 117:4022–4033
32. Markram H et al. (2015) Reconstruction and simulation of neocortical microcircuitry. *Cell* 163:456–492 (rat: ~31,000 neurons, ~37 M synapses ≈ 1,200 local synapses per neuron, plus long-range input; used for relative sizes of interneuron types)
33. McDonald AJ (1982) Neurons of the lateral and basolateral amygdaloid nuclei: a Golgi study in the rat. *J Comp Neurol* 212:293–312 (pyramid-like principal cells). Golgi spine density ~0.19/µm × ~10 mm would give only ~1,900 spines, which is surely an undercount
34. Baufreton J et al. (2009) Sparse but selective and potent synaptic transmission from the globus pallidus to the subthalamic nucleus. *J Neurophysiol* 102:532–545 (no per-cell total)
35. De Zeeuw CI et al. (1998) Microcircuitry and function of the inferior olive. *Trends Neurosci* 21:391–400 (glomeruli; each spine gets 1 excitatory + 1 GABA terminal)
36. Aston-Jones G et al. (1991) Afferent regulation of locus coeruleus neurons: anatomy, physiology and pharmacology. *Prog Brain Res* 88:47–75 (no per-cell count for LC or other modulatory nuclei)
37. Iovino M et al. (2016) review of afferent pathways to magnocellular PVN/SON neurons. *Endocr Metab Immune Disord Drug Targets* (no per-cell count)
38. Schwarz C, Thier P (1999) Binding of signals relevant for action: towards a hypothesis of the functional role of the pontine nuclei. *Trends Neurosci* 22:443–451
39. Weiss T, Veh RW (2011) Morphological and electrophysiological characteristics of neurons within identified subnuclei of the rat lateral habenula. *Neuroscience* 172:74–93; Li B et al. (2011) Synaptic potentiation onto habenula neurons in the learned helplessness model of depression. *Nature* 470:535–539
40. Sürmeli G et al. (2015) Molecularly defined circuitry reveals input-output segregation in deep layers of the medial entorhinal cortex. *Neuron* 88:1040–1053 (L5a output vs L5b input)

# Neocortex: neuron types and proportions

Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain / derived by us.
"% of neurons" means all neurons in the area unless stated. Species is always labeled.

## 0. Headline numbers

| Quantity | Value | Species | Source | Conf. |
|---|---|---|---|---|
| Neurons in the cerebral cortex | ~16 billion (19% of the brain's 86 billion) | human | Azevedo 2009 | ● |
| Neurons in the neocortex | ~23 billion (men), ~19 billion (women) | human (stereology) | Pakkenberg & Gundersen 1997 | ◐ (method differs from Azevedo) |
| GABAergic share of neurons, M1 | **human 33%**, marmoset 23%, **mouse 16%** | snRNA-seq | Bakken 2021 | ● |
| GABAergic share, general | primates ~25–34%; rodents ~15–25% | review | Gabbott & Bacon 1996 via Clowry 2018 | ◐ |
| GABAergic share, rat barrel cortex | ~12% (older estimates were ~2× too high) | rat | Meyer 2011 | ● |
| GABAergic share, rodents (review) | ~10–15% | rodent | Tremblay, Lee & Rudy 2016 | ◐ |
| Area variation | excitatory:inhibitory ratio in human **V1 is more than double** that of other areas; elsewhere similar | human | Jorstad 2023 | ● |
| IT neurons as share of excitatory | ~70% | mouse MOp (MERFISH) | Zhang 2021 | ● |

**Working split to model:** excitatory 67% / inhibitory 33% (human association/motor cortex); 80/20 for V1-like
sensory areas (○, derived from Jorstad's "E:I more than double" in V1); 84/16 if we ever want a mouse-like cortex.

## 1. Excitatory types (all glutamate)

Layer shares below come from the only complete per-layer neuron count we found: one rat barrel column
(Meyer 2010: L1 63, L2 2,039, L3 3,735, L4 4,447, L5A 1,737, L5B 2,235, L6A 3,786, L6B 1,066; total ~19,100 neurons,
including ~12–15% interneurons). Subclass shares inside a layer come from separate sources and are combined by us (○ where so).

| Type (marker) | % of neurons | Species | Axon target | Conf. |
|---|---|---|---|---|
| **L2/3 IT** (Cux2; human: many types) | ~30% of all neurons in a rat S1 column (L2 10.7% + L3 19.5%); **relatively more in human** (L2/3 IT significantly more common in human than marmoset/mouse M1) | rat (count); human (qualitative) | ipsi + contralateral cortex (callosal), striatum; no subcortical-below-telencephalon | ● counts / ◐ human share |
| **L4 IT / spiny stellate** (Rorb, Rspo1) | ~23% in rat S1 column; **sparse in M1** ("L4-like cells, less diverse, much sparser") ; thick and subdivided (4A/4B/4Cα/4Cβ) in primate V1 | rat; human/mouse M1 | local: mostly to L2/3 and within L4; receives thalamic core input | ● |
| **L5 IT** (Rorb/Deptor, "L4/5 IT", "L5 IT") | L5 = ~21% of rat S1 column; IT ≈ 70–80% of L5 excitatory in mouse, ~94–98% in human (complement of ET) | rat, mouse, human | cortex (both hemispheres), striatum; feedback to lower areas | ◐ |
| **L5 PT / ET** (Fam84b, Pou3f1; Betz cells in M1) | **mouse 20–30% of L5 excitatory; human 2–6% (~5%)** → ≈4–6% of all neurons in mouse S1, <1–2% in human (○ our arithmetic) | mouse vs human | thalamus (higher-order, driver), superior colliculus, pons, brainstem, spinal cord; collaterals to striatum | ● ratio / ○ absolute |
| **L6 CT** (Foxp2, Syt6; Ntsr1 in mouse) | 30–50% of L6 pyramidal cells → ≈6–10% of a rat S1 column (○); **rarer in primates than mouse** | review (mostly rodent/cat) | thalamus (relay nucleus of same modality + TRN); modulator, not driver | ◐ |
| **L6 IT** (incl. L6 IT Car3) | the other ~50–70% of L6 pyramidal cells (CC cells) | review | cortex, claustrum (Car3 type strongly to claustrum) | ◐ |
| **L6b** (Cplx3, Ctgf/Ccn2) | ≈5–6% of a rat S1 column (L6B layer count, includes some non-L6b cells) | rat | L1 and L6 of cortex, some thalamus; subplate remnant | ○ |
| **L5/6 NP** (near-projecting; Tshz2) | a few %; **no reliable number found** | mouse | local, within a few hundred μm; little long-range | ○ |

Notes:
- "IT ≈ 70% of all excitatory neurons" (mouse MOp, Zhang 2021) is the most trustworthy single number; the remaining
  ~30% are ET + CT + NP + L6b.
- Human: L5 ET and L6 CT are **significantly rarer** than in mouse (Bakken 2021) → human cortex is even more IT-dominated.

## 2. Inhibitory types (all GABA, all local)

Shares are of all interneurons, mouse S1 unless stated (Rudy 2011; Tremblay 2016).

| Type (marker) | % of interneurons | Species | Targets | Conf. |
|---|---|---|---|---|
| **PV group** (Pvalb) total | **~40%** | mouse S1 | — | ● |
| · PV basket | the large majority of PV ("most abundant interneuron type in all areas and layers") | mouse | soma + proximal dendrites of pyramids and other PV cells | ● |
| · PV chandelier / axo-axonic (Pvalb ± ; Vipr2, Unc5b) | a small minority of PV; **no reliable % found**; more numerous in L2/3 than L4–6 | mouse | axon initial segment of pyramidal cells only | ○ |
| **SST group** (Sst) total | **~30%** | mouse S1 | — | ● |
| · SST Martinotti (Calb2+ subset 15–30%) | "vast majority" of supragranular SST; ~60% of SST is quoted but unverified | mouse | axon to L1 → apical tufts | ◐ / ○ number |
| · SST non-Martinotti (L4-targeting, X94-like; long-range Chodl/Nos1) | remainder (~40%?) | mouse | L4 cells stay in L4 and prefer **PV cells** (disinhibit L4); Chodl = long-range sleep-active | ○ |
| **5HT3aR group** (Htr3a) total | **~30%** | mouse S1 | — | ● |
| · VIP (Vip; 50–70% also CR, 10–30% CCK) | ~40% of 5HT3aR → **~12% of interneurons** | mouse | mainly **SST cells** (disinhibition); densest in L2/3 | ● |
| · LAMP5 / neurogliaform (Lamp5, Ndnf in L1) | part of the ~60% non-VIP 5HT3aR (~18% of interneurons, shared with SNCG) | mouse | volume release of GABA onto nearby dendrites; GABA-B | ◐ |
| · other 5HT3aR (Sncg / CCK basket, single bouquet) | rest of the non-VIP 5HT3aR | mouse | soma (CCK basket) or dendrites | ○ |
| L1 overall | non-VIP 5HT3aR = ~90% of L1 interneurons | mouse | — | ● |

Primates: the extra interneurons are mostly in the **5HT3aR/VIP/calretinin** family. Calretinin interneurons are ≤4% of
neurons in rodents but up to ~12% in higher-order primate areas; PV and SST shares are fairly constant across mammals
(Clowry review; Hladnik 2014). Individual human GABAergic types are each rare (<4.5% of neurons; Bakken 2021).
**No verified human PV/SST/VIP/LAMP5 percentages found**; the Allen human MTG/SEA-AD taxonomy tables would have them.

## 3. Human-specific (or primate-enriched) cells

| Cell | Where | Numbers | Species | Conf. |
|---|---|---|---|---|
| **Rosehip neuron** (GAD1+, CCK+; SST−, CALB2−, PVALB−, CNR1−) | **layer 1**, human temporal cortex (studied in MTG) | **10–15% of L1 interneurons** (FISH, 408 GAD1+ cells, 2 subjects; 10/76 morphologies) | human; no mouse match | ◐ |
| **Von Economo neuron (VEN)**, large bipolar/spindle L5 projection cell | layer 5 of **frontoinsular cortex (FI, anterior insula)** and **anterior cingulate (limbic anterior area, LA)**; also some dlPFC area 9 | adult FI: ~35,000–90,000 (left), ~48,000–128,000 (right) per hemisphere, **~0.9–1.6% of FI neurons**; adult ACC (LA): ~67,000–180,000 (left), ~127,000–213,000 (right), **~0.6–0.9% of LA neurons**; 20–40% more in the **right** hemisphere; absent before 36 weeks post-conception, rise in first 8 months | human (stereology, Allman 2010); also great apes (fewer), macaque insula (homolog, Evrard 2012), whales, elephants | ● counts / ◐ function |
| Betz cell (giant L5 ET) | L5b of primary motor cortex | inside 2 L5 ET clusters; not separately counted | human, primates | ◐ |
| Double-bouquet SST cell | L2/3 | not counted | primate-specialised | ○ |

## 4. Area differences

| Feature | Primary sensory (V1, S1, A1) | Primary motor (M1) | Prefrontal (dlPFC / ACC) |
|---|---|---|---|
| Layer 4 | thick, granular; primate V1 subdivided 4A/4B/4Cα/4Cβ; ~23% of neurons in rat S1 | **agranular**: L4-like cells exist but sparse and less diverse | primate dlPFC dysgranular/granular (thin L4); ACC and rodent PFC agranular |
| L5 ET / PT | present, fewer | **largest and most numerous** (Betz cells, corticospinal) | present; ACC/insula have VENs |
| E:I ratio | human V1 **>2× other areas** (fewer interneurons) | human ~2:1 (33% GABA) | similar to M1 |
| Interneuron flavour | V1 has its own SST types | — | more calretinin/VIP in higher-order primate areas |
| Thalamic input | core, first-order (LGN, VPM, MGB) into L4 | VL/VA (cerebellar) into L3/L5 | mediodorsal (MD) + matrix into L1/L3 |

## 5. Signature behaviors to model (testable)

| Type | Behavior | Test idea |
|---|---|---|
| L2/3 IT | sparse firing, **spike-frequency adaptation**; human: **dendritic Ca²⁺ action potentials** whose amplitude is largest at threshold and drops with stronger input (Gidon 2020 → XOR-like) | constant drive: peak then settle (already in our tests); human variant: response to 1 input > response to 2 inputs on same dendrite |
| L4 stellate | no apical tuft; strong, synchronous thalamic synapses (~10% of synapses); fast, regular-spiking | thalamic burst drives it alone; cortical input alone weak |
| L5 IT | regular spiking, thinner tuft, no BAC burst | basal+apical sum ≈ linear |
| L5 ET | **BAC firing** (basal + apical coincidence → burst), strong h-current (Ih), intrinsic bursting | already in our tests; plus: sag in response to hyperpolarizing step |
| L6 CT | low firing, **facilitating** synapses onto thalamus; sparse spontaneous activity | paired-pulse ratio >1 at its thalamic target |
| L6b | responsive to orexin/hypocretin (wake-promoting) | rate rises with orexin signal |
| PV basket | fast-spiking (up to several hundred Hz), no adaptation, **depressing** excitatory input, perisomatic divisive inhibition, gamma generator | rate keeps rising with drive; gain of target divided |
| Chandelier | inhibits/controls spike initiation at the axon initial segment (sign can be depolarizing in some conditions) | veto of target output regardless of dendritic input |
| SST Martinotti | low-threshold spiking, **facilitating** excitatory input, inhibits apical tufts | ramp under sustained drive; gates apical input |
| SST L4 non-Martinotti | targets PV cells → disinhibition of L4 | activating it raises L4 pyramidal output |
| VIP | irregular spiking, high input resistance, driven by top-down + ACh (nicotinic), inhibits SST | VIP on → SST down → apical input passes |
| LAMP5 / NGF | late-spiking; slow GABA-A + **GABA-B** volume inhibition; electrically coupled | slow, long-lasting, non-specific dampening of nearby cells |
| Rosehip | compact "rosehip" bouton clusters on apical dendritic shafts of human L3 pyramids; segment-specific control of dendritic electrogenesis | inhibits apical-driven bursts of a single target without changing somatic response |
| VEN | large fast-conducting L5 projection cell; output to frontal pole, insula, and subcortex; delayed maturation (postnatal); right-lateralised | build with fewer, larger cells; long axon, low-latency relay of interoceptive/salience signals (◐ function) |

## 6. Main connections

- **Feedforward** (lower → higher area): L2/3 IT (and L4 recipient) → L4/L3 of target.
- **Feedback** (higher → lower): L5 IT and L6 IT → L1 (apical tufts) and L6 of target.
- **Cortico-thalamic:** L6 CT → same-modality relay nucleus (modulator) + TRN; L5 ET → higher-order thalamus (driver) =
  basis for transthalamic cortico-cortical paths (Sherman).
- **Subcortical outputs:** L5 ET → striatum (collaterals), superior colliculus, pons, brainstem, spinal cord; L5/L2/3 IT → striatum.
- **Inputs:** thalamic core → L4 (and L3/L5b); thalamic matrix → L1; claustrum → most areas (mostly to interneurons);
  neuromodulators: ACh (basal forebrain), NE (LC), 5-HT (raphe, onto 5HT3aR cells), DA (mostly frontal).
- **Local inhibitory motifs:** PV ↔ pyramids (feedback/feedforward), SST → tufts, VIP → SST, L1 NGF → tufts.

## 7. Human vs rodent differences that matter

1. **About twice the interneuron fraction** (33% vs 16% in M1), mostly from the CGE/5HT3aR/VIP/calretinin family.
2. **L2/3 IT expanded**, L5 ET and L6 CT reduced (L5 ET 2–6% of L5 excitatory vs 20–30% in mouse).
3. Human L2/3 pyramidal dendrites are longer and more compartmentalized (dendritic Ca²⁺ APs, more Ih).
4. **Human-only or hominid-enriched cells**: rosehip (L1), VENs (FI, ACC), Betz cells (also in other primates).
5. Area differences are sharper in humans: V1 has >2× the E:I ratio of other areas and its own cell types.

## 8. Sources

- Azevedo et al. 2009, *J Comp Neurol* 513:532 — 86 billion neurons, 19% in cortex. https://doi.org/10.1002/cne.21974
- Pakkenberg & Gundersen 1997, *J Comp Neurol* 384:312 — neocortical neuron number by sex.
- Bakken et al. 2021, *Nature* 598:111 — human/marmoset/mouse M1; 33/23/16% GABAergic; ET & CT rarer in primates. https://pmc.ncbi.nlm.nih.gov/articles/PMC8494640/
- Zhang et al. 2021, *Nature* 598:137 — MERFISH MOp; IT ≈ 70% of excitatory. https://pmc.ncbi.nlm.nih.gov/articles/PMC8494645/
- Jorstad et al. 2023, *Science* 382:eadf6812 — human areal differences; V1 E:I >2×. https://pubmed.ncbi.nlm.nih.gov/37824655/
- Hodge et al. 2019, *Nature* 573:61 — human MTG taxonomy.
- Kalmbach et al. 2021, *Neuron* (bioRxiv 2020.11.02.365080) — human L5 ET = 2–6% of L5 excitatory vs 20–30% mouse. https://pmc.ncbi.nlm.nih.gov/articles/PMC8570452/
- Tasic et al. 2018, *Nature* 563:72 — mouse VISp/ALM taxonomy (IT/PT/CT/NP/L6b nomenclature). https://pmc.ncbi.nlm.nih.gov/articles/PMC6456269/
- Meyer et al. 2010, *Cereb Cortex* 20:2277 — neurons per layer in rat barrel column. https://pubmed.ncbi.nlm.nih.gov/20534784/
- Meyer et al. 2011, *PNAS* 108:16807 — interneurons ~12% in rat barrel cortex. https://doi.org/10.1073/pnas.1113648108
- Rudy et al. 2011, *Dev Neurobiol* 71:45 — PV 40 / SST 30 / 5HT3aR 30%. https://pmc.ncbi.nlm.nih.gov/articles/PMC3556905/
- Tremblay, Lee & Rudy 2016, *Neuron* 91:260 — VIP ≈ 40% of 5HT3aR; L1 composition; signature properties. https://pmc.ncbi.nlm.nih.gov/articles/PMC4980915/
- Thomson 2010, *Front Neuroanat* 4:13 — CT = 30–50% of L6 pyramidal cells. https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2010.00013/pdf
- Clowry 2018 / Hladnik et al. 2014 — primate interneuron fraction 25–34%, calretinin expansion. https://pmc.ncbi.nlm.nih.gov/articles/PMC4072090/
- Boldog et al. 2018, *Nat Neurosci* 21:1185 — rosehip cells, 10–15% of L1 interneurons. https://pmc.ncbi.nlm.nih.gov/articles/PMC6130849/
- Allman et al. 2010, *Brain Struct Funct* 214:495 — VEN stereology (Tables 2–3). https://pmc.ncbi.nlm.nih.gov/articles/PMC3140770/ (Annals version) / https://doi.org/10.1007/s00429-010-0254-0
- Evrard, Forro & Logothetis 2012, *Neuron* 74:482 — VENs in macaque insula.
- Gidon et al. 2020, *Science* 367:83 — dendritic Ca²⁺ APs in human L2/3.
- Sherman 2016 / chapter "Thalamocortical circuitry matters" — L5 → higher-order thalamus drivers.

**Not found / uncertain:** NP and L6b shares; chandelier share of PV; Martinotti share of SST (60% unverified);
human subclass percentages for PV/SST/VIP/LAMP5; absolute human counts per excitatory subclass.

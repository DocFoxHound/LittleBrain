# Hippocampal formation: neuron types and proportions

Scope: entorhinal cortex (EC), dentate gyrus (DG), CA3, CA2, CA1, subiculum.
Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain.
"Rat" and "mouse" are always labeled. Unlabeled human counts are **per hemisphere**.

## 0. Whole-formation census (principal-cell layers)

| Subfield | Human (West & Gundersen 1990; 5 hippocampi, CV in brackets) | Rat (Arellano & Rakic 2025 meta-analysis of 89 stereology studies, adult, per hemisphere) | Human / rat |
|---|---|---|---|
| EC, all layers | ~7 M (cognitively normal, Gómez-Isla et al. 1996); ~13 M in West & Slomianka 1998 (method differences) ◐ | ~700,000 (≈60% MEC, 40% LEC) | ~10–20× |
| EC layer II | ~650,000 (Gómez-Isla 1996); ~1 M (West & Slomianka 1998) ◐ | 108,000 (MEC ~60k, LEC ~40k) | ~6–9× |
| EC layer III | ~5 M (West & Slomianka 1998) ○ | 270,000 | |
| DG granule cell layer | **15 M** (0.28) ● | **~1.03 M** | ~15× |
| Hilus (mossy cells + hilar interneurons) | **2.0 M** (0.16) ● | ~48,000–50,000 | ~40× |
| CA3 + CA2 | **2.7 M** (0.22) ● | ~240,000 (CA3 ~210k, CA2 ~30k, CA2 called unreliable) | ~11× |
| CA1 | **16 M** (0.32); range 24 → 11 M, falls with age (47–85 y) ● | ~350,000 | ~45× |
| Subiculum | ~4.5 M ◐ (my recollection of West & Gundersen 1990; the abstract I could access did not show it) | ~280,000–300,000 | ~15× |

Derived ratios that matter for modelling (from the table above):
- **CA1 : CA3 ≈ 6 : 1 in human vs ≈ 1.5 : 1 in rat.** CA1 is the part that expanded most in primates.
- **Granule cells per EC layer II cell: ≈ 15–23 in human, ≈ 10 in rat.** The expansion step into DG is stronger in humans.
- Granule cells : CA3 ≈ 5–6 : 1 in human and ≈ 4–5 : 1 in rat. This is the convergence onto CA3.

**Interneuron share.** Rodent models assume **~11% of CA1 and CA3 neurons are GABAergic** (Bezaire & Soltesz 2013; older reviews give ~10%, Freund & Buzsáki 1996) ◐. For human, Hungarian stereology (J Neurosci 2025, bioRxiv 2023 "Absolute number of three populations of interneurons and all GABAergic synapses in the human hippocampus") counted only three marker classes per hemisphere: **PV ≈ 49,400, SST ≈ 141,500, CR ≈ 250,600** (≈ 441,000 together). That is **≈ 1.1% of ~40 M principal-layer neurons**. All interneurons together must be several times that. **No full human interneuron census exists**, so I found no total human percentage ○. The same study estimates **~2.5 × 10¹⁰ GABAergic synapses per hemisphere**, of which **PV boutons are ~14%**. Calretinin cells are the largest of the three human classes. In rodent, CR cells are mostly interneuron-specific (IS) cells and are a minority. This points to more disinhibitory circuitry in humans ○.

---

## 1. Entorhinal cortex (EC)

| Type (marker) | % of EC neurons | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **MEC L2 stellate** (Reelin+) | roughly 2/3 of MEC L2 principal cells | rat/mouse | MEC L2 total ~60k (rat) | Glu | E · projection → **DG + CA3** (perforant path, outer/middle molecular layer) | ◐ |
| **MEC L2 pyramidal, "island" cells** (Calbindin+, Wfs1+) | roughly 1/3 of MEC L2 principal cells, in ~ 100 µm clusters | rat/mouse | — | Glu | E · projection → **CA1 stratum lacunosum-moleculare** (temporoammonic) and contralateral EC; contact L1 interneurons | ◐ |
| **LEC L2 fan cells** (Reelin+) | majority of LEC L2 | rat/mouse | LEC L2 ~40k (rat) | Glu | E · → DG/CA3 (lateral perforant path) | ◐ |
| **L3 pyramidal** | ~38% of all EC neurons | rat | ~270k (rat); ~5 M (human, ○) | Glu | E · → **CA1** (proximal-distal topography) and subiculum; LEC → distal CA1, MEC → proximal CA1 | ◐ |
| **L5/6 pyramidal** | ~47% of EC neurons | rat | ~340k (rat) | Glu | E · receive CA1/subiculum output; → neocortex, and back into superficial EC | ◐ |
| **EC interneurons** (PV basket, SST, CR, VIP, CCK) | not well quantified; assume cortical-typical ~10–20% | rodent | — | GABA | I · local. PV cells mediate stellate↔stellate lateral inhibition, since direct stellate–stellate excitation is almost absent (Couey et al. 2013) | ○ |

**Functional (not anatomical) types**, rat MEC:
| Functional type | Share | Where | Conf. |
|---|---|---|---|
| Grid cells | Often quoted as ~10–30% of MEC L2/3 cells, varying strongly by layer, dorsal-ventral position and recording method. Present in both stellate ("ocean") and island cells in similar proportions (Sun et al. 2015) | MEC L2 > L3, L5 | ○ (I could not verify one figure) |
| Border cells | **~10%** of recorded cells, all layers (Solstad et al. 2008) | MEC + parasubiculum | ◐ |
| Head-direction / conjunctive grid×HD | common in L3, L5, L6; rare in L2 (Sargolini et al. 2006); fewer sharply tuned HD cells ventrally | deep layers | ◐ |
| Speed cells, object-vector cells | minority | MEC | ○ |

Human: grid-like cells have been recorded intracranially (Jacobs et al. 2013), with grid-like fMRI signals as well. There are no proportions.

### Signature behaviors (EC)
- **L2 stellate:** HCN/Ih **sag** of ≥ 20% to hyperpolarizing steps. Persistent Na⁺ current giving **subthreshold membrane-potential oscillations at theta (4–10 Hz)**. Resonance frequency is **higher dorsally than ventrally**, and this gradient tracks the grid-spacing gradient. *Test:* a step to −10 mV gives sag; a ZAP/chirp input gives peak resonance at theta. Dorsal cell f_res > ventral cell f_res.
- **Grid cell (network property):** firing fields on a hexagonal lattice (60° symmetry). Spacing grows in discrete **modules** along the dorsoventral axis, with ratio ≈ 1.4 (Stensola et al. 2012). *Test:* the gridness score of the rate map is > 0.3, and the autocorrelogram has 6 peaks.
- **L2 island cell:** fires on theta phase earlier than stellate cells, with strong theta modulation. Output targets CA1 (temporal-association/"when" learning; Kitamura et al. 2014).
- **L3 pyramidal:** **persistent firing** for seconds after brief input, under cholinergic (muscarinic, CAN-current) drive (Egorov et al. 2002). *Test:* with ACh on, a 1 s input leaves rate > 0 for > 5 s, and without ACh it decays. This is useful as a working-memory buffer.
- **L5:** graded persistent activity, also cholinergic.

---

## 2. Dentate gyrus (DG)

| Type (marker) | % of DG neurons | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **Granule cell, mature** (Prox1, Calbindin) | ~95% | rat (≈1.03 M of ≈1.08 M); human similar | **15 M human**; 1.0 M rat | Glu (+Zn²⁺, dynorphin) | E · **mossy fibers → CA3** (each cell contacts only ~10–15 CA3 pyramids via giant boutons) + hilar mossy cells + interneurons | ● |
| **Adult-born granule cell, 4–8 weeks** (DCX, PSA-NCAM while immature) | rodent: thousands per day, a few % of GCs young at any time. Human: **disputed**, from ~0 to a sizeable fraction (see below) | rat / human | Human: **~700 new neurons per hippocampus per day** (Spalding et al. 2013) | Glu (early on, GABA is depolarizing for them) | E · same targets as mature cells | rodent ●, human ○ |
| **Mossy cell** (GluR2/3, calretinin in ventral mouse) | **~57–65% of hilar neurons** (Buckmaster & Jongen-Rêlo 1999: 64%) → ~3% of DG | rat | ~30,000 rat (derived: 0.64 × ~48k) | Glu | E · **associational/commissural**: inner molecular layer of DG, **mostly far along the long axis and contralateral** (rodent). Excite GCs directly and inhibit them via interneurons | ◐ |
| **Hilar interneurons**, total | ~35–45% of hilar neurons; DG interneurons overall ~2–4% of DG neurons | rat | ~15–20k hilar (rat, derived) | GABA | I · local | ◐ |
| ↳ PV basket (pyramidal basket) | largest share of GCL-border interneurons | rat | — | GABA | → GC somata | ◐ |
| ↳ PV axo-axonic (chandelier) | small | rat | — | GABA | → GC axon initial segment | ● (type), ○ (%) |
| ↳ **HIPP** (hilar perforant-path-associated, SST+) | most numerous hilar interneuron | rat | — | GABA | → **outer molecular layer** (GC distal dendrites, gating perforant-path input) | ◐ |
| ↳ HICAP (CCK+), MOPP, neurogliaform, IS cells | minority | rat | — | GABA | inner/outer molecular layer, other interneurons | ○ |

Human DG: the hilus ("CA4") holds 2.0 M neurons, which is relatively larger than in rat. Human mossy-cell vs interneuron split is **not known** ○. Primates have **weak or absent commissural** mossy-cell projections, while ipsilateral longitudinal projections remain.

### The human adult-neurogenesis controversy
| Study | Method | Claim |
|---|---|---|
| Spalding et al. 2013, *Cell* | ¹⁴C bomb-pulse dating of neuronal nuclei (55 people) | **~700 new neurons per hippocampus per day**. **1.75% yearly turnover within the renewing fraction** (about 1/3 of DG neurons). Modest decline with age. Rate similar to middle-aged mice |
| Sorrells et al. 2018, *Nature* | DCX/PSA-NCAM immunostaining | Neurogenesis falls to **undetectable after childhood (~7–13 y)** |
| Boldrini et al. 2018, *Cell Stem Cell* | stereology, 14–79 y | Thousands of immature neurons **persist across adulthood**. Quiescent stem-cell pool shrinks |
| Moreno-Jiménez et al. 2019, *Nat Med* | optimized tissue fixation | **Tens of thousands of DCX+ cells** into the 80s. Fall sharply in Alzheimer's disease |
| Franjic et al. 2022, *Neuron* | snRNA-seq (human, macaque, pig, mouse) | Neurogenic trajectories found in the other species, **not in adult human**. DCX is expressed broadly, not marking a young population |
| Zhou et al. 2022, *Nature* | snRNA-seq + machine learning | **Rare** immature granule cells and rare proliferating progenitors in adults, fewer in AD |

**Modelling stance:** treat human adult neurogenesis as **low-rate and uncertain**. A young-cell pool of 0–3% of granule cells is defensible. Rodent rates are an upper bound.

### Signature behaviors (DG)
- **Mature granule cell:** very hyperpolarized rest (≈ −80 mV), low input resistance, strong feedback inhibition. As a result **only ~1–5% are active in an environment** (sparse code), and most are silent. *Test (exists):* input overlap → lower output overlap (pattern separation).
- **Mossy fiber → CA3 synapse ("detonator"):** **strong frequency facilitation**, about 5–10× over a train (low release probability at rest). **Presynaptic, NMDA-independent, cAMP/PKA LTP.** *Test:* a 10-pulse train at 20 Hz gives a final EPSP ≥ 3× the first; a single GC burst can fire a CA3 cell.
- **Young granule cell (4–6 weeks):** higher input resistance, lower threshold, **enhanced LTP** (lower induction threshold, larger magnitude), weaker inhibition, broadly tuned. *Test (exists):* threshold and learning rate.
- **Mossy cell:** **high spontaneous activity and multi-field (multi-place) firing**, unlike sparse GCs (GoodSmith et al. 2017; Senzai & Buzsáki 2017). Drives distant GCs and interneurons. *Test:* given the same input, a mossy cell has more active fields than a granule cell.
- **HIPP (SST):** facilitating input; gates distal perforant-path input on GC dendrites, much like Martinotti cells in cortex.

---

## 3. CA3

| Type (marker) | % of CA3 neurons | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **CA3 pyramidal** | **~89%** | rat (modelling assumption) | ~210k rat; human CA3+CA2 = 2.7 M | Glu | E · **recurrent collaterals within CA3** (both sides in rodents), **Schaffer collaterals → CA1**, back-projection to the DG hilus, lateral septum | ● (type), ◐ (%) |
| ↳ thorny-excrescence ("thorny") vs athorny | athorny cells are a minority in deep CA3a/b (Hunt et al. 2018). They fire first in sharp waves | mouse | — | Glu | E | ○ |
| **Interneurons**, total | **~11%** | rat (Bezaire & Soltesz assumption) | ~25k rat (derived) | GABA | I · local; same families as CA1 (PV basket, axo-axonic, bistratified, O-LM, CCK, ivy/NGF, IS), plus **mossy-fiber-driven feedforward interneurons** in stratum lucidum | ◐ |

**Recurrent connectivity:** pairwise connection probability between CA3 pyramids is **~0.9% in rat** (Guzman et al. 2016, *Science*: 0.92%, flat over 400 µm, rich in disynaptic motifs). Human also shows < 1% (Watson et al. 2024). A 2025 optical-recording preprint argues it is higher and random. A rat CA3 cell gets the order of **~12,000 recurrent synapses, ~4,000 perforant-path synapses and ~46 mossy-fiber inputs** (classic estimates used in Treves & Rolls models) ◐.

### Signature behaviors (CA3)
- **Autoassociative recall (pattern completion):** a partial cue (e.g., 30% of a stored pattern) recalls the whole through recurrent excitation. Guzman et al. show it works even with sparse connectivity, because of motifs and dendritic integration. *Test:* store K patterns with Hebbian recurrent weights; a 30% cue converges to correlation > 0.9 with the stored pattern.
- **Recurrent synapses: NMDA-dependent, symmetric-STDP-like LTP** (at CA3–CA3 the STDP window is symmetric, Mishra et al. 2016), which suits associative storage. Mossy-fiber input is presynaptic LTP (see DG).
- **Intrinsic bursting** in a subset of pyramids. CA3 generates **sharp-wave ripples** (150–250 Hz on CA1 output) during rest and sleep, replaying sequences.
- Input logic: **mossy fibers = "teacher"/storage drive** (strong, few). **Perforant path = recall cue** (weak, many).

## 4. CA2

| Type (marker) | % | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **CA2 pyramidal** (RGS14, PCP4, STEP, AVPR1b) | ~10% of CA3+CA2 pyramids (rat ~30k, unreliable) | rat | human counted with CA3 (2.7 M together) | Glu | E · → **CA1 stratum oriens** (deep CA1 cells), CA2/CA3, contralateral, **supramammillary/septum**; gets EC L2 direct input and **hypothalamic (vasopressin, SuM) input** | ◐ |
| CA2 interneurons (dense PV basket plexus) | high PV density | rodent | — | GABA | I | ○ |

### Signature behaviors (CA2)
- **Schaffer collateral (CA3→CA2) synapses resist LTP** (RGS14, high Ca²⁺ buffering, strong feedforward inhibition) (Zhao et al. 2007). *Test:* a tetanus that potentiates CA1 by > 40% gives < 10% in CA2. EC L2 → CA2 distal inputs do potentiate.
- **Social memory:** needed for remembering a familiar conspecific (Hitti & Siegelbaum 2014). Place fields drift ("remap") over time, giving a temporal code.
- Can **drive CA1 and ripples during waking immobility**; vasopressin (AVPR1b) increases output.

---

## 5. CA1

| Type (marker) | % of CA1 neurons | % of CA1 interneurons | Species | Abs. count (rat) | Transmitter | Axon target | Conf. |
|---|---|---|---|---|---|---|---|
| **Pyramidal** (deep vs superficial sublayers) | **89%** | — | rat | **311,500** (human ~16 M) | Glu | E · projection → **subiculum, EC deep layers**, prefrontal cortex (ventral CA1), amygdala, nucleus accumbens, septum. Weak recurrence (<1%) | ● |
| **All interneurons** | **11%** | 100% | rat | **38,500** | GABA | I | ◐ |
| **PV basket** (PV, fast-spiking) | 1.6% | 14.4% | rat | 5,530 | GABA | soma and proximal dendrites | ◐ |
| **Axo-axonic / chandelier** (PV) | 0.4% | 3.8% | rat | 1,470 | GABA | **axon initial segment** | ◐ |
| **Bistratified** (PV, SST, NPY) | 0.6% | 5.7% | rat | 2,210 | GABA | oriens + radiatum dendrites (co-aligned with CA3 input) | ◐ |
| **O-LM** (SST, mGluR1α) | 0.5% | 4.3% | rat | 1,640 | GABA | **stratum lacunosum-moleculare** (distal tufts receiving EC input) | ◐ |
| Other SST long-range (double-projection, oriens-retrohippocampal, septal-projecting) | 0.5% | 4.1% | rat | 1,590 | GABA | **septum, subiculum, retrohippocampal cortex** (long-range inhibitory) | ○ |
| **CCK basket** (CCK, CB1R, VIP in some) | 1.0% | 9.4% | rat | 3,600 | GABA | soma; **asynchronous release, endocannabinoid-suppressed** | ◐ |
| CCK dendrite-targeting (Schaffer-associated, perforant-path-associated, apical-dendrite-innervating, misc.) | 0.5% | 4.5% | rat | 1,770 | GABA | radiatum / lacunosum-moleculare dendrites | ○ |
| **Ivy** (NPY, nNOS) | **2.5%** | **22.9%** (most numerous type) | rat | 8,810 | GABA | dense **dendritic** plexus in oriens/radiatum; slow GABA_A + GABA_B | ◐ |
| **Neurogliaform** (NPY, nNOS, reelin) | 1.0% | 9.3% | rat | 3,580 | GABA | lacunosum-moleculare; **volume transmission**, slow GABA_B | ◐ |
| **Interneuron-specific IS-I / II / III** (CR; VIP; VIP+CR) | 2.1% | 19.4% (11.0 / 5.1 / 3.2) | rat | 7,470 | GABA | **other interneurons only** (IS-III → O-LM; IS-II → dendrite-targeting cells) | ◐ |
| Other / unclassified (trilaminar, back-projection, radiatum-retrohippocampal, large CB) | ~1% | ~1.2% | rat | ~480 + uncounted | GABA | multiple | ○ |

Source: Bezaire & Soltesz 2013, *Hippocampus*. Their totals are a **modelling knowledge base** built from mixed stereology, and they assumed 11%. Treat single-type values as ± 50%.

Human CA1: no per-type census. Human interneurons are relatively **richer in CR+ (IS-type) cells** than rodent ○. Human pyramidal cells are larger and more compartmentalized, with longer dendrites.

### Signature behaviors (CA1)
- **Pyramidal, place cell:** one or a few place fields; **theta phase precession** (spikes move earlier in the theta cycle as the animal crosses the field). *Test:* phase vs position slope < 0.
- **Pyramidal, plasticity rule:** classic NMDA-dependent Schaffer LTP/LTD (Hebbian, STDP). Also **behavioral timescale synaptic plasticity (BTSP)**: one dendritic **plateau potential** (from coincident EC + CA3 input) potentiates inputs active **over a seconds-wide window (−3 to +2 s)**, so a place field forms in one trial (Bittner et al. 2017). *Test:* one plateau at position x creates a field centred slightly before x; 100 ms STDP pairing is not required.
- **Pyramidal, comparator:** proximal dendrites get CA3 (recalled/predicted), distal tufts get EC L3 (current sensory input). **Mismatch** produces a novelty signal. Deep vs superficial sublayer cells differ in place-field stability and in reward/landmark coding.
- **PV basket:** fast-spiking (> 100 Hz, no adaptation); fires **just after the pyramids** at the theta trough; paces **gamma (30–80 Hz)**. Perisomatic, divisive inhibition. *Test:* same as cortical PV (gain control) plus rhythm.
- **Axo-axonic:** fires **at the theta peak** and **is silenced during sharp-wave ripples** (Klausberger et al. 2003). Can veto output at the axon initial segment.
- **Bistratified:** fires strongly **during ripples** and at the theta trough. It controls Schaffer-input integration in dendrites.
- **O-LM:** fires **at the theta trough, phase-locked**. Facilitating input from local pyramids, so it is recruited by sustained CA1 output (feedback). It **inhibits EC input to distal tufts**, so it favours CA3 (recall) over EC (encoding) (Leão et al. 2012). *Test:* sustained CA1 activity raises O-LM, which suppresses tuft-driven responses but not Schaffer-driven ones.
- **CCK basket:** regular-spiking, adapting. **Asynchronous GABA release.** **Depolarization-induced suppression of inhibition (DSI)** through CB1 receptors (retrograde endocannabinoids). Excited by 5-HT3 and nicotinic input. Shows mood/state-dependent tuning. *Test:* a postsynaptic burst transiently cuts CCK inhibition onto that cell by > 50% for seconds.
- **Ivy and neurogliaform:** late-spiking. **Slow, long-lasting inhibition (GABA_A slow + GABA_B, hundreds of ms)** via volume transmission. Electrically coupled. Ivy cells fire at theta-trough phase. *Test:* IPSC decay τ ≫ PV IPSC (e.g., > 50 ms vs ~5 ms).
- **IS cells (VIP/CR):** disinhibition by targeting O-LM and other interneurons. This is the same motif as cortical VIP. *Test:* reuse `test_vip_disinhibition` with O-LM in place of SST.
- **Theta-phase map:** different classes fire at different theta phases (Klausberger & Somogyi 2008). This is a good integration test for a theta-generating network.

---

## 6. Subiculum

| Type | % | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **Bursting pyramidal** (intrinsically bursting) | ~50–80% of pyramids, more distally | rat | rat ~280–300k total neurons; human ~4.5 M ◐ | Glu | E · → **EC deep layers, mammillary bodies, anterior thalamus, nucleus accumbens, lateral septum, prefrontal, amygdala, hypothalamus** | ◐ |
| **Regular-spiking pyramidal** | remainder, more proximally | rat | — | Glu | E · ditto, with different target bias | ◐ |
| Interneurons (PV, SST, CCK…) | ~10% assumed | rodent | — | GABA | I · local | ○ |

### Signature behaviors (subiculum)
- **Burst firing** (Ca²⁺/Na⁺ spike after-depolarization) is the main output mode. Bursting and regular-spiking cells follow **different LTP rules**: in bursting cells LTP is presynaptic (cAMP/PKA); in regular-spiking cells it is NMDA-dependent and postsynaptic (Wozny et al. 2008).
- **Boundary-vector cells, axis and direction cells:** firing set by distance and direction to boundaries. Broad, stable spatial fields.
- It is the **main output** of the hippocampus, sending it to the cortex and subcortex.

---

## 7. Main connections (trisynaptic loop and shortcuts)

```
EC L2 stellate/fan ──perforant path──► DG granule ──mossy fibers──► CA3 ──Schaffer──► CA1 ──► subiculum ──► EC L5/6 ──► neocortex
     │                                    ▲   │                      ▲ │recurrent         ▲                      │
     ├──────── direct ───────────────────────────────────────────────┘                    │                      ├► mammillary/ant. thalamus (Papez)
EC L3 pyramidal ──temporoammonic─────────────────────────────────────────────────────────►CA1 distal tufts   ├► accumbens, septum, PFC, amygdala
EC L2 island ───────────────────────────────────────────────────────────────────────────► CA1 SLM (via L1 IN)
DG mossy cell ⇄ granule cells (long-axis, contralateral in rodent); CA3 back-projection → hilus
CA2 ◄ EC L2, hypothalamus (SuM, vasopressin)  ;  CA2 ► CA1 deep pyramids
```
Modulatory inputs: **medial septum** (cholinergic plus GABAergic **theta pacemaker**, mostly onto PV/O-LM interneurons), **supramammillary nucleus**, **VTA and locus coeruleus dopamine** (novelty → LTP and replay), **raphe 5-HT** (onto CCK/5-HT3 interneurons), **nucleus reuniens** (PFC ↔ CA1).
Long-axis gradient: **dorsal (rodent)/posterior (human) = spatial, fine scale**; **ventral/anterior = emotion and context**, linked to the amygdala, hypothalamus and mPFC.

## 8. Human vs rodent differences that matter
1. **CA1 is massively expanded** (CA1:CA3 ≈ 6:1 in human vs 1.5:1 in rat). The hilus is relatively larger too.
2. **Commissural connections are weak in primates.** The two hippocampi are more independent; rodent DG/CA3 is bilaterally wired.
3. **Adult neurogenesis is uncertain** in humans (see DG). Rodent rates are an upper bound.
4. The human interneuron mix appears **CR-rich** (more IS/disinhibitory cells). Human PV cells account for only ~14% of GABAergic synapses ○.
5. Human CA3 recurrent connectivity is also sparse (<1%; Watson et al. 2024). Human pyramidal dendrites are larger, with more compartmental (non-linear) integration.
6. **Long axis:** primate anterior hippocampus ≈ rodent ventral. The human posterior hippocampus is relatively smaller.
7. **Age:** human CA1 loses neurons with age (24 → 11 M across 47–85 y in West & Gundersen; later work attributes much of it to pathology). EC layer II loses ~60% in very mild Alzheimer's disease (Gómez-Isla 1996). The model's "aging" or "disease" knobs should hit EC L2 first.

## 9. Sources
- West MJ, Gundersen HJ (1990) Unbiased stereological estimation of the number of neurons in the human hippocampus. *J Comp Neurol* 296:1–22. https://onlinelibrary.wiley.com/doi/abs/10.1002/cne.902960102
- West MJ, Slomianka L, Gundersen HJ (1991) Optical fractionator, rat hippocampus. *Anat Rec* 231:482–497. https://doi.org/10.1002/ar.1092310411
- Arellano JI, Rakic P (2025 preprint) Updated neuronal numbers of the rat hippocampal formation. https://www.biorxiv.org/content/10.1101/2025.04.16.649153v2.full
- Gómez-Isla T et al. (1996) Profound loss of layer II entorhinal cortex neurons in very mild AD. *J Neurosci* 16:4491–4500.
- West MJ, Slomianka L (1998) Total number of neurons in the layers of the human entorhinal cortex. *Hippocampus* 8:69–82 (+ corrigendum).
- Absolute number of three populations of interneurons and all GABAergic synapses in the human hippocampus. *J Neurosci* 2025, 45(10):e0372242024 (https://www.jneurosci.org/content/45/10/e0372242024); preprint https://www.biorxiv.org/content/10.1101/2023.09.26.559559v3
- Bezaire MJ, Soltesz I (2013) Quantitative assessment of CA1 local circuits. *Hippocampus* 23:751–785. https://pmc.ncbi.nlm.nih.gov/articles/PMC3775914/
- Freund TF, Buzsáki G (1996) Interneurons of the hippocampus. *Hippocampus* 6:347–470.
- Klausberger T, Somogyi P (2008) Neuronal diversity and temporal dynamics. *Science* 321:53–57.
- Pelkey KA et al. (2017) Hippocampal GABAergic inhibitory interneurons. *Physiol Rev* 97:1619–1747.
- Guzman SJ et al. (2016) Synaptic mechanisms of pattern completion in the hippocampal CA3 network. *Science* 353:1117–1123. https://doi.org/10.1126/science.aaf1836
- Buckmaster PS, Jongen-Rêlo AL (1999) Highly specific neuron loss preserves lateral inhibitory circuits in the dentate gyrus. *J Neurosci* 19:9519–9537.
- GoodSmith D et al. (2017) Spatial representations of granule cells and mossy cells of the dentate gyrus. *Neuron*. https://www.cell.com/neuron/fulltext/S0896-6273(16)30999-0
- Spalding KL et al. (2013) Dynamics of hippocampal neurogenesis in adult humans. *Cell* 153:1219–1227. https://pubmed.ncbi.nlm.nih.gov/23746839/
- Sorrells SF et al. (2018) *Nature* 555:377–381. Boldrini M et al. (2018) *Cell Stem Cell* 22:589–599. Moreno-Jiménez EP et al. (2019) *Nat Med* 25:554–560.
- Franjic D et al. (2022) Transcriptomic taxonomy and neurogenic trajectories of adult human, macaque, and pig hippocampal and entorhinal cells. *Neuron* 110:452–469. Zhou Y et al. (2022) *Nature* 607:527–533. https://www.med.upenn.edu/minglab/assets/user-content/documents/Nature_July2022.pdf
- Kitamura T et al. (2014) Island cells control temporal association memory. *Science* 343:896–901. Ray S et al. (2014) Grid-layout and theta-modulation of layer 2 pyramidal neurons in MEC. *Science* 343:891–896. Sun C et al. (2015) *PNAS* (islands vs oceans). https://web.stanford.edu/group/schnitzerlab/publications/Sun2015_PNAS.pdf
- Rowland DC et al. (2018) Functional properties of stellate cells in medial entorhinal cortex layer II. *eLife* 7:e36664. https://elifesciences.org/articles/36664
- Couey JJ et al. (2013) Recurrent inhibitory circuitry as a mechanism for grid formation. *Nat Neurosci* 16:318–324.
- Sargolini F et al. (2006) *Science* 312:758–762. Solstad T et al. (2008) *Science* 322:1865–1868. Stensola H et al. (2012) *Nature* 492:72–78. Jacobs J et al. (2013) *Nat Neurosci* 16:1188–1190.
- Egorov AV et al. (2002) Graded persistent activity in EC neurons. *Nature* 420:173–178.
- Bittner KC et al. (2017) Behavioral time scale synaptic plasticity. *Science* 357:1033–1036.
- Mishra RK et al. (2016) Symmetric spike timing-dependent plasticity at CA3–CA3 synapses. *Nat Commun* 7:11552.
- Zhao M et al. (2007) Synaptic plasticity (and the lack thereof) in CA2. *J Neurosci* 27:12025–12032. Hitti FL, Siegelbaum SA (2014) *Nature* 508:88–92.
- Leão RN et al. (2012) OLM interneurons differentially modulate CA3 and EC inputs. *Nat Neurosci* 15:1524–1530.
- Klausberger T et al. (2003) Brain-state- and cell-type-specific firing during network oscillations. *Nature* 421:844–848.
- Wozny C et al. (2008) Differential cAMP signaling at hippocampal output synapses. *J Neurosci* 28:14358–14362.
- Watson JF et al. (2024/2025, Jonas lab) Human hippocampal CA3 uses specific functional connectivity rules for efficient associative memory. *Cell*. (Reports <1% CA3–CA3 connectivity in human tissue.)

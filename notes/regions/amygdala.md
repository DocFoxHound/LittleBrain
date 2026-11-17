# Amygdala: neuron types and proportions

Scope: basolateral complex (BLA = lateral LA, basal BA, accessory basal AB), intercalated cell masses (ITC),
central nucleus (CeA: lateral/capsular CeL/CeC, medial CeM), plus a short note on the medial and cortical nuclei.
Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain. Unlabeled human counts are **per amygdala (one hemisphere)**.

**Organizing idea for the model:** the BLA is **cortex-like** (pyramidal-like glutamate cells plus a ~15–20% interneuron
minority, the same PV/SST/VIP/CCK families as cortex). The CeA is **striatum-like** (almost all GABAergic, with
mutually inhibiting projection cells). ITCs are a GABAergic **switch** between the two.

## 0. Whole-amygdala census (human)

| Nucleus | Neurons (×10⁶), mean ± SD | % of amygdala neurons | Source | Conf. |
|---|---|---|---|---|
| Lateral (LA) | **4.00 ± 0.40** | 32.8% | Schumann & Amaral 2005 (10 adult male controls) | ● |
| Basal (BA) | **3.24 ± 0.52** | 26.5% | " | ● |
| Accessory basal (AB) | **1.28 ± 0.20** | 10.5% | " | ● |
| Central (CeA) | **0.36 ± 0.08** | 2.9% | " | ● |
| Remaining nuclei (medial, cortical, periamygdaloid cortex, anterior amygdaloid area, NLOT, amygdalohippocampal area, **ITCs**) | 3.33 ± 0.53 | 27.3% | " | ● |
| **Total** | **12.21 ± 1.28** | 100% | " | ● |
| Total, other cohort | **15.3** (volume ~956 mm³) | — | García-Amado & Prensa 2012 (7 brains) | ◐ |

- **BLA ≈ 70% of human amygdala neurons; CeA ≈ 3%.**
- **Monkey vs rat** (Chareyron et al. 2011): monkeys have **14× (LA), 11× (BA), 16× (AB)** more neurons than rats, but only **2.3× (CeA) and 1.5× (medial)**. Primate expansion is concentrated in the cortex-connected BLA. Human BLA is further enlarged relative to apes (Barger et al. 2007/2012). Neuron density is 2.4–3.7× lower in monkey (larger cells, more neuropil).

---

## 1. Basolateral complex (LA, BA, AB)

| Type (marker) | % of BLA neurons | % of BLA interneurons | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|---|
| **Principal neuron** ("pyramidal-like", CaMKIIα; subpopulations Rspo2 (BA, negative valence) vs Ppp1r1b (BA, positive valence)) | **~80–85%** | — | rat/mouse (McDonald 1992; Sah et al. 2003) | human BLA total ≈ 8.5 M neurons → ~7 M principal if the rodent ratio holds ○ | Glu | E · **projection** → CeA (CeL, CeM), ITC, BNST, nucleus accumbens, mPFC/OFC, hippocampus (ventral CA1/subiculum), entorhinal and sensory cortex. Also local recurrent collaterals | ● type, ◐ % |
| **All GABAergic interneurons** | **~15–20%** | 100% | rat | — | GABA | I · local (some long-range SST/NPY cells project out) | ◐ |
| ↳ **PV** (often CB+; fast-spiking basket and **chandelier/axo-axonic**) | ~6–9% | **~40–50%** (range 19–43% by nucleus in some counts) | rat (McDonald & Mascagni 2001) | — | GABA | soma + axon initial segment of principal cells | ◐ |
| ↳ **SST** (often CB+, some NPY+) | ~2–3% | ~10–20% | rat | — | GABA | **distal dendrites** of principal cells; some long-range (→ entorhinal, basal forebrain) | ○ |
| ↳ **CCK, large type L** (CB+, CB1R+) | ~1–3% | ~10–15% | rat | — | GABA | soma (CCK basket); endocannabinoid-sensitive | ○ |
| ↳ **CR / VIP / small CCK type S** (VIP and/or CR, bipolar) | ~4–6% | **CR ~25–30%** (overlapping with VIP and small CCK cells) | rat (McDonald & Mascagni 2001; Mascagni & McDonald 2003) | — | GABA | mostly **other interneurons** (interneuron-selective, disinhibitory) | ◐ |
| ↳ NPY neurogliaform, others | small | ~5% | rodent | — | GABA | slow, volume inhibition | ○ |

Notes:
- The interneuron subtype shares above **overlap** (CB is co-expressed with both PV and SST, VIP with CR and small CCK). They do not sum neatly to 100%.
- **Primate:** I found no stereological interneuron fraction for monkey or human BLA ○. Cross-species single-cell data (Yu et al. 2023) report **LAMP5+ interneurons (neurogliaform-like) are much more abundant in primates**. Assume a 15–25% interneuron share in human, but flag it as uncertain.
- The BA contains **fear neurons** (respond to CS after conditioning) and **extinction neurons** (become active after extinction; Herry et al. 2008). These are functional, intermingled principal-cell sets with different projection targets: fear → prelimbic PFC; extinction → infralimbic PFC.

### Signature behaviors (BLA)
- **Principal cell, firing:** regular-spiking with **spike-frequency adaptation** (most cells). A minority burst. Slow AHP set by Ca²⁺-activated K⁺ channels; **noradrenaline (β) and ACh (muscarinic) reduce the AHP**, so the same input yields more spikes in arousal. *Test:* a 1 s step gives an adapting train; with NE on, the spike count rises ≥ 1.5×.
- **Principal cell, plasticity (core fear-learning rule):** **Hebbian, NMDA-dependent LTP** at thalamic (auditory MGm/PIN) and cortical (auditory cortex) inputs onto LA cells. It occurs when the **CS input coincides with strong US-driven (shock) depolarization**: the US acts as the "teacher", not the CS alone (Blair et al. 2001; optogenetic pairing in Johansen et al. 2010). It is **gated by neuromodulators**: needs β-adrenergic (NE) and dopamine signals at US time, and GABA_A inhibition must be released (VIP→PV/SST disinhibition during the US; Krabbe et al. 2019). *Test:* CS alone ×20 → no change; CS + US paired → CS response ≥ +50%; CS + US with NE blocked → much less LTP; US alone → no CS-specific change.
- **Memory allocation:** cells with higher **CREB-raised excitability** at training are recruited into the engram (~10–20% of LA principal cells; Han et al. 2007; Yiu et al. 2014). *Test:* raise excitability in a random 10% → those cells hold the memory; ablating them erases recall.
- **Extinction:** new learning, not erasure. BA extinction neurons and **IL-PFC → ITC** inhibition suppress fear output. **Spontaneous recovery, renewal (context change, hippocampus-dependent) and reinstatement** must still be possible. *Test:* after extinction, CS in a new context → fear returns.
- **PV interneuron:** fast-spiking, perisomatic. **Gamma/theta pacing** across BLA–PFC–hippocampus during fear. **During the US, PV cells are inhibited (disinhibiting principal-cell dendrites).** During the CS they inhibit SST cells, which also opens dendrites (Wolff et al. 2014).
- **SST interneuron:** dendritic gating of sensory input. **Inhibited by the CS** (via PV) to allow plasticity in principal-cell dendrites.
- **VIP interneuron:** **strongly excited by the US (aversive and unexpected)**. Disinhibits principal cells (VIP → SST/PV) to set the learning window. Its response shrinks as the US becomes expected, so it acts as a teaching signal like a prediction error (Krabbe et al. 2019). *Test:* reuse the cortical VIP disinhibition test, add "response to the 10th predicted US < response to the 1st".
- **CCK basket:** inhibition suppressed by endocannabinoids, which favours extinction (CB1 knockouts fail to extinguish; Marsicano et al. 2002).

---

## 2. Intercalated cell masses (ITC)

| Type | % | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **ITC neuron** (FoxP2+, μ-opioid receptor, D1 receptor; small, densely packed, medium spiny-like) | ~100% GABAergic. Share of total amygdala neurons not established (part of the 3.33 M "remaining" in human) | rodent; primate | **no reliable count** found ○ | GABA | I · **projection** (local but cross-nucleus): **medial paracapsular (mpITC) → CeM and CeL** (inhibits fear output); **lateral ITC → BLA**; ITC clusters chain-inhibit each other lateral → medial. Main ITC (between BLA and CeA) → CeA | ● type, ○ count |

- Rodents have **discrete clusters** (lateral paracapsular, medial paracapsular, main ITC nucleus). **Primates have a continuous net of ITC neurons around the whole amygdala**, including inside the BLA (Zikopoulos et al. 2016). The primate ITC system is therefore more widespread.
- Inputs: BLA principal cells (glutamate), **infralimbic PFC** (extinction), dopamine (D1, from VTA/SNc), opioid peptides.

### Signature behaviors (ITC)
- **Feedforward inhibitor of CeM:** BLA → ITC → CeM, so BLA activity can **reduce** fear output when ITCs are engaged. ITC lesions impair **extinction recall** (Likhtik et al. 2008).
- **Plasticity:** BLA→ITC synapses show **NMDA-dependent LTP and LTD**, with the direction set by postsynaptic activity (Royer & Paré 2002). LTP after extinction training increases ITC drive.
- **Dopamine (D1) hyperpolarizes / reduces ITC firing**, which disinhibits CeM → more fear (Marowsky et al. 2005). μ-opioids also inhibit ITCs.
- **Two functional groups** (Asede et al. 2022 and later): dorsal ITC (fear) vs ventral ITC (extinction), which inhibit each other. *Test:* activate IL input → ITC up → CeM output down → freezing down; add D1 → reversed.
- Firing: **"late-firing"/sustained** with slow, deep AHP, plus a pronounced **inward rectification (Kir)** typical of spiny projection cells.

---

## 3. Central amygdala (CeA)

| Type (marker) | % of CeA (or subdivision) neurons | Species | Abs. count | Transmitter | E/I · axon | Conf. |
|---|---|---|---|---|---|---|
| **All GABAergic** | **~95%** (a few glutamatergic cells; Vglut2 minority) | rodent | human CeA total **0.36 M** | GABA | I | ◐ |
| **PKCδ+** (overlaps with **Drd2**, enkephalin, oxytocin-receptor and **CGRP-receptor** cells) | **~50% of CeL GABA neurons** (Haubensak et al. 2010); ~61% of CeL cells projecting to CeM | mouse | — | GABA (+ enkephalin) | I · local: **→ CeM output cells**; reciprocal inhibition with SST cells; some → BNST; a fraction projects long range | ◐ |
| **SST+** (largest CeL/CeC marker population in situ; overlaps with **CRF, neurotensin, Tac2** in posterior CeL; **only ~13% of SST cells are PKCδ+**) | ~30–40% of CeL (**number not verified**) | mouse (Li et al. 2013; McCullough et al. 2018) | — | GABA (+ SST, often CRF/NTS) | I · local → PKCδ cells; **long-range → PAG, PBN, BNST, lateral hypothalamus** | ○ % / ● non-overlap |
| **CRF+** (Crh; subset overlaps SST) | ~10–20% of CeL (variable by AP level) | mouse/rat | — | GABA + CRF | I · → BNST, LC, brainstem, CeM. **CRF release drives sustained anxiety/conditioned flight** | ○ |
| **Neurotensin, Tac2 (NKB), Drd1, Htr2a** subsets | overlapping minorities | mouse | — | GABA + peptide | various | ○ |
| **CeM output neurons** (various peptides; few PKCδ) | majority of CeM | rodent | — | GABA | I · **projection → PAG (vlPAG: freezing), lateral hypothalamus, parabrachial, dorsal vagal complex (heart rate), BNST, basal forebrain (ACh), LC, VTA/SN** | ● |

Note: the CeA is small in primates (rat→monkey only ×2.3). Its peptide map is mostly from mouse ○.

### Signature behaviors (CeA)
- **CeL "ON" vs "OFF" cells:** after fear conditioning, **CeL-on cells (largely SST+) increase** their CS response and **CeL-off cells (largely PKCδ+) decrease**. On/off cells inhibit each other, **so CeL works like a flip-flop** (Ciocchi et al. 2010; Haubensak et al. 2010). *Test:* a two-population mutual-inhibition circuit with two stable states; a CS after conditioning flips it to "on".
- **Disinhibitory output:** CeL-on (SST) → inhibits CeL-off (PKCδ) → which normally inhibits CeM → so **CeM is disinhibited → freezing**. *Test:* silence PKCδ → more freezing, CeM ↑.
- **Plasticity rule:** **LA→CeL-SST synapses potentiate after fear conditioning** and LA→PKCδ synapses weaken (Li et al. 2013). The potentiation is **presynaptic** and needed for conditioned fear. CeL also stores some fear memory itself (PKCδ, ERK-dependent).
- **Active vs passive defense:** CeA **CRF** neurons favour **active (flight, escape)** responses and SST favours **passive (freezing)** responses, with mutual inhibition (Fadok et al. 2017). *Test:* high threat imminence → CRF wins → flight.
- **PKCδ:** also gates **anorexia** and pain-related responses (CGRP from the parabrachial nucleus; Cai et al. 2014). Firing: late-firing type in CeL. SST/CRF cells: regular-spiking or low-threshold bursting.
- **Sustained vs phasic fear:** CeA mediates phasic, cue-specific fear; the BNST (extended amygdala), driven by CRF, mediates sustained anxiety.

---

## 4. Medial and cortical nuclei (brief)

| Region | Main types | Transmitter | Role | Conf. |
|---|---|---|---|---|
| **Medial nucleus (MeA)** | mostly **GABAergic** projection cells (Lhx6+ lineage, Dbx1-derived and Foxp2 sets) plus a glutamatergic minority (Lhx9+); steroid-receptor rich (AR, ER, PR) | GABA ≫ Glu (rodent) | **Vomeronasal / accessory olfactory** input. Social behavior: mating, aggression, parenting, predator odour. → hypothalamus (VMH, MPOA), BNST. Sexually dimorphic | ◐ |
| **Cortical nuclei + periamygdaloid cortex** | layered, pyramidal-like glutamate cells plus interneurons (cortex-like) | Glu | **Main olfactory** input (posterolateral cortical nucleus: innate odour responses). → BLA, hypothalamus | ◐ |

In primates both are small relative to the BLA (medial ×1.5 neurons from rat to monkey). Human counts are pooled within the "remaining 3.33 M".

---

## 5. Main connections

```
Sensory thalamus (MGm/PIN) ─fast "low road"─┐
Sensory/assoc. cortex, insula ─"high road"──┤
Hippocampus (vCA1, subiculum: context) ─────┼──► LA ──► BA/AB ──► CeL/CeM, ITC, BNST, accumbens, mPFC/OFC, hippocampus, cortex
mPFC: PL (fear↑) / IL (extinction) ─────────┘          │
                                                   LA ──► CeL(SST/PKCδ) ──► CeM ──► PAG (freeze), LH/PVN (HPA, stress hormones),
IL-PFC ──► ITC ──┤ CeM                                                   PBN/DVC (heart, breathing), LC (NE), basal forebrain (ACh), VTA/SN (DA)
Parabrachial (CGRP: pain/US) ──► CeL PKCδ        MeA ◄ accessory olfactory bulb ; cortical nuclei ◄ main olfactory bulb
Neuromodulators in: LC-NE (β, gates BLA LTP), VTA-DA (D1 on ITC; BLA learning), BF-ACh (US salience), raphe 5-HT
```
- **US (pain) pathway:** spinal cord → parabrachial nucleus / PAG / thalamus → LA (and CeL directly via CGRP).
- **BLA ↔ ventral hippocampus and BLA ↔ mPFC** are reciprocal, with theta (4 Hz in rodent freezing) synchrony during fear recall.
- **BLA → accumbens** carries positive-valence and reward learning. The BLA is not only for fear.

## 6. Human vs rodent differences that matter
1. **BLA expansion:** primate BLA neurons ×11–16 vs rat, CeA ×2.3, medial ×1.5. The human amygdala is about 70% BLA by neuron count. The model's amygdala should be mostly cortex-like BLA, with a small striatum-like CeA.
2. **ITC net instead of clusters** in primates (Zikopoulos et al. 2016), which gives more widespread inhibitory gating, including within the BLA.
3. **LAMP5/neurogliaform-type interneurons are much more abundant in primate amygdala** (cross-species single-cell data) ○.
4. **Human BLA neuron number keeps rising from childhood into adulthood** (Avino et al. 2018), unusually, possibly from late maturation of paralaminar-nucleus immature neurons. This could be modelled as late-arriving BLA capacity ○.
5. Human fear and extinction are strongly **PFC-controlled** (vmPFC ≈ rodent IL). Human amygdala cells also respond to **faces, eyes, novelty and ambiguity**, not only threat (single-unit recordings).
6. Most cell-type percentages (PKCδ, SST, CRF, interneuron classes) are **rodent-only**. There is no human stereology by subtype.

## 7. Sources
- Schumann CM, Amaral DG (2005) Stereological estimation of the number of neurons in the human amygdaloid complex. *J Comp Neurol* 491:320–329. https://pmc.ncbi.nlm.nih.gov/articles/PMC2572713/
- García-Amado M, Prensa L (2012) Stereological analysis of neuron, glial and endothelial cell numbers in the human amygdaloid complex. *PLoS One* 7:e38692. https://pmc.ncbi.nlm.nih.gov/articles/PMC3374818/
- Chareyron LJ et al. (2011) Stereological analysis of the rat and monkey amygdala. *J Comp Neurol* 519:3218–3239. https://pmc.ncbi.nlm.nih.gov/articles/PMC4342351/
- Barger N et al. (2012) Neuronal populations in the basolateral nuclei of the amygdala are differentially increased in humans compared with apes. *Front Hum Neurosci*. https://schumannlab.faculty.ucdavis.edu/wp-content/uploads/sites/673/2020/08/Neuronal-populations-in-the-basolateral-nuclei-of-the-amygdala-are-differentially-increased-in-humans-compared-with-apes-A-stereological-study.pdf
- Avino TA et al. (2018) Neuron numbers increase in the human amygdala from birth to adulthood. *PNAS* 115:3710–3715. https://pmc.ncbi.nlm.nih.gov/articles/PMC5889677/
- McDonald AJ (1992) Cell types and intrinsic connections of the amygdala. In: *The Amygdala* (Aggleton, ed.).
- McDonald AJ, Mascagni F (2001) Colocalization of calcium-binding proteins and GABA in neurons of the rat basolateral amygdala. *Neuroscience* 105:681–693. Mascagni F, McDonald AJ (2003) *Brain Res* 976:171–184.
- Sah P et al. (2003) The amygdaloid complex: anatomy and physiology. *Physiol Rev* 83:803–834.
- Spampanato J, Polepalli J, Sah P (2011) Interneurons in the basolateral amygdala. *Neuropharmacology* 60:765–773.
- Krabbe S et al. (2019) Adaptive disinhibitory gating by VIP interneurons permits associative learning. *Nat Neurosci* 22:1834–1843. Wolff SBE et al. (2014) *Nature* 509:453–458.
- Blair HT et al. (2001) Synaptic plasticity in the lateral amygdala: a cellular hypothesis of fear conditioning. *Learn Mem* 8:229–242. Johansen JP et al. (2010) *PNAS* 107:12692–12697.
- Han JH et al. (2007) *Science* 316:457–460. Yiu AP et al. (2014) *Neuron* 83:722–735.
- Herry C et al. (2008) Switching on and off fear by distinct neuronal circuits. *Nature* 454:600–606.
- Marsicano G et al. (2002) The endogenous cannabinoid system controls extinction of aversive memories. *Nature* 418:530–534.
- Likhtik E et al. (2008) Amygdala intercalated neurons are required for expression of fear extinction. *Nature* 454:642–645. Royer S, Paré D (2002) *Neuroscience* 115:455–462. Marowsky A et al. (2005) *Neuron* 48:1025–1037.
- Zikopoulos B et al. (2016) Specialized inhibitory network of the primate intercalated masses. *Neuroscience*. https://sites.bu.edu/brainlab/files/2020/07/2016_IMamygdala-ZikopoulosEtAl-Neuroscience2016.pdf
- Asede D et al. (2022) Amygdala intercalated cells: gate keepers and conveyors of internal state. https://pmc.ncbi.nlm.nih.gov/articles/PMC9761677/
- Haubensak W et al. (2010) Genetic dissection of an amygdala microcircuit that gates conditioned fear. *Nature* 468:270–276. https://ideas.repec.org/a/nat/nature/v468y2010i7321d10.1038_nature09553.html
- Ciocchi S et al. (2010) Encoding of conditioned fear in central amygdala inhibitory circuits. *Nature* 468:277–282.
- Li H et al. (2013) Experience-dependent modification of a central amygdala fear circuit. *Nat Neurosci* 16:332–339. https://pmc.ncbi.nlm.nih.gov/articles/PMC3581751/
- McCullough KM et al. (2018) Quantified coexpression analysis of central amygdala subpopulations. *eNeuro* 5:ENEURO.0010-18.2018. https://www.eneuro.org/content/eneuro/5/1/ENEURO.0010-18.2018.full.pdf
- Fadok JP et al. (2017) A competitive inhibitory circuit for selection of active and passive fear responses. *Nature* 542:96–100.
- Cai H et al. (2014) Central amygdala PKC-δ⁺ neurons mediate the influence of multiple anorexigenic signals. *Nat Neurosci* 17:1240–1248.
- Yu B et al. (2023) Molecular and cellular evolution of the amygdala across species analyzed by single-nucleus transcriptome profiling. *Cell Discovery* (cross-species; LAMP5 interneurons primate-enriched). https://db.cngb.org/data_resources/project/PRJNA800681

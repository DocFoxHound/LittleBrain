# Neuromodulatory nuclei: neuron types and proportions

Scope: VTA (dopamine), locus coeruleus (noradrenaline), dorsal and median raphe (serotonin), basal forebrain (nucleus basalis of
Meynert, medial septum / diagonal band; acetylcholine), pedunculopontine and laterodorsal tegmental nuclei (acetylcholine).
SNc dopamine is in `basal-ganglia.md`.

Main lesson: **none of these "chemical" nuclei is mostly made of its namesake cell.** GABA and glutamate neurons are a third to
over half of most of them, and they project too. Model each nucleus as a mix, not as one broadcast unit.

Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain.

---

## Summary table

| Nucleus | Modulator cells | GABA | Glutamate | Human count of modulator cells | Species of % | Conf. |
|---|---|---|---|---|---|---|
| VTA | **DA ~64%** | ~33% | ~2–3% (later work: more) | not retrieved | rat | ● / ◐ glu |
| Locus coeruleus | **NE ~all of the core** | peri-LC shell (unquantified) | few | **~45,000–54,000** | human | ● |
| Dorsal raphe | **5-HT roughly 1/3–1/2** (○) | large lateral population | VGluT3 (mostly co-5-HT), VGluT2 | **165,000 ± 34,000** | human | ● count / ○ % |
| Median raphe | **5-HT 2–9%** | **37–61%** | VGluT3 7–26% | **64,400** | mouse % / human count | ◐ |
| Basal forebrain (whole) | **ACh ~5%** | ~35% | glutamate-capable ~90% (overlaps) | — | rat | ◐ |
| Nucleus basalis (Ch4) | **ACh ~90%** | minority | minority | **~200,000 per hemisphere** (ChAT+ 210,000) | human | ● |
| PPN | **ACh ~25%** (rat); 80–90% of pars compacta (human) | ~32% | ~43% | not retrieved | rat / human | ◐ |
| LDT | **ACh ~22%** | ~46% | ~32% | not retrieved | rat | ◐ |

---

## 1. Ventral tegmental area (VTA)

### Composition (rat, Nair-Roberts et al. 2008, bilateral stereology)

| Type (marker) | % of VTA neurons | Count (rat, bilateral) | Transmitter | E/I; projection | Conf. |
|---|---|---|---|---|---|
| **Dopaminergic** (TH, DAT) | **~64%** (65% of TH+GAD) | **40,174** | dopamine (subsets co-release glutamate or GABA) | modulatory projection → **nucleus accumbens**, olfactory tubercle, **prefrontal cortex**, amygdala, hippocampus, septum, lateral habenula | ● |
| **GABAergic** (GAD) | **~33–35%** | **21,011** | GABA | local inhibition of DA cells **and** long-range projection (→ NAc cholinergic interneurons, PFC, lateral habenula, ventral pallidum) | ● |
| **Glutamatergic** (VGluT2) | **~2–3%** (≈ 1,600 cells), mostly rostromedial | ~1,600 | glutamate | projection → NAc, PFC, lateral habenula, ventral pallidum; local → DA cells | ◐ (later mouse/rat work finds larger and co-releasing VGluT2 populations; Morales & Margolis 2017) |

Subnuclei differ: GABA share is **12% in the interfascicular**, **30% parabrachial pigmented**, **45% parainterfascicular**
(rostral VTA more dopaminergic, caudal VTA more GABAergic). For comparison: SNc 29% GABA; retrorubral field 58% GABA.
Caudal tail of VTA = **RMTg ("tVTA")**, a GABA nucleus that brakes DA cells and relays lateral habenula "disappointment" signals.

**Human count:** I could not find a stereological human VTA dopamine count. Humans have relatively fewer midbrain DA neurons for brain size than other primates (Hardman 2002). ○

### Signature behaviors
- **DA cells:** pacemaker ~1–8 Hz, **phasic burst** (15–30 Hz, ~100–200 ms) on unexpected reward or reward-predicting cue, **pause** on omitted reward = **reward prediction error** (already modelled in note 18). Mesocortical subset (→ PFC) lacks/has little DAT and D2 autoreceptor, fires faster and less regular; aversive/salience-responding subsets exist (Lammel 2014). *Test:* already exists (first reward 1.00 → 15th 0.04; omission −0.96). Add: PFC-projecting subset doesn't show autoreceptor self-inhibition.
- **VTA GABA cells:** fire 10–40 Hz, **encode expected reward** (sustained activity during cue → reward delay) and subtract it from DA cells (Cohen et al. 2012, Eshel 2015). *Test:* GABA rate ramps up with learned expectation; silencing them abolishes the learned DA dip.
- **VTA glutamate cells:** signal aversive or rewarding salience; drive lateral habenula and NAc.

---

## 2. Locus coeruleus (LC)

| Type (marker) | Share | Species | Count | Transmitter | Projection | Conf. |
|---|---|---|---|---|---|---|
| **Noradrenergic principal cell** (TH, DBH, NET; many co-express galanin, NPY) | essentially **all** neurons of the compact LC core | human, rodent | **human ≈ 45,000–46,000** (Sharma 2010, two subjects: 46,423 and 45,567, optical fractionator) and **53,900 TH+ in LC + 6,260 in subcoeruleus** (Baker 1989); rat classic figure ~1,500–1,600 per side (○, not re-verified) | noradrenaline (+ dopamine co-release in hippocampus) | **whole forebrain and cerebellum and spinal cord**: neocortex, hippocampus, thalamus, amygdala, cerebellum; the *only* NE source for cortex and hippocampus | ● |
| **Peri-LC GABA neurons** | in the dendritic zone around the core; not quantified | rat, mouse | — | GABA | local inhibition of LC (arousal, exploration control; Luskin et al. 2025 *Nature*) | ◐ |

Human LC count is higher than other primates in absolute terms but lower than predicted from neocortex/cerebellum size (Sharma 2010). Whether Sharma's numbers are per side or bilateral is not stated in the retrieved text. LC loses neurons early in Alzheimer's and Parkinson's.

### Signature behaviors
- **Tonic mode** 0.5–5 Hz scaling with arousal (silent in REM sleep, low in NREM).
- **Phasic mode**: brief bursts (~10–20 Hz) to salient/surprising stimuli, ~100 ms latency; phasic is strongest at *moderate* tonic rates (inverted U; Aston-Jones & Cohen 2005 adaptive gain theory). High tonic → distractible/exploratory.
- Electrotonic coupling → population synchrony; α2 autoreceptor self-inhibition gives a post-burst pause.
- NE effect on targets: **gain increase** (multiplicative), plus β-adrenergic boost of plasticity.
*Test:* tonic rate follows an "arousal" input; phasic burst on surprise is largest at mid arousal; REM/sleep input → ~0 Hz.

---

## 3. Raphe nuclei

### 3a. Dorsal raphe (DR)

| Type (marker) | Share | Species | Count | Transmitter | Projection | Conf. |
|---|---|---|---|---|---|---|
| **Serotonergic** (TPH2, SERT, Pet1) | roughly **one-third to one-half** of DR neurons (no stereological % retrieved) | rodent | **human 165,000 ± 34,000** (Baker et al. 1991, 10 brains) | 5-HT; many co-release **glutamate (VGluT3)**; human DR neurons often co-contain **substance P** | **forebrain-wide**: prefrontal and other cortex, striatum, NAc, amygdala, VTA/SNc, thalamus, hypothalamus | ● count / ○ % |
| Substance P neurons | human **74,600 ± 17,600**, rostral, largely the *same* cells as 5-HT | human | | SP (+5-HT) | | ◐ |
| Catecholamine (DA) neurons | human **5,600 ± 3,400**, rostral DR | human | | dopamine | → extended amygdala; wake-promoting | ◐ |
| **GABAergic** | large population, mostly **lateral** to 5-HT cells | rat, mouse | — | GABA | local inhibition of 5-HT; some long-range | ◐ |
| Glutamatergic (VGluT2, non-5-HT; VGluT3-only) | minority | mouse | — | glutamate | → VTA (rewarding), forebrain | ◐ |

### 3b. Median raphe (MR)

| Type | % (mouse MR nucleus proper; whole MRR region) | Human count | Transmitter | Projection | Conf. |
|---|---|---|---|---|---|
| **5-HT only** | **8.5%**; whole region 2.1% | **64,400 5-HT (PH8+) neurons** (Baker 1991, *Synapse*) | 5-HT | → **hippocampus, medial septum**, entorhinal, hypothalamus | ◐ |
| 5-HT + VGluT3 | whole region 3.6% | — | 5-HT + glutamate | same | ◐ |
| **VGluT3 (glutamate, non-5-HT)** | **26%**; whole region 7% | — | glutamate | → hippocampal interneurons, septum, habenula | ◐ |
| **GABAergic** | **37%**; whole region **61%** | — | GABA | local + projection | ◐ |
| Triple-negative | ~25% | — | (VGluT2 etc.) | | ○ |

(Sós et al. 2017, mouse stereology.) MR cells control **hippocampal theta and desynchronization**; MR VGluT2 cells drive aversive memory/“stop” behaviour.

### Signature behaviors (5-HT cells)
- **Slow, very regular** firing ~0.5–3 Hz, highest in waking, lower in NREM, **silent in REM**; broad spikes; 5-HT1A autoreceptor self-inhibition.
- Respond tonically to **reward and delayed reward**; activity sustains **patience/waiting** for delayed reward (Miyazaki et al. 2014; Fonseca 2015); also aversive/punishment signals in some subsets. Matches the "slow average of reward (patience)" model in note 18.
*Test:* regular low-rate firing; rate stays elevated across a waiting period and predicts how long the agent waits; drops to ~0 in REM.

---

## 4. Basal forebrain

Mesulam's cholinergic groups: **Ch1** medial septum, **Ch2** vertical diagonal band → hippocampus; **Ch3** horizontal diagonal band → olfactory bulb/piriform; **Ch4** nucleus basalis of Meynert (nbM) → neocortex and amygdala.

### 4a. Whole basal forebrain (rat, Gritti et al. 2006)

| Type | % of ~355,000 BF neurons | Count | Note | Conf. |
|---|---|---|---|---|
| **Cholinergic** (ChAT) | **~5%** | ~22,000 | large, projection | ◐ |
| **GABAergic** (GAD) | **~35%** | ~119,000 | incl. PV projection neurons to cortex/hippocampus | ◐ |
| **Glutamate-capable** (PAG) | **~90%** | ~316,000 | overlaps with ACh and GABA cells; categories not exclusive | ◐ |

### 4b. Nucleus basalis of Meynert (human)

| Type | Share | Count | Transmitter | Projection | Conf. |
|---|---|---|---|---|---|
| **Magnocellular cholinergic (Ch4)** (ChAT, p75NTR, VAChT) | **~90%** of nbM neurons | **~200,000 per hemisphere**; ChAT+ **210,000**, VAChT+ 174,000 per hemisphere (Gilmor 1999); no left–right difference | ACh | **entire neocortex** (topographic) and basolateral amygdala | ● |
| GABA / glutamate / peptide cells | ~10% | — | GABA, glutamate | cortex, TRN | ◐ |

### 4c. Medial septum / diagonal band (Ch1–Ch2)

| Type | Share | Species | Transmitter | Projection | Conf. |
|---|---|---|---|---|---|
| Cholinergic | minority; ratio to GABA cells disputed (Brashear 1986: ACh ~2× GABA; others the reverse) | rat | ACh | → hippocampus, entorhinal (slow, sets encoding mode) | ○ % |
| **GABAergic PV pacemakers** | large population | rat | GABA | → **hippocampal interneurons** (disinhibition) = **theta pacemaker** | ● |
| Glutamatergic (VGluT2) | ~16,000 cells in rat medial septal region | rat | glutamate | local (drive PV/ACh cells) + hippocampus; locomotion-speed related | ◐ |

### Signature behaviors
- **Cholinergic BF cells:** slow tonic firing, higher in **waking and REM**, low in NREM; **phasic bursts** to reinforcement/punishment and salient cues (Hangya et al. 2015) — fast "teaching" signal, not only slow arousal. ACh shifts cortex/hippocampus to **encoding mode** (boost feedforward input, suppress recurrent recall; Hasselmo). Matches note-18 model (novel 0.97 / familiar 0.15).
- **MS PV GABA cells:** rhythmic bursting at **theta 4–10 Hz**, phase-locked; drive hippocampal theta. *Test:* MS-PV population oscillates at theta and entrains target interneurons.
- **BF PV/GABA cortical projection:** drive gamma and cortical activation on waking.

---

## 5. Pedunculopontine (PPN) and laterodorsal tegmental (LDT) nuclei (Ch5, Ch6)

### Rat stereology (Luquin et al. 2018)

| Nucleus | Cholinergic | Glutamatergic | GABAergic | % ACh / Glu / GABA | Conf. |
|---|---|---|---|---|---|
| **PPN** | 3,360 | 5,910 | 4,439 | **25 / 43 / 32%** | ◐ |
| **LDT** | 3,650 | 5,190 | 7,599 | **22 / 32 / 46%** | ◐ |

>95% of cholinergic cells lack VGluT2/GAD mRNA (Wang & Morales 2009); a small subset co-expresses VGluT2 (8% PPN, 5% LDT).
Human: **80–90% of neurons in PPN pars compacta are cholinergic**; humans lose ~36–38% of PPN cholinergic neurons in Parkinson's
(linked to gait/falls). Absolute human counts were not retrieved. ◐

### Projections
- **PPN:** → **thalamus** (intralaminar and relay, arousal), **SNc and VTA** (burst triggering), STN, GPi, striatum (cholinergic interneurons), brainstem/spinal reticular (locomotion = mesencephalic locomotor region).
- **LDT:** → **VTA** (strong, triggers DA bursts), thalamus, lateral habenula, medial prefrontal, hippocampal/septal.

### Signature behaviors
- Cholinergic cells: **"REM-on" / "wake-REM-on"** — high firing in waking and REM, low in NREM; drive thalamic tonic mode and PGO waves (REM). Gate the thalamic relay out of burst mode (fits note 18's rule that relay bursting needs low ACh).
- Glutamatergic PPN cells: locomotion initiation and speed; GABAergic cells: stop locomotion.
- PPN/LDT → VTA/SNc input is needed for **phasic dopamine bursts** to reward cues.

---

## 6. Human vs rodent

- Human LC (~45–54 k) and DR (~165 k 5-HT) are 10–30× rodent populations; counts scale with brainstem size more than with neocortex (LC: Sharma 2010).
- Human DR 5-HT neurons widely co-express **substance P**; mouse 5-HT cells lack the NOS/galanin co-expression seen in rat — species differ even among rodents.
- Human nbM is a large, dense magnocellular Ch4 complex (~90% cholinergic); rodent "NB" is small and intermixed (~5% ACh of all BF).
- Humans have relatively fewer midbrain DA cells per brain size (Hardman 2002).
- Most %-of-nucleus data (VTA, MR, PPN/LDT, BF) are **rodent**; human cell-type proportions for these nuclei are largely unknown.

---

## 7. Sources

- Nair-Roberts RG et al. (2008) Stereological estimates of dopaminergic, GABAergic and glutamatergic neurons in the VTA, SN and RRF in the rat. *Neuroscience* 152:1024. https://pmc.ncbi.nlm.nih.gov/articles/PMC2575227/
- Morales M, Margolis EB (2017) Ventral tegmental area: cellular heterogeneity, connectivity and behaviour. *Nat Rev Neurosci* 18:73.
- Cohen JY et al. (2012) Neuron-type-specific signals for reward and punishment in the VTA. *Nature* 482:85.
- Lammel S, Lim BK, Malenka RC (2014) Reward and aversion in a heterogeneous midbrain dopamine system. *Neuropharmacology* 76:351.
- Sharma Y et al. (2010) Comparative anatomy of the locus coeruleus in humans and nonhuman primates. *J Comp Neurol* 518:963. https://pmc.ncbi.nlm.nih.gov/articles/PMC2820586
- Baker KG, Törk I, Hornung JP, Halasz P (1989) The human locus coeruleus complex. *Exp Brain Res* 77:257. https://serval.unil.ch/notice/serval:BIB_721253531130
- Luskin AT et al. (2025) A diverse network of pericoerulear neurons controls arousal. *Nature*. https://ideas.repec.org/a/nat/nature/v643y2025i8071d10.1038_s41586-025-08952-w.html
- Aston-Jones G, Cohen JD (2005) An integrative theory of locus coeruleus–norepinephrine function. *Annu Rev Neurosci* 28:403.
- Baker KG et al. (1991) Distribution, morphology and number of monoamine-synthesizing and substance P-containing neurons in the human dorsal raphe nucleus. *Neuroscience* 42:757. https://doi.org/10.1016/0306-4522(91)90043-N (PMID 1720227)
- Baker KG et al. (1991) Cytoarchitecture of serotonin-synthesizing neurons in the pontine tegmentum of the human brain. *Synapse* 7:301. https://serval.unil.ch/notice/serval:BIB_73978BAD4D52
- Sós KE et al. (2017) Cellular architecture and transmitter phenotypes of neurons of the mouse median raphe region. *Brain Struct Funct* 222:287. https://pmc.ncbi.nlm.nih.gov/articles/PMC5225166 (doi:10.1007/s00429-016-1217-x)
- Miyazaki KW et al. (2014) Optogenetic activation of dorsal raphe serotonin neurons enhances patience for future rewards. *Curr Biol* 24:2033.
- Gritti I et al. (2006) Stereological estimates of the basal forebrain cell population in the rat. *Neuroscience* 143:1051. https://bioweb.supagro.inrae.fr/ESTHER/paper/Gritti_2006_Neurosci_143_1051
- Gilmor ML et al. (1999) Preservation of nucleus basalis neurons containing choline acetyltransferase and the vesicular acetylcholine transporter in the elderly with mild cognitive impairment and early Alzheimer's disease. *J Comp Neurol* 411:693. https://scholars.uky.edu/es/publications/preservation-of-nucleus-basalis-neurons-containing-choline-acetyl/
- Liu AKL et al. (2015) Nucleus basalis of Meynert revisited. *Acta Neuropathol* 129:527. https://pmc.ncbi.nlm.nih.gov/articles/PMC4366544
- "~200,000 neurons per hemisphere, nearly 90% cholinergic" figure: secondary statement in a *J Alzheimer's Dis* review (doi:10.3233/JAD-180133); primary source not traced. ◐
- Hangya B et al. (2015) Central cholinergic neurons are rapidly recruited by reinforcement feedback. *Cell* 162:1155.
- Colom LV et al. (2005) Characterization of medial septal glutamatergic neurons. https://scholars.uthscsa.edu/en/publications/characterization-of-medial-septal-glutamatergic-neurons-and-their/
- Luquin E et al. (2018) Stereological estimates of glutamatergic, GABAergic, and cholinergic neurons in the pedunculopontine and laterodorsal tegmental nuclei in the rat. *Front Neuroanat* 12:34. https://pmc.ncbi.nlm.nih.gov/articles/PMC5958217/
- Wang HL, Morales M (2009) PPN and LDT contain distinct populations of cholinergic, glutamatergic and GABAergic neurons in the rat. *Eur J Neurosci* 29:340. https://pmc.ncbi.nlm.nih.gov/articles/PMC3833361
- French IT, Muthusamy KA (2018) A review of the pedunculopontine nucleus in Parkinson's disease. *Front Aging Neurosci* 10:99. https://pmc.ncbi.nlm.nih.gov/articles/PMC5933166/
- Human PPN cholinergic stereology in PD (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12352924/
- Hardman CD et al. (2002) *J Comp Neurol* 445:238. https://pubmed.ncbi.nlm.nih.gov/11920704

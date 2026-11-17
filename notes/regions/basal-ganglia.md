# Basal ganglia: neuron types and proportions

Scope: striatum (caudate, putamen, nucleus accumbens), external and internal globus pallidus (GPe, GPi),
subthalamic nucleus (STN), substantia nigra pars reticulata (SNr) and pars compacta (SNc).
The VTA is covered in `neuromodulatory-nuclei.md`.

Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain.
"Rat"/"mouse" figures are rodent and labeled as such; prefer human/primate rows when they exist.

---

## 1. Striatum

### 1a. Composition table

| Type (marker) | % of striatal neurons | Species | Absolute count | Transmitter | E/I; local/projection; axon target | Conf. |
|---|---|---|---|---|---|---|
| **All projection neurons, medium spiny neurons (MSN / SPN)** | **~95%** rodent; **~74–83%** human (interneurons 17–26%, see below) | rat/mouse; human | human striatum ≈ **82 million** "small" neurons (caudate + putamen + accumbens, n = 5; Heinsen et al. 1994) | GABA (+ peptides) | inhibitory projection | ● rodent / ◐ human |
| ↳ **dSPN, D1-MSN** (D1R, substance P, dynorphin) | ~50% of MSNs (≈ 50% in matrix; ~70% in substance-P-rich striosomes) | mouse | | GABA, SP, dynorphin | inhibitory projection → **GPi, SNr** (+ collaterals to GPe) = *direct pathway* | ● |
| ↳ **iSPN, D2-MSN** (D2R, enkephalin, A2A) | ~50% of MSNs (complementary to D1) | mouse | | GABA, enkephalin | inhibitory projection → **GPe** only = *indirect pathway* | ● |
| ↳ D1/D2 co-expressing MSNs | ~5% dorsal striatum, higher (~15–17%) in accumbens shell; one tree-shrew study ~30% | mouse (BAC reporters); tree shrew | | GABA | projection (eccentric "eSPN" class in scRNA-seq) | ◐ |
| **Cholinergic interneuron (ChAT; "TAN")** | **0.5–1%** rodent; **~1–2%** human | rat; human | human density 361 cells/mm³; total unresolved, ~0.8–7 M (see note) | acetylcholine (some co-release glutamate via VGluT3) | modulatory local; giant aspiny, axon spans ~1 mm | ● |
| **PV fast-spiking interneuron (FSI)** | **~0.7%** | rat (stereology) | | GABA | inhibitory local → MSN somata (feed-forward from cortex) | ● |
| **SST/NPY/NOS (PLTS) interneuron** | **~0.6%** | rat | | GABA, NO, SST, NPY | inhibitory local → MSN distal dendrites; vasomotor (NO) | ● |
| **Calretinin (CR) interneuron** | **~0.5%** rodent; **most abundant interneuron in primates/humans** | rat; human/monkey | | GABA | inhibitory local | ● qualitative / ○ human % |
| **TH interneuron (THIN)** | small; not stereologically fixed (~a few thousand per hemisphere in mouse) | mouse | | GABA (TH+ but little/no dopamine release) | inhibitory local | ○ |
| Other newer classes (NPY-neurogliaform, fast-adapting, spontaneously active bursty) | small, unquantified | mouse | | GABA | inhibitory local | ○ |

**Interneuron total.** Rodent: GABAergic interneurons ≈ 3–4% plus cholinergic ≈ 0.5–1% → total ≈ 5% (Tepper et al. 2010).
Human: the classic figure is **~26%** (Graveland et al. 1985 lineage; "5% in rodents, 26% in humans"). The only unbiased
stereology (Lecumberri et al. 2018, *Brain Struct Funct*, doi:10.1007/s00429-017-1579-8) gives **17.1% on average**:
associative territory 21.9%, sensorimotor 12.8%, limbic 11.1%. Use **17–26%**, territory-dependent. ◐

**Note on the human cholinergic count.** Bernácer et al. 2007 measured 361 ChAT+ cells/mm³ and multiplied by a reference striatal volume
of 6.3 × 10⁴ mm³ (63 cm³, roughly 3× a real bilateral human striatum of ~20 cm³) to get 2.27 × 10⁷. With a realistic volume the
same density gives ~7 M, which would be ~9% of 82 M neurons, far above the classic 1–2%. The figures don't reconcile; the range is
**~0.8 M (1% of 82 M) to ~7 M (density × volume)**. Treat human cholinergic counts as unresolved. ○

Interneuron shares *within* the human interneuron pool: CR > PV ≈ ChAT ≈ SST/NPY is the reported order (Wu & Parent 2000; Cicchetti et al. 2000; Bernácer 2012); exact percentages not stereologically settled. ○
Primate-only: **giant CR+ cells**, most of which are also ChAT+ (absent in rodents). ◐

### 1b. Compartments

| Compartment | Share | Notes | Conf. |
|---|---|---|---|
| **Matrix** | ~85–90% of volume | calbindin-rich; inputs from sensorimotor/associative cortex (IT and PT layer 5); D1 ≈ D2 | ◐ |
| **Striosome (patch)** | ~10–15% of volume, larger rostrally, shrinking caudally | MOR (μ-opioid)-rich, calbindin-poor; inputs from limbic/prefrontal cortex (deep layer 5); striosomal dSPNs are the only MSNs that project **directly to SNc dopamine cells** (inhibit them); D1 share ≈ 70% in SP-positive striosomes | ◐ |
| Dorsal (caudate/putamen) vs ventral (accumbens core/shell) | — | accumbens shell has more D1/D2 co-expressing cells and the islands of Calleja | ● |

Cholinergic interneurons sit at striosome/matrix borders and are ~1.8× denser in caudate than putamen in humans (Bernácer 2007).

### 1c. Signature behaviors to model (testable)

- **MSN (both).** Bistable membrane: **down-state** near −85 mV (K_ir), **up-state** ~−55 mV only with *convergent* cortical input; fires sparsely (≈ 0–5 Hz) and only in the up-state. *Test:* a single input cluster doesn't fire it; ≥ N coincident inputs drive it into a sustained up-state; rate stays near 0 at rest.
- **D1-MSN plasticity.** Three-factor rule: corticostriatal **LTP when pre/post coincide AND dopamine burst** (D1 → cAMP/PKA); LTD when dopamine dips. *Test:* weight change sign follows sign of DA × eligibility trace.
- **D2-MSN plasticity.** Mirror image: **LTP when dopamine dips** (D2 released, A2A active), LTD with DA present (endocannabinoid-dependent). Dopamine *lowers* D2-MSN excitability, *raises* D1-MSN excitability. *Test:* the same DA burst moves D1 and D2 weights in opposite directions.
- **Lateral inhibition between MSNs.** Weak, sparse (~10–15% connection probability nearby), asymmetric, dendritic. Model as weak competitive inhibition, not winner-take-all.
- **PV FSI.** Fast-spiking up to 200+ Hz, gap-junction coupled, strong **feed-forward inhibition** onto MSN somata (one FSI contacts hundreds of MSNs); sets spike timing. *Test:* a cortical volley activates FSIs first and delays/vetoes weakly driven MSNs.
- **Cholinergic TAN.** Tonically active **~2–10 Hz** pacemaker; on salient/reward cues it shows a **burst–pause–rebound**; the pause coincides with the dopamine burst and is a learning *gate* (opens a plasticity window). ACh via nicotinic receptors on dopamine terminals triggers local DA release; M1 raises MSN (esp. D2-MSN) excitability. *Test:* intrinsic firing without input; a reward cue produces a pause of ~200 ms.
- **SST/NPY/NOS (PLTS).** Low-threshold Ca²⁺ spikes, persistent plateau after depolarization; slow dendritic inhibition and NO release.
- **CR, TH interneurons.** Little functional data; treat as generic local GABA cells.

### 1d. Connections

In: cortex (glutamate; IT bilateral and PT ipsilateral, layer 5), thalamus (centromedian/parafascicular → strongly onto ChIs and MSNs), SNc/VTA dopamine, amygdala and hippocampus (ventral striatum), dorsal raphe 5-HT, GPe arkypallidal cells.
Out: D1-MSN → GPi/SNr; D2-MSN → GPe; striosome D1 → SNc.

---

## 2. Pallidum, STN, SNr

### 2a. Composition table

| Nucleus / type (marker) | % of nucleus | Species | Absolute count | Transmitter | E/I; target | Conf. |
|---|---|---|---|---|---|---|
| **GPe prototypic** (PV, Nkx2-1, Lhx6) | **~2/3** (55–68%) | rat (Abdi 2015), mouse (Hernandez 2015: PV 55%) | | GABA | inhibitory projection → **STN, SNr/GPi**, SNc, thalamic reticular, and back to GPe | ● |
| **GPe arkypallidal** (FoxP2, Npas1, PPE) | **~1/4–1/3** (≈ 25–30%) | rat, mouse | | GABA | inhibitory projection → **striatum only** (MSNs and FSIs, dense) | ● |
| GPe other (Npas1+/Nkx2-1+ ; ChAT+ cells near border; Lhx6+ PV−) | ~5–15% | mouse | | GABA / ACh | → cortex, striatum | ◐ |
| **GPi** (PV+ projection neurons) | ~100% one main type | primate | human GPi ≈ STN ≈ SNc in count (Hardman 2002) — exact human figure not retrieved | GABA | inhibitory projection → motor thalamus (VA/VL), centromedian, lateral habenula (border cells, glutamate/GABA co-release), PPN | ● |
| **STN** | ~100% one principal type | human | **~240,000–560,000 per side** (Lévesque & Parent 2005: 239,500; Füssenich 1967: 520,300; Bokulić 2021 range ~270–550 k) | glutamate | **excitatory** projection → GPe, GPi, SNr, (SNc, PPN) | ● type / ◐ count |
| **SNr** (PV+) | ~70% GABAergic, ~30% TH+ (displaced DA cells) by Nair-Roberts counts | rat | rat SNr GAD+ ≈ 21,700 bilateral | GABA | inhibitory projection → motor thalamus, superior colliculus (saccades), PPN, brainstem | ● |

Human GPe has **more** neurons than GPi, STN or SNc; GPi, STN and dopaminergic SN are similar to one another within each species
(Hardman et al. 2002, rats to humans). Humans have proportionally *more* STN and GPe neurons and *fewer* dopamine neurons relative to brain size.

### 2b. Signature behaviors

- **GPe prototypic.** Autonomous pacemaker **~30–60 Hz** in vivo (HCN + persistent Na⁺); pauses when D2-MSNs fire; strong reciprocal loop with STN. *Test:* fires regularly with no synaptic input; striatal inhibition produces a pause.
- **GPe arkypallidal.** Lower rate (~5–10 Hz), fire *more* when prototypic cells pause; send a broad **"stop" signal** back to striatum (Mallet 2016: cancelling an action).
- **GPi / SNr.** High tonic rate **~60–90 Hz** (autonomous); this *tonic inhibition* is the brake on thalamus/colliculus. The direct pathway produces a *pause* → disinhibition → movement. *Test:* output rate high at rest; a D1-MSN burst produces a brief pause.
- **STN.** Autonomous ~10–20 Hz; **rebound burst** after release from GPe inhibition (T-type Ca²⁺); fast relay of cortex (hyperdirect). In Parkinsonism the GPe–STN loop locks into **beta (13–30 Hz)** oscillation. *Test:* hyperpolarize then release → burst; coupled GPe–STN with low dopamine → beta oscillation.

---

## 3. SNc dopamine neurons

| Type (marker) | % of SNc neurons | Species | Count | Transmitter | Target | Conf. |
|---|---|---|---|---|---|---|
| **Dopaminergic** (TH, DAT, Sox6/Aldh1a1 subtypes; calbindin-poor ventral tier) | **~71%** (TH+ 15,772 vs GAD+ 6,400 bilateral) | rat (Nair-Roberts 2008) | rat SNc ≈ **15,800 bilateral**; mouse SNc ≈ **8,300**; human **~550,000 pigmented neurons** in the whole SN (bilateral, Pakkenberg 1991), i.e. ~275,000 per side; other human series lower (~160,000 SNc pigmented, Alzheimer-control study) | dopamine (some co-release GABA via ALDH1a1 pathway) | **dorsal striatum** (nigrostriatal), less to cortex | ● type / ◐ counts |
| GABAergic | **~29%** | rat | rat 6,400 bilateral; human non-pigmented SN neurons ~260,000 (Pakkenberg 1991, whole SN) | GABA | local + projection | ● |
| Glutamatergic (VGluT2) | essentially none in SN (VTA only) | rat | | glutamate | | ◐ |

Sub-tiers: **ventral tier** (calbindin−, Sox6+, Aldh1a1+) → dorsolateral striatum, most vulnerable in Parkinson's; **dorsal tier** (calbindin+) → ventromedial striatum, closer to VTA in character.

**Signature behaviors.** Pacemaking **~1–5 Hz** (L-type Ca²⁺, very wide spikes ~2–3 ms); **phasic burst 15–30 Hz** for better-than-expected outcomes, **pause** for worse-than-expected (reward prediction error, Schultz 1997); SNc cells also carry movement-initiation and salience signals. Huge axon arbor (one rat SNc cell: ~4.5 × 10⁵ striatal boutons, ~4–5% of striatal volume; Matsuda 2009). Somatodendritic DA release + D2 autoreceptor self-inhibition. *Test:* omitted reward → rate dips below baseline; tonic firing continues with all inputs removed.

---

## 4. Pathways

| Pathway | Route | Net effect on thalamus | Conf. |
|---|---|---|---|
| **Direct** | cortex → **D1-MSN** ⊣ GPi/SNr ⊣ thalamus → cortex | disinhibits → "go", select the action | ● |
| **Indirect** | cortex → **D2-MSN** ⊣ GPe ⊣ STN → GPi/SNr ⊣ thalamus (plus GPe ⊣ GPi/SNr directly) | increases output inhibition → "no-go", suppress competitors | ● |
| **Hyperdirect** | cortex (layer 5 PT, esp. frontal/pre-SMA/IFG) → **STN** → GPi/SNr | fastest (~few ms), broad brake: "hold / stop everything" | ● |
| Arkypallidal stop | GPe arky ⊣ striatum | cancels an initiated action | ◐ |
| Nigrostriatal / striosomal loop | striosome D1-MSN ⊣ SNc → striatum | lets learned value regulate dopamine | ◐ |

Modern caveats (still model-relevant): D1-MSNs also send collaterals to GPe; both pathways are co-active at movement onset (Cui 2013), so model **selection/competition** (centre-surround) rather than strict go vs stop.

---

## 5. Human vs rodent

- **Interneurons**: ~5% of striatal neurons in rodents vs **17–26% in humans**; calretinin cells are the most common primate interneuron and rare in rodents; primate-only giant CR/ChAT cells. ◐
- **Caudate and putamen** are separated by the internal capsule in primates (one "caudate-putamen" in rodents); associative territory (caudate) is much bigger in humans.
- **Dopamine neurons are relatively fewer** in humans vs brain size (Hardman 2002), with more GPe/STN neurons — relatively more weight on pallidal control.
- **Ventral SNc tier** is larger in primates; rats have very few ventrally located SND neurons.
- GPe arkypallidal/prototypic split is defined mostly in rodents; primate GPe has comparable FoxP2+ cells but less-quantified shares. ○

---

## 6. Sources

- Tepper JM, Tecuapetla F, Koós T, Ibáñez-Sandoval O (2010) Heterogeneity and diversity of striatal GABAergic interneurons. *Front Neuroanat* 4:150. https://www.frontiersin.org/articles/10.3389/fnana.2010.00150/full (rodent 95% MSN; ChI 0.5–1%; PV 0.7%, CR 0.5%, SOM/NOS/NPY 0.6%)
- Tepper JM et al. (2018) Heterogeneity and diversity of striatal GABAergic interneurons: update 2018. *Front Neuroanat* 12:91. https://www.frontiersin.org/articles/10.3389/fnana.2018.00091/full
- Lecumberri A, Lopez-Janeiro A, Corral-Domenge C, Bernacer J (2018) Neuronal density and proportion of interneurons in the associative, sensorimotor and limbic human striatum. *Brain Struct Funct*: 17.1% average, 21.9/12.8/11.1% by territory. https://doi.org/10.1007/s00429-017-1579-8
- Heinsen H et al. (1994) Cortical and striatal neurone number in Huntington's disease (82 × 10⁶ small striatal neurons in controls). https://opus.bibliothek.uni-wuerzburg.de/frontdoor/index/index/year/2011/docId/4739
- Bernácer J, Prensa L, Giménez-Amaya JM (2007) Cholinergic interneurons are differentially distributed in the human striatum. *PLoS One* 2:e1174. https://pmc.ncbi.nlm.nih.gov/articles/PMC2137841/
- Cicchetti F, Prensa L, Wu Y, Parent A (2000) Chemical anatomy of striatal interneurons in normal individuals and in patients with Huntington's disease. *Brain Res Rev* 34:80–101.
- Garas FN et al. (2018) Calretinin interneurons in primate striatum. https://ora.ox.ac.uk/objects/uuid:2435f3a4-b656-4200-ba07-73c4cbbf4e86
- Striosome-based map of mouse striatum (D1 ≈ 50% matrix, ~70% SP-positive striosomes). https://link.springer.com/doi/10.1007/s00429-019-01928-3
- Bertran-Gonzalez J et al. (2008) Opposing patterns of signaling activation in D1 and D2 receptor-expressing striatal neurons. *J Neurosci* 28:5671 (D1/D2 co-expression; recalled, not re-fetched). ◐
- Abdi A et al. (2015) Prototypic and arkypallidal neurons in the dopamine-intact external globus pallidus. *J Neurosci* 35:6667. https://www.bndu.ox.ac.uk/papers/prototypic-and-arkypallidal-neurons-dopamine-intact-external-globus-pallidus
- Hernández VM et al. (2015) Parvalbumin+ neurons and Npas1+ neurons are distinct neuron classes in the mouse external globus pallidus. *J Neurosci* 35:11830.
- Mallet N et al. (2012) Dichotomous organization of the external globus pallidus. *Neuron* 74:1075; Mallet N et al. (2016) Arkypallidal cells send a stop signal to striatum. *Neuron* 89:308.
- Hardman CD et al. (2002) Comparison of the basal ganglia in rats, marmosets, macaques, baboons, and humans. *J Comp Neurol* 445:238. https://pubmed.ncbi.nlm.nih.gov/11920704
- Bokulić E et al. (2021) The stereological analysis and spatial distribution of neurons in the human subthalamic nucleus. *Front Neuroanat* 15:749390. https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2021.749390/full
- Nair-Roberts RG et al. (2008) Stereological estimates of dopaminergic, GABAergic and glutamatergic neurons in the VTA, SN and RRF in the rat. *Neuroscience* 152:1024. https://pmc.ncbi.nlm.nih.gov/articles/PMC2575227/
- Pakkenberg B et al. (1991) The absolute number of nerve cells in substantia nigra in normal subjects and in patients with Parkinson's disease. *JNNP* 54:30. https://jnnp.bmj.com/content/54/1/30
- Mouse SNc count (8,305 ± 540). https://pmc.ncbi.nlm.nih.gov/articles/2705113/
- Oorschot DE (1996) Total number of neurons in the rat basal ganglia. *J Comp Neurol* 366:580 (rat counts for every nucleus; full text not retrieved here).
- Matsuda W et al. (2009) Single nigrostriatal dopaminergic neurons form widely spread and highly dense axonal arborizations. *J Neurosci* 29:444.
- Schultz W, Dayan P, Montague PR (1997) A neural substrate of prediction and reward. *Science* 275:1593.
- Cui G et al. (2013) Concurrent activation of striatal direct and indirect pathways during action initiation. *Nature* 494:238.

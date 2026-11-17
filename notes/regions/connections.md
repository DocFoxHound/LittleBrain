# Connections between regions (cell type → cell type)

Specification for roadmap step 7 (federation). It covers the pathways *between* the regions in
`prototype/brain/regions.py`. Wiring *inside* the cortex follows note 17. Every row is meant to become one
entry in a projection table.

Confidence: ● well established · ◐ supported, details or numbers vary · ○ uncertain, our estimate, or from
background knowledge not re-checked in this search. **Species is given for every number.** Almost no
inter-regional synapse share has been measured in humans.

## How to read the tables

- **Cell names** are the `name` fields in `prototype/brain/celltypes.py` (for example `L5 PT`, `CA1 pyramidal`,
  `D1 MSN`). Cells we don't model yet are in *italics* with a suggested stand-in.
- **Compartment** maps onto our neuron's input slots:
  - `basal` = basal dendrites, feedforward and driving input;
  - `apical` = apical tuft, context and feedback (layer 1, CA1 stratum lacunosum-moleculare);
  - `prox` = proximal dendrites or soma, strong "driver" terminals;
  - `soma` and `ais` = inhibitory targets;
  - `volume` = neuromodulator sent as a diffuse broadcast to a region (a gain or learning-rate signal, not a synapse slot).
- **Strength:**
  - **driver vs modulator** in Sherman & Guillery's sense: a driver is a few large, depressing, ionotropic synapses that carry the message; a modulator is many small, facilitating synapses that also act through metabotropic receptors and tune the target;
  - otherwise a share of the target's synapses, or a convergence number.
- **Topo.** = topography. `map` means point-to-point (keep the map position); `diffuse` means broad.
- **IDs** (for example `CT-3`) are meant to be used as keys in code.

---

## 1. Cortex ↔ thalamus

| ID | Source → target (cell types) | Compartment | Sign / transmitter | Strength / share | Topo. / convergence | Delay | Plasticity | Conf. |
|---|---|---|---|---|---|---|---|---|
| CT-1 | First-order thalamus `TC core` → sensory cortex `L4 stellate`, `L2/3 IT` (L3), `L5 PT` (L5B), `L6 CT`; collateral → `PV basket` (feedforward inhibition) | basal / prox | + glutamate (AMPA/NMDA) | **driver, yet a small share:** thalamic synapses are only about **6% (cat V1) to 10–20% (mouse)** of a layer 4 cell's excitatory synapses [1]. They work by being strong, convergent and synchronous (Bruno & Sakmann 2006) | map (topographic, one area) | a few ms (rodent, ○) | Hebbian, mostly in a critical period; ACh and NE gate it in adults | ◐ |
| CT-2 | `TC matrix` (calbindin) → many areas, layer 1: `L5 PT`/`L5 IT`/`L2/3 IT` tufts; `LAMP5 neurogliaform`; also `VIP` | apical | + glutamate | modulator-like, broad. No share measured | **diffuse**, crosses area borders | ○ | Hebbian on tufts | ◐ |
| CT-3 | `L6 CT` → the same thalamic nucleus that feeds it: `TC core` (distal dendrites) **and** collateral → `TRN` | apical-like (distal) | + glutamate, ionotropic + **mGluR1** | **modulator: 30–50% of relay-cell synapses** (cat/primate LGN). Retinal drivers are only ~5% (2–7%) [2]. L6 axons outnumber the sensory input to the thalamus by about **10×** [3] | map, reciprocal ("returns to where it came from") | slow axons (○) | paired-pulse **facilitation** | ● share; ○ delay |
| CT-4 | `L6 CT` → `TRN` → `TC core`/`TC matrix` (GABA-A + GABA-B) | soma / prox | − GABA | TRN is ~15% of relay synapses (cat LGN; pairing ○) [2] | sector-wise map, wider than the CT-3 patch, so it gives **surround inhibition** | — | — | ◐ |
| CT-5 | **Transthalamic:** `L5 PT` (L5B, area A) → higher-order thalamus `TC core` (pulvinar, POm, MD) → area B, layer 4/3 (as CT-1) | prox | + glutamate | **driver**: large terminals, ionotropic, depressing. They are only ~2% of a HO relay cell's synapses, yet they drive it (from `thalamus.md`) [2,3]. Mouse visual cortex: L5 drivers to the pulvinar come mostly from V1, while L6 modulators come from many areas [4] | map, one area to the next | ○ | ○ | ◐ |
| CT-6 | `L5 PT` axon branch → same targets as CT-5 **plus** brainstem, pons (CB-1), striatum (BG-2) | — | + | each L5 PT axon to the thalamus is a **copy of a motor/output command** (efference copy) | — | — | — | ◐ |
| CT-7 | Inhibitory control of HO thalamus from outside the TRN: `GPi`/`SNr` (BG-8), zona incerta (*not modeled*), pretectum | prox / soma | − GABA | strong, large terminals (from `thalamus.md`) | ○ | — | — | ◐ |
| CT-8 | Brainstem `PPN/LDT ACh` (+ `LC noradrenaline`, `raphe serotonin`) → `TC core`, `TC matrix`, `TRN` | volume + synaptic | modulatory (ACh nicotinic + muscarinic) | **~25%** of LGN relay synapses (cat) [2]. ACh depolarizes relay cells (tonic, "awake" mode) and hyperpolarizes TRN | diffuse | — | — | ◐ |

**Model notes.**
- Relay cells switch between **tonic** mode (depolarized; ACh, NE) and **burst** mode (hyperpolarized; TRN, sleep) through `burst_gain`.
- Drive mode from CT-8.
- TRN is the gate: attention means L6 CT (or prefrontal cortex) excites TRN sectors that *don't* carry the attended input. ○ as a mechanism.

---

## 2. Cortex ↔ hippocampal formation

### 2a. Into the hippocampus

| ID | Source → target | Compartment | Sign | Strength / share | Topo. / convergence | Delay | Plasticity | Conf. |
|---|---|---|---|---|---|---|---|---|
| HC-1 | Neocortex association areas → **perirhinal / parahippocampal** (*not modeled: use `L2/3 IT` of a "parahippocampal" area*) → entorhinal cortex | basal | + glu | **about 2/3 of the EC's cortical input arrives via perirhinal + parahippocampal cortex** (macaque) [5] | perirhinal ("what") → lateral EC; parahippocampal/postrhinal ("where") → medial EC; rostro-caudal bands | ○ | Hebbian | ● |
| HC-2 | Direct neocortex → EC (orbitofrontal, insula, cingulate, retrosplenial, STS) | superficial (L1–3) **or** deep (L5–6), depending on the source area [5] | + glu | the remaining ~1/3 | rostral EC: orbitofrontal/insula/ACC; caudal EC: parietal/retrosplenial [5] | ○ | Hebbian | ◐ |
| HC-3 | **Perforant path:** `EC L2 stellate` (MEC), fan cells (LEC; use `EC L2 stellate`) → `DG granule` outer 2/3 of the molecular layer; → `HIPP`, `PV` (DG feedforward inhibition) | apical-like distal dendrite (DG granule cells have no basal tree) | + glu | the main excitatory input to granule cells. Medial EC → middle third, lateral EC → outer third | **divergent, roughly random** across the septo-temporal level | ○ | NMDA-LTP; neurogenesis ages (`DG granule (young)` learns more) | ● |
| HC-4 | `EC L2 stellate` → `CA3 pyramidal` (stratum lacunosum-moleculare), and → `CA2 pyramidal` | apical | + glu | **~3,750 perforant-path synapses per CA3 cell** (rat; Treves & Rolls 1992 from Amaral 1990; Rolls & Treves 1994 give ~4,000) [6]. Many weak synapses = the **recall cue** | divergent | ○ | NMDA-LTP; CA2 tuft inputs potentiate | ◐ |
| HC-5 | **Temporoammonic:** `EC L3 pyramidal` → `CA1 pyramidal` (distal tufts, SLM); → `subiculum bursting`/`subiculum regular`; → SLM interneurons (`HC NGF`, `O-LM` targets the same zone) | **apical** | + glu | a **minority** of CA1's **~30,000 excitatory synapses** (rat; total 30,000 E + 1,700 I, ~40% perisomatic) [7]. SLM share: number not verified ○. Alone it rarely fires CA1; it provides the **plateau / instructive signal** for BTSP together with CA3 input | topographic: proximal CA1 ← LEC, distal CA1 ← MEC | ○ | **BTSP** (plateau → potentiate inputs within −3 to +2 s); `O-LM` gates it | ● circuit; ○ share |
| HC-6 | `EC L2 island` (MEC calbindin) → CA1 SLM interneurons (feedforward inhibition) | soma (of interneurons) | + glu | — | — | — | — | ◐ (from `hippocampus.md`) |

### 2b. Inside the hippocampus (trisynaptic loop)

| ID | Source → target | Compartment | Sign | Strength / share | Topo. / convergence | Plasticity | Conf. |
|---|---|---|---|---|---|---|---|
| HC-7 | **Mossy fibers:** `DG granule` → `CA3 pyramidal` (thorny excrescences, stratum lucidum) + stratum lucidum interneurons (`CA PV basket`, use for feedforward inhibition; ~10× more interneuron than pyramid contacts, ○) | **prox** | + glu (+ zinc, dynorphin) | **~46 mossy inputs per CA3 cell** (rat; Treves & Rolls 1992, from Amaral 1990) [6]. Each granule cell contacts only ~10–15 CA3 pyramids. **"Conditional detonator":** at baseline, single spikes rarely fire CA3, but trains do (in vivo; Henze 2002). After post-tetanic potentiation, a **single spike fires CA3 ("full detonation")** for tens of seconds (Vyleta 2016) [8] | sparse, lamellar | **presynaptic, NMDA-independent cAMP/PKA LTP**; 5–10× frequency facilitation. Role: **the "teacher" / storage drive** | ● |
| HC-8 | **Recurrent collaterals:** `CA3 pyramidal` → `CA3 pyramidal` (both sides in rodents; mostly ipsilateral in primates) | basal + apical oblique (stratum oriens/radiatum) | + glu | **~12,000 recurrent synapses per CA3 cell** (rat) [6]; pair connection probability **~0.9%** (rat, Guzman 2016) and <1% in human (Watson 2024) | divergent along the long axis | **NMDA-LTP, symmetric STDP window** → autoassociative storage. ACh (muscarinic) suppresses it during encoding (Hasselmo, ○) | ● |
| HC-9 | **Schaffer collaterals:** `CA3 pyramidal` → `CA1 pyramidal` (stratum radiatum = apical **oblique**, and oriens = **basal**); → `bistratified`, `CA PV basket`, `ivy` | basal / proximal-apical oblique (**not** the tuft) | + glu | the **bulk** of CA1's ~30,000 excitatory synapses (rat) [7] | divergent, longitudinal | classic NMDA-LTP/LTD (Hebbian/STDP) + **BTSP** with HC-5 | ● |
| HC-10 | `CA3 pyramidal` → `CA2 pyramidal` | basal / oblique | + glu | — | — | **resists LTP** (RGS14) | ◐ |
| HC-11 | `CA2 pyramidal` → `CA1 pyramidal` (deep sublayer) | basal | + glu | — | — | — | ◐ |
| HC-12 | `CA3 pyramidal` → hilus: `DG mossy cell`, hilar interneurons (back-projection); `DG mossy cell` → `DG granule` (inner 1/3 of the molecular layer, far along the long axis) | apical (DG) | + glu (direct) and − (via interneurons) | primates: commissural branch weak or absent | long-axis | — | ◐ |

### 2c. Out of the hippocampus

| ID | Source → target | Compartment | Sign | Strength / topography | Plasticity | Conf. |
|---|---|---|---|---|---|---|
| HC-13 | `CA1 pyramidal` → `subiculum bursting`, `subiculum regular` | basal | + glu | topographic: proximal CA1 → distal subiculum | NMDA-LTP; burst-dependent | ● |
| HC-14 | `CA1 pyramidal` + `subiculum regular`/`subiculum bursting` → **`EC L5/6 pyramidal` (layer 5b)** | basal / apical | + glu | hippocampal output preferentially targets **L5b over L5a** (mouse) [9] | ○ | ◐ |
| HC-15 | EC **L5b → L5a** → neocortex (and back to CA1): `EC L5/6 pyramidal` → association cortex `L2/3 IT`, `L5 IT`, and layer 1 (feedback-like) | apical (feedback rule from note 17) | + glu | **Only L5a sends the EC's long-range telencephalic output** (mouse, Tsoi 2022). L5a also copies it to CA1 [9]. Split `EC L5/6 pyramidal` into L5b (receiver) and L5a (sender) | consolidation in sleep (replay, ripples) | ◐ |
| HC-16 | `subiculum bursting` → mammillary bodies → anterior thalamus (*not modeled*) → cingulate (Papez circuit); `subiculum regular` → nucleus accumbens, lateral septum, amygdala, hypothalamus | basal | + glu | ventral/anterior subiculum carries emotion and context | — | ◐ |
| HC-17 | **Hippocampus → prefrontal:** ventral/intermediate `CA1 pyramidal` + proximal `subiculum regular` → prelimbic, infralimbic, medial orbital (`L5 IT`, `L5 PT`, `L2/3 IT`; + `PV basket` feedforward) | basal | + glu | **monosynaptic, ipsilateral, one-way** (rat; Jay & Witter 1991). **No direct return:** PFC → hippocampus goes via nucleus reuniens (*not modeled*: stand-in `TC matrix`) and the EC [10] | Hebbian LTP, synchrony-dependent (theta) | ● route; ○ latency (only population responses of 20–40 ms found [10]) |

### 2d. Medial septum → hippocampus (theta and acetylcholine)

| ID | Source → target | Compartment | Sign | Strength / topography | Conf. |
|---|---|---|---|---|---|
| HC-18 | `septal PV pacemaker` → hippocampal **interneurons only** (`CA PV basket`, `O-LM`, `bistratified`, `HIPP`), also MEC interneurons | soma | − GABA (rhythmic, theta 4–10 Hz) | **targets GABAergic interneurons exclusively** (Freund & Antal 1988). The theta pacemaker works by **disinhibiting** pyramids rhythmically [11] | ● |
| HC-19 | Medial septum cholinergic (`PPN/LDT ACh` used as a stand-in; better: `basal forebrain ACh`) → all hippocampal subfields + EC | volume (+ some synapses on pyramids and interneurons) | modulatory: ACh, muscarinic (M1 on pyramids ↑ excitability; presynaptic suppression of CA3 recurrent and Schaffer synapses, ○) + nicotinic (α4β2 on interneuron-specific `interneuron-specific` cells → disinhibition [12]) | **sets encoding mode:** high ACh favors EC input and learning, low ACh favors CA3 recall (Hasselmo, ○) | ◐ |
| HC-20 | `VTA glutamate`-type septal glutamate cells → hippocampal pyramids/interneurons; septum ← `CA3`/`CA1` via lateral septum (*not modeled*) | soma | + glu | — | ○ |

---

## 3. Amygdala

| ID | Source → target | Compartment | Sign | Strength / topography | Plasticity | Conf. |
|---|---|---|---|---|---|---|
| AM-1 | **Low road:** auditory/somatosensory thalamus MGm/PIN (`TC matrix` stand-in) → lateral amygdala `BLA principal (+)`/`BLA principal (-)` + `PV` | basal / prox | + glu | fast, coarse | **Hebbian NMDA-LTP gated by NE (β) and DA**; this is where CS–US pairing is stored | ● |
| AM-2 | **High road:** sensory association cortex, insula (`L2/3 IT`, `L5 IT`) → `BLA principal` | basal | + glu | slower, detailed | same as AM-1 | ● |
| AM-3 | **US / pain:** parabrachial CGRP (*not modeled*) → `CeL PKCδ (off)`; spinal → PAG/thalamus → LA | prox | + glu/CGRP | teaching signal for fear | — | ◐ |
| AM-4 | LA → BA (`BLA principal` → `BLA principal`) | basal | + glu | — | Hebbian | ● |
| AM-5 | `BLA principal` → **medial ITC** (`intercalated`) → `CeM output` (feedforward inhibition) | soma | + glu → − GABA | ITC is the **BLA/CeA gate**. **Extinction strengthens BLA → ITC synapses**, which needs IL activity during extinction (Amano 2010) [13] | LTP at BLA → ITC, D1-modulated | ● |
| AM-6 | `BLA principal` → `CeM output` (direct) and → `CeL SST (on)` / `CeL PKCδ (off)` | soma | + glu | **direct drive of fear output** | Hebbian (LA → CeL also plastic) | ● |
| AM-7 | Lateral ITC → `BLA principal` (feedforward inhibition of BLA) | soma | − GABA | separate from medial ITC [13] | — | ◐ |
| AM-8 | **Infralimbic** `L5 IT`/`L5 PT` → `intercalated` (strong) + `BLA` | soma / basal | + glu | IL **inhibits** amygdala output via ITC → **extinction** (IL inactivation impairs extinction; microstimulation reduces fear) [14] | IL plasticity during extinction consolidation | ● |
| AM-9 | **Prelimbic** `L5 IT`/`L5 PT` → `BLA principal` | basal | + glu | PL **excites** amygdala output → **fear expression** (PL inactivation blocks learned but not innate fear) [14] | — | ● |
| AM-10 | `BLA principal` → PL/IL (`L2/3 IT`, `L5 IT`, + `PV`) and → OFC | basal | + glu | reciprocal with AM-8/9; 4 Hz theta synchrony during fear recall (rodent) | Hebbian | ◐ |
| AM-11 | `BLA principal (-)` → ventral `CA1 pyramidal` (+ vHPC interneurons) | basal | + glu, plus disynaptic − | **monosynaptic glutamatergic** onto vHPC pyramids, plus polysynaptic feedforward inhibition. Activating it ↑ anxiety (mouse, Felix-Ortiz 2013) [15] | — | ● |
| AM-12 | `BLA principal (+)` → nucleus accumbens `D1 MSN`/`D2 MSN` (ventral striatum) | basal | + glu | positive-valence and reward seeking (Stuber 2011, ○ not re-checked) | DA-gated three-factor (BG rules) | ◐ |
| AM-13 | **Context:** ventral `CA1 pyramidal` / `subiculum regular` → `BLA principal` (+ `PV`) | basal | + glu | context gates fear and its extinction | Hebbian | ● |
| AM-14 | `CeM output` → PAG (freezing; *not modeled*), lateral hypothalamus, PVN (via BNST; `PVN CRH`), parabrachial / vagal complex (heart, breathing), `LC noradrenaline`, `basal forebrain ACh`, `dopamine` (VTA/SNc) | soma (GABA, so **disinhibitory**) | − GABA (+ CRF, neurotensin) | CeM → PAG mediates freezing (lesion/tracing reviews) [16]. Distinct CeM populations for freezing vs flight (○) | — | ● |
| AM-15 | `CeA CRF` → `LC noradrenaline`, parabrachial, BNST | soma / volume | CRF (excites LC) + GABA | major CeA CRF targets (rat, Crh-Cre) [16]. Sustained anxiety and arousal | slow peptide | ◐ |
| AM-16 | Inside CeA: `CeL SST (on)` ⊣ `CeL PKCδ (off)` ⊣ `CeM output` | soma | − GABA | disinhibitory chain (already tested in note 21) | Hebbian at LA → CeL | ● |

Neuromodulators into the amygdala:
- `LC noradrenaline` β → BLA principals: gates LTP. ◐
- `dopamine` D1 → ITC and BLA: needed for learning. ◐
- `basal forebrain ACh` → BLA: US salience. ◐
- `raphe serotonin`. ◐

All of these are volume input (from `amygdala.md`).

---

## 4. Basal ganglia loops

| ID | Source → target | Compartment | Sign | Strength / share | Topo. | Plasticity | Conf. |
|---|---|---|---|---|---|---|---|
| BG-1 | Cortex `L5 IT` (+ `L2/3 IT`; both hemispheres) → `D1 MSN` > `D2 MSN`; → `striatal PV FSI` (strong) | basal (MSN spines) | + glu | IT terminals are **~2× as common as PT on D1 spines** (rat EM; Lei 2004) [17]. Cortex is the largest single input to MSNs, the same order as thalamus but more abundant [18] | **map** (corticostriatal funnels: motor → putamen, PFC → caudate, limbic → accumbens), with convergence from many areas | **three-factor:** pre × post (up-state) × dopamine; D1: DA burst → LTP; D2: DA dip → LTP / burst → LTD (eCB-LTD) | ● |
| BG-2 | Cortex `L5 PT` (ipsilateral collaterals of corticofugal axons) → `D2 MSN` > `D1 MSN` | basal | + glu | PT terminals are **~4× as common as IT on D2 spines** (rat) [17]. Efference copy of motor commands | map | as BG-1 | ◐ (selectivity degree unknown [17]) |
| BG-3 | Cortex → `striatal PV FSI` → MSNs (soma) | soma | − GABA | **feedforward inhibition**. One FSI contacts many MSNs (number ○) | local | — | ● |
| BG-4 | **Thalamostriatal:** intralaminar centromedian/parafascicular (CM/Pf, `TC matrix` stand-in) → `cholinergic TAN` (strong), MSN dendritic shafts | prox / dendrite | + glu | Pf input to TANs gives a **burst, then a >0.5 s pause** (Ding 2010; pause through Kv1, Tubert 2024) → transient gating of corticostriatal input; signals **salient events / attention shifts** [19] | partly topographic | — | ● |
| BG-5 | Other thalamic → striatum (VA/VL, MD; VGluT2) → MSN spines and shafts | basal | + glu | VGluT2 terminals: 87% axospinous in striosomes vs 55% in matrix (rat) [18] | — | — | ◐ |
| BG-6 | `cholinergic TAN` → MSNs (muscarinic M1 on D1/D2, M4 on D1), dopamine terminals (nicotinic → local DA release), `striatal PV FSI`, NGF-type interneurons | volume (local) | modulatory ACh | TAN **pause coincides with the DA burst** and defines the learning window (○ as a rule) | local | — | ◐ |
| BG-7 | **Direct:** `D1 MSN` → `GPi`, `SNr` | soma / dendrite | − GABA (+ substance P, dynorphin) | **"Go":** a pause in GPi/SNr releases the thalamus | topographic | — | ● |
| BG-8 | `GPi`/`SNr` → thalamus VA/VL (motor), MD (prefrontal), CM/Pf (`TC core` / `TC matrix`); `SNr` → superior colliculus, PPN | prox | − GABA, **tonic** | tonic inhibition of the motor thalamus (GPi ~60–80 Hz in primate, ○) | topographic: parallel loops (motor, oculomotor, prefrontal, limbic) | — | ● |
| BG-9 | Thalamus VA/VL/MD `TC core` → cortex (motor/premotor/PFC `L2/3 IT`, `L5 PT`; also layer 1 via `TC matrix`) → BG-1 | basal (L3/L5), apical (L1) | + glu | **closes the loop** | map (loop-specific) | — | ● |
| BG-10 | **Indirect:** `D2 MSN` → `GPe prototypic` | soma | − GABA (+ enkephalin) | "No-go" | topographic | — | ● |
| BG-11 | `GPe prototypic` → `STN`, `GPi`/`SNr`, `striatal PV FSI`; `GPe arkypallidal` → striatum (MSNs, interneurons) ("stop") | soma | − GABA | — | — | — | ● |
| BG-12 | `STN` → `GPi`/`SNr` (+ `GPe prototypic` reciprocal) | soma / prox | + glu | the only excitatory BG nucleus. **Divergent** (one STN cell → many pallidal cells) | diffuse-ish | — | ● |
| BG-13 | **Hyperdirect:** cortex `L5 PT` (motor, SMA, IFG) → `STN` | prox | + glu | Cortical stimulation in monkey gives an **early pallidal excitation at ~8–11 ms** (via STN), inhibition at 15–19 ms (direct pathway), late excitation at 26–32 ms (indirect) [20]. Human STN stimulation evokes antidromic cortical potentials at **2.8 / 5.8 / 7.7 ms** [20] | topographic | — | ● |
| BG-14 | **SNc** `dopamine` → dorsal striatum (MSNs, `cholinergic TAN`, FSIs), small GPe branch | volume | modulatory DA | **one SNc axon covers 0.45–5.7% (mean 2.7%) of the striatum** (rat, Matsuda 2009). Total axon ~0.5 m per cell (secondary source, ○); every cell innervates both striosome and matrix [21] | **diffuse within a sector** | sets the third factor of BG-1/2 | ● |
| BG-15 | **VTA** `dopamine` → ventral striatum (accumbens) **dense**; → PFC `L5`/`L6` pyramids and interneurons **sparse**; also amygdala, hippocampus, septum (and `VTA GABA`/`VTA glutamate` co-projections) | volume | modulatory DA (the PFC subset has little DAT and D2 autoreceptor) | striatum ≫ PFC by density (●, qualitative) | diffuse | reward prediction error | ● |
| BG-16 | **Striosome** `striosome MSN` → SNc `dopamine` (bouquets around DA dendrites in SNr, "striosome–dendron bouquets") | dendrite / soma | − GABA | strong inhibition of SNc firing; release from it → **rebound** (Crittenden 2016; Evans 2020) [22] | topographic | — | ● (matrix vs striosome strength debated) |
| BG-17 | **Inputs to SNc/VTA DA** (see NM-1) | | | | | | |

**Missing cell types:**
- **GPi/SNr → thalamus**: needs a "VA/VL" and an "MD" subregion inside the higher-order thalamus (use `TC core`).
- **Lateral habenula**, **RMTg** and **superior colliculus** are not modeled.

---

## 5. Cerebellar loop

| ID | Source → target | Compartment | Sign | Strength / share | Topo. / conv. | Plasticity | Conf. |
|---|---|---|---|---|---|---|---|
| CB-1 | Cortex `L5 PT` (motor, premotor, **prefrontal**, parietal) → **pontine nuclei** (*not modeled*: stand-in `VTA glutamate`-like relay, or a new `pontine` type) | basal | + glu | **~19 M corticopontine fibers per side → ~23 M pontine cells** (human, Tomasch 1968, via secondary source; another review gives 40 M, ○) [23]. In humans, a relatively **large prefrontal contribution** (Ramnani 2006) | topographic: PFC medial in the peduncle, motor middle third, posterior areas lateral | — | ◐ |
| CB-2 | Pontine → (crossed) **mossy fibers** → `cerebellar granule` (each granule cell has 4 dendrites, **one mossy fiber each**), `unipolar brush`, `Golgi`; collaterals → `DCN projection` | prox (glomerulus) | + glu | expansion recoding: one mossy fiber → hundreds of granule cells (○) | patchy, fractured somatotopy | MF→GC LTP (minor) | ● |
| CB-3 | `Golgi` → `cerebellar granule` (glomerulus) | soma | − GABA (A + tonic) | sets granule sparseness | local | — | ● |
| CB-4 | **Parallel fibers:** `cerebellar granule` → `Purkinje`, `cerebellar stellate`, `cerebellar basket`, `Golgi` | Purkinje distal spiny dendrites (`basal` slots in our model) | + glu | **rat ~175,000 PF synapses per Purkinje cell** (~150–200 k; `cerebellum.md`). **Human: not measured.** Morphology gives ~7.5× the spines of mouse (Masoli 2024). A 2025 review proposes **300–500 k in human vs 30–50 k in mouse** [24]. The often-quoted "≈200,000 in humans" is the rat-range number and is **not** a human count. Most PF synapses are silent (~85%, Isope & Barbour 2002) | **beam-like:** PF runs 2–3 mm (rat) to ~6 mm (human) along a folium, perpendicular to Purkinje planes | **PF-LTD when PF precedes CF by ~50–200 ms**; PF-LTP without CF | ◐ rat, ○ human |
| CB-5 | **Climbing fiber:** `inferior olive` → `Purkinje` (proximal dendrite); collaterals → `DCN projection`, MLIs (spillover) | prox | + glu | **1 CF per adult Purkinje cell (rodent classic).** **But** ~25% of multi-branched mouse Purkinje cells receive >1 CF, and **most human Purkinje cells have 2–3 primary dendrites**, so multi-innervation is likely in humans (Busch & Hansel 2023, human evidence indirect) [25]. One olive cell → ~14–17 Purkinje cells (human), ~7 (rat) | **strict microzones** (parasagittal stripes, zebrin+/−) | the CF *is* the teaching/error signal (~1 Hz complex spikes) | ● (1 CF), ◐ (human multi-CF) |
| CB-6 | `cerebellar stellate` → Purkinje dendrites; `cerebellar basket` → Purkinje AIS (pinceau) | dendrite / ais | − GABA | — | off-beam (lateral) inhibition | — | ● |
| CB-7 | `Purkinje` → `DCN projection`, `nucleo-olivary`, DCN interneurons; flocculonodular → vestibular nuclei | soma / prox | − GABA | **functional convergence 20–50 Purkinje → 1 DCN cell** (rodent, Person & Raman 2012); anatomical estimates 100–860 [26]. Synchronous Purkinje pauses → time-locked DCN spikes / **rebound** | microzone → DCN region | PC→DCN and MF→DCN plasticity (○) | ◐ |
| CB-8 | `DCN projection` (dentate, interposed) → crossed **VL thalamus** (`TC core`, "motor thalamus") → motor, premotor, **prefrontal**, parietal cortex (L3/L5, the same areas that sent CB-1) | prox (VL) | + glu | **driver-like:** large perisomatic terminals, large depressing EPSCs, denser in VL than in VM/CL. L6 inputs to the same cells are small and facilitating (mouse; Gornati 2018, Schäfer 2021) [27] | **closed loops:** a cortical area gets back the cerebellar output it fed (Strick) | — | ● |
| CB-9 | `DCN projection` → red nucleus, brainstem; → intralaminar thalamus (CL) → striatum (see CB-11) | — | + glu | — | — | — | ● |
| CB-10 | `nucleo-olivary` → `inferior olive` (same microzone) | dendrite (gap-junction region) | − GABA | **closes the error loop**: once an error is learned, the DCN silences the olive (already tested in note 21) | microzone-specific | — | ● |
| CB-11 | **Cerebellum ↔ basal ganglia (disynaptic):** (a) dentate `DCN projection` → thalamus (intralaminar, CL/VL) → striatum MSNs (Hoshi 2005); (b) `STN` → pontine nuclei (likely; also PPN) → cerebellar cortex (Crus II, VIIB) (Bostan 2010) [28] | basal (MSN); prox (granule) | + glu | **substantial** (macaque rabies tracing). Topographic. The intermediate nuclei of (b) are unconfirmed | topographic | — | ◐ |

---

## 6. Neuromodulatory nuclei: inputs, outputs and receptor effects

### 6a. What drives each nucleus

| ID | Source → nucleus (cell types) | Sign | Effect / share | Conf. |
|---|---|---|---|---|
| NM-1 | Into **VTA** `dopamine`/`VTA GABA`: **lateral hypothalamus** (strongest input to VTA DA in mouse rabies maps; `orexin` + LH GABA/glu), **striosome/ventral striatum patches** (direct to DA; matrix → GABA cells), ventral pallidum, **PPN/LDT** (`PPN/LDT ACh` + glu: **needed for phasic bursts**), **lateral habenula → RMTg (GABA) → DA** (reward omission / aversive: LHb excited, DA inhibited), prefrontal cortex (`L5 PT`), BNST, `CeM output`, dorsal raphe, `VTA GABA` (local, subtracts expected reward) [29,30] | mixed | burst = PPN/LDT + LH glutamate + disinhibition; dip = LHb → RMTg | ● (sources), ◐ (weights) |
| NM-2 | Into **SNc** `dopamine`: motor and somatosensory cortex, **`STN`** (relatively strong excitation), `striosome MSN`, `GPe prototypic`, `SNr` (local), PPN [29] | mixed | — | ● |
| NM-3 | Into **LC** `LC noradrenaline`: `CeA CRF` (excites), **`orexin` (densest extrahypothalamic orexin target)**, prefrontal/ACC cortex (top-down control of phasic mode, Aston-Jones & Cohen), nucleus paragigantocellularis, PVN CRH (○) [16,31] | + | surprise/salience → phasic burst | ● |
| NM-4 | Into **dorsal raphe** `raphe serotonin`: mPFC (strong top-down), LHb, `orexin` (OX1R + OX2R), LH, VTA-like inputs similar to dopamine cells (Uchida lab) [31] | mixed | — | ◐ |
| NM-5 | Into **nucleus basalis** `basal forebrain ACh`: `CeM output`/`CeA` (US salience), PFC, `orexin`, `LC noradrenaline`, brainstem | mixed | phasic bursts to reinforcement/punishment (Hangya 2015) | ◐ |
| NM-6 | Into **TMN** (histamine; *not modeled*): `orexin` via **OX2R only** [31] | + | wake | ● |

### 6b. Where they project (all `volume`)

| ID | Nucleus → targets | Density / topography | Conduction | Conf. |
|---|---|---|---|---|
| NM-7 | `dopamine` SNc → dorsal striatum (BG-14); VTA → accumbens, PFC, amygdala, hippocampus (BG-15) | striatum dense, sector-wise; PFC sparse, deep layers | — | ● |
| NM-8 | `LC noradrenaline` → **whole cortex** (all layers), hippocampus, amygdala, thalamus, cerebellum, spinal cord; ~no input to the striatum | **diffuse, single cells branch widely** | unmyelinated; **1–5 m/s** (monkey estimate); latency varies by >20 ms with recent firing (rat) [32] | ● |
| NM-9 | `raphe serotonin` DR → forebrain, cortex, striatum, amygdala; MR → **hippocampus and septum** | diffuse | slow (○) | ● |
| NM-10 | `basal forebrain ACh` nbM (Ch4) → **neocortex + amygdala**; septum/DB (Ch1–2) → hippocampus/EC (HC-19) | **more topographic than NE**: a nbM sector → particular cortical areas (○) | — | ● |
| NM-11 | `PPN/LDT ACh` → thalamus (relay + TRN, CT-8), VTA/SNc (NM-1), STN, GPi, striatum TANs, brainstem locomotor | diffuse | — | ● |

### 6c. Receptor effects on target cells (what to model)

| Transmitter / receptor | Where (our types) | Effect to model | Conf. |
|---|---|---|---|
| **DA D1** (Gs/olf, low affinity → responds to **phasic bursts**, ○) | `D1 MSN`; PFC `L5` pyramids (deep layers); `intercalated` | ↑ excitability in the up-state; **burst → LTP** at corticostriatal synapses (BG-1); PFC: inverted-U on working memory (○) | ● |
| **DA D2** (Gi, high affinity → senses **tonic** levels, ○) | `D2 MSN`; DA cell autoreceptors (absent on the PFC-projecting subset); `cholinergic TAN` | ↓ excitability; **dip → LTP, burst → LTD** in D2 MSNs (`dopamine_sign = −1`, already in code); D2 on TANs contributes to the pause | ● |
| **NE α2A** (Gi; dendritic spines with HCN channels) | PFC `L2/3 IT`/`L5` pyramids | **moderate NE strengthens PFC recurrent networks** (closes HCN → ↑ working-memory connectivity) [33] | ● |
| **NE α1** (Gq, lower affinity) | PFC pyramids; thalamus | **high NE (stress) impairs PFC** working memory; shifts control to posterior/subcortical areas [33] | ● |
| **NE β** (Gs) | `BLA principal`, hippocampus, cortex | ↑ gain and **↑ plasticity** (gates BLA LTP; memory of emotional events) | ◐ (β1 vs β2 split not re-checked) |
| **5-HT1A** (Gi; at the **axon initial segment** of pyramids, ○ exact site) | PFC `L5 PT`/`L5 IT`; also the raphe autoreceptor | **inhibition**: 84% of mouse mPFC L5 pyramids show a 1A-dependent inhibitory response [34] | ◐ |
| **5-HT2A** (Gq; **apical dendrites**) | `L5 PT` (incl. PFC → DR/VTA projectors), `L5 IT` | **excitation** of the apical tuft, so it boosts the "context" input; only ~9% of mouse mPFC L5 pyramids show a net 2A response [34]. ~80% of 1A/2A-expressing PFC neurons co-express both (rodent) [34] | ◐ |
| **5-HT3A** (ionotropic) | `VIP`, `LAMP5 neurogliaform` (the 5HT3aR interneuron class); hippocampal `CCK basket` | fast excitation of disinhibitory interneurons | ◐ [12] |
| **ACh nicotinic** (α4β2; α7) | `VIP`/`interneuron-specific`, layer 1 interneurons (`LAMP5 neurogliaform`); thalamocortical terminals; DA terminals in the striatum | **fast disinhibition** (VIP ⊣ SST → tufts open); ↑ thalamocortical (feedforward) release [12] | ● |
| **ACh muscarinic M1** (Gq) | cortical/hippocampal pyramids; `D1 MSN`/`D2 MSN` | ↑ excitability, ↑ plasticity; presynaptic M2/M4 suppress intracortical/recurrent synapses → **favors feedforward input (encoding mode)** (Hasselmo, ○) | ◐ |
| **Orexin OX1R / OX2R** (Gq, excitatory) | LC (OX1R), TMN (**OX2R only**), raphe (both), cortex L6b, basal forebrain | tonic excitation that **stabilizes wake** [31] | ● |

---

## 7. Hypothalamus

| ID | Source → target | Compartment | Sign | Strength / notes | Conf. |
|---|---|---|---|---|---|
| HY-1 | `AgRP/NPY` → PVN MC4R satiety cells (*stand-in*: `PVN CRH` / a PVN-MC4R glutamate type) | soma | − GABA + NPY; AgRP antagonizes MC4R | AgRP → PVN inhibition is sufficient to drive feeding (Atasoy 2012, ○ not re-checked). Arcuate glutamate and AgRP converge on PVN MC4R cells [35] | ● |
| HY-2 | `POMC` → PVN MC4R (α-MSH), LH | volume / soma | + peptide (slow) | satiety; slow (minutes–hours) | ● |
| HY-3 | PVN glutamate (TRH/PACAP) → `AgRP/NPY` | soma | + glu | **reverse loop**: activating these cells in sated mice drives intense feeding (Krashes 2014) [35] | ● |
| HY-4 | `AgRP/NPY` → LH, PVT, parabrachial (⊣ CGRP "meal-end" cells) | soma | − GABA | — | ◐ |
| HY-5 | `orexin` → `LC noradrenaline` (densest), `raphe serotonin`, TMN (histamine), `basal forebrain ACh`, `dopamine` (VTA), cortex (**L6b**), thalamus (midline) | volume | + orexin (+ glu, dynorphin) | **wake stabilizer**; loss → narcolepsy [31] | ● |
| HY-6 | `MCH` → wide (cortex, hippocampus, LC, raphe) | volume | − (GABA + MCH) | promotes REM/sleep | ◐ |
| HY-7 | LH GABA (*not modeled*) → `VTA GABA` (disinhibits DA) and LH glutamate → VTA; LH ⇄ accumbens shell | soma | ± | LH is the strongest single input to VTA DA cells (mouse) [29] | ● |
| HY-8 | `SCN clock` → **SPZ** (main target) → **DMH** (GABA/glu) → VLPO (sleep), LH `orexin` (wake), PVN (`PVN CRH`: cortisol rhythm; melatonin via sympathetic → pineal) | soma | − GABA (SCN) | **direct SCN → VLPO/orexin input is sparse; most of it is relayed via SPZ → DMH.** DMH lesions almost abolish the circadian sleep rhythm (Saper) [36] | ● |
| HY-9 | `PVN CRH` → median eminence → pituitary ACTH → adrenal **cortisol** | blood (global) | hormone | **slow global signal:** minutes to hours. Model as a scalar that rises slowly and lingers (already in `PVN CRH`, `slow`). Feedback: cortisol on hippocampal and PFC GR/MR receptors → ↓ PVN | ● |
| HY-10 | `PVN oxytocin`, `PVN/SON vasopressin` → posterior pituitary (blood) **and** central axons → amygdala (CeA: oxytocin ↓ fear), accumbens, VTA, septum, `CA2 pyramidal` (vasopressin: social memory) | blood + volume | peptide | bonding, caregiving, social memory | ◐ |
| HY-11 | Amygdala/BNST → PVN (stress); `CeM output` → LH/PVN (AM-14) | — | ± | — | ● |

---

## 8. The minimal federated brain

Here is the smallest set of pathways that still closes the main loops, in order of importance. Build them in
this order. Every step should still run (and pass the recall tests of note 20) before the next is added.

1. **Sensory–thalamocortical core.**
   - Pathways: CT-1 (core → L4/L3, driver); CT-3 + CT-4 (L6 CT → relay + TRN, modulator + gate); note-17 feedforward (L2/3 IT → next area, basal) and feedback (L5/L6 IT → L1, apical).
   - *Already built in note 19.*
2. **Hippocampal memory loop.**
   - Pathways: HC-1 (cortex → EC superficial); HC-3 (EC L2 → DG); HC-7 (DG → CA3, ~46 strong mossy inputs = teacher); HC-8 (CA3 recurrent, <1%); HC-4 (EC L2 → CA3, cue); HC-9 (CA3 → CA1 basal/oblique); HC-5 (EC L3 → CA1 tuft, BTSP); HC-13/14 (CA1 → subiculum → EC L5b); HC-15 (EC L5a → neocortex, apical).
   - This is *sensory → cortex → hippocampus → cortex*.
3. **Neuromodulator broadcast (minimal).** Two global scalars at first, as `volume`:
   - **dopamine** (VTA/SNc; RPE: burst/dip) → striatum (third factor) + PFC + amygdala + hippocampus (BG-14/15);
   - **acetylcholine** (BF/septum) → cortex + hippocampus; sets encoding vs recall (HC-19, NM-10).

   Add NE (LC: gain + plasticity, surprise) and 5-HT (slow average reward) next. Their inputs (NM-1, NM-3) close the loop: the brain's own errors and surprises drive them.
4. **Cortex–basal ganglia–thalamus loop.**
   - Pathways: BG-1/2 (L5 IT → D1, L5 PT → D2); BG-7 (D1 → GPi); BG-10 → BG-11 → BG-12 (D2 → GPe → STN → GPi); BG-13 (hyperdirect "stop"); BG-8 (GPi → VA/VL/MD); BG-9 (thalamus → cortex); BG-16 (striosome → SNc).
   - Learning: three-factor rule with dopamine.
5. **Amygdala ↔ cortex/hippocampus.**
   - Pathways: AM-1/2 (thalamus/cortex → LA); AM-13 (vCA1 → BLA, context); AM-5/6 (BLA → ITC/CeM); AM-8/9 (IL → ITC extinction, PL → BLA expression); AM-11/12 (BLA → vCA1, BLA → accumbens); AM-14/15 (CeM/CRF → LC, BF, VTA, PVN).
   - The last of these makes the amygdala's "overreaction" **global** through the neuromodulators.
6. **Cortex–cerebellum–thalamus loop.**
   - Pathways: CB-1/2 (L5 PT → pons → mossy fibers → granule); CB-4 (PF → Purkinje); CB-5 (olive → Purkinje, error); CB-7 (Purkinje → DCN); CB-8 (DCN → VL → the same cortex, driver); CB-10 (DCN ⊣ olive).
   - Can be added last, since the brain runs without it. It is the slowest to scale (the granule expansion).
7. **Hypothalamic drives** as slow global scalars:
   - HY-5 (orexin: wake);
   - HY-8 (clock → sleep/wake, cortisol rhythm);
   - HY-9 (cortisol);
   - HY-1/3 (hunger).

   These feed NM-1/NM-3, which is the link between *need* and *how much the brain cares*.

**New cell types this needs** (not in `celltypes.py`):
- pontine relay neuron;
- VA/VL and MD subregions of the thalamus (can reuse `TC core`);
- lateral habenula and RMTg (for the dopamine dip);
- PAG (freezing output);
- PVN MC4R cell;
- split `EC L5/6 pyramidal` into L5b (receiver) and L5a (sender).

---

## Main uncertainties (numbers to treat as soft)

- **Human parallel fibers per Purkinje cell:**
  - not measured;
  - the rat figure is ~175 k;
  - human estimates come from models and run from ~200 k (old) to 300–500 k (Masoli 2025).
- **One climbing fiber per Purkinje cell** may not hold in humans (multiple primary dendrites).
- **Mossy fibers per CA3 cell (~46), recurrent synapses (~12,000) and perforant path synapses (~3,750)** are all rat estimates from the early 1990s.
- The **CA1 tuft (SLM) share** of excitatory synapses was not found. Only the total (~30,000, rat) is verified.
- The **thalamic synapse share** onto L4 runs from 6% to 20%, depending on species and method.
- The **L6 share** of relay synapses (30–50%) and the "10×" outnumbering come from cat/primate work and secondary citations.
- **Corticostriatal selectivity** (IT → D1 2×, PT → D2 4×) comes from one rat EM study, and the degree of selectivity is unknown.
- **Purkinje → DCN convergence:** 20–50 (functional) vs 100–860 (anatomical).
- **Latencies** are mostly unknown for our purposes. Only the hyperdirect pathway (8–11 ms pallidal excitation, monkey) and the STN–cortex antidromic latencies (human) are verified. Hippocampus → PFC latency was not found.
- **Receptor sites:** 5-HT1A at the axon initial segment, and D1 vs D2 affinity (phasic vs tonic) are background knowledge, marked ○/◐.

---

## Sources

1. Thalamocortical share of L4 synapses: Ahmed et al. 1994; Bruno & Sakmann 2006 *Science*, summarized in [PMC6623786](https://pmc.ncbi.nlm.nih.gov/articles/PMC6623786), [PMC4078833](https://pmc.ncbi.nlm.nih.gov/articles/PMC4078833), [PMC2680913](https://pmc.ncbi.nlm.nih.gov/articles/PMC2680913/)
2. Driver/modulator shares on relay cells: Sherman & Guillery; Van Horn et al. 2000. See `notes/regions/thalamus.md` and its sources.
3. L6 outnumbering and the transthalamic route: Llano & Sherman 2010 ([PMC2774389](https://pmc.ncbi.nlm.nih.gov/articles/PMC2774389)); [Frontiers Neurosci 2012](https://www.frontiersin.org/articles/10.3389/fnins.2012.00053/pdf); Borbély et al. 2025 [Sci Rep](https://doaj.org/article/9278df9042eb436d87b494cecf077ce7)
4. Cassidy et al. 2024, L5 vs L6 inputs to the pulvinar: [Kavli preprint](https://preprints.kavlimeetings.org/2024/07/17/all/neuro/ucsd-salk/204315/)
5. Neocortex → entorhinal cortex, macaque (Insausti, Amaral, Mohedano-Moriano): [PMC3313676](https://pmc.ncbi.nlm.nih.gov/articles/PMC3313676), [PMC5070464](https://pmc.ncbi.nlm.nih.gov/articles/PMC5070464)
6. CA3 input numbers: Treves & Rolls 1992 [PDF](https://www.oxcns.org/papers/163_Treves+Rolls92.pdf); Rolls & Treves 1994 [PDF](https://www.oxcns.org/papers/182%20Rolls%20Treves%201994%20Neural%20networks%20involved%20in%20memory.pdf); Amaral, Ishizuka & Claiborne 1990; Guzman et al. 2016 *Science*
7. Megías et al. 2001, *Neuroscience* 102:527, CA1 synapse totals: [PubMed 11226691](https://pubmed.ncbi.nlm.nih.gov/11226691/)
8. Mossy fiber detonation: Henze et al. 2002 [PDF](https://itb.biologie.hu-berlin.de/%7Ekempter/Hippocampus_Journal_Club/Articles/henze02.pdf); Vyleta et al. 2016 [eLife 17977](https://elifesciences.org/articles/17977)
9. Entorhinal L5a/L5b: Sürmeli et al. 2015 *Neuron* [PMC4675718](https://pmc.ncbi.nlm.nih.gov/articles/PMC4675718); Tsoi et al. 2022 [eLife 73162](https://elifesciences.org/articles/73162)
10. Hippocampus → PFC: Jay & Witter 1991 *J Comp Neurol* 313:574; Thierry et al. 2000 *Cereb Cortex*; [PMC5975431](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5975431/); [Sci Rep 2018](https://www.doi.org/10.1038/S41598-018-19540-6)
11. Septal GABA → hippocampal interneurons: Freund & Antal 1988 *Nature*, cited in [PMC4666911](https://pmc.ncbi.nlm.nih.gov/articles/PMC4666911), [PMC6554630](https://pmc.ncbi.nlm.nih.gov/articles/PMC6554630)
12. VIP / 5-HT3A / nicotinic disinhibition: Guet-McCreight et al. 2020 [PMC7296096](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7296096/); Bell et al. 2015 [Front Cell Neurosci](https://doaj.org/article/3d9097dde0284dec9388c596a5119124)
13. ITC cells and extinction: Amano et al. 2010 [PMC2847017](https://pmc.ncbi.nlm.nih.gov/articles/PMC2847017); [PMC3118749](https://pmc.ncbi.nlm.nih.gov/articles/PMC3118749); lateral vs medial ITC, [Cell Rep 2015](https://doaj.org/article/db7fe00f3e034d1bad5302ab529d3100)
14. PL vs IL: Sierra-Mercado et al. 2011 *Neuropsychopharmacology*; Vidal-Gonzalez et al. 2006 [Learn Mem](https://learnmem.cshlp.org/content/13/6/728.abstract); Laurent & Westbrook 2009 [Learn Mem](https://learnmem.cshlp.org/content/16/9/520.full)
15. BLA → ventral hippocampus: Felix-Ortiz et al. 2013 *Neuron* [PDF](https://tyelab.org/wp-content/uploads/2013-bla-to-vhpc-inputs-modulate-anxiety-related-behaviors.pdf); BLA → accumbens: Stuber et al. 2011 *Nature* (not re-checked)
16. CeA outputs: Pomrenze et al. 2015, CeA CRF → LC/PBN/BNST, [Front Neurosci](https://www.frontiersin.org/articles/10.3389/fnins.2015.00487/text); CeA → PAG freezing reviews [ANE](https://ane.pl/index.php/ane/article/view/1449)
17. IT/PT → D1/D2: Reiner et al. 2010 [Front Neuroanat](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2010.00142/full); Lei et al. 2004 *J Comp Neurol* 473
18. Cortical vs thalamic input to MSNs: Doig et al. 2010 [PMC6633626](https://pmc.ncbi.nlm.nih.gov/articles/PMC6633626); Huerta-Ocampo et al. 2014; Raju et al. 2006 [PDF](https://med.aws.emory.edu/departments/pharmacology-chemical-biology/labs/hall/_documents/publications/Raju_et_al_2006.pdf)
19. Thalamus → TANs: Ding et al. 2010 *Neuron* 67:294; Tubert et al. 2024 [eLife 102184](https://elifesciences.org/articles/102184); [Front Neural Circuits 2015](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00015/full)
20. Hyperdirect pathway timing: Nambu et al. 2000; Kita et al. 2004, summarized in [Front Hum Neurosci 2021](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.590251/pdf) and [PMC3221766](https://pmc.ncbi.nlm.nih.gov/articles/PMC3221766/figure/F1); human evoked potentials [PMC6199405](https://pmc.ncbi.nlm.nih.gov/articles/PMC6199405)
21. Single nigrostriatal axons: Matsuda et al. 2009 *J Neurosci* 29:444 [PMC6664950](https://pmc.ncbi.nlm.nih.gov/articles/6664950/)
22. Striosome → SNc: Crittenden et al. 2016 *PNAS* 113:11318; Evans et al. 2020 *Cell Rep* [PDF](https://www.med.upenn.edu/ngg/assets/user-content/documents/jc-2020-2021/evansetal-cellreports2020-paper1.pdf); Lazaridis et al. 2024 [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.01.596922.full.pdf)
23. Corticopontine counts: Tomasch 1968 [notes](https://pseudocerebellum.readthedocs.io/en/latest/notes/TomaschJ-1968.html); [Wikipedia: corticopontine fibers](https://en.wikipedia.org/wiki/Corticopontine_fibers); Ramnani et al. 2006 *Cereb Cortex*
24. Human Purkinje cells: Masoli et al. 2024 *Commun Biol* [PMC10761885](https://pmc.ncbi.nlm.nih.gov/articles/PMC10761885); Masoli et al. 2025 [Front Physiol](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1671271/pdf); Isope & Barbour 2002
25. Climbing fiber multi-innervation: Busch & Hansel 2023 *Science* [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.27.534425.full.pdf); [NIH Director's blog](https://directorsblog.nih.gov/2023/11/28/the-amazing-brain-turning-conventional-wisdom-on-brain-anatomy-on-its-head/); human follow-up [eLife 105013](https://elifesciences.org/articles/105013v1)
26. Purkinje → DCN convergence: Person & Raman 2012 [PMC3268051](https://pmc.ncbi.nlm.nih.gov/articles/PMC3268051); Luthman et al. 2011 [PDF](https://uhra.herts.ac.uk/id/eprint/2645/1/LuthmanCerebellum2011.pdf)
27. Cerebellar nuclei → thalamus: Gornati et al. 2018 [Cell Rep](https://doaj.org/article/7958ff97741b40bd9eafa8a7509b0b4a); Schäfer et al. 2021 [PMC8048920](https://pmc.ncbi.nlm.nih.gov/articles/PMC8048920)
28. Basal ganglia ↔ cerebellum: Bostan, Dum & Strick 2010 *PNAS* [PDF](https://web.math.princeton.edu/~sswang/literature_general_unsorted/bostan_strick10_BG-CB_PNAS.pdf); Hoshi et al. 2005 *Nat Neurosci* [PDF](https://web.math.princeton.edu/~sswang/CB-nonmotor/hoshi_strick05_NN_BG-CB.pdf); Jwair et al. 2017 [Front Neuroanat](https://www.frontiersin.org/articles/10.3389/fnana.2017.00013/full)
29. Inputs to dopamine neurons: Watabe-Uchida et al. 2012 *Neuron* [PDF](https://web.math.princeton.edu/~sswang/CB-VTA/watabe-uchida_uchida12_neuron.pdf)
30. Lateral habenula → RMTg → VTA: Jhou et al. 2009; Hong et al. 2011, reviewed in [PMC3873746](https://pmc.ncbi.nlm.nih.gov/articles/PMC3873746), [PMC3055682](https://pmc.ncbi.nlm.nih.gov/articles/PMC3055682)
31. Orexin projections and receptors: Peyron et al. 1998 [PubMed 9822755](https://pubmed.ncbi.nlm.nih.gov/9822755/); receptor review [PMC5767120](https://pmc.ncbi.nlm.nih.gov/articles/PMC5767120)
32. LC axon conduction: Aston-Jones et al. (Rutgers record) [link](https://www.researchwithrutgers.org/en/publications/brain-aminergic-axons-exhibit-marked-variability-in-conduction-ve/); Aston-Jones 1981 thesis [PDF](https://thesis.caltech.edu/18481/01/Aston-Jones_G_1981.pdf)
33. NE and PFC (α2A / α1, inverted U): Wang et al. 2007 *Cell*; Arnsten lab [summary](https://medicine.yale.edu/lab/arnsten/research/invertedu/); review [PMC4876052](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876052)
34. 5-HT1A/2A in PFC: Santana et al. 2004 *Cereb Cortex*; Celada/Artigas review [Front Integr Neurosci 2013](https://frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2013.00025/text); mouse L5 responses [Front Neural Circuits 2012](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2012.00012/text)
35. Arcuate ⇄ PVN: Atasoy et al. 2012 *Nature* 488:172 (not re-checked); Krashes et al. 2014 *Nature* (PVN → AgRP); Garfield et al. 2015 [Nat Neurosci](https://www.Nature.com/articles/nn.4442)
36. SCN output relays: Saper group; Deurveilher & Semba 2005 [J Biol Rhythms](https://journals.sagepub.com/doi/reader/10.1177/0748730405280924); [VLPO overview](https://en.wikipedia.org/wiki/Ventrolateral_preoptic_nucleus)

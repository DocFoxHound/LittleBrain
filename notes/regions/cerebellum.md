# Cerebellum: neuron types and proportions

Scope: cerebellar cortex (granular, Purkinje and molecular layers), deep cerebellar nuclei (DCN), and the inferior olive (IO)
as the source of climbing fibers.

Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain.

---

## 0. Headline numbers (human)

| Quantity | Value | Source | Conf. |
|---|---|---|---|
| Neurons in whole human brain | **86.1 ± 8.1 billion** | Azevedo et al. 2009 (isotropic fractionator) | ● |
| Neurons in cerebellum | **69.03 ± 6.65 billion = ~80% of all brain neurons**, in ~10% of brain mass | Azevedo et al. 2009 | ● |
| Neurons in cerebral cortex | ~16 billion, ~19% | Azevedo et al. 2009 | ● |
| Granule cells (stereology) | **~101 billion** (5 elderly men) | Andersen, Korbo & Pakkenberg 1992 | ◐ |
| Purkinje cells | **15–30.5 million** (15.4 M fractionator 1989; 19.7–27 M Agashiwala 2008; 30.5 M Andersen 1992) | several | ◐ |
| Dentate nucleus neurons | **~5.0 million** | Andersen et al. 1992 | ◐ |

The two methods disagree: the stereological granule count (101 B) exceeds Azevedo's total cerebellar count (69 B). Granule cells are
>99% of cerebellar neurons either way, so for modelling **granule cells ≈ 99% of cerebellar neurons ≈ 75–80% of all brain neurons**. ◐

---

## 1. Cerebellar cortex: composition table

Percentages below are of cerebellar-cortex neurons. Granule cells dominate so completely that every other type is a fraction of a percent;
**ratios per Purkinje cell** are more useful for building circuits.

| Type (marker) | % of cortex neurons | Per Purkinje cell | Species | Absolute count | Transmitter | E/I; local/projection; axon | Conf. |
|---|---|---|---|---|---|---|---|
| **Granule cell** (Gabra6, NeuroD1) | **>99%** | **~430 (rat); ~3,300 (human, 101 B / 30.5 M)** | rat (Korbo 1993: 265 M); human | human ~10¹¹ | glutamate | excitatory local: ascending axon + **parallel fiber** (2–3 mm in rat, ~6 mm human) → Purkinje, MLIs, Golgi | ● |
| **Purkinje cell** (Calb1, Pcp2, L7) | ~0.02–0.2% | 1 | human, rat | human 15–30 M; rat 0.61 M | GABA | **inhibitory projection** → DCN and vestibular nuclei (sole cortical output); recurrent collaterals to Purkinje/MLIs/Lugaro | ● |
| **Golgi cell** (mGluR2, Neurogranin; some GlyT2) | ~0.2–0.25% (rat Golgi:granule ≈ 1:430) | **~1** (rat 0.64 M vs 0.61 M Purkinje); cat ~1:3 | rat; cat | rat 0.64 M; human ~81 M reported (Andersen 1992 via compilation) | GABA ± glycine | inhibitory local → granule-cell dendrites in glomeruli (feedforward + feedback) | ◐ |
| **Basket cell** (PV, molecular layer, lower third) | ~0.3% (cat) | **~6** (cat: 7.5 M / 1.25 M) | cat | cat 7.5 M | GABA | inhibitory local → Purkinje soma; **"pinceau"** around axon initial segment (ephaptic inhibition) | ◐ |
| **Stellate cell** (PV, outer molecular layer) | ~0.9% (cat) | **~17** (cat: 20.9 M / 1.25 M) | cat | cat 20.9 M | GABA | inhibitory local → Purkinje dendrites | ◐ |
| Molecular-layer interneurons total (basket + stellate; MLI1/MLI2 by scRNA-seq) | | **~20+** | cat; mouse | | GABA | MLI2 inhibit MLI1 (disinhibition of Purkinje) | ◐ |
| **Lugaro cell** | rare | ≪ 1 | rat | not quantified | GABA + glycine | inhibitory → Golgi cells, MLIs; driven by serotonin and Purkinje collaterals | ○ |
| **Unipolar brush cell (UBC)** (Tbr2/Eomes; ON = mGluR1, OFF = calretinin/mGluR2) | rare overall; **dense in vestibulocerebellum** (lobules IX–X, flocculus), comparable to Golgi there | — | rat, mouse | not quantified | glutamate | **excitatory local** → granule cells and other UBCs (feedforward amplifier) | ◐ qualitative |
| Candelabrum cell, globular cell | rare | — | mouse, rat | — | GABA | inhibitory local (Purkinje layer) | ○ |

Human vs rat: human has ~50× more Purkinje cells than rat but 13× lower density (Andersen & Korbo 1993).

### Signature behaviors to model (testable)

- **Granule cell.** Only **4 short dendrites ("claws")**, each receiving one mossy fiber rosette → each cell samples ~4 inputs; very low resting rate, high threshold; under tonic Golgi inhibition (GABA_A δ-subunit). Implements **sparse expansion recoding** (Marr 1969, Albus 1971). *Test:* with Golgi inhibition, only a small fraction (≈ few–10%) of granule cells fire for a given mossy-fiber pattern, and two overlapping input patterns produce less-overlapping outputs (like DG granule pattern separation).
- **Purkinje cell.** Huge planar dendrite with **~150,000–200,000 parallel-fiber synapses** (rat ~175 k) and **one climbing fiber**. Fires intrinsic **simple spikes ~40–100 Hz** even without input; the climbing fiber produces a **complex spike** (~1 Hz) = a burst plus a dendritic Ca²⁺ spike, followed by a pause in simple spikes. Output is *inhibition* of DCN. *Test:* tonic firing in isolation; a CF input yields one all-or-none event and a pause of tens of ms; learning lowers the response to a PF pattern paired with CF.
- **Golgi cell.** Pacemaker ~5–10 Hz; gap-junction coupled; feedback (from PFs) and feedforward (from mossy fibers) inhibition of granule cells → sets the **time window** and sparseness of granule firing. *Test:* removing Golgi inhibition raises the fraction of active granule cells and degrades pattern separation.
- **Basket/stellate.** Spontaneous 5–30 Hz; feedforward inhibition from the same PFs that excite Purkinje (sharpens timing, sign inversion — lets a PF beam *reduce* Purkinje output); basket pinceau can silence Purkinje within ~1 ms.
- **UBC.** Turns one brief mossy-fiber input into a **prolonged burst** (hundreds of ms to seconds) via mGluR1 (ON type) or a prolonged pause (OFF type, mGluR2/3); continuum of delays (Huson & Regehr 2025). *Test:* brief input → long, delayed output; a population of UBCs spans a range of delays (a temporal basis set).
- **Lugaro.** Silent until serotonin; then inhibits Golgi cells (disinhibits granule layer).

---

## 2. Plasticity at parallel fiber → Purkinje synapses

| Rule | Condition | Mechanism | Conf. |
|---|---|---|---|
| **PF-LTD** | PF activity **paired with CF** (within ~50–200 ms; PF slightly before CF is optimal) | large dendritic Ca²⁺ + mGluR1 → PKCα → AMPA receptor endocytosis (Ito; Linden) | ● |
| **PF-LTP (postsynaptic)** | PF activity **without** CF | lower Ca²⁺ + NO → AMPA insertion; reverses LTD | ● |
| Presynaptic PF-LTP | high-frequency PF bursts | cAMP/PKA | ● |
| MLI → Purkinje rebound potentiation | CF-evoked depolarization | GABA_A upregulation | ◐ |
| Mossy fiber → DCN LTP; Purkinje → DCN LTD | Purkinje pause/rebound + MF input | consolidation site: memory moves from cortex to nuclei over time | ◐ |

Model as a **supervised error rule**: climbing fiber = teaching/error signal; weight change Δw ∝ −(PF eligibility trace × CF) + small potentiation for PF without CF. Most PF synapses are silent at steady state (≈ 85% in some estimates; Isope & Barbour 2002). *Test:* pairing a PF pattern with CF for N trials reduces Purkinje response to that pattern; the same pattern without CF slowly recovers.

---

## 3. Deep cerebellar nuclei (DCN)

Fastigial (medial), interposed (emboliform + globose) and dentate (lateral, ~5 M neurons in human, much expanded in primates).

| Type (marker) | Share | Species | Transmitter | Target | Conf. |
|---|---|---|---|---|---|
| **Large projection neurons** (non-GABAergic, "GADnL") | majority of DCN neurons (exact share not retrieved) | mouse | **glutamate** | **excitatory projection** → thalamus (VL → motor/prefrontal cortex), red nucleus, brainstem, vestibular nuclei; collaterals to granule layer as mossy fibers | ● type / ○ % |
| **Small nucleo-olivary neurons** (GAD+, "GAD+IO") | minority | mouse | GABA | **inhibitory projection → inferior olive** (closes the olivo-cerebellar loop, suppresses CF signals for learned errors) | ● |
| Local interneurons (GABA / glycine) | minority | mouse | GABA, glycine | local; some glycinergic cells project to cerebellar cortex Golgi cells | ◐ |
| Glycinergic projection neurons (fastigial) | small | mouse | glycine | → ipsilateral brainstem | ◐ |

Uusisaari & Knöpfel classify **six types**: four projection, two local. Proportions are not settled; treat projection glutamatergic cells as the main population.

**Signature behaviors.** Glutamatergic DCN cells fire spontaneously **~10–50 Hz**, are inhibited by convergent Purkinje input (~30–50 Purkinje cells per DCN neuron in rodents), and show **rebound bursting** after a release from inhibition (T-type Ca²⁺). *Test:* tonic firing; synchronous Purkinje pause → rebound burst.

---

## 4. Inferior olive and climbing fibers

| Item | Value | Species | Conf. |
|---|---|---|---|
| Cell type | one principal type, **glutamatergic**, electrically coupled by dendro-dendritic gap junctions (Cx36) in glomeruli | all | ● |
| Purkinje cells per olivary neuron (CF branches) | **~4–7 rat; ~14–17 human** | rat; human (secondary citation) | ◐ |
| CFs per adult Purkinje cell | **exactly 1** (after developmental pruning) | all | ● |
| Human total IO neuron number | not retrieved (stereology exists, e.g. Porzionato 2008; totals not found) | human | ○ |

**Signature behaviors.** Subthreshold oscillations **~3–10 Hz**; low firing **~1 Hz**; gap-junction synchrony across clusters (modulated by GABA from nucleo-olivary cells) → coherent complex spikes in parasagittal zones. Carries **error / unexpected sensory event** signals. *Test:* coupled IO cells phase-lock; DCN GABA input decouples them and reduces CF rate after learning.

---

## 5. Main connections

- **Inputs:** mossy fibers (pontine nuclei relaying cortex — the largest; spinal, vestibular, reticular, also DCN collaterals) → granule cells and UBCs, collaterals to DCN. Climbing fibers (IO) → Purkinje, collaterals to DCN. Modulatory: LC noradrenaline, raphe serotonin (Lugaro), cholinergic beaded fibers.
- **Within cortex:** MF → granule → PF → Purkinje + MLIs + Golgi; Golgi ⊣ granule; MLI ⊣ Purkinje; Purkinje ⊣ DCN.
- **Outputs:** DCN → thalamus → cortex (motor, prefrontal, parietal) ; → red nucleus; → brainstem; DCN ⊣ IO; flocculonodular Purkinje → vestibular nuclei directly.
- Modular **microzones**: an IO subnucleus → a parasagittal Purkinje strip (zebrin+ or zebrin−) → a DCN region → back to that IO subnucleus. Zebrin+ Purkinje cells fire faster (~90 Hz) than zebrin− (~60 Hz) in mice (Zhou et al. 2014).

---

## 6. Human vs rodent

- Human cerebellum holds ~80% of neurons (rodents: ~60–70%, similar scaling; the cerebellum:cortex neuron ratio ~4:1 is conserved across mammals, Herculano-Houzel 2010). ◐
- **Lateral cerebellum (crus I/II) and dentate nucleus are greatly expanded in primates/humans**, connected to prefrontal cortex: cognitive, not only motor.
- Purkinje number ~50× rat, density 13× lower; CF branching ~2–3× rat.
- Parallel fibers longer in humans (~6 mm vs ~2–3 mm), so each Purkinje cell samples more distant granule cells.

---

## 7. Sources

- Azevedo FAC et al. (2009) Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain. *J Comp Neurol* 513:532. Summary: https://bionumbers.hms.harvard.edu/bionumber.aspx?id=106322
- Andersen BB, Korbo L, Pakkenberg B (1992) A quantitative study of the human cerebellum with unbiased stereological techniques. *J Comp Neurol* 326:549. Notes: https://pseudocerebellum.readthedocs.io/en/latest/notes/AndersonBB+2-1992.html
- Andersen BB, Korbo L (1993) human vs rat Purkinje comparison; Agashiwala RM et al. (2008) Purkinje 19.7–27 M. https://mobile.jci.org/references/scholar/38903/B62
- Fractionator Purkinje count 15.4 M (1989). https://pubmed.ncbi.nlm.nih.gov/2613942/
- Korbo L et al. (1993) Total numbers of various cell types in rat cerebellar cortex. *Brain Res* 609:262. Compiled table: https://pseudocerebellum.readthedocs.io/en/latest/tables/cell_counts.html
- Palkovits M et al. (1971) cat cerebellar cell counts (basket 7.5 M, stellate 20.9 M, Purkinje 1.25 M), via the same compilation.
- Uusisaari M, Knöpfel T (2011/2012) Diversity of neuronal elements and circuitry in the cerebellar nuclei. *Cerebellum*. https://link.springer.com/article/10.1007/s12311-010-0240-3 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3215887
- Huson V, Regehr WG (2025) Mossy fiber input patterns to unipolar brush cells evoke a continuum of temporal responses. (PMID 39819796)
- Ito M (2001) Cerebellar long-term depression: characterization, signal transduction, and functional roles. *Physiol Rev* 81:1143.
- Isope P, Barbour B (2002) Properties of unitary granule cell→Purkinje cell synapses. *J Neurosci* 22:9668.
- Marr D (1969) A theory of cerebellar cortex. *J Physiol* 202:437; Albus JS (1971) *Math Biosci* 10:25.
- Zhou H et al. (2014) Cerebellar modules operate at different frequencies. *eLife* 3:e02536.
- Herculano-Houzel S (2010) Coordinated scaling of cortical and cerebellar numbers of neurons. *Front Neuroanat* 4:12.
- Climbing fiber branching (4–7 rat, 14–17 human; secondary citation of Rashed et al. 2005) and Porzionato et al. 2008 human IO stereology: https://www.research.unipd.it/handle/11577/2443160

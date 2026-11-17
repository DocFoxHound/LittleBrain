# 21: Brain regions, organized (roadmap step 5)

Every region of the model now has a **cell makeup with percentages**, taken from the research in
`notes/regions/*.md` (three research agents, sources in each file). Each number records the species it
comes from and a confidence mark: ● established · ◐ supported, varies · ○ uncertain or our estimate.

- Code: `prototype/brain/regions.py` (data + `allocate()` + `build()`)
- Full printout: `notes/regions/composition-report.txt`, or run `PYTHONPATH=prototype:prototype/brain python3 prototype/brain/regions.py`
- **89 cell types** in `prototype/brain/celltypes.py`, each naming its real human counterpart.
- **31 signature tests** pass: `test_cell_types.py` (12) and `test_cell_types_regions.py` (19).

## Regions and their makeup (summary; full tables in the report)

| Region | Subregions | Key proportions | Human neurons |
|---|---|---|---|
| **Neocortex** | sensory; motor/prefrontal; cingulate/insula | **67% excitatory / 33% inhibitory** (human; mouse 16%). Interneurons PV 40 : SST 30 : 5HT3aR 30 (VIP ≈12%). Layer 5 PT only ~0.5–0.7% of neurons in humans. **Von Economo neurons ~1%** in cingulate/insula. **Rosehip** cells in layer 1 | ~16 billion |
| **Thalamus** | first-order; higher-order; TRN | relay 75% / interneurons 25% (primate LGN); TRN 100% GABA. Driver synapses only ~2–7% of relay inputs; layer 6 provides 30–50% | — |
| **Claustrum** | — | ~88% projection neurons (estimate) | — |
| **Hippocampal formation** | entorhinal; dentate gyrus; CA3; CA2; CA1; subiculum | DG 95% granule cells (incl. 2.5% young, disputed in adult humans); CA1 89% pyramidal + 11% interneurons in **9 types** (ivy cells most numerous) | DG 15 M granule; CA3+CA2 2.7 M; CA1 16 M |
| **Amygdala** | basolateral; intercalated; central | BLA ~83% principal (positive/negative valence) / ~17% interneurons; central amygdala ~95% GABA (PKCδ "off", SST "on", CRF, output) | 12.2 M |
| **Basal ganglia** | striatum; GPe; GPi; STN; SNr; SNc | Striatum (human): **83% MSN / 17% interneurons** (rodent 5%), cholinergic 1.5%. GPe 73% prototypic / 27% arkypallidal. SNc 71% dopamine | striatum 82 M; STN ~0.5–1.1 M; SN ~550 k pigmented |
| **Cerebellum** | cortex; deep nuclei; inferior olive | **granule 99.2%**, Purkinje 0.03%, Golgi 0.08%, basket 0.18%, stellate 0.51% (≈3,300 granule cells per Purkinje cell) | **69 billion (80% of all brain neurons)** |
| **Neuromodulatory nuclei** | VTA; LC; dorsal raphe; median raphe; nucleus basalis; medial septum; PPN/LDT | VTA 64% dopamine / 34% GABA / 2.5% glutamate; LC ~95% noradrenergic; nucleus basalis ~90% cholinergic | LC ~50 k; DR 165 k serotonin; nbM ~400 k |
| **Hypothalamus** | arcuate; lateral; PVN; SCN | modeled populations only: AgRP/POMC; orexin/MCH (1:2); CRH/oxytocin/vasopressin; SCN clock (80% AVP / 20% VIP) | orexin ~70 k; SCN ~17 k |

## New cell behaviors (all tested)

| Cell | Signature behavior | Test result |
|---|---|---|
| Purkinje | tonic firing; parallel-fiber input paired with a climbing-fiber error → LTD | rest 0.80; trained response 1.00 → 0.72 |
| Cerebellar granule | expansion recoding (4 dendrites, one mossy fiber each) | overlap 0.71 → 0.36 |
| Deep cerebellar nuclei | rebound burst when Purkinje inhibition ends | 0.30 → 0.80 |
| Inferior olive | fires for errors; silenced when the error is learned (nucleo-olivary inhibition) | 0.00 / 1.00 / 0.00 |
| Medium spiny neuron | silent "down state" until strong convergent input | 0.00 vs 0.97 |
| D1 vs D2 MSN | a dopamine burst strengthens D1 and weakens D2 synapses | +1.00 vs −0.20 |
| GPi / SNr | tonic inhibition; pauses when the striatum fires (releases the thalamus) | 0.45 → 0.00 |
| STN | excitatory rebound burst | 0.23 → 0.80 |
| Cholinergic TAN | tonic, pauses on a salient event, rebounds | 0.50 → 0.03 → 0.57 |
| Chandelier | vetoes output at the axon initial segment | 0.68 → 0.22 |
| Neurogliaform / ivy | slow volume inhibition outlasts firing | 0.12 vs PV 0.00, 10+ steps later |
| CA1 pyramidal | **one-shot plateau learning (BTSP)**, even without firing | 0.240 vs L2/3 0.000 |
| Von Economo | fast relay | 1 step vs 3 for an L5 pyramid |
| Central amygdala | SST "on" cells disinhibit the fear output | 0.02 → 0.64 |
| Dopamine cell | baseline, burst, dip | 0.30 / 1.00 / 0.00 |
| Serotonin cell | slow (average reward) | 0.34 → 0.75 over 20 rewards |
| CRH (stress hormone) | slow rise; lingers long after the stressor | still above half 60 steps later |
| Septal pacemaker | theta-like rhythm | period 8.0 |
| SCN clock | light shifts it depending on timing | +0.15 rad at dawn vs +1.44 rad in the evening |

## Caveats
- **Human interneuron censuses mostly don't exist** (hippocampus, amygdala, thalamus). Rodent or primate
  shares are used and labeled.
- Cortical excitatory shares per layer are our arithmetic from a rat column count (○).
- The cerebellum's real ratio (~3,300 granule cells per Purkinje cell) can't be kept at small scale.
  `allocate()` guarantees at least one cell of every type, so rare types exist, but small models
  compress the granule expansion.
- Cells are rate models. Bursting, pauses and rebounds are approximations of spiking behavior.
- Regions aren't wired to each other yet. That's roadmap step 7 (federation). Only cortex + thalamus
  (note 19) is wired as a circuit so far.

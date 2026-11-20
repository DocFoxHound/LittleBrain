# Connection plan compared with the game brain

The [regional connection plan](regions/connections.md) describes more pathways than the default game brain implements. The comparisons below use the source, destination, direction and compartment of the actual connection. Some historical IDs in the Python descriptions have different meanings from the IDs in the plan. For example, the EC island cell is present, but it drives the wrong class of CA1 cell for HC-6.

These are implementation findings at runtime revision `b296613`. A represented route still needs a physiological or behavioral experiment to establish its function. [Three-seed direct response tests](56-regional-verification.md) · [Game specification](../prototype/brain/brainspec.py) · [Additional terminals](../prototype/brain/regional_circuits.py)

## Cortex and thalamus

| Plan | Game implementation and gap |
|---|---|
| CT-1 | Sensory relay drives cortical L4 and L6. The full set of direct cortical targets and the feedforward PV collateral are incomplete. |
| CT-2 | Pulvinar matrix drives sensory L2/3 through a pooled excitatory terminal. The diffuse, layer-1 tuft and interneuron targeting is absent. |
| CT-3, CT-4 | Sensory L6 drives both the relay and TRN. TRN inhibits the relay. Inhibition of all core and matrix populations and separate GABA receptor kinetics remain approximations. |
| CT-5 | The implemented transthalamic route runs B L5 PT → pulvinar matrix → A L2/3. The plan also calls for a driver from A to higher-order core relay to B. |
| CT-6 | A L5 PT reaches pons and striatum. Common branching of a single axon to the planned destinations is unmodeled. |
| CT-7 | GPi → VA and SNr → MD both inhibit their recipients after the repair. The additional external inhibitory nuclei are unmodeled. |
| CT-8 | PPN excites the sensory relay. LC, raphe and TRN branches of this proposed state-control circuit are incomplete. |

## Hippocampal formation

| Plan | Game implementation and gap |
|---|---|
| HC-1, HC-2 | Association cortex drives superficial EC. Perirhinal and parahippocampal subdivisions are combined in the cortical stand-in. Distinct insular, cingulate and orbitofrontal inputs to EC are absent. |
| HC-3, HC-4 | EC stellate drives dentate and CA3. Dentate interneurons receive granule feedback. The direct EC-driven inhibitory branch is absent. The modeled EC input to CA2 comes from layer 3. |
| HC-5 | EC L3 reaches the CA1 tuft and CA1 neurogliaform cells. Direct L3 projections to the two subiculum populations are absent. |
| HC-6 | **Target mismatch.** EC island cells directly excite CA1 pyramids. The plan specifies excitation of CA1 lacunosum-moleculare interneurons, producing feedforward inhibition. |
| HC-7 | Mature and young granules reach CA3 pyramids. Mature mossy-fiber weights can reverse sign after development. Direct dentate drive to CA3 PV basket cells is absent. [Matched sign intervention](55-hippocampal-transmission.md) |
| HC-8 | CA3 pyramids have basal and apical recurrent inputs with acetylcholine-dependent suppression. |
| HC-9 | CA3 reaches CA1 pyramids, PV and bistratified cells. CA1 ivy cells receive CA1 input in this model. |
| HC-10, HC-11 | CA3 → CA2 → CA1 is present. CA2 uses a cortical pyramidal learning rule. Resistance to synaptic potentiation and social recognition need separate tests. |
| HC-12 | CA3 reaches hilar mossy cells, which feed dentate granules. Laminar and longitudinal detail is reduced. |
| HC-13, HC-14 | CA1 reaches both subiculum types. Bursting subiculum drives explicit EC L5b. CA1 and regular subiculum also use the older combined EC L5 group. |
| HC-15 | EC L5b reaches L5a, which drives B L2/3. The older combined EC L5 group also feeds cortical apical synapses. The explicit L5a terminal is pooled and lacks the planned return to CA1. |
| HC-16, HC-17 | The Papez relay, regular-subiculum projections to the listed limbic destinations, and a direct CA1/subiculum → PFC pathway are absent. Hippocampal output can still reach cortex through EC. |
| HC-18 | The septal pacemaker inhibits CA1 PV cells. Its other planned hippocampal and entorhinal targets are absent. |
| HC-19, HC-20 | Acetylcholine is a supplied scalar. Septal cholinergic and glutamatergic projections are absent. CA1 drives the septal pacemaker directly in the reduced feedback path. |

## Amygdala

| Plan | Game implementation and gap |
|---|---|
| AM-1, AM-2 | Thalamic and cortical input reaches BLA principals. The full brain also learns visual cue input to separate positive and negative ensembles. Insula projects to negative BLA. |
| AM-3 | Signed external outcomes drive the BLA valence groups. A distinct parabrachial CGRP pathway into CeL PKCδ is absent. |
| AM-4 | Separate lateral-to-basal principal-cell populations are absent. |
| AM-5, AM-6 | Positive BLA recruits ITC, which inhibits CeM. Negative BLA recruits CeL SST. The other planned direct BLA → CeM and BLA → CeL PKCδ branches are absent. |
| AM-7, AM-8, AM-9, AM-10 | Lateral ITC → BLA, infralimbic → ITC, prelimbic → BLA, and the return projection to frontal cortex are absent. A generic PFC relay cannot establish extinction learning. |
| AM-11, AM-12, AM-13 | Dedicated reciprocal BLA–ventral hippocampal projections and direct positive BLA → accumbens are absent. BLA cue value can influence actions indirectly through dopamine. |
| AM-14, AM-15 | CeM reaches PAG, LC, nucleus basalis and PVN. CeL CRF reaches LC. Detailed autonomic outputs and the extended-amygdala relays are absent. |
| AM-16 | CeL SST inhibits PKCδ, which inhibits CeM. |

## Basal ganglia

| Plan | Game implementation and gap |
|---|---|
| BG-1, BG-2, BG-3 | IT/PT cortical input reaches both MSN populations and recruits FSI inhibition. The code shares afferent fractions between D1 and D2 rather than assigning the planned differential IT/PT preference. |
| BG-4, BG-5, BG-6 | Sensory thalamus drives TAN and MSNs. TAN drives FSI. VA activity also enters the reduced striatal computation. Distinct CM/Pf input, local dopamine-terminal modulation and the full TAN target set are absent. |
| BG-7, BG-8, BG-9 | D1 inhibits GPi and SNr. GPi inhibits VA and SNr inhibits MD. Both thalamic populations reach PFC. The model also includes SNr → PPN. Additional thalamic, tectal and motor cortical targets remain incomplete. |
| BG-10, BG-11, BG-12, BG-13 | The sequence D2 → GPe → STN → GPi and the L5 PT → STN hyperdirect branch are present. GPe also inhibits GPi, and arkypallidal output reaches D1. The return STN → GPe and several SNr/FSI branches are absent. |
| BG-14, BG-15, BG-16, BG-17 | Striosomes inhibit SNc, which supplies a local FSI input. VTA supplies the global phasic learning signal. Dorsal versus ventral release fields and full SNc target selectivity are unmodeled. See NM-1 and NM-2. |

## Cerebellum

| Plan | Game implementation and gap |
|---|---|
| CB-1, CB-2 | Cortical input now reaches the pontine projection ensemble. Pons drives granules, unipolar brush cells and the ordinary deep-nucleus synapse bank. At reference scale the specialized DCN projection cell replaces its ordinary output with a granule/Purkinje calculation, so the bank alone does not establish a functional direct mossy collateral. |
| CB-3, CB-4 | Golgi inhibition reaches granules. Granules reach Purkinje, basket, stellate and Golgi cells. |
| CB-5, CB-6 | Climbing-fiber excitation, basket inhibition and stellate inhibition reach timing-mode Purkinje cells after repair. Olive collaterals to deep nuclei and molecular-layer interneurons are absent. |
| CB-7, CB-8 | Purkinje inhibits deep projection and nucleo-olivary cells. Deep output reaches VA and MD, then frontal cortex. The model lacks the complete cortical return map and vestibular outputs. |
| CB-9, CB-10, CB-11 | Nucleo-olivary cells inhibit the olive. A reduced thalamostriatal pathway carries cerebellar timing activity. Red nucleus, the detailed intralaminar pathway and STN → pons are absent. |

## Neuromodulatory nuclei

| Plan | Game implementation and gap |
|---|---|
| NM-1, NM-2 | LHb → RMTg → VTA dopamine, local VTA GABA inhibition and striosome → SNc are represented. The timed VTA receives signed outcome and value-change signals. Many anatomical inputs, including direct STN → SNc and PPN → dopamine, are absent. |
| NM-3, NM-4, NM-5 | CeM, CeL CRF and PVN CRH reach LC. PFC drives dorsal raphe. CeM drives nucleus basalis. Orexin reaches median raphe, but the broader arousal network in the plan is incomplete. |
| NM-6 | The TMN histamine population is explicitly outside the modeled catalog. |
| NM-7, NM-8 | Global phasic dopamine is computed from VTA. LC excites negative BLA locally. Computed global arousal uses CeM firing directly. A cell-driven LC broadcast to the designated cortical and hippocampal targets is absent. |
| NM-9, NM-10, NM-11 | Dorsal raphe reaches BLA PV and cerebellar Lugaro, median raphe reaches CA1 PV, nucleus basalis drives cortical VIP, and PPN excites the sensory relay. Global waking acetylcholine is supplied by the agent. The complete terminal fields and receptor-specific effects remain unverified. |

## Hypothalamus

| Plan | Game implementation and gap |
|---|---|
| HY-1, HY-2, HY-3 | POMC excites PVN MC4R. The proposed AgRP → MC4R inhibitory terminal is absent. The current return projection is inhibitory MC4R → AgRP. The documented excitatory TRH/PACAP → AgRP feedback would require a distinct population. |
| HY-4, HY-5, HY-6 | AgRP disinhibits orexin through a net positive stand-in. Orexin drives median raphe and VTA glutamate. MCH adjusts acetylcholine during replay. Wider orexin and MCH projection fields are absent. Some game computations read the instantaneous AgRP/POMC firing rates directly. |
| HY-7, HY-8 | Separate LH GABA/glutamate reward loops are absent. SCN drives orexin directly. The SPZ–DMH relay and circadian scheduling of the agent's sleep are absent. |
| HY-9, HY-10, HY-11 | CeM drives PVN CRH. CRH reaches LC. Oxytocin reaches CeM and vasopressin reaches CA2. The pituitary–adrenal cortisol feedback loop and peripheral endocrine channels are absent. |

## Targeted checks of discrepancies

Each row reports the difference between a unit source pulse and a quiet source in the same developed brain. Only the named receiver advances. The additional tests are included in the regional audit's follow-up output.

| Pulse and readout | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| EC island → CA1 pyramidal membrane drive | +0.2554 | +0.2527 | +0.2589 |
| EC island → CA1 neurogliaform output | 0 | 0 | 0 |
| Pontine relay → deep cerebellar projection output, direct collateral | 0 | 0 | 0 |

The clearest target discrepancy is HC-6. The largest omitted learning loops include hippocampus-to-PFC context, bidirectional amygdala–hippocampus context, and frontal control of extinction. Adding any of these involves choosing a local recipient population and testing its contribution with a matched task.

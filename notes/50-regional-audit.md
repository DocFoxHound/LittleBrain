# Regional audit of the default game brain

The reference brain has 14,656 neurons in 120 populations. We tested all 45 functional territories, including the reticular nucleus separately from the sensory thalamus. This audit measures the CPU default at revision `bfc1c6a`. All three seeds were calibrated, developed, slept and woken before testing. Each trial starts from a copy of the same developed state.

## Which questions each test answers

* **Wiring.** We checked every destination and every sampled synapse against its declared source population and compartment, and checked that each pooled terminal uses a channel understood by its receiver. There are 127 sampled pathway declarations and 167 pooled declarations. The signed pulse experiment below goes one step further and reads the receiving cells.
* **Response.** The same 24 simulation steps were run with and without patterned light, movement, positive and negative outcomes, hunger, satiety and peptide-related inputs. A change greater than one millionth of the model's activity scale counted as a response. We also silenced every population in one territory at a time, held all external inputs fixed and checked which connected territories changed. A lesion effect can travel by an indirect path.
* **Direct effect.** The following predictions specify a source, a target and a sign independently of the wiring table. The source output is clamped to 1 or 0. Only the receiving population advances from the matched state. Numbers give the difference in its mean emitted activity on seeds 0 / 1 / 2. For VTA the readout is signed phasic dopamine. For learned, sparse populations the sign of the mean rate can differ from the sign of the incoming current.
* **Behavior.** A successful direct response establishes that particular connection under the tested conditions. Learning a task requires a separate behavioral comparison. Note 48 records those comparisons for the earlier full runtime. Many territories have no game that isolates their characteristic function, including CA2 social memory, ITC extinction, claustral coordination and circadian control of sleep.

## Direct responses by territory

| Territory | Function being checked | Source → receiver | Predicted sign | Measured change, seeds 0 / 1 / 2 |
|---|---|---|:---:|---:|
| TH | Relay sensory input to cortex | TH.TC → A.L4 | + | 0.6632 / 0.6612 / 0.6677 |
| TRN | Inhibit the sensory relay | TH.TRN → TH.TC | − | -0.7061 / -0.6216 / -0.6436 |
| A | Send sensory features up the cortical hierarchy | A.L23 → B.L23 | + | 0.4417 / 0.4685 / 0.4786 |
| B | Send association-cortical input into memory | B.L23 → EC.stellate | + | 0.619 / 0.6592 / 0.662 |
| PFC | Maintain a cue with MD | PFC.IT → MD.relay | + | 0.8 / 0.8 / 0.8 |
| ACC | Influence striatal action selection | ACC.IT → STR.D1 | + | 0.5 / 0.5 / 0.5 |
| INS | Forward internal-state signals to ACC | INS.IT → ACC.IT | + | 0.3229 / 0.4358 / 0.01875 |
| PUL | Relay cortical feedback | PUL.matrix → A.L23 | + | 0.3688 / 0.4618 / 0.4063 |
| MD | Support prefrontal maintenance | MD.relay → PFC.IT | + | 0.2039 / 0.216 / 0.1663 |
| VA | Return motor-loop signals to cortex | VA.relay → PFC.IT | + | 0.1553 / 0.2154 / 0.09688 |
| CLA | Recruit cortical inhibition | CLA.pyr → A.PV | + | 0.06792 / 0.1004 / 0.1089 |
| EC | Provide input to the dentate gyrus | EC.stellate → DG.gc | + | 0.04428 / 0.04335 / 0.04422 |
| DG | Drive a sparse CA3 ensemble | DG.gc → CA3.pyr | + | 0.00334 / -0.0007397 / 0.003959 |
| CA3 | Pass hippocampal activity to CA2 | CA3.pyr → CA2.pyr | + | 0.4998 / 0.4992 / 0.4994 |
| CA2 | Relay to CA1 | CA2.pyr → CA1.pyr | + | 0.001038 / 0.001931 / 0.0009313 |
| CA1 | Send hippocampal output to subiculum | CA1.pyr → SUB.pyr | + | 0.4988 / 0.4981 / 0.4945 |
| SUB | Return hippocampal output to deep EC | SUB.pyr → EC.L5b | + | 0.5 / 0.5 / 0.5 |
| BLA | Associate value and recruit the central amygdala | BLA.neg → CeL.SST | + | 0.6 / 0.5999 / 0.6 |
| ITC | Gate central-amygdala output | ITC.gate → CeM | − | -0.2436 / -0.06152 / -0.06055 |
| CeL | Gate the medial central amygdala | CeL.PKCD → CeM | − | -0.3605 / -0.3616 / -0.3594 |
| CeM | Control the defensive output relay | CeM → PAG.out | − | -0.05864 / -0.05864 / -0.05864 |
| STR | Release the brake on a selected action | STR.D1 → GPi | − | -0.4642 / -0.2065 / -0.2058 |
| GPe | Inhibit the subthalamic nucleus | GPe → STN.proj | − | -0.7976 / -0.7986 / -0.7982 |
| GPi | Inhibit the motor thalamus | GPi → VA.relay | − | 0 / 0 / 0 |
| STN | Increase the pallidal brake | STN.proj → GPi | + | 0.2409 / 0.2705 / 0.2333 |
| SNr | Inhibit thalamic output | SNr.out → MD.relay | − | 0 / 0 / 0 |
| SNc | Modulate striatal activity | SNc.DA → STR.FSI | + | 0 / 0.02608 / 0 |
| PONS | Relay mossy-fiber input to cerebellum | PONS.relay → CB.granule | + | 0.2263 / 0.1367 / 0.3048 |
| CB | Regulate deep nuclear output | CB.Purkinje → DCN.proj | − | -0.2544 / -0.2587 / -0.2609 |
| DCN | Send cerebellar output to motor thalamus | DCN.proj → VA.relay | + | 0.8 / 0.8 / 0.8 |
| IO | Trigger a Purkinje teaching response | IO.olive → CB.Purkinje | + | 0.7359 / 0.7265 / 0.7334 |
| VTA | Supply phasic dopamine | VTA.DA → phasic dopamine | + | 2 / 2 / 2 |
| LC | Modulate amygdala excitability | LC.NE → BLA.neg | + | 0.07561 / 0.1926 / 0.239 |
| DR | Modulate amygdala inhibition | DR.5HT → BLA.PV | + | 0 / 0 / 0 |
| MR | Modulate hippocampal inhibition | MR.5HT → CA1.PV | + | 0.05762 / 0.05622 / 0.04164 |
| NB | Recruit cortical VIP cells | NB.ACh → A.VIP | + | 0.1896 / 0.1475 / 0.146 |
| MS | Inhibit hippocampal interneurons rhythmically | MS.theta → CA1.PV | − | -0.06841 / -0.07413 / -0.04998 |
| PPN | Modulate the sensory thalamus | PPN.ACh → TH.TC | + | 0.566 / 0.5769 / 0.4865 |
| LHb | Recruit the negative-outcome pathway | LHb.proj → RMTg.GABA | + | 0.4102 / 0.4102 / 0.4102 |
| RMTg | Suppress dopamine-cell activity | RMTg.GABA → VTA.DA | − | -0.2764 / -0.2768 / -0.277 |
| PAG | Relay a defensive-state signal | PAG.out → LH.orexin | + | 0.00659 / 0.006747 / 0.006655 |
| HYP | Signal hunger to lateral hypothalamus | HYP.AgRP → LH.orexin | + | 0.0107 / 0.01083 / 0.01074 |
| LH | Provide state input to raphe | LH.orexin → MR.5HT | + | 0.035 / 0.035 / 0.035 |
| PVN | Provide peptide input to CA2 | PVN.vasopressin → CA2.pyr | + | 0.4745 / 0.476 / 0.4603 |
| SCN | Send clock output toward wake circuitry | SCN.clock → LH.orexin | + | 0.005907 / 0.00584 / 0.005872 |

## Matched interventions

| Seed | Direct predictions confirmed above threshold | Populations responding to the external sequence | Territories whose lesion changed a declared downstream readout |
|---:|---:|---:|---:|
| 0 | 41/45 | 116/120 | 45/45 |
| 1 | 41/45 | 115/120 | 45/45 |
| 2 | 41/45 | 115/120 | 44/45 |

Three populations stayed below the evoked-response threshold on all three seeds. They were the sensory chandelier cells, CA1 axo-axonic cells and association-cortical VIP cells. The claustral projection cells also stayed below threshold on seeds 0 and 1, PFC L6 IT on seeds 1 and 2, and PFC NP on seed 2. Direct stimulation recruited all six populations on every seed. The VIP population received the same acetylcholine level in the two original conditions. Raphe and SNc pulses also reached their targets when the recipient PV/FSI cells received additional excitatory input. The dentate pulse increased the mean CA3 membrane drive on every seed, even where competition reduced the mean output rate.

| Follow-up pulse or condition | Change, seeds 0 / 1 / 2 |
|---|---:|
| DR → amygdala PV, with BLA principal co-drive | 0.1 / 0.1 / 0.1 |
| SNc → striatal FSI, with cortical co-drive | 0.15 / 0.15 / 0.15 |
| Dentate → CA3, membrane drive before sparse competition | 0.6854 / 0.1399 / 0.5533 |
| PFC IT → claustral projection cells | 0.2175 / 0.2107 / 0.2376 |
| PFC IT → PFC L6 IT | 0.2129 / 0.2045 / 0.2460 |
| PFC IT → PFC NP | 0.1803 / 0.2003 / 0.2387 |
| Sensory L2/3 → chandelier cells | 0.25 / 0.25 / 0.25 |
| CA1 pyramids → axo-axonic cells | 0.25 / 0.25 / 0.25 |
| Acetylcholine → association-cortical VIP cells | 0.3418 / 0.3418 / 0.3418 |
| Cortical L5 PT → pontine relay | 0 / 0 / 0 |
| Deep cerebellar nuclei → MD | 0 / 0 / 0 |
| Olive → Purkinje with the timing ensemble active | 0 / 0 / 0 |
| Cerebellar basket → Purkinje with timing active | 0 / 0 / 0 |
| Cerebellar stellate → Purkinje with timing active | 0 / 0 / 0 |

The reference-scale MD, VA, pontine and Purkinje populations have one cell per tile. Their specialized computations in association.py can overwrite the ordinary cell response in every tile. The pulse tests isolate seven ineffective direct projections in these states, including both basal-ganglia brakes. The olive still supplies a separate learning signal during timing. A successful timing game can therefore coexist with a missing instantaneous complex spike. CT-7/BG-8 and CB-1/CB-5/CB-6 in the [connection plan](regions/connections.md) describe the biological expectations.

## Regional composition

The 36 catalog subregions specify 167 distinct local cell-type entries. At reference scale, 100 have an exact local match, leaving 67 missing across 22 subregions. This count merges maps assigned to a shared catalog entry (A and B sensory-like columns, ACC and insula, MD and pulvinar, CeL and CeM). B uses the sensory column as a proxy. A missing entry in either constituent map can therefore be hidden by this aggregate. Specialized runtime names count separately from catalog types. The five additional connective relays in the audit have no regional composition table. The audit JSON reports planned percentages and actual counts for reproducing proportion comparisons. Presence alone does not imply the planned ratio.

| Catalog subregion | Present / planned unique types | Missing locally |
|---|---:|---|
| sensory cortex (e.g. V1, S1) | 14/16 | L5/6 NP, L6 IT |
| motor / prefrontal cortex | 4/16 | CCK basket, L2/3 IT, L4 stellate, L5 PT, L6 CT, L6b, LAMP5 neurogliaform, PV chandelier, SST Martinotti, SST non-Martinotti, VIP, rosehip |
| anterior cingulate / frontoinsular cortex | 3/17 | CCK basket, L2/3 IT, L4 stellate, L5 PT, L5/6 NP, L6 CT, L6 IT, L6b, LAMP5 neurogliaform, PV chandelier, SST Martinotti, SST non-Martinotti, VIP, rosehip |
| first-order thalamus (LGN, VPM/VPL, MGBv) | 2/3 | TC matrix |
| higher-order thalamus (pulvinar, MD, POm) | 1/3 | TC core, thalamic interneuron |
| thalamic reticular nucleus | 1/1 | — |
| claustrum | 1/4 | PV basket, SST Martinotti, VIP |
| entorhinal cortex | 5/8 | LAMP5 neurogliaform, SST Martinotti, VIP |
| dentate gyrus | 5/6 | PV chandelier |
| CA3 | 2/9 | CA axo-axonic, CCK basket, O-LM, bistratified, hippocampal neurogliaform, interneuron-specific, ivy |
| CA2 | 2/2 | — |
| CA1 | 8/9 | CCK basket |
| subiculum | 3/5 | O-LM, hippocampal neurogliaform |
| basolateral amygdala (lateral + basal + accessory basal) | 3/7 | CCK basket, LAMP5 neurogliaform, SST Martinotti, interneuron-specific |
| intercalated cell masses | 1/1 | — |
| central amygdala | 4/5 | VTA glutamate |
| striatum (caudate, putamen, accumbens) | 8/8 | — |
| globus pallidus externa | 2/2 | — |
| globus pallidus interna | 1/1 | — |
| subthalamic nucleus | 1/1 | — |
| substantia nigra pars reticulata | 1/2 | dopamine |
| substantia nigra pars compacta | 1/2 | VTA GABA |
| cerebellar cortex | 7/7 | — |
| deep cerebellar nuclei | 2/3 | Golgi |
| inferior olive | 1/1 | — |
| ventral tegmental area | 3/3 | — |
| locus coeruleus | 1/2 | VTA GABA |
| dorsal raphe | 1/4 | VTA GABA, VTA glutamate, dopamine |
| median raphe | 1/3 | VTA GABA, VTA glutamate |
| nucleus basalis of Meynert | 1/2 | VTA GABA |
| medial septum / diagonal band | 1/3 | PPN/LDT ACh, VTA glutamate |
| pedunculopontine / laterodorsal tegmental | 1/3 | VTA GABA, VTA glutamate |
| arcuate nucleus (modeled populations) | 2/2 | — |
| lateral hypothalamus (modeled populations) | 2/2 | — |
| paraventricular nucleus (modeled populations) | 3/3 | — |
| suprachiasmatic nucleus | 1/1 | — |

## Reproduce

From the repository root with the project Python environment, run the following once for each seed. The JSON includes every measured upstream and downstream response in the assay and the per-population inventory.

```bash
for seed in 0 1 2; do
    OPENBLAS_NUM_THREADS=1 python prototype/brain/check_regional_audit.py --seed "$seed" > "/tmp/regional-$seed.json"
done
```

The audit also accepts `--load-brain` for an already developed, trusted reference-scale checkpoint. For the archived results, use the code and specification of this audit commit. Future repairs should rerun the implicated direct pulse, the matched lesion and the relevant task before updating the claim.

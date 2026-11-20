# Verification after the regional wiring repairs

Three default brains were built, calibrated, developed for two days, and woken on CPU at runtime revision `b296613`. Their seeds were 0, 1 and 2. All three contain 14,656 neurons in 120 populations. The implementation and the evidence for each repaired mechanism are in notes 49 and 51–54. The composition comparison is in [note 50](50-regional-audit.md). A [comparison of all 94 connection-plan entries](57-connection-plan-audit.md) identifies missing projections, substitute targets and limits of the reduced circuits.

## Network and cell checks

| Measure | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Sampled synapses | 1,547,264 | 1,547,072 | 1,547,200 |
| Correct pooled currents, pulse and quiet controls | 209/209 | 209/209 | 209/209 |
| Previously masked receiver responses restored | 7/7 | 7/7 | 7/7 |
| Populations responding to the 24-step stimulus versus quiet | 117/120 | 117/120 | 117/120 |
| Territories whose lesion changed a connected downstream readout | 45/45 | 45/45 | 45/45 |
| Representative direct predictions with the expected sign at the developed operating point | 43/45 | 44/45 | 43/45 |

All 36 original cell and learning checks passed. All three integrated runtime checks passed, including the eleven slow outputs, nine existing paired regional probes, scheduled sleep with partner changes, and the external game bridge checkpoint round trip. The three quiet populations all responded to their targeted pulse. These are activity and transmission checks.

## Direct effects by region

Each number is the mean receiving output with the source clamped to one minus the output with the source clamped to zero. Other sources are held fixed. An inhibitory projection should give a negative value. Note 50 defines the assay and its biological interpretation for each territory.

| Territory | Source → receiver | Expected | Seed 0 | Seed 1 | Seed 2 |
|---|---|:---:|---:|---:|---:|
| TH | TH.TC → A.L4 | + | 0.6732 | 0.6663 | 0.6752 |
| TRN | TH.TRN → TH.TC | − | -0.6435 | -0.6272 | -0.6272 |
| A | A.L23 → B.L23 | + | 0.4726 | 0.4362 | 0.3803 |
| B | B.L23 → EC.stellate | + | 0.6721 | 0.6774 | 0.682 |
| STR | STR.D1 → GPi | − | -0.213 | -0.2062 | -0.2228 |
| GPe | GPe → STN.proj | − | -0.7991 | -0.7989 | -0.7999 |
| GPi | GPi → VA.relay | − | -0.2387 | -0.2464 | -0.2479 |
| VTA | VTA.DA → phasic dopamine | + | 2 | 2 | 2 |
| BLA | BLA.neg → CeL.SST | + | 0.5991 | 0.5999 | 0.6 |
| CeL | CeL.PKCD → CeM | − | -0.361 | -0.3628 | -0.3577 |
| CeM | CeM → PAG.out | − | -0.05864 | -0.05864 | -0.05864 |
| HYP | HYP.AgRP → LH.orexin | + | 0.01091 | 0.01096 | 0.01101 |
| LH | LH.orexin → MR.5HT | + | 0.035 | 0.035 | 0.035 |
| PFC | PFC.IT → MD.relay | + | 0.5765 | 0.5731 | 0.5733 |
| MD | MD.relay → PFC.IT | + | 0.1659 | 0.2208 | 0.2639 |
| PONS | PONS.relay → CB.granule | + | 0.3129 | 0.213 | 0.04146 |
| CB | CB.Purkinje → DCN.proj | − | -0.233 | -0.242 | -0.2436 |
| IO | IO.olive → CB.Purkinje | + | 0.7324 | 0.7314 | 0.7213 |
| DCN | DCN.proj → VA.relay | + | 0.584 | 0.6049 | 0.5597 |
| EC | EC.stellate → DG.gc | + | 0.04424 | 0.04109 | 0.04185 |
| DG | DG.gc → CA3.pyr | + | -0.002422 | 0.0003547 | -0.001424 |
| CA3 | CA3.pyr → CA2.pyr | + | 0.4996 | 0.4991 | 0.499 |
| CA1 | CA1.pyr → SUB.pyr | + | 0.4972 | 0.4825 | 0.4983 |
| SUB | SUB.pyr → EC.L5b | + | 0.5 | 0.5 | 0.5 |
| CA2 | CA2.pyr → CA1.pyr | + | 0.002519 | 0.0008802 | 0.0003428 |
| INS | INS.IT → ACC.IT | + | 0.4428 | 0.3817 | 0.1461 |
| ACC | ACC.IT → STR.D1 | + | 0.5 | 0.5 | 0.5 |
| PUL | PUL.matrix → A.L23 | + | 0.346 | 0.4763 | 0.3958 |
| CLA | CLA.pyr → A.PV | + | 0.06227 | 0.1622 | 0.1249 |
| STN | STN.proj → GPi | + | 0.2703 | 0.2714 | 0.27 |
| SNr | SNr.out → MD.relay | − | -0.139 | -0.1345 | -0.1403 |
| VA | VA.relay → PFC.IT | + | 0.2223 | 0.225 | 0.2633 |
| ITC | ITC.gate → CeM | − | -0.07181 | -0.1204 | -0.06759 |
| PAG | PAG.out → LH.orexin | + | 0.00659 | 0.006747 | 0.006655 |
| LC | LC.NE → BLA.neg | + | 0.1762 | 0.2415 | 0.2236 |
| NB | NB.ACh → A.VIP | + | 0.168 | 0.1847 | 0.1716 |
| MS | MS.theta → CA1.PV | − | -0.07238 | -0.07921 | -0.07938 |
| DR | DR.5HT → BLA.PV | + | 0 | 0 | 0 |
| MR | MR.5HT → CA1.PV | + | 0.05979 | 0.06477 | 0.06022 |
| PPN | PPN.ACh → TH.TC | + | 0.6652 | 0.7451 | 0.5712 |
| SNc | SNc.DA → STR.FSI | + | 0.02354 | 0.15 | 0.06679 |
| LHb | LHb.proj → RMTg.GABA | + | 0.4102 | 0.4102 | 0.4102 |
| RMTg | RMTg.GABA → VTA.DA | − | -0.28 | -0.28 | -0.28 |
| PVN | PVN.vasopressin → CA2.pyr | + | 0.4763 | 0.4844 | 0.4647 |
| SCN | SCN.clock → LH.orexin | + | 0.006017 | 0.005908 | 0.006013 |

Dorsal raphe required background excitation in its BLA PV target. With that co-drive, the same pulse increased PV activity by 0.100 on each seed. Dentate stimulation reduced mean CA3 membrane drive on all three seeds. The separate [mossy-fiber intervention](55-hippocampal-transmission.md) traces that sign reversal to learned negative contact weights. A successful region lesion does not establish that each of its synapses has the intended biological sign.

## Reproduction

Use the commands in note 50 to develop or load each seed. Run `python prototype/brain/check_full_runtime.py --seed N --load-brain PATH` against the same checkpoint. The standard CPU environment for these measurements used NumPy and pygame-ce. The baseline and repaired audit can be compared using their respective commits.

# Regional placements in the expanded reference brain

The expansion preserves the 120 existing population counts and adds one 64-cell map per missing catalog entry. The comparison uses the 36 regional composition tables, including their shared cortical and thalamic entries. New composite entries are distributed between constituent maps. Within-region percentages at this resolution reflect the one-map minimum. Named historical stages and saved topologies remain available.

Each row in catalog_placements.py identifies an afferent, the new population, its target and the terminal. Tuft afferents are specified separately for new pyramidal types with apical dendrites. Existing types provide their cellular dynamics. Minority populations represented with a type from another region carry a local human description. The choices of individual recipient, gain, and single-source representation are approximations. The dorsal raphe dopamine projection uses the extended amygdala as its receiver (Matthews et al. 2016). [Cell composition and regional sources](regions/composition-report.txt) · [Connection research and references](regions/connections.md)

For each addition, three new brains took 12 visual-stimulation and settling steps. A unit source pulse was paired with a quiet source in a copy of the same state. The readout was receiver membrane drive for dendritic input or emitted activity for axonal vetoes and non-pyramidal receivers. A bias current could bring a receiver above threshold. The regional script reports that bias and each individual effect. Cutting only the new output terminal gave zero in every pair. Run `python prototype/brain/check_placements.py --region REGION --seed N` for seeds 0, 1 and 2. Omit the region for the complete panel. Pass a developed reference checkpoint with `--load-brain PATH` for post-development measurements.

The numeric columns give the smallest absolute afferent and efferent effects among the populations added at each stage, in seed order. Each individual change had its predicted sign.

| Catalog cohort | New populations | Reference neurons | Minimum afferent effect, 0 / 1 / 2 | Minimum efferent effect, 0 / 1 / 2 |
|---|---|---:|---:|---:|
| sensory cortex | A.NP, B.L6IT | 14,784 | 0.38523 / 0.39891 / 0.37576 | 0.15 / 0.15 / 0.15 |
| prefrontal cortex | PFC.L4, PFC.L23, PFC.L5PT, PFC.L6CT, PFC.L6b, PFC.CCK, PFC.chandelier, PFC.SST, PFC.SSTnonMartinotti, PFC.VIP, PFC.NGF, PFC.rosehip | 15,552 | 0.020509 / 0.020509 / 0.020509 | 0.003655 / 0.0011367 / 1.55e-05 |
| cingulate and insular cortex | INS.L4, ACC.L23, ACC.L5PT, INS.NP, ACC.L6CT, INS.L6IT, INS.L6b, ACC.CCK, INS.chandelier, ACC.SST, INS.SSTnonMartinotti, ACC.VIP, INS.NGF, ACC.rosehip | 16,448 | 0.020509 / 0.020509 / 0.020509 | 0.0002474 / 0.0002491 / 0.0002465 |
| first-order thalamus | TH.matrix | 16,512 | 3.4675 / 3.3282 / 3.2285 | 0.096335 / 0.093634 / 0.094114 |
| higher-order thalamus | PUL.core, PUL.interneuron | 16,640 | 0.41018 / 0.41018 / 0.41018 | 0.34987 / 0.34992 / 0.34982 |
| claustrum | CLA.PV, CLA.SST, CLA.VIP | 16,832 | 0.035343 / 0.035349 / 0.035355 | 0.0027748 / 0.0027748 / 0.0027748 |
| entorhinal cortex | EC.SST, EC.VIP, EC.NGF | 17,024 | 0.020509 / 0.020509 / 0.020509 | 0.0010568 / 0.0014989 / 0.0007085 |
| dentate gyrus | DG.chandelier | 17,088 | 0.25 / 0.25 / 0.25 | 3.6e-06 / 3.6e-06 / 3.4e-06 |
| CA3 | CA3.axoaxonic, CA3.bistratified, CA3.OLM, CA3.CCK, CA3.ivy, CA3.NGF, CA3.IS | 17,536 | 0.020509 / 0.020509 / 0.020509 | 1.03e-05 / 1.66e-05 / 1.39e-05 |
| CA1 | CA1.CCK | 17,600 | 0.27345 / 0.27345 / 0.27345 | 0.31691 / 0.31532 / 0.31649 |
| subiculum | SUB.OLM, SUB.NGF | 17,728 | 0.020509 / 0.020509 / 0.020509 | 0.030453 / 0.030936 / 0.031859 |
| basolateral amygdala | BLA.CCK, BLA.SST, BLA.IS, BLA.NGF | 17,984 | 0.020509 / 0.020509 / 0.020509 | 0.0027721 / 0.0027721 / 0.0027721 |
| central amygdala | CeL.Glu | 18,048 | 7.6964 / 7.2284 / 6.9078 | 0.094631 / 0.09463 / 0.094632 |
| nigra pars reticulata | SNr.DA | 18,112 | 0.7 / 0.7 / 0.7 | 0.19286 / 0.19285 / 0.19287 |
| nigra pars compacta | SNc.GABA | 18,176 | 0.41018 / 0.41018 / 0.41018 | 0.175 / 0.175 / 0.175 |
| deep cerebellar nuclei | DCN.local | 18,240 | 0.34182 / 0.34182 / 0.34182 | 0.0043169 / 0.0043121 / 0.0043185 |

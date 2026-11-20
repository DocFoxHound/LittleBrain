# Expanded reference brain after development and sleep

The default game brain now contains **18,944 neurons**, **187 populations** and **98 distinct modeled cell types**. All **167 catalog subregion/type entries** across **36 subregions** have an exact local match. This counts shared catalog entries consistently with note 50. The full brain has 45 functional territories including connective relays outside those composition tables.

Each of the 67 new populations responded to its principal input with increased emitted activity on each of three independently developed brains. All 201 tests also found a signed change in the receiving population's emitted activity. Muting only the added output terminal abolished the measured change. In 64, 62 and 62 cases respectively, the receiver showed this effect at its developed operating point. The other receivers needed a matched background drive of 0.8 model units. The paired measurements cover each afferent, including specified tuft input. [Individual measurements and controls](expanded-placement-probes.csv) · [Incremental region checks](65-regional-placements.md)

| CPU seed | Neurons | Sampled synapses | Resident NumPy array storage | Compressed checkpoint |
|---:|---:|---:|---:|---:|
| 0 | 18,944 | 1,821,824 | 65.65 MiB | 17.90 MiB |
| 1 | 18,944 | 1,822,336 | 65.66 MiB | 17.89 MiB |
| 2 | 18,944 | 1,821,056 | 65.62 MiB | 17.88 MiB |

The array census counts distinct backing arrays reachable from the loaded CPU brain. Total process memory also includes Python, temporary arrays and the allocator. GPU memory was unmeasured. Sampled counts exclude pooled projections and external receptors. The one-map minimum preserves rare populations and gives them relatively large shares at small scale. For example CA3 contains 256 pyramids out of 768 cells (33.3%) against an 89% catalog target. The 14,656-neuron model had 256 pyramids out of 320 cells. [Scaling rules](../docs/scaling.md)

## Regional function

Silencing each of the 45 territories changed a downstream readout on every seed. Of the 45 prespecified directional output probes, 43, 42 and 44 exceeded the expected-sign threshold at the resting operating point. Dorsal raphe needed background input to its target on all three seeds. The seed-1 nigral dopamine target also needed background input. In seed 1 the locus coeruleus pulse raised amygdala membrane drive by 0.254 but barely changed firing at rest. A matched unit background current revealed a firing increase of 0.0361. In seed 0 dentate stimulation slightly reduced the mean CA3 firing rate after sparse competition, despite raising membrane drive. These distinctions matter when interpreting a quiet population.

| Paired readout, source on minus off | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Dentate granule input, CA3 membrane | +0.1276 | +0.7549 | +0.2825 |
| Entorhinal island input through CA1 neurogliaform cells, pyramidal membrane | −0.000352 | −0.000348 | −0.000165 |
| Direct pontine collateral, ordinary deep nuclear output | +0.1438 | +0.1509 | +0.1549 |
| Direct pontine collateral, deep nuclear output during timing | +0.1620 | +0.1679 | +0.1727 |
| Sensory thalamic input, striosome membrane | +0.5100 | +0.3576 | +0.4097 |

The two repaired glutamatergic projections had zero negative transmitted contacts. Other sampled glutamatergic pathways still use signed weights. The wider audit found 35,016, 32,631 and 33,416 negative contacts among roughly 1.78 million positive-gain glutamatergic contacts. The current measurements establish specific pathway effects. They leave open the biological interpretation of signed contacts elsewhere.

All 36 cell and learning checks and all three integrated runtime checks passed. Scheduled sleep changed 44,595, 44,722 and 44,747 sampled partners respectively. The external game bridge, action controls, learned checkpoint round trip and frontal retention checks passed. The viewer mapped all 187 populations and rendered a Forage turn. A saved 14,656-neuron checkpoint also resumed play at its original size.

## Reproduction

For each seed N in 0, 1 and 2, run the following from the repository root using the project Python and CPU backend.

```sh
export OPENBLAS_NUM_THREADS=1
python prototype/brain/check_full_games.py --seed "$N" --prepare-only "$PATH_TO_BRAIN"
python prototype/brain/check_placements.py --seed "$N" --load-brain "$PATH_TO_BRAIN"
python prototype/brain/check_full_runtime.py --seed "$N" --load-brain "$PATH_TO_BRAIN"
python prototype/brain/check_regional_audit.py --seed "$N" --load-brain "$PATH_TO_BRAIN"
```

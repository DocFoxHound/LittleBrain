# Direct pontine input to deep cerebellar output

Pontine mossy fibers send excitatory collaterals into the deep cerebellar nuclei. The original full game brain computed their current, then replaced the projection cell's output with a granule/Purkinje calculation. Blending in the ordinary spike output failed on all three developed brains. A pontine pulse changed membrane drive while the ordinary spike output remained at zero. For example, on seed 0 the maximal pulsed membrane drive was 2.72 and the lowest spike threshold was 7.87 in the dentate-repaired checkpoint. [Cerebellar connection plan, CB-2](regions/connections.md)

The reduced ensemble now also receives the positive part of ordinary somatic drive above the cell's tonic baseline, after its synaptic weights and inhibition are applied. For an ensemble rate of r and net drive d, its output becomes (r + d) / (1 + d). This bounded rate approximation uses the existing sampled synapses in both ordinary and timing states.

Three default CPU brains developed from scratch with this repair. Each value is a pulse minus a quiet control in an identical copy of the brain. Only the deep nuclear receiver advances. The contact lesion mutes only pontine synaptic gains in that receiver.

| Seed | Pontine response, ordinary | Pontine response, timing | Pontine contacts muted, ordinary / timing | Purkinje response, ordinary / timing |
|---:|---:|---:|---:|---:|
| 0 | +0.1565 | +0.1681 | 0 / 0 | −0.2810 / −0.4452 |
| 1 | +0.1529 | +0.1663 | 0 / 0 | −0.2632 / −0.4264 |
| 2 | +0.1514 | +0.1736 | 0 / 0 | −0.2745 / −0.4315 |

The previous projection output gave zero pontine response in either state on all three seeds. In the same saved brains opened with the repair, the timing responses were +0.1615, +0.1601 and +0.1734. The old checkpoints used here had completed development with the entorhinal repair.

Prepare each seed using `check_full_games.py --seed N --prepare-only PATH`. Run `check_full_runtime.py --seed N --load-brain PATH` to exercise both modes and their contact lesions. The regional audit also reports both pontine pulse measurements. These tests measure transmission directly.

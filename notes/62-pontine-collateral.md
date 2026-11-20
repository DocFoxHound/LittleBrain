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

## Timing behavior over 300 turns

For each seed, both conditions started with the same saved brain developed at `6123d25`. Only the deep nuclear readout differed during the game. The control used that revision's readout, and the active condition used the repaired one at `3d087f8`. Both conditions learned. Every 50 turns the game reported mean normalized error among episodes completed in the preceding 100 turns, or 50 at the first measurement. Perfect timing is zero. Uniform random action selection averages 0.745.

| Seed | Readout | 50 | 100 | 150 | 200 | 250 | 300 | Episodes in final window |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Prior | 0.267 | 0.161 | 0.101 | 0.350 | 0.383 | 0.303 | 22 |
| 0 | Repaired | 0.267 | 0.161 | 0.028 | 0.238 | 0.367 | 0.286 | 21 |
| 1 | Prior | 0.191 | 0.123 | 0.174 | 0.317 | 0.300 | 0.300 | 20 |
| 1 | Repaired | 0.485 | 0.236 | 0.026 | 0.153 | 0.349 | 0.270 | 21 |
| 2 | Prior | 0.191 | 0.115 | 0.101 | 0.317 | 0.383 | 0.286 | 21 |
| 2 | Repaired | 0.222 | 0.136 | 0.013 | 0.130 | 0.350 | 0.350 | 20 |

The repaired circuit had lower final timing error on seeds 0 and 1 and higher error on seed 2. All three repaired brains were more precise halfway through training than at the end. The direct transmission repair therefore leaves late timing drift open. Reproduce the comparison by preparing each checkpoint at the control revision, then running `check_full_games.py --game timing --seed N --load-brain PATH --condition learning --turns 300` at each revision.

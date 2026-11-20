# Keep dentate mossy fibers excitatory

Dentate granule cells release glutamate onto CA3 pyramidal cells. The model allowed synaptic depression to reverse that current. Their sampled basal contacts now have a zero lower bound at transmission. Learning, consolidation and the choice of new partners still use the original state. Both synapse storage formats use the same bound. Loading a trusted old checkpoint identifies the contacts by their saved source indices. [HC-7, Henze et al. 2002](regions/connections.md)

The measurement is the change in CA3 membrane drive from a dentate pulse relative to silence. Each pair below uses identical cells, contacts and learned state. The signed control temporarily removes the transmission floor.

| Seed | Old checkpoint, signed / repaired | Developed from scratch with repair, signed / repaired |
|---:|---:|---:|
| 0 | −0.171 / +0.297 | +0.290 / +0.727 |
| 1 | −0.251 / +0.153 | −1.149 / +0.098 |
| 2 | −0.304 / +0.188 | −0.579 / +0.204 |

There were zero negative transmitted weights among 2,560 dentate contacts on every seed in both sets. This repair is limited to HC-7. Note 55 identified signed contacts elsewhere whose net physiological effects require individual testing.

In a separate one-presentation recall experiment, four images were shown and their half-image cues were tested in shuffled order. The table reports identification accuracy in CA3 for a trained brain and a copy that slept without seeing the patterns. Chance was 25%. These comparisons started from the old checkpoints. The sign repair did not establish a recall improvement and the trained score fell on seeds 0 and 2.

| Seed | Signed, trained / untrained | Repaired, trained / untrained |
|---:|---:|---:|
| 0 | 50% / 50% | 25% / 50% |
| 1 | 0% / 0% | 0% / 25% |
| 2 | 75% / 0% | 25% / 0% |

For each seed, prepare with `check_full_games.py --seed N --prepare-only PATH`, then run `check_regional_audit.py --seed N --load-brain PATH` and inspect `glutamate`. The integration check in `check_full_runtime.py` asserts the conductance and membrane response. All three integrated runs and all 36 cell and learning checks passed on CPU. For the recall comparison use `diagnostics.memory_vs_control(brain, seed=N, epochs=1, n=4, group="CA3.pyr", wake_steps=60)` on a separate load of each condition. Set `brain.g["CA3.pyr"].basal.conductance_floor = None` for the signed control.

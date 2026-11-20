# Slow regional outputs

Eleven populations in the default game brain compute a filtered output for slow inhibition or peptide release. Their cell models already returned that value. The federated simulation discarded it and published the immediate firing rate to downstream pathways. This affected populations including cortical neurogliaform cells, CA1 ivy cells and hypothalamic AgRP cells. The simulator now publishes the cell model's returned signal and keeps the firing rate available for recording. Silencing the population zeros its output.

In a newly built reference brain, a one-step hunger input of 1 gave the following AgRP values. The original runtime serves as the matched control. The largest outgoing error over all eleven populations fell from 0.08 to zero on each seed.

| Seed | Firing rate | Expected slow output | Published before | Published after |
|---:|---:|---:|---:|---:|
| 0 | 0.013447 | 0.000269 | 0.013447 | 0.000269 |
| 1 | 0.013447 | 0.000269 | 0.013447 | 0.000269 |
| 2 | 0.013447 | 0.000269 | 0.013447 | 0.000269 |

All 36 cell and learning checks passed. The full runtime check exercised all eleven terminals on previously developed checkpoints for seeds 0, 1 and 2, including the silenced control, nine regional probes, scheduled sleep and the game bridge. These measurements establish the time course of the signal transmitted through the network. They do not measure a change in task learning.

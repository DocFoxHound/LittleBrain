# Basal ganglia inhibition of thalamic relay cells

Both GPi and SNr already sent inhibition to thalamus. The specialized VA motor relay copied deep cerebellar activity and the specialized MD relay copied frontal activity after the conventional cell update. At reference scale these ensembles use every cell in the two relays, so neither population responded to the incoming brake. This conflicts with BG-8 and CT-7 in the [connection plan](regions/connections.md).

The relay now divides the specialized excitatory response by one plus its incoming inhibitory drive, before the rate update. This is a reduced inhibitory transfer function. Other cells retain the conventional calculation. Each isolated probe kept excitatory drive at 0.5, clamped the upstream brake to zero or one, and read the actual receiving rate after one update.

| Receiver | Without brake, seeds 0 / 1 / 2 | With brake before fix | With brake after fix |
|---|---|---|---|
| VA, GPi input | 0.400 / 0.400 / 0.400 | 0.400 / 0.400 / 0.400 | 0.200 / 0.200 / 0.200 |
| MD, SNr input | 0.400 / 0.400 / 0.400 | 0.400 / 0.400 / 0.400 | 0.286 / 0.286 / 0.286 |

The changed inhibitory gain affected a working-memory signal. In a seed-0 frozen probe, the Euclidean distance between the two frontal cue representations at choice fell from 0.645 with the nigral terminal blocked to 0.012 with inhibition active. The learned-choice comparison is reported separately because distance alone cannot establish whether the brain chose correctly.

At turn 150 the delayed-choice game scored 100% / 100% / 100% with both brakes active and 100% / 100% / 100% with only those two terminals silenced (seeds 0 / 1 / 2, 25 scored choices per seed in the rolling window, chance 50%). Both conditions learned from the same developed checkpoints. This intermediate comparison tests whether the reduced signal could still support a choice.

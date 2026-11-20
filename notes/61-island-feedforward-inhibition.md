# Entorhinal island cells recruit CA1 feedforward inhibition

The HC-6 terminal now excites a CA1 interneuron in stratum lacunosum-moleculare. The existing CA1 neurogliaform population stands in for the recipient interneurons in the reduced circuit. Its filtered inhibitory output reaches CA1 pyramidal cells. Kitamura et al. (2014) describe the island-cell feedforward pathway in mouse. The choice of this single interneuron type is a modeling assumption. [Hippocampal references](regions/hippocampus.md)

Three default CPU brains developed with the repair. Each measurement compares a unit island-cell pulse with silence in the same developed state. The last column restores the old direct excitatory bypass on a disposable copy. For the feedforward measurement only the interneuron and then the pyramid advance, in the order used by the runtime.

| Seed | Interneuron output change | Pyramid membrane change via interneuron | Island terminal silenced | Old direct bypass to pyramid |
|---:|---:|---:|---:|---:|
| 0 | +0.00348 | −0.000473 | 0 | +0.2561 |
| 1 | +0.00365 | −0.000507 | 0 | +0.2528 |
| 2 | +0.00470 | −0.000640 | 0 | +0.2584 |

Holding interneuron output fixed gave zero direct pyramid response on every seed. The effect through the slow inhibitory terminal is small on the first step. This experiment tests the route and its direction. Temporal-association learning would require a behavior that depends on this pathway.

The full reference population order and counts are preserved. New partial stages containing island cells include their target. Loading an older full checkpoint repairs its pooled terminal without resampling any synapses. Archived partial checkpoints lacking the receiver retain their saved circuit.

Run `check_full_games.py --seed N --prepare-only PATH`, then `check_regional_audit.py --seed N --load-brain PATH` and inspect the three EC island entries in `followup`. Repeat with N equal to 0, 1 and 2. Adding `EC.island -> CA1.NGF` to the brain's ablation set silences the terminal. The integrated runtime passed for each seed.

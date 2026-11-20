# Admit cerebellar input to the mediodorsal relay

The game specification supplies deep cerebellar excitation to the mediodorsal relay. The reduced relay previously copied frontal activity, discarding the cerebellar contribution. The projection is a reduced representation of cerebellar output returning to frontal cortex through thalamus (see CB-8 in the [connection plan](regions/connections.md)).

During cue acquisition the relay combines frontal activity with the incoming excitatory current, bounding the response by dividing their sum by one plus the incoming current. The established memory-hold gate protects a maintained cue from new external input, and the reset still clears the ensemble. The normally computed nigral inhibition applies after this excitatory step. The transfer function and the distinction between acquisition and maintenance are modeling choices.

| Direct pulse, difference from quiet | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| DCN to MD before fix | 0 | 0 | 0 |
| DCN to MD during acquisition after fix | 0.02272 | 0.01769 | 0.02176 |
| DCN to MD during protected maintenance after fix | 0 | 0 | 0 |
| SNr inhibition of MD after fix | −0.2024 | −0.2070 | −0.2021 |

These paired pulses began from the same three developed reference checkpoints used for the regional audit. They establish the response of the receiving cells in both states.

# Restore cortical input to the pontine timing relay

At reference scale the timing relay has one pontine neuron per tile. The normal cell computed its corticopontine input from layer 5 PT neurons, then the timing code replaced the result with visual thalamic activity. A pulse from cortex had zero effect on pontine firing on every seed.

The specialized ensemble now retains 15% of the ordinary cell response and 85% of the sustained visual drive. These fractions reuse the existing sensory-relay approximation. They are modeling coefficients. The biological route is cortex to pons to mossy fibers (CB-1 and CB-2 in the [connection plan](regions/connections.md)).

| Direct pulse, difference from quiet | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Cortex to pons before fix | 0 | 0 | 0 |
| Cortex to pons after fix | 0.0980 | 0.0805 | 0.1003 |
| Pons to granule after fix | 0.2263 | 0.1367 | 0.3048 |

The matched pulse advances the receiving population once, with all other sources fixed. It demonstrates that the sensory timing ensemble can now carry cortical input.

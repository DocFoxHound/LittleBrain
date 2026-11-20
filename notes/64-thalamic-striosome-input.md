# Restore excitatory thalamic input to striosomes

The full brain uses a thalamic relay as a reduced source of glutamatergic striosome input. Pulsing it lowered striosome membrane drive on three developed brains. Muting just its negative contacts restored a small positive response. The first attempted fix bounded transmission at zero. During fresh development, every thalamic contact could still become silent, and all three fresh brains failed the transmission test. Cortical plasticity could depress the entire thalamic input field through the shared heterosynaptic competition rule. [Thalamostriatal anatomy, BG-4 and BG-5](regions/connections.md)

Thalamic and cortical contacts on each striosome cell now compete within their own input class, and thalamic transmission has a zero lower bound. Independent competition budgets are a modeling assumption. With only cortical eligibility in a controlled update, the old rule decreased mean thalamic fast weight by 0.012 on seed 0. Within-class competition gave zero change to the idle thalamic class.

| Seed | Old checkpoint, signed membrane response | Same state, negative contacts muted | Freshly developed with both repairs, membrane / emitted response | Same fresh brain, thalamic contacts muted |
|---:|---:|---:|---:|---:|
| 0 | −0.2752 | +0.00161 | +0.7864 / +0.4984 | 0 / 0 |
| 1 | −0.5069 | +0.00043 | +0.5433 / +0.4231 | 0 / 0 |
| 2 | −0.3590 | +0.00254 | +0.6373 / +0.4669 | 0 / 0 |

A 50-turn signal game started from the same pre-repair checkpoint in every condition. Chance accuracy was 25%. The mean phasic dopamine values are from the repaired learning condition.

| Seed | Prior accuracy | Zero floor only | Both repairs | Learning off | Dopamine on miss / reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 78% | 88% | 86% | 32% | −0.492 / +0.478 |
| 1 | 88% | 88% | 94% | 22% | −0.181 / +0.551 |
| 2 | 88% | 90% | 92% | 34% | −0.264 / +0.494 |

For the game comparison, prepare checkpoints at revision 3d087f8, then run `check_full_games.py --game signal --turns 50 --seed N --load-brain PATH` at each revision. Use `--condition off` for the learning control. To isolate the transmission-only candidate on the new code, set the striosome basal bank's `thalamic_afferents` to None. For fresh development use `check_full_games.py --prepare-only PATH --seed N`, then run the regional audit and full runtime checks with that checkpoint. All three reference-scale integrated checks passed. Silencing each of the 45 territories affected a downstream target on every seed. Signed contacts remain on other learned projections.

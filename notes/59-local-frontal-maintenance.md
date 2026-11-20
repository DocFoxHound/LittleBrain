# Restore local recurrent support for a held frontal cue

The reduced frontal holding ensemble now combines its own previous activity with MD relay activity during passive viewing. It uses the existing prefrontal basal afferent fractions, 0.84 for recurrent PFC and 0.08 for MD, normalized over those two remaining inputs. The 0.999 per-step decay and the explicit gate for clearing a cue are preserved. The relative gains are a simulation assumption. The previous update copied MD alone on each held step. With the restored nigral brake, this repeatedly attenuated the only retained representation. [Matched diagnosis](58-frontal-cue-retention.md)

## Paired cue separation at choice

The 64 frontal projection cells saw the left or right cue with identical starting states, visual delays and passive motor copies. The following values are Euclidean distances between their two activity patterns at choice. The MD lesion was applied to a copy of the same developed brain.

| Seed | Before repair | Local recurrence active | Local recurrence with MD silenced |
|---:|---:|---:|---:|
| 0 | 0.00816 | 0.44259 | 0.00611 |
| 1 | 0.00939 | 0.49105 | 0.00659 |
| 2 | 0.00756 | 0.44623 | 0.00599 |

## Learning and matched controls

Each seed trained on the delayed game for 300 turns from the same developed checkpoint as note 56. The final window contains 25 two-way choices, with chance accuracy 50%. The training-time PFC/MD lesion bypasses the changed holding calculation in both versions. Its 300-turn control is reused from note 56. The frozen probe independently silences the loop only after training and measures both cues from matched starting states.

| Seed | Final correct before repair | Final correct after repair | PFC/MD silenced throughout training | Correct probability for left / right after repair | Same trained brain, acute PFC/MD lesion |
|---:|---:|---:|---:|---:|---:|
| 0 | 80% | 92% | 84% | 100.00% / 100.00% | 100.00% / 0.01% |
| 1 | 88% | 96% | 68% | 100.00% / 100.00% | 0.00% / 100.00% |
| 2 | 92% | 92% | 100% | 100.00% / 99.99% | 100.00% / 0.00% |

All three trained brains distinguished both cues in the frozen probe. Silencing the same trained loop produced a strong one-sided preference. The seed-2 brain trained with the lesion reached 100%, so the intact circuit did not outperform compensating pathways on every seed.

Choice accuracy was sampled after 50, 100, 150, 200, 250 and 300 turns.

| Seed | Accuracy by block |
|---:|---|
| 0 | 0.917 / 0.960 / 1.000 / 1.000 / 1.000 / 0.920 |
| 1 | 0.833 / 0.920 / 1.000 / 1.000 / 0.960 / 0.960 |
| 2 | 0.917 / 0.920 / 0.920 / 0.960 / 0.960 / 0.920 |

The change applies only when an unapplied action has put the agent in its held-context state. Standard development and the other four games never set that state. A paired two-turn smoke run in each of those games on each seed produced identical actions, probabilities, dopamine, final population activity and sensory-striatal weights across revisions. The full 300-turn comparisons for those games are in [note 56](56-regional-verification.md).

Reproduce the direct assay with `check_full_games.py --probe-held-cue` and the game with `check_full_games.py --game delayed --condition learning --turns 300`, supplying the reference seed and developed checkpoint to both. The integrated runtime check also exercises both frontal inputs and the clear gate.

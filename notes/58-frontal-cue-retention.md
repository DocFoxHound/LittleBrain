# Holding a cue across two blank turns

The delayed-choice task exposed a loss of cue information after restoring inhibition to mediodorsal thalamus. In the reference-scale game brain the first frontal projection cell on each tile maintained its activity by copying MD at every step. MD copied frontal activity back through the nigral brake. This made the same inhibitory gate attenuate a remembered cue repeatedly during a blank delay. The separate sampled local recurrent PFC pathway reached the ordinary cells, while the dedicated holding ensemble overwrote its computed result.

To isolate the effect, each developed brain saw both possible cues from identical states. The two passive movements and their motor copies were fixed across trials. The table shows the Euclidean distance between the 64 frontal projection rates for left and right cues. Larger distances indicate a more distinguishable cue. These distances precede game training.

| Seed | Intact after seeing cue | Intact at choice | At choice with SNr → MD blocked | At choice with DCN → MD blocked | At choice with MD silenced |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.71844 | 0.00816 | 0.43521 | 0.01158 | 0.00000 |
| 1 | 0.78907 | 0.00939 | 0.44066 | 0.01450 | 0.00000 |
| 2 | 0.70740 | 0.00756 | 0.40010 | 0.01248 | 0.00000 |

The acute SNr terminal intervention preserved much more cue separation on all three seeds. Removing the cerebellar input to MD had a smaller effect. Completely silencing MD eliminated the surviving cue difference at choice. The trained frozen-policy and behavior measurements are in [note 56](56-regional-verification.md).

Reproduce the four-condition probe with `python prototype/brain/check_full_games.py --probe-held-cue --seed N --load-brain PATH` on each developed reference checkpoint. The output gives separation after viewing the cue, both intervening blanks, and the final choice image.

## Training intervention on the failed seed

An exploratory seed-1 run blocked each terminal separately throughout 300 turns of training, starting from the same checkpoint as the original failure. The intact and single-terminal comparisons below use the version before the local recurrence repair. Chance choice accuracy is 50%.

| Training condition | Correct in final 25 choices | Frozen probability correct for left / right |
|---|---:|---:|
| Both terminals active | 88% | 0% / 100% |
| SNr → MD blocked | 92% | 100% / 100% |
| DCN → MD blocked | 100% | 100% / 2.17% |

Blocking the nigral terminal restored both cue responses on this seed. Blocking the cerebellar terminal improved the training-window score while leaving a one-sided frozen response. These interventions help explain why the successful repair retained both afferents and restored the missing local holding current. [Three-seed repair comparison](59-local-frontal-maintenance.md)

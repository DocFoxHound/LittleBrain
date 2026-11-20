# Full-brain reconstruction

The earlier integrated checkout ended at `b4b378f` and was reported as 58 commits ahead. The replacement checkout contains the public baseline and the new reconstruction history. The available source excerpts establish the following provenance.

| Recovered file | What could be restored directly |
|---|---|
| `federation.py`, `agent.py`, `brainview.py`, `run_signal_game.py` | The supplied lines match the earlier public sources at `96f368c` after line-ending normalization. The newer circuit and game code was reconstructed. |
| `21-brain-regions-built.md` | The regional catalog note was already present in the public baseline. |
| `47-littlebrainworld-design.md` | Recovered in full. SHA-256 `0af2cc9c10c5e2ba28178c22be2b2677652293f1e3ac30cb5cbf12c4abdb2159`. |
| Chat transcript | Supplied goals and design decisions. The uploaded text ends during the game-interface discussion. The longer surviving conversation record identifies the former revision and the foraging demo. Neither contained the missing diffs or trained state. |

## Implementation

The default viewer develops a 14,656-neuron brain and opens the foraging task. The default game runner and external bridge use the same full specification. The reference scale has 120 simulated populations covering 98 distinct catalogued cell types. Retina, motor efference copy and the unconditioned outcome are external inputs. Population means, timing channels and diffuse neuromodulation are computational approximations. The viewer depicts approximate anatomical territories in a side projection. Its dots group neighboring cells.

The cumulative factories preserve the intermediate BG, dopamine, amygdala, hypothalamus, PFC/MD, cerebellum, hippocampus, CA2 and insula/cingulate stages. The full factory adds the regional terminals, including higher-order thalamus, brainstem neuromodulators and rare cell types. Map minima and the additional cerebellar granule maps reproduce the reference total. Original per-population counts and individual synapse partners could not be recovered. [Scaling rules](../docs/scaling.md)

A preliminary reconstruction still chose visual actions at chance. Successive presentations of one cue evoked divergent association-cortical activity, and the initial striatal current could be overwhelmed before reaching the motor readout. The repaired relay combines a sustained retinotopic component with the original transient firing. Opponent D1/D2 projection ensembles receive cue activity, a copy of the chosen motor channel and phasic dopamine from VTA. The existing GPi motor readout still selects the action. The game supplies pixels and consequences. It supplies canonical answers only to the experiment log.

A second failure was a dopamine dip when hiding successfully. Appetitive and aversive thalamic inputs to BLA now acquire and extinguish separately. The cue changes the expected value, and the outcome is compared with that expectation. AgRP and POMC activity sets the food component of the outcome. Other population approximations include a recurrent PFC/MD projection ensemble, pontine and granule temporal responses and cerebellothalamic output. The specific time constants, weights and ensembles are reconstruction choices.

After 64 learning turns the full game agent reactivates up to eight waking sensory patterns at reduced acetylcholine, then runs the existing consolidation and value-guided partner selection. The waking log preserves signed phasic dopamine. MCH activity adjusts ACh during replay. The default lean banks consider up to 3% of eligible slots for turnover. Source pathways, partner budgets, recently important synapses and newborn protection constrain selection. This establishes the mechanism. A long-term performance advantage from rewiring remains an experiment.

The biological precedents include dopamine modulation of striatal projection neurons (Gerfen and Surmeier, 2011), distinct amygdala circuits for positive and negative associations (Namburi et al., 2015), and mediodorsal thalamic support for prefrontal working memory (Bolkan et al., 2017). The overlapping granule timing responses and error-driven parallel-fiber adjustments are reduced stand-ins for cerebellar timing (Medina et al., 2000). In the timing game an actual movement can receive early or late feedback after its outcome. Directional climbing fibers update the active and neighboring phase ensembles. The phase widths, shifts and reduced projection from cerebellothalamic output to striatum are modeling assumptions. Branch-specific spine formation during post-learning sleep has experimental precedent (Yang et al., 2014). The simulator's fixed turnover budget and signed partner-selection rule remain hypotheses.

## Verification

Run from the repository root with NumPy and pygame-ce installed. Each CPU game comparison starts from a normally calibrated and developed checkpoint, using the same brain and game seed for both conditions. For the control, `off` disables reinforcement and scheduled sleep. Intrinsic activity regulation continues, as it does in the original game runner. A lesion suppresses the named populations during the game after normal development.

```sh
export OPENBLAS_NUM_THREADS=1
python prototype/brain/test_cell_types.py
python prototype/brain/test_cell_types_regions.py
python prototype/brain/test_learning_by_region.py
python prototype/brain/check_terminal_series.py
python prototype/brain/check_full_games.py --seed 0 --prepare-only /tmp/seed0.brain
python prototype/brain/check_full_runtime.py --seed 0 --load-brain /tmp/seed0.brain
python prototype/brain/check_full_games.py --seed 0 --load-brain /tmp/seed0.brain --game signal --condition learning
python prototype/viz/brainview.py --shots 6 --out /tmp/forage-shots
```

Repeat the checkpoint, runtime check and comparisons for seeds 1 and 2. Use `--game signal --condition off` for the frozen control and `--game danger --condition lesion` for the amygdala lesion. The other games accept the same arguments. Delayed trials score only the final decision. Timing scores normalized error only at episode completion. Foraging reports spontaneous intake by hunger and a matched post-training probe that presents the same image while varying only hunger.

All 36 original neuron and plasticity checks passed. All 55 cumulative terminal additions delivered the specified signed input against a silenced-path control on each of three seeds. These tests isolate wiring. They do not establish a behavioral benefit from every individual cell type.

| Seed | Sampled synapses | Partners changed at the scheduled-night probe |
|---:|---:|---:|
| 0 | 1,547,264 | 37,160 |
| 1 | 1,547,072 | 37,194 |
| 2 | 1,547,200 | 37,236 |

The external Python client completed an observation/outcome round trip on each seed. It rejected a stale turn, applied the corrected outcome and reloaded a checkpoint with changed striatal weights. Seed 0 used 157 MiB peak process memory when loading the developed CPU brain and playing one turn headlessly. The measurement includes Python, NumPy and transient allocations. GPU memory was not measured.

The paired activity probes in the integrated brain reached CA2 from both CA3 and entorhinal layer III, then reached CA1 from CA2 and subiculum from CA1. They also exercised insula to cingulate, cingulate to striatum, pulvinar to cortical layer II/III, claustrum to cortical PV cells and Purkinje to deep cerebellar nuclei. Each probe changed measured target firing on all three seeds. At quarter scale, the allocation floors retained 8,256 neurons and seed 0 had 417,280 sampled synapses. The scaled brain exercised all eight motor slots and continued into the timing game.

In an additional sleep control, identical copies received the same five waking Signal turns. Reversing the sign of the stored dopamine in one copy before calling `sleep()` changed 36,325, 36,320 and 36,381 of the selected partners relative to the original signs (seeds 0–2). The samples and source populations were otherwise matched. At double scale, a freshly wired seed-0 brain handled eight actions and timing outcomes with finite activity in 29,312 neurons and 4,957,696 sampled synapses. The double-scale smoke test did not measure learning performance.

## Failures during integration

On the first full-scale visual projection, seed 0 scored 22% at turn 200 in the trailing 100 Signal turns (four-action chance is 25%). The initial timing repair also looked promising over a short run. Directional olivary feedback alone reached errors of 0.167, 0.136 and 0.115 by turn 100, then regressed to 0.831, 0.845 and 0.825 at turn 300. Inspecting the relay revealed that reading only the motor stripes had erased the temporal pattern from the striatal learning input. Reward for a correctly timed movement could subsequently release that movement at the wrong time. The revised projection carries interval and motor correction on separate axes of the existing map. It saturates the associative current before combining them.

## Behavioral measurements

In Signal, the same four pictures have a different randomly assigned button mapping for each seed. In Danger, the action that avoids damage also forfeits food. Each entry below covers the last 100 turns of 300.

| Seed | Signal correct, learning | Signal correct, learning off | Danger hurt, intact | Food collected on food trials, intact |
|---:|---:|---:|---:|---:|
| 0 | 100% | 26% | 0% | 100% |
| 1 | 100% | 27% | 3.64% | 88.89% |
| 2 | 97% | 15% | 0% | 100% |

The Signal chance rate is 25%. Random selection in Danger has a 75% hurt rate and a 25% food collection rate. The reference gates are at least 60% Signal accuracy and at most 40% hurt on every seed.

Foraging also required a matched intervention. Before outcome-specific revaluation, seed 0 ate on 37.5% of naturally hungry trials and 17.3% of sated trials in the last 100. Its trained copy then gave the same food image an eating probability of zero under both sustained hunger and sustained satiety. Natural visits alone had confounded appetite with the trajectory through the game. The new food-contact afferents learn which visual and motor inputs predict food. Current AgRP/POMC activity changes the incentive value of that learned outcome even before another meal. This is a reduced version of sensory-specific outcome revaluation (Balleine and Dickinson, 1998).

With interval and motor activity retained separately, normalized timing error in the first 50 turns was 0.262, 0.308 and 0.485. In the final 100 turns it was 0.197, 0.288 and 0.258. A one-turn discrepancy in this three-turn task has normalized error 0.333. These are means over completed episodes. Individual training blocks fluctuated.

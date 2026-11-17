# 29: The first game: learning which button goes with which picture

*2026-11-17. Option 2 from note 28: make the brain learn what game characters need, reward-driven
action learning, and measure it honestly against an identical brain with learning switched off.*

## What was built
- **Game interface** (`prototype/games/core.py`): `reset() -> Observation`, `step(action) -> (obs, reward, done, info)`.
  An observation is a 2D picture in [0, 1] plus optional body signals. Games know nothing about neurons.
- **Signal game** (`prototype/games/signal_game.py`): one of 4 pictures (3 random discs each); each picture
  has one correct button (random per game seed). Reward 1 for the right button. A contextual bandit, the
  classic test of striatal, dopamine-taught learning.
- **Basal ganglia action loop** (`brainspec.stage_bg`, 5,632 neurons): stage 1 cortex + thalamus →
  striatum (D1 "go", D2 "no-go", fast-spiking interneurons) → GPe and GPi. Corticostriatal input converges
  broadly. The 8×8 map is split into 4 vertical stripes = 4 **action channels** kept parallel through
  striatum and pallidum. GPi holds a tonic brake on every action; D1 cells release their channel's brake,
  D2 cells (via GPe) tighten it.
- **Efference copy** (`MOT` input): the chosen action excites its own striatal channel while the outcome
  arrives (motor thalamus/cortex → striatum), so dopamine strengthens or weakens only the chosen action's
  synapses (three-factor rule: input × cell activity × dopamine; D2 cells learn from dips).
- **Agent** (`prototype/brain/agent.py`): look at the picture for 6 steps → read GPi per channel → choose
  with a softmax (exploration = neural noise) → act for 4 steps, dopamine at step 3 → rest 3 steps.
  **Dopamine is a stand-in critic**: reward minus a running average of reward (not cue-specific yet).

## Problems fixed on the way
- Striatal cells silent because the cortex was still in its post-sleep silence (note 28): a 360-step wake
  period after development, plus cortical learning ×0.05 (`CORTEX_LRF`, keeps L2/3 alive after sleep).
- GPi already off at rest (GPe → GPi too strong): weakened to −0.25 so GPi idles at ~0.3 and choosing an
  action drops its channel to ~0.04.
- Learning went the right way (correct channel had the strongest D1 drive for 3 of 4 pictures) but was
  ~0.02 where ~0.4 is needed: striatal fast weights forgot with a half-life of ~11 turns. Corticostriatal
  plasticity lasts hours, so MSNs now use lr 0.6 (×4) and fast decay 0.99995 per step (`MSN_LR`, `MSN_DECAY`).

## Results (`run_signal_game.py`, 400 turns, chance 0.25)
| seed | learning: last 100 turns | learning off |
|---|---|---|
| 0 | 0.47 | 0.24 |
| 1 | 0.57 | 0.27 |
| 2 | 0.59 | 0.26 |

- **The character learns**, on every seed, about twice chance.
- It **plateaus** around 0.45–0.6: a 1,000-turn run (seed 0) stays at ~0.44.

## Likely reasons for the plateau (next)
1. The cortical picture codes are sparse (2–3% active) and the 4 pictures overlap; each D1 cell sees only a few active inputs.
2. The critic isn't cue-specific: once some pictures are learned, their rewards are "expected" and stop teaching.
3. Homeostasis raises the threshold of striatal cells that learned to fire, partly undoing the learning.
4. Exploration temperature is fixed.

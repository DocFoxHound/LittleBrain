# 30: Raising the ceiling on the signal game

*2026-11-17. Note 29's character learned (~0.5 correct, chance 0.25) but plateaued. Three suspects, tested.*

## Suspects that weren't it (3 seeds each, 400 turns, last 100 turns)
| config | seed 0 / 1 / 2 |
|---|---|
| as note 29 | 0.59 / 0.57 / 0.47 |
| `CRITIC=cortex`: expected reward read out of the cortex per picture (stand-in for striosomes / ventral striatum) | 0.56 / 0.40 / 0.42 |
| `HOMEO_SKIP=STR.D1,STR.D2`: striatal cells left out of homeostasis | 0.52 / 0.49 / 0.41 |
| both | 0.13 / 0.26 / 0.45 (collapses on seed 0) |

Neither helps; both are kept as options.

## What it was: one look at the picture is too noisy
Over the last 150 turns, **the correct button had the strongest average release for all 12 picture/brain
pairs**: the striatum had learned every answer. But on single turns, the best-released button was right only
47–67% of the time, and a simple decoder could tell which picture was shown from one look's cortical activity
only 63–71% of the time. The cortex's response to the same picture still varies from turn to turn (note 28's
reliability is ~0.5, not 1).

**Fix: look longer and add up the evidence** (the basal ganglia integrate over time before releasing an action).
| look / steps counted | reward, last 150 (seeds 0 / 1 / 2) | picture decoding |
|---|---|---|
| 6 / 3 (note 29) | 0.47 / 0.56 / 0.52 | 0.63–0.71 |
| 12 / 8 | 0.57 / 0.71 / 0.69 | 0.68–0.80 |
| **20 / 16 (new default)** | 0.74 / 0.67 / 0.61 | 0.80–0.93 |
| 20 / 16, temperature 0.01 | 0.72 / 0.72 / 0.62 | |

Default run (`run_signal_game.py`, seed 0): **learning 0.73 vs learning off 0.24** over the last 100 turns.
A lower choice temperature trades exploration for exploitation and breaks even, so it stays at 0.02.

## Notes for the tool
- Looking time is a natural **speed–accuracy knob**: a character that must act fast in a game will be less accurate.
  That's a realistic, testable "environmental factor".
- The real ceiling is cortical reliability. Improving it (note 28's open items, area B) should help every game.

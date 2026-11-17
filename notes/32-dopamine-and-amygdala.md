# 32: Dopamine cells and the amygdala

*2026-11-17. First two regions wired into the game brain after the basal ganglia (order agreed: dopamine cells,
amygdala, hypothalamus, prefrontal cortex, cerebellum).*

## Dopamine made by neurons (`brainspec.stage_da`, 5,760 neurons; `BRAIN=da`)
- **Reward input** (`US`): the reward signal arriving from the body (lateral hypothalamus / brainstem), set by the
  agent from the game's reward. It excites the dopamine cells (`VTA.DA`).
- **Expectation:** striosome MSNs (`STR.SO`) learn from cortex how much reward to expect and inhibit the dopamine
  cells (striosome → SNc/VTA; Crittenden et al. 2016). They also get the **copy of the chosen action**, so the
  expectation arrives when the outcome is due and is specific to (situation, action): an action value.
- **Prediction error:** `Brain._own_dopamine` = the dopamine cells' firing relative to their own slow (tonic)
  level, used as the learning signal. Bursts = better than expected, dips = worse. Dips are shallower than bursts,
  as in real dopamine cells (they can't fire below zero; Bayer & Glimcher 2005).
- Getting there:
  | setup | what went wrong |
  |---|---|
  | striosomes without the action copy | flat activity all turn; absorbed into the slow baseline: no dips |
  | strong brake (weight 4) | tonic striosome firing pinned dopamine at zero |
  | no homeostasis on striosomes | they ran away (0.6+) or went silent |
  | normal homeostasis | erased the learned expectation (it rises for every picture at once as the character improves) |
  | **slow homeostasis on striosomes (`HOMEO_SLOW=STR.SO:0.1`), weight 4, action copy** | ✅ |
- **Result (signal game, 300 turns):** misses dip deeper as the character improves (−0.06 → −0.21; −0.07 → −0.22)
  and reward bursts shrink (0.95 → 0.85; 0.92 → 0.82): expected reward is being learned. Learning 0.63–0.69 correct
  (chance 0.25), close to the computed stand-in (0.61–0.74). One seed of the earlier variant: 0.43.

## Amygdala (`brainspec.stage_amy`, 6,272 neurons; `BRAIN=amy`)
- **BLA principal cells**, positive and negative valence, hear the thalamus ("low road") and cortex ("high road").
  Pain or reward from the body drives them; they learn in a few tries, gated by arousal. Negative cells learn from
  dopamine dips, positive from bursts.
- **Central amygdala chain:** BLA(−) → CeL SST ("fear-on") ⊣ CeL PKCδ ("fear-off") ⊣ CeM output. CeM above its
  resting level raises **arousal** (`Brain._own_arousal`, CeM → locus coeruleus), which gates BLA learning.
- **BLA → dopamine cells:** a learned danger cue lowers dopamine, a learned good cue raises it.
- **Danger game** (`games/danger_game.py`): food pictures (right button +1) and danger pictures (anything but
  "hide" −1, hiding 0).

### Results (`run_danger_game.py`, 300 turns, 2 seeds)
- **The amygdala learns which pictures are dangerous** (seed 0, after 300 turns; before training no difference):
  | | food | danger |
  |---|---|---|
  | BLA(−) | 0.070 | 0.087 |
  | BLA(+) | 0.106 | 0.093 |
  | CeM (alarm) | 0.150 | 0.213 |
  | extra arousal | 0.013 | 0.091 |
- **But the character doesn't learn to hide, with or without the amygdala:** hurt on danger stays 0.64–0.76
  (random: 0.75); last 100 turns 0.71 / 0.73 intact vs 0.64 / 0.76 silenced.
- **Why:** punishment learning is weak in this brain. (1) Dopamine dips are ~5x weaker than bursts.
  (2) Hiding successfully produces *nothing*: no relief signal. Real avoidance learning relies on relief, a dopamine
  burst when an expected danger doesn't come (two-factor theory; Mowrer). That needs dopamine computed over time
  (temporal difference: the cue itself causes a dip, and the safe outcome a burst relative to it). Ours compares with
  a slow baseline, so it can't produce relief.

## Next
1. Temporal-difference dopamine: the dopamine cells compare the expectation now with the expectation a moment ago
   (striosome / ventral striatum activity at cue vs outcome). Gives relief bursts, cue dips, and a fair test of the amygdala.
2. Then hypothalamus (drives), prefrontal cortex (holding things in mind), cerebellum (timing).

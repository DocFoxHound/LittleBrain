# 80: Credit and harm across steps (plan phase 1 and track N1)

*2026-11-22. Before this, the brain learned only "this situation and action pays off now". Opt-in with
`TD_GAMMA` (e.g. 0.9); 0 keeps one-turn learning.*

## What was built
- **A value critic** (`striatal_projection.critic_value/critic_learn`): a reduced ventral-striatum readout of how good
  the current situation is, including what it leads to.
- **Temporal-difference error, split over two moments.** For turn t: at the outcome, error = reward − V(now); at the
  next turn's cue, error = γ·V(next), which also credits the **previous** action and situation. Together:
  reward + γ·V(next) − V(now) (Schultz, Dayan & Montague 1997). Episodes (`done`) are not chained.
- **The critic computes the error; it drives the dopamine cells** (bursts directly; pauses through the lateral
  habenula → RMTg, Matsumoto & Hikosaka 2007) **and teaches**. A negative error teaches only if the habenula → RMTg
  pathway is intact, so silencing it removes learning about harm. In the vision brain: an `aversive value -> LHb` path
  and a stronger `LHb -> RMTg` relay (the reconstructed one saturated at about a third of a pause; now a full loss of
  value pauses dopamine about as much as an equal gain bursts it: −0.96 vs +1.15).
- **Both action systems get the credit**: the reduced striatal learner, and the striatum's own synapses (at the cue,
  their plasticity is credited to the previous action's channel via `credit_motor`).
- **Games** (`games/planning_games.py`): **key and door** (3-step chain, reward only at the end; wrong actions stay put)
  and **slow poison** (a berry pays +0.3 now, costs −1 two turns later).

## What it took (seed 0, slow poison)
1. The habenula path first did nothing: its cells are excitatory and ignore the generic "drive" input (use `exc`).
2. Cue and outcome dopamine were read against an unusual instant before them, or a biased resting level: the +0.3 berry
   read as +0.45-0.9 and sickness as −0.5, so eating looked good. Reading against a slow resting level, then (final)
   teaching with the critic's own error instead of reading it back from the cells.
3. The critic needed its own input: the actor's trial-centred input erased frequent situations' values; un-centred
   input made all pictures look alike. Final: minus a very slow average, then a **sparse code** (top 5% of features,
   all-or-none; pattern separation) so situations barely overlap, learned on exactly the code it predicted from.
4. Then the critic behaved like dopamine: V(after eating) −0.02 → −0.60; the cue after eating dipped −0.18 → −0.99;
   the sickness dip shrank −0.94 → −0.30 as it became expected. But eating still rose: the striatum's own synapses
   learned only at the outcome (+0.3) and never got the later dip. Crediting the cue dopamine to the previous action's
   channel fixed it (eating 0.47 → 0.12).

## Results (3 seeds, vision brain, last 100 of 600 turns)
| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| poison, eat rate: TD off | 0.58 | 0.27 | 0.67 |
| poison, eat rate: **TD on** | **0.12** | **0.21** | **0.33** |
| poison, eat rate: TD on, habenula silenced | 0.48 | 0.42 | 0.03 |
| key-door, per-step / first step / chains done: TD off | 0.19 / 0.09 / 6 | 0.16 / 0.12 / 6 | 0.39 / 0.33 / 13 |
| key-door: **TD on** | **0.41 / 0.32 / 14** | **0.37 / 0.30 / 12** | 0.37 / 0.25 / 12 |
(random: eat 0.25; per-step 0.25; ~33 berry decisions per 100 turns, so poison rates are noisy.)

- **Delayed harm is learned**: with TD the brain avoids the poison berry on every seed (TD off it is drawn to the +0.3
  on two seeds); silencing the habenula brings eating back on two of three seeds (seed 2: 0.03, unexplained).
- **Delayed reward is learned partly**: key-door per-step accuracy roughly doubles on seeds 0-1; seed 2's TD-off run
  was already good. **Below the plan's bar** (≥ 0.6 on a 3-step chain).
- TD stays **opt-in**: in one-turn games that don't mark episode ends (signal, colour) bootstrapping between unrelated
  trials only adds noise. Games should mark `done` before TD becomes a default.

## TD(λ) (eligibility that fades over earlier turns, λ = 0.6, now the TD default; `TD_LAMBDA`)
| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| poison eat rate: TD | 0.12 | 0.21 | 0.33 |
| poison eat rate: **TD(λ)** | **0.15** | **0.06** | **0.12** |
| key-door per-step: TD | 0.41 | 0.37 | 0.37 |
| key-door per-step: TD(λ) | 0.35 | 0.26 | 0.49 |
Harm avoidance is sharper on every seed; the chain is no better overall, so its limit lies elsewhere (the actor's
input for similar pictures, exploration): the world model of phase 3 is the next tool for it.

## Next
- Stronger chain learning: eligibility across more than one step (TD(λ)), more turns, and the world model of phase 3.
- Games mark episodes; then TD on by default for the game brain.

# 78: Plan: a brain that can learn to play Minecraft, and shipping it

*2026-11-22. Agreed direction: ship a **blank** brain and a **Minecraft template** (a brain already trained in
Minecraft), selectable in the app, with "save as new template" and "overwrite". Before any template is trained, the
brain must be able to learn what Minecraft needs: above all, to **think ahead**. Phases are in build order; each ends
with a test game, controls (learning off, region silenced), 3 seeds, a note, and a commit (AGENTS.md rules).*

## What "plays Minecraft" means for template v1
Realistic first goals, on a small local world with the bot doing the fine motor control:
1. **Eat when hungry**, not when full (food from animals or berries).
2. **Gather and craft**: wood → planks → crafting table → wooden pickaxe (a 4-7 step chain with the reward at the end).
3. **Survive the first night**: notice mobs, keep distance or get into shelter, recover.
4. **Find its way back** to a remembered place (base, food spot).
Not in v1: building, mining deep, combat strategy, language/chat.

## Where we are (start of this plan)
Learns from immediate outcomes (habits) through the striatum; dopamine made by neurons, with cue responses and relief
(notes 32, 48); amygdala, hypothalamus drives, prefrontal working memory, cerebellar timing; vision with colour, motion
(MT), a ventral stream and eye movements (notes 71-77; colour and motion learned, shape-at-any-position not yet).
Missing for Minecraft: credit across many steps, a world model, look-ahead, goals/routines, spatial maps, object
recognition regardless of position, and the Minecraft connection itself.

---

## Phase 0: groundwork (small)
- **Design version in every save file**: brain kind (`full`, `vision`, `human100k`...), spec hash, code version, date.
  Loading refuses a mismatched design with a clear message (needed before templates exist).
- **One "game brain"**: the vision brain becomes the base for games; a size option (human-proportion scaling, note 70)
  that includes the visual areas.
- **Speed budget**: today a decision takes ~27 brain steps (~1-3 s on CPU). Target ≤ ~1 s per decision for the Minecraft
  brain; Minecraft's tick rate can be lowered on a private server (`/tick rate`, Java 1.20.3+) so the world waits for it.
- Done when: old and new saves load or are refused correctly; speed measured and recorded.

## Phase 1: credit across steps (temporal-difference dopamine)
- **What**: dopamine = reward + (how good the next situation looks) − (how good this one looked) (Schultz, Dayan &
  Montague 1997). The expectation comes from striosomes / ventral striatum, now compared **between turns**, so value flows
  backwards through a chain. Builds on note 32's dopamine cells and note 48's cue/relief responses.
- **Test game: key and door** (new): pick up a key, then open a door for the reward; chains of 2, 3 and 5 steps;
  distractor actions. Also the danger game's avoidance (relief) as a regression.
- **Done when** (3 seeds): 3-step chain ≥ 0.6 success (chance computed per chain), clearly above learning off and above
  the brain without TD; dopamine moves earlier in the chain with training (to the key, then to the start).

## Phase 2: knowing where things are (spatial maps)
- **What**: head-direction cells (anterior thalamus/postsubiculum), grid cells (entorhinal cortex) driven by self-motion
  (the action copy: path integration), place cells (CA3/CA1) tied to landmarks (Moser & Moser; O'Keefe). The hippocampal
  loop exists (notes 26, 55, 60-61) but is not yet spatial.
- **Test game: LittleBrainWorld** (note 47's design, top-down): go home after foraging; remember where food was; with
  the hippocampus silenced as a control.
- **Done when**: place-like cells (stable firing fields) on every seed; return-home success clearly above silenced
  hippocampus.

## Phase 3: a world model and look-ahead
- **What**:
  - **Predictive map**: the hippocampus learns where each situation tends to lead (successor representation,
    Stachenfeld et al. 2017) with the TD signal of phase 1.
  - **Imagining before acting**: a short "imagine" phase before each choice; hippocampal sweeps forward along each option
    (Johnson & Redish 2007), each outcome scored by orbitofrontal/ventromedial prefrontal value **under the current
    drive** (hypothalamus: hungry → food paths valuable). The basal ganglia pick.
- **Test games**: **detour** (block the usual path: find another without retraining; Tolman 1948), **latent learning**
  (explore without reward, then a goal appears), **revaluation** (food made worthless when full: a planner changes at
  once, a habit doesn't).
- **Done when** (3 seeds): detour and revaluation solved within a few trials, clearly better than the brain with
  hippocampus or OFC/vmPFC silenced (which should behave like a habit learner).

## Phase 4: goals and routines
- **What**: prefrontal cortex holds the current **goal and sub-goal** (extends note 59's working memory); the basal
  ganglia **chunk** practised sequences into single choices ("routines", Graybiel 1998); a sub-goal finishing releases the next.
- **Test game: crafting** (new, abstract): items and recipes like Minecraft's (logs → planks → table → pickaxe), sparse
  reward at the end; later, recipes it has seen used in a new order.
- **Done when**: completes the 4-step crafting chain ≥ 0.6 of episodes on every seed; practised chains get faster
  (fewer decision steps) with practice; PFC silenced → can't hold the goal.

## Phase 5: recognising things anywhere (object vision)
- **What**: temporal-trace learning in V4/IT: views seen close together in time (an object moving, or the eye
  moving over it) become one object (Földiák 1991; Wallis & Rolls 1997). Saccades (note 76) supply the views.
- **Tests**: shape game at random positions (note 75: ~chance now); then Minecraft-like block and mob pictures.
- **Done when**: shape at any position learned ≥ 0.6 on every seed; IT identifies shapes across positions.

## Phase 6: learning in sleep
- **What**: sleep replays whole sequences, and value spreads backwards offline (the hippocampus-to-striatum replay of
  rodents; "Dyna" in machine learning). Sleep and replay already exist (notes 09, 48); extend replay to sequences with TD.
- **Test**: the same games with fewer waking trials needed after nights than without them.
- **Done when**: on every seed, a night of replay improves the next day's success clearly more than an awake rest.

## Phase 7: the Minecraft connection
- **Server**: a private local Minecraft Java server (Paper or vanilla). **Owner decision**: accepting Mojang's EULA;
  a Minecraft Java account to watch in-game (or watch in a browser with prismarine-viewer).
- **Bot**: Mineflayer (Node.js) speaks Game Bridge v1 to the brain. The bot does the fine motor work (pathfinding,
  aiming), the brain chooses among **macro-actions** (≤ 8 per menu; larger sets through control layouts, note 46):
  move toward / away from nearest X, look around, mine or attack the target, place, eat, open crafting, craft item N.
- **Senses**:
  - Vision: a low-resolution first-person picture (rendered with prismarine-viewer headless, or ray-cast colour/depth),
    plus a small top-down local map as a fallback.
  - Body: hunger, health (damage → pain), time of day (light), air.
  - Outcomes: food eaten, items gained, damage taken, death.
- **World**: seeded worlds and slowed ticks for repeatable experiments; a curriculum: flat peaceful → trees and
  animals → night with mobs.
- **Done when**: the blank brain plays episodes end to end through the bridge, with logged observations, actions and
  outcomes, and the viewer can show it live.

## Phase 8: training the Minecraft template
- Curriculum episodes, nightly sleep, checkpoints each night, a fixed **evaluation suite** for v1's four goals (success
  rates on held-out seeded worlds) compared against the blank brain.
- Expect long runs: hours to days of CPU time; scale chosen in phase 0.
- **Done when**: the template beats the blank brain on all four goals on held-out worlds, on 3 independently trained brains.

## Phase 9: shipping templates in the app
- **Templates**: `blank` (developed, untrained) and `minecraft-v1`, each a save file plus metadata (design version,
  size, training summary, date, evaluation scores).
- **In the app (viewer and bridge)**: choose a template at start or from a menu; switch brains while running (with a
  "save first?" prompt); **save as new template**; **overwrite** another (with confirmation); stored in the player's save
  folder (SLAPorSLOP packaging, note 34). Built-in templates are read-only.
- **Checks**: a template only loads into a matching design (phase 0); saves are data-only (note 68).
- **Done when**: a fresh install offers both templates, loads each, saves and overwrites correctly, and refuses an
  incompatible file.

---

## Order and dependencies
0 → 1 → (2, 5 in parallel) → 3 → 4 → 6 → 7 → 8 → 9. Phase 7 (the connection) can start in parallel with 3-6 using
the blank brain, so the plumbing is ready when the brain is.

## Risks
- **Sample efficiency**: biological learning rules are slow; Minecraft may need many thousands of episodes. Mitigation:
  macro-actions, curriculum, sleep replay, slowed world.
- **Speed**: decision latency vs. a live world (slowed ticks).
- **Honest scope**: v1 goals are deliberately small; each phase can show a capacity is not there yet, and the notes
  will say so.

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
5. **Say a few things** in chat when they matter ("food" when hungry, "help" in danger, names of things it knows),
   and understand a few words from the player (track L).
6. **Cooperate sensibly**: help a known partner when it costs little, prefer partners who reciprocate, stop helping
   one who never does (track C).
Not in v1: building, mining deep, combat strategy, conversation beyond a few words.

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

## Track L: communication (sound and text), grown like a child's, not an LLM
*Added 2026-11-22 at the owner's request. Language grows out of hearing, babbling and interaction (README: "from cries
into sounds into words"), learned with the same biological rules as everything else. No language model.*

**Senses and voice.** Sound and text share one route in: text is turned into the same sound-like patterns a word
would make, so a word read in chat and a word heard are the same word to the brain (reading glyphs with the eye can
come later). Out: a small **vocal motor** system whose units are syllable-like sounds; the host turns them into audio
or text.

| step | what | real counterpart | test (3 seeds, controls) |
|---|---|---|---|
| L1 hearing | an ear like the retina: a filter bank (cochlea) → auditory thalamus (MGB) → auditory cortex (A1, belt) on a tonotopic map | cochlea, MGBv, A1 | tell syllables apart; recurring sound patterns become familiar from mere exposure (statistical learning, Saffran et al. 1996) |
| L2 babbling and imitation | the vocal units babble; the brain hears its own voice (efference copy) and a basal ganglia loop learns from the mismatch with sounds it has heard | larynx motor cortex, songbird-style vocal learning with dopamine error (Gadagkar et al. 2016) | imitation accuracy rises; silenced auditory feedback → no improvement |
| L3 words mean things | a caregiver names what the character is looking at; heard word + seen object link in temporal association cortex and hippocampus | cross-situational word learning (Smith & Yu 2008); joint attention | picks the named object among several; learns words from ambiguous scenes |
| L4 using words | saying a word changes the world: "food" when hungry makes the caregiver bring food; "help" in danger | requests, social reward | asks appropriately (hungry → food word) above chance and above a no-language control |
| L5 combining and turn-taking | two-word messages (sequence via prefrontal goals and basal ganglia chunks, phase 4); taking turns | Broca's area, PFC | understands and produces simple two-word combinations; waits its turn |
| L6 two brains talking | two LittleBrains invent a shared code in a signalling game (Lewis signalling; emergent communication) | | shared code above chance; a fresh brain can learn it |

**The caregiver.** A scripted caregiver in the games (names things, answers requests, praises), plus the **player**
through the app: type or speak to the character, read or hear its replies.

**In Minecraft.** Chat is the channel: player messages arrive as heard words; the character's vocal output is sent as
chat. v1 template goal: a handful of words used appropriately ("food", "help", names of things it knows).

**Where it fits.** L1 can start right away, in parallel with phase 1. L2 needs phase 1 (credit across steps). L3 needs
phase 5 (objects). L4 needs phase 3 (goals and drives). L5 needs phase 4. L6 any time after L4. Template training
(phase 8) includes L1-L4 so the Minecraft template can say a few things.

---

## Track C: cooperation, learned on top of built-in social drives
*Added 2026-11-22 at the owner's request: brains should come to value cooperative actions that benefit the group and
the self. Mammals have built-in social drives (company is rewarding, another's distress is unpleasant) and learn from
experience **who** and **when** to cooperate with. So: a few innate drives, learned value, and no hard-coded "always
cooperate": self-interest stays, and cooperating with someone who never reciprocates should stop.*

Already in the brain: oxytocin neurons (PVN) driven by a `social_contact` body signal; CA2 (social memory);
insula and anterior cingulate cortex.

| step | what | real counterpart | test (3 seeds; controls) |
|---|---|---|---|
| C1 others are rewarding | recognise other characters (vision: phase 5; individuals: CA2); being near a familiar, friendly other is mildly rewarding (oxytocin → dopamine cells) and its absence a mild need, like hunger | oxytocin-VTA interaction in social reward (Hung et al. 2017); CA2 social memory (Hitti & Siegelbaum 2014) | seeks a familiar partner more than a stranger; CA2 silenced → no preference |
| C2 others' outcomes matter | seeing another's reward or distress partly activates one's own (anterior cingulate, insula); weighted by familiarity; helping that ends another's distress is rewarding | vicarious reward and pain (ACC gyrus, Chang et al. 2015; Lamm et al. 2011); rats free trapped cagemates (Ben-Ami Bartal et al. 2011) | frees a trapped partner without any food reward, more for familiar partners; ACC/insula silenced → stops |
| C3 shared rewards | joint tasks where reward needs both (pull a rope together; a stag hunt); the TD value of phases 1-3 learns that together pays | cooperative pulling in chimpanzees and elephants (Hirata & Fuwa 2007; Plotnik et al. 2011) | coordinates and waits for the partner; learns which tasks need two |
| C4 reciprocity and trust | remembers what each individual did (CA2, orbitofrontal value per partner); a betrayal is a negative prediction error (insula); trust builds and breaks | trust games (King-Casas et al. 2005); unfairness and insula (Sanfey et al. 2003) | iterated prisoner's dilemma: cooperates with reciprocators, stops cooperating with persistent defectors; forgives occasional mistakes |
| C5 the group | several brains in one world: shared resources, a common pool that collapses if over-used, division of labour | common-pool resource games (Ostrom) | groups keep the pool alive longer than selfish baselines; members still look after themselves |
| C6 coordinating by talking | uses track L words to coordinate ("help", "here", "wait") | | coordinated joint tasks succeed more with communication than without |

**Measures.** Cooperation rate over time; payoff to self **and** to the group (both should rise); resistance to
exploitation (no blind cooperation with defectors); partner preference. Compared against a brain with the social drives
off (oxytocin input and vicarious signals silenced), which should cooperate only when it directly pays.

**Infrastructure.** Several brains in the same world: LittleBrainWorld supports multiple characters (note 47); in
Minecraft, several bots on one server, plus the player. Smaller brains for group runs (compute grows with the number
of brains).

**Where it fits.** C1 after phase 5 (recognising others) and phase 2 (where they are); C2 after phase 1; C3 and C4
after phase 3; C5 after C4; C6 after L4. Template v1 goal: **helps a known partner when it costs little, prefers
partners who reciprocate, and stops helping one who never does.**

---

## Order and dependencies
0 → 1 → (2, 5 in parallel) → 3 → 4 → 6 → 7 → 8 → 9. Phase 7 (the connection) can start in parallel with 3-6 using
the blank brain, so the plumbing is ready when the brain is. Track L runs alongside: L1 with phase 1; L2 after 1;
L3 after 5; L4 after 3; L5 after 4. Track C: C1 after 2 and 5; C2 after 1; C3, C4 after 3; C5 after C4; C6 after L4.

## Risks
- **Sample efficiency**: biological learning rules are slow; Minecraft may need many thousands of episodes. Mitigation:
  macro-actions, curriculum, sleep replay, slowed world.
- **Speed**: decision latency vs. a live world (slowed ticks).
- **Honest scope**: v1 goals are deliberately small; each phase can show a capacity is not there yet, and the notes
  will say so.

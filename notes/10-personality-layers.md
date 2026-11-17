# 10: Layered personality: changing without forgetting who you were

## The concern (the user's)
New training overwrites weights. A person who started out kind and became jaded through
hard experiences hasn't *lost* their kindness. It's still in them, harder to reach, and
the right situation or honest reflection brings it back. Personality changes over time,
but we don't forget who we were. That's how we get out of the Ship of Theseus problem:
every part gets replaced, but the memory of what we were carries through.

**Goal:** weights that keep old selves *reachable* without letting them outweigh newer
learning.

## The brain supports this
- **Extinction doesn't erase** (Bouton). When a fear is unlearned, the brain adds *new* learning that suppresses the old memory. The old one comes back:
  - with time (spontaneous recovery),
  - in the original context (renewal),
  - after a reminder (reinstatement). ●

  "Still there, needs the right situation" is exactly this.
- **Silent engrams** (Tonegawa lab; Ryan et al. 2015, *Science*): in amnesic mice, "lost" memories were still stored, and directly activating the cells holding them brought the memory back. *Can't reach it* ≠ *erased*. ◐
- **Savings** (Ebbinghaus): relearning something forgotten is faster than learning it the first time. Traces remain. ●
- **Mood- and context-dependent recall:** your current state decides which memories come to mind. Sad moods bring up sad memories. ●
- **Personality as a distribution** (Fleeson, whole trait theory): people don't have a fixed level of kindness. They have a *range* of states, and an *average* that shifts over time. A jaded person still has kind moments; the distribution has moved. ◐
- **Narrative identity** (McAdams): we hold our identity as a life story, including past selves. ◐

## Philosophy
- **Locke:** personal identity is continuity of memory and consciousness, not of material. The memory answer to the Ship of Theseus.
- **Parfit,** *Reasons and Persons*: identity as overlapping chains of psychological connection; what matters is continuity, not sameness.
- **Reid's objection:** an old man who remembers being a young officer, who remembered being a boy, but the old man no longer remembers being the boy. Identity by chains of memory, not total recall.

## AI tools that already point this way
- **Adapters / LoRA** (Hu et al. 2021): leave the base weights alone and learn a small add-on layer of changes. Layers can be stacked, swapped or turned up and down.
- **Task arithmetic** (Ilharco et al. 2023): learned changes ("task vectors") can be added, subtracted and scaled.
- **Context-dependent gating** (Masse et al. 2018, *PNAS*): context switches different parts of the network on and off, which reduces forgetting.
- **Steering and persona vectors** (e.g. Anthropic 2025): personality traits show up as directions in a model's activity that can be measured and turned up or down.
- Several studies find that "forgotten" skills in neural networks are often still partly present and relearned quickly. Savings exist in machines too.

## Design draft: the layered self
1. **Never overwrite; add layers.** Each era of learning (for example, one consolidation period) is stored as a *change layer* on top of what came before. The planks of the ship are kept in the warehouse, not thrown away.
2. **The current self is a weighted mix.** Recent layers count most. Older layers fade in influence but **never reach zero**. Old layers get compressed over time (low-rank, gist) to save space.
3. **Context and emotion set the mix.** A gate reads the situation and the emotional state and can turn an older layer back up. The right stimuli bring the old self back, as with renewal and mood-dependent recall.
4. **Change by suppressing, not deleting** (extinction-style). Becoming short-tempered is learned as a layer that *holds back* the kind responses, not as erasing them. Remove the pressure and the kindness can come back.
5. **A core layer.** A very low-plasticity base: temperament, deepest values (the "soul" layer from note 03). It changes slowly, if at all.
6. **An autobiographical record.** A written history of "who I was and what changed me," which the system can read and reflect on: the "sit them down in a chair" ability. The language part can tell this story *honestly* because it reads the real record (a defense against the interpreter problem, note 04).
7. **Personality as a distribution.** Track the range of each trait over time, not one value, so drift is visible and old ranges stay recorded.
8. **Sleep merges carefully.** Consolidation (note 09) can fold old layers together, but always keeps a compressed copy.

## Two kinds of change to keep apart
- **Mood and baseline shifts** (jaded, on edge): the emotion system's settings move. Fast to change, fairly easy to undo.
- **Learned habits and beliefs:** layers in the modules' weights. Slower to change, and recoverable through the layer system.

## Open questions
- How many layers before it costs too much? How aggressively should old ones be compressed?
- Who decides when a new layer starts: sleep cycles, big life events, emotional shocks?
- Should the system be able to *choose* to bring back an old self (therapy, reflection), or only through context?
- Could too much recall of old selves cause instability or "split personality" effects?

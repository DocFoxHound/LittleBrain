# 04: The modular brain

## The argument
An LLM is one network trained on one objective. A brain is many parts with different
jobs, different ways of learning and different speeds, all reacting to the same moment
and then combining. Training one part on everything gets surprisingly far, maybe even
to some form of consciousness, but it would be the consciousness of that one part, not
of a whole being.

Emotion probably isn't one part's job. It looks like deep structures plus global
chemical signals: one region over- or under-reacting, with the effect spreading to all
the others.

## Evidence from the brain
- **Split brain** (Sperry, Gazzaniga): either half can perceive and act on its own.
- **The interpreter:** the left (language) hemisphere confidently makes up reasons for actions the right hemisphere chose.
  - **Design warning:** an LLM placed on top of a decision brain *will* become the interpreter. Pass it the real reasons.
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995): you need a fast episodic learner (hippocampus) and a slow general learner (cortex), because one network can't do both without overwriting what it knows.
- **Basal ganglia as the answer to the selection problem** (Redgrave, Prescott & Gurney 1999): choosing between competing actions or tools.
- **Doya (1999)**, *What are the computations of the cerebellum, the basal ganglia and the cerebral cortex?*:
  - cortex learns **without labels** (finds patterns on its own)
  - basal ganglia learn from **reward**
  - cerebellum learns from **error correction** (compares what it predicted to what happened)
- **Doya (2002)**, *Metalearning and neuromodulation*: the brain's chemical messengers act as global settings:
  - dopamine = surprise about reward
  - serotonin = how far ahead to care
  - noradrenaline = how random or exploratory to be
  - acetylcholine = how fast to learn

  This is a concrete version of "one organ's reaction spreads to the rest."

## Modular AI research
- **LeCun (2022), *A Path Towards Autonomous Machine Intelligence*:** perception, world model, short-term memory, actor, cost (built-in drives), configurator. The closest existing design to this project.
- **Shared workspace:** Goyal, Bengio et al. 2021; VanRullen & Kanai 2021. Modules compete to write to a small shared space that all of them read.
- **Recurrent Independent Mechanisms** (Goyal, Bengio): modules that wake up only when relevant.
- **Older theory:** Minsky, *Society of Mind*; Fodor, *Modularity of Mind*; Hawkins, *A Thousand Brains*.
- **Not the same idea:** Mixture of Experts in LLMs. Its "experts" are chosen per word fragment, not real specialists.
- **The argument against:** Sutton, *The Bitter Lesson*. Hand-built structure tends to lose to one big system trained on more data.
- **Likely compromise:** design the *structure* by hand (which modules, each one's learning goal and speed, how they connect) and let the *content* be learned.

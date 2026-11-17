# 05: Brain map

What each part of the brain is generally understood to do, how it learns, and what kind
of small model could stand in for it. Neuroscience is messier than any table: regions
overlap, cooperate and argue. Treat this as a starting point.

## Old and deep: staying alive, drives, arousal

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Brainstem** (reticular formation) | wake/sleep, arousal, breathing, reflexes | mostly built in | global "arousal" level; hard-coded reflexes |
| **Hypothalamus** | drives and keeping the body in balance: hunger, thirst, temperature, sex, stress hormones | mostly built in, can be tuned | a vector of needs; rewards for keeping needs in balance (homeostatic RL) |
| **Chemical messenger centers** (dopamine, serotonin, noradrenaline, acetylcholine) | broadcast global signals to the whole brain | built in, adapts over time | global settings for every module (Doya 2002): surprise about reward, how far ahead to plan, how much to explore, how fast to learn |

## Emotional and evaluative

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Amygdala** | threat and importance; tags memories as emotionally significant | fast association (fear conditioning) | small "importance / threat" classifier; its score decides how strongly a memory is stored |
| **Insula** | sense of the body's internal state; disgust; "how do I feel" | learns predictions of body signals | encoder for internal state (needs, energy, damage) |
| **Anterior cingulate** | notices conflict and errors; judges whether effort is worth it | learns from errors | monitor that decides when to stop and think harder |
| **Orbitofrontal / ventromedial prefrontal** | values options; brings emotion into choices (Damasio's somatic markers) | learns from reward | critic or value function |

## Memory

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Hippocampus** | episodic memory, one-shot learning, spatial maps (place cells), replays memories during sleep | **fast**, from single events | episodic store (nearest-neighbor or vector database), replay buffer, Neural Episodic Control |
| **Neocortex** (in general) | slow, general knowledge; concepts | **slow**, without labels | small networks trained from hippocampal replay ("sleep") |

## Perception and space

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Occipital lobe** | vision | without labels | small image encoder (CNN or autoencoder) |
| **Temporal lobe** | hearing, recognizing objects, meaning | without labels | object and meaning encoder |
| **Parietal lobe** | touch, spatial layout, attention, where things are | without labels | spatial map / attention module |

## Action

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Basal ganglia** (striatum) | choosing actions, habits, picking between competitors | **from reward** (dopamine) | actor-critic policy; the selector that gates which action or tool wins |
| **Cerebellum** | timing, smooth movement, predicting the results of actions and correcting them | **from error correction** | forward model; motor controller |
| **Motor cortex** | carrying out movements | mixed | action output head |

## Executive, language and self

| Region | What it does | How it learns | Possible stand-in |
|---|---|---|---|
| **Prefrontal cortex** | planning, working memory, goals, holding back impulses | slow, from reward and context | planner (MCTS / world-model rollouts) plus a working-memory buffer |
| **Broca's and Wernicke's areas** | producing and understanding language | slow, without labels | a small LLM, possibly frozen and pre-trained |
| **Default mode network** | daydreaming, imagination, thinking about self and others | — | offline "imagination" rollouts; a self-model |

## Wiring and coordination

| Region | What it does | Possible stand-in |
|---|---|---|
| **Thalamus** | relays and gates information; directs attention | router / gate into the shared workspace |
| **Corpus callosum** | connects the two hemispheres | communication channel (how much bandwidth each part gets) |
| **Sleep** | consolidates memory: hippocampus replays to cortex | offline training phase |

## First sketch of a framework

```
             ┌──────── global signals (arousal, chemical-messenger settings) ────────┐
             ↓                                                                       ↓
World → Perception encoders ─→ Shared workspace (thalamus-like gate) ←─→ Episodic memory
             ↑                     ↑        ↓                                (amygdala tags importance)
   Internal needs (insula,         │   Evaluators: value, threat, conflict
   hypothalamus) ──────────────────┘        ↓
                               Selector (basal ganglia) → action | planner | LLM (speech)
                                            ↓
                              Forward model (cerebellum) predicts the result → error signals
```

**Learning together:** each module keeps its *own* learning goal and its *own* learning
speed. They share experience through the workspace, and the global signals set how they
all behave in the moment. Sleep phases move memories from the fast store into the slow
modules.

**Start small:** a MiniGrid or Crafter world with needs (energy, safety), a perception
encoder, an episodic memory, an importance tagger, a value critic and a selector. Add
language last.

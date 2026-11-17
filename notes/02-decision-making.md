# 02: How AIs make decisions

## Reinforcement learning (RL)
An agent observes a **state**, takes an **action**, gets a **reward**, and learns a **policy** (situation → action).
- Milestones: DQN (Atari), AlphaZero (Go and chess), MuZero (plans using a world model it learned itself).
- **POMDP:** the agent doesn't see the whole world, so it keeps a *belief* about it. This is the formal home of "unknown."

## World models: imagining "what if I do X"
- Ha & Schmidhuber, *World Models* (2018); the Dreamer series (V1–V3); LeCun's JEPA.
- Genie (DeepMind) and DIAMOND (a diffusion-based world model of games).

## Diffusion for decisions
A diffusion model generates rather than chooses, but it can generate plans or actions.
- Diffuser (Janner et al. 2022); Diffusion Policy (Chi et al. 2023, robotics).
- Decision Transformer: decisions treated as predicting the next action in a sequence.

## LLM agents with memory
- **CoALA** (2023): working, episodic, semantic and procedural memory, plus a decision loop around an LLM.
- **Generative Agents** (Park et al. 2023): a memory stream retrieved by recency, importance and relevance.
- **Voyager** (Minecraft skill library); ReAct, Reflexion.
- Older cognitive architectures: SOAR, ACT-R. Also **case-based reasoning** (reusing past scenarios).

## Classic game AI
Behavior trees, utility AI, GOAP, MCTS. Cheap, explainable, and often enough in simple worlds.

## Framing decision
The brain is the **controller**; the LLM is a **tool** it uses for language, not a preprocessor sitting in front of it.

```
World → Perception → State + uncertainty
                       ↓
     Episodic memory ⇄ World model (imagine outcomes)
                       ↓
          Planner / policy (RL, MCTS, diffusion)
                       ↓
        Action in world   or   instructions → LLM
```

## Places to experiment
MiniGrid (2D grid), Crafter (2D, Minecraft-like), NetHack, Unity ML-Agents (3D).

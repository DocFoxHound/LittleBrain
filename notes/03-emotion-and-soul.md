# 03: Emotion and soul

## Emotion is needed for decisions
- **Damasio, *Descartes' Error*:** patients with damage to emotion-related areas could still reason but couldn't decide. "Somatic markers" tag options as good or bad before any reasoning happens.
- **Picard, *Affective Computing* (1997):** founded the field.

## Computational models of emotion
- **Appraisal:** emotion comes from evaluating an event against your goals.
  - OCC model (22 emotions); EMA (Gratch & Marsella); FAtiMA.
- **Dimensional:** PAD (pleasure, arousal, dominance); Russell's circumplex (valence × arousal). A few numbers that can adjust a decision system.
- **Emotion in RL:** Moerland et al. 2018 survey; curiosity and intrinsic motivation (Pathak et al. 2017); homeostatic RL (Keramati & Gutkin).
- **Architectures:** MicroPsi (Bach; Dörner's Psi theory), where needs → urges → changes in how it thinks; LIDA; Soar's emotion extension; Sloman's CogAff.
- **Games:** *Creatures* (Steve Grand) simulated hormones and biochemistry; *The Sims* uses needs-based utility AI.
- **Deep-brain emotion systems:** Jaak Panksepp's seven primary systems: SEEKING, RAGE, FEAR, LUST, CARE, PANIC/GRIEF, PLAY.
- **The opposing view:** Lisa Feldman Barrett, *How Emotions Are Made*. Emotions are built by prediction from body signals, not stored in dedicated organs.

## Soul
- **Aristotle:** the soul is the organizing form or function of a living thing. A functionalist definition, the friendliest one for engineers.
- **Consciousness theories:** Global Workspace (Baars); Integrated Information Theory (Tononi); Butlin, Long et al. 2023 (checks AI against these theories).
- **Self-models:** Hofstadter, *I Am a Strange Loop*; Metzinger, *Being No One*; Minsky, *The Emotion Machine*.
- **Hard problem (Chalmers):** you can simulate everything a soul does and still not know whether anything is felt.
- Working position: everyone has a soul, regardless of how much brain they have left. A soul isn't the same thing as the machinery, but the machinery may be what lets it act in the world.

## Proposed layers

| Layer | Timescale | Role |
|---|---|---|
| "Soul": values and personality | almost fixed | what matters; adjusts how events are appraised |
| Mood | minutes to days | slow average of emotions; biases everything |
| Emotion | moments | fast appraisal: valence and arousal |
| Decision | each step | action or tool choice |

Emotion acting on decisions: arousal → urgency; fear → risk aversion; curiosity → exploration;
valence ← "better or worse than expected." **Emotional intensity decides what gets remembered.**

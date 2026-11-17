# 08: Learning while acting (continual learning)

## The problem
Brains learn and act at the same time, for their whole lives, cheaply (about 20 watts).
Standard deep learning doesn't:
- **Training and use are separate phases.** A model is trained, frozen, then deployed.
- **Backpropagation is global and costly.** Every weight waits for an error signal sent back from the very end, and the system must store its intermediate activity to compute it.
- **It assumes shuffled data.** Real life arrives in order and in streaks, so new learning overwrites old learning (**catastrophic forgetting**).
- **It loses the ability to learn.** Networks trained for a long time on changing tasks gradually lose plasticity (Dohare et al. 2024, *Nature*).
- The underlying tension is the **stability–plasticity dilemma**: changeable enough to learn, stable enough to remember.

## What brains seem to do differently
- **Local learning:** each connection changes based on what's happening right at it, plus a few broadcast signals (the chemical messengers). No global backpropagation.
- **Sparse:** only a small fraction of neurons are active at once, so learning touches only a few connections.
- **Gated by surprise and importance:** dopamine, noradrenaline and acetylcholine decide *when* and *how much* to learn.
- **Several timescales per connection:** changes start fragile and short-lived, and only some become permanent ("synaptic tagging and capture," Frey & Morris 1997; multi-timescale synapse models, Fusi et al. 2005, Benna & Fusi 2016).
- **Fast store plus slow store** (Complementary Learning Systems), joined by offline consolidation. → parked; see "Dreaming" in the README.
- **New units:** some new neurons are born in the hippocampus, and unused connections are pruned and regrown.

## Existing AI work to know
| Approach | Idea | Examples |
|---|---|---|
| Protect important weights | slow down changes to weights that mattered for old skills | EWC (Kirkpatrick et al. 2017), Synaptic Intelligence (Zenke et al. 2017) |
| Replay | mix old experience back in | experience replay; generative replay (Shin et al. 2017) |
| Grow the network | add new capacity for new skills | Progressive Networks (Rusu et al. 2016) |
| Keep plasticity alive | reset units that stopped being useful | Continual Backprop (Dohare et al. 2024) |
| Fast weights | a second, quickly changing and quickly fading set of weights | Ba, Hinton et al. 2016 |
| Local learning rules | learn without global backpropagation | Forward-Forward (Hinton 2022), e-prop (Bellec et al. 2020), predictive coding networks, Hebbian rules |
| Brain-inspired hardware | event-driven, always learning | spiking networks, Intel Loihi |

## Starting hypotheses (our own; to test)
1. **Learn from surprise only.** Most steps just *act*. A module updates only when its prediction was wrong by enough. This is cheap, and it means learning and acting can happen in the same step.
2. **Two weights per connection.** A *fast* weight that learns immediately and fades, and a *slow* weight that changes only when the fast one keeps being useful, or when a strong "this matters" signal arrives (tag and capture).
3. **No backpropagation across modules.** Each module learns only from signals at its own edges: its inputs, its own prediction error, and the broadcast signals. Modules exchange *predictions and errors*, not gradients. This keeps modules independent and the system always responsive.
4. **Emotion controls plasticity.** The global signals (surprise, arousal, "learn this now") set every module's learning rate in the moment. Emotion doesn't only shape *what you do*; it shapes *what you become*.
5. **Each connection has its own plasticity budget.** Plasticity drops for connections that are used and important (a critical period closing), and can be reopened by strong novelty, as when an old memory is recalled and becomes changeable again (reconsolidation).
6. **Recycle dead capacity.** Units that stop contributing are reset and become available again (neurogenesis / continual backprop).
7. **Separate fast learning from slow learning.** The episodic store learns instantly while awake; slow modules learn mostly offline from replay. → parked with dreaming.

## First experiment idea
A small world that **changes over time**: rules shift, new objects appear, old ones return.
Measure:
- **forgetting:** does it still do the old things?
- **plasticity:** can it still learn new things after a long run?
- **compute per step:** is it cheap enough to run live?

Compare plain online backpropagation against hypotheses 1 + 2 + 4 combined.

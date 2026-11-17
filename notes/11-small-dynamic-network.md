# 11: A small dynamic network: brain-like neurons on a fixed budget

## Deliberate restoration (design position)
The emotional response still fires (a sudden loud bang is frightening), but **how to act on it is a
choice**. Choosing is hard, but possible: deciding to be the version of yourself that's
present for the people who need you.

**Design consequence:** separate the **response** from the **action**.
- The emotion system (amygdala-like) fires automatically. It isn't suppressed or deleted.
- The executive (prefrontal-like) can choose which *self-state* to act from, including an
  earlier one, and can override the automatic response.
- **Choosing costs effort** (anterior cingulate: effort versus value). That cost is why it
  doesn't happen "willy-nilly." The system needs a reason, either internal or from the
  environment, worth paying for.
- Whether it develops this ability may be **emergent**; the architecture only has to make it possible.

## The constraint
- 8–16 GB of graphics memory (VRAM). Small models, narrow and targeted tests.
- **No growing the network forever.** Like the brain: a roughly fixed number of neurons,
  **repurposed** without completely destroying old connections.

## Encouraging scale check
- The adult fruit fly brain: about **140,000 neurons and ~50 million synapses** (FlyWire connectome, 2024).
  Flies learn, remember, sleep, and use dopamine to gate learning.
- At about 10 bytes per synapse (see below), a fly-sized brain is roughly **500 MB**. It fits on an 8 GB card.
- **The real bottleneck is speed, not memory.** Graphics cards are fast at big dense blocks of math
  and slow at scattered, irregular connections. Keep connectivity in blocks, or use dense matrices
  plus masks at small sizes.

## Biological template: the fly mushroom body
The fly's learning center is a small, well-mapped version of almost everything we want:
- about 2,000 **Kenyon cells** with **sparse** coding (only a few active per smell);
- dopamine neurons split into **compartments**, each with **its own learning rate and memory
  lifespan**: some learn fast and forget fast, others learn slowly and keep memories long
  (Aso & Rubin 2016, *eLife*);
- both short- and long-term memory in the same circuit.

It's worth studying as the first blueprint.

## The synapse: where most of the design lives
Each connection stores a small record instead of one number:

| Field | Size | Purpose |
|---|---|---|
| target neuron | 2–4 B | sparse connections that can be rewired |
| **fast weight** | 1 B (int8) | learns immediately, fades quickly ("fire and forget") |
| **cascade weights** (3 levels) | 3 B | progressively slower hidden copies of the value; the deep levels hold *old* values |
| tag / stability | 1 B | "this mattered" tag from emotion; how consolidated it is |
| age / usage | 1 B | for pruning and recycling |

About **8–10 bytes per synapse.**

### Cascade synapse: how old selves stay without new layers
Based on Fusi, Drew & Abbott (2005) and Benna & Fusi (2016):
- The effective weight comes mostly from the shallow levels.
- Each level slowly leaks into the next deeper one, and deeper ones slowly pull the shallow ones back.
- New learning changes the surface quickly. **The deep levels still remember the old value.**
- Remove the pressure, or recall the old context, and the surface drifts back toward the old value.
  That is spontaneous recovery and renewal, built into every connection.
- **The layered personality lives *inside* each synapse**, so nothing new is added.

## Neuron types as settings, not separate models
All neurons share one structure; their **type is a small set of parameters**:

| Type | Plasticity | Memory span | Connections | Role |
|---|---|---|---|---|
| **Core / anchor** | very low | very long | moderate | temperament, deep values; hard to break |
| **Workhorse** | medium, cascade | long | many | skills and knowledge |
| **Branching hub** | medium | long | very many, across modules | links distant parts together |
| **Transient** | very high | seconds to minutes | few, quickly rewired | scratch work; reassigned constantly |
| **Inhibitory** | high | medium | local | **suppression**: change by holding back, not erasing (extinction) |
| **Modulatory** | — | — | broadcast | emotion and global signals; set everyone's learning rate |

A neuron's plasticity can also change with **age** (critical periods) and be reopened by strong
novelty plus emotional tagging.

## Rewiring on a fixed budget (structural plasticity)
- Every neuron has a **fixed number of connection slots.**
- Regularly (mostly during sleep):
  - **prune** connections that are weak, unused **and** not consolidated;
  - **regrow** connections into the freed slots, to new targets (random, or toward active neurons).
- Early life: high prune and regrow rate (lots of new paths). Later: low.
- Consolidated connections are protected; their deep cascade levels survive even if the surface weakens.
- AI precedent: SET (Mocanu et al. 2018) and RigL (Evci et al. 2020) prune and regrow during training with a fixed parameter count.

## Cheap "layers" when you do want them
- **Binary masks** (PackNet, Mallya & Lazebnik 2018; Supermasks in Superposition, Wortsman et al. 2020):
  a past self is stored as a 1-bit-per-connection mask over the *same* weights. That's 1/32 the
  size of a full-precision copy.
- **Low-rank change layers:** an old era stored as a tiny rank-4 add-on; merge old ones into a single "past self" over time.
- **Context-dependent gating** (Masse et al. 2018): a context vector picks which neurons are active, so the same network holds several modes.

## Learning rule (local, three factors)
For each synapse, each step:

```
eligibility  = trace of (pre activity × post activity or error)     # what just happened here
modulator    = global signal from emotion (surprise, arousal, "learn now")
fast_weight += type.lr × modulator × eligibility
fast_weight *= type.fast_decay
cascade: each level moves a little toward its neighbors
if strong emotion: tag = high  →  during sleep, fast is absorbed into cascade level 1
effective weight = fast + level_1 (+ small pull from deeper levels)
```

No backpropagation across modules. The modulator is where emotion controls learning (note 08).

## Memory budget example
| Scale | Neurons | Synapses | Memory at ~10 B each |
|---|---|---|---|
| Tiny test | 5,000 | 500,000 | ~5 MB |
| Fly mushroom body scale | 20,000 | 2 M | ~20 MB |
| Small multi-module brain | 150,000 | 50 M | ~500 MB |
| Upper practical limit (8 GB card) | ~1 M | ~400 M | ~4 GB (leaves room for the world and the rest) |

## Open questions
- How deep should the cascade be? Each extra level is one more byte and one more, slower memory span.
- How fast should deep levels pull the surface back? Too fast and the system can't change; too slow and old selves are lost.
- Can the *executive* intentionally boost a deep layer (deliberate restoration), and how is that effort paid for?
- Should neuron types be assigned at birth, or should neurons drift between types?

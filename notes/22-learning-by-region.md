# 22: Learning by region, and dendritic clustering (roadmap step 6)

Code: `prototype/brain/cells.py` (mechanisms), `prototype/brain/celltypes.py` (who uses what),
`prototype/neuron.py` (new rule fields). Test: `PYTHONPATH=prototype python3 prototype/brain/test_learning_by_region.py`: **5/5 pass**.
All earlier tests still pass (12 + 19 cell tests, learning rules v1/v2).

## New mechanisms (from dendrite biology)
| Mechanism | Biology | What it does in the model |
|---|---|---|
| **Branch gating by context** | SST cells can silence single branches; different memories sit on different branches (Cichon & Gan 2015) | a closed branch neither drives the cell nor learns |
| **Branch strength plasticity** | a branch that spikes before a learning signal becomes more excitable as a whole (Losonczy et al. 2008) | per-branch gain, learned |
| **Clustered plasticity** | plasticity proteins are shared along a branch; branches with more co-active synapses capture more (Govindarajan et al. 2006) | synapses learning together on one branch amplify each other |

## Learning rules by region
| Region / cell | Rule | Biology |
|---|---|---|
| Cortex pyramids | v2 three-factor + cascade (unchanged) | note 16 |
| **CA3** | fast (lr ×2.7) + clustering + branch strength | rapid autoassociative storage |
| **CA1** | one-shot plateau learning (BTSP), even without firing | Bittner et al. 2017 (note 21) |
| **Basolateral amygdala** | fast, and **gated by arousal** (noradrenaline) | fear conditioning in one or a few trials, only when it matters |
| **Striatum (MSNs)** | dopamine-signed (D1 +, D2 −), **long eligibility trace** | dopamine can arrive seconds after the action (Yagishita et al. 2014) |
| **Purkinje cells** | supervised: climbing-fiber error → LTD | note 21 |
| **Thalamic relay** | slow (lr 0.03) | relay synapses are comparatively stable |
| PV inhibition | Vogels inhibitory plasticity | note 18 |

## Results
| Test | Result |
|---|---|
| **One cell holds two eras of life** (A good in era 1 on branches 1–4, A bad in era 2 on branches 5–8) | **era-1 context: 0.95 · era-2 context: 0.05 · without branch gating: 0.09** (old self overwritten) |
| Clustered plasticity | share of learning on the top 2 of 8 branches 0.41 → 0.47 (modest); branch-strength spread 0 → 0.05 |
| Amygdala: 3 pairings | calm +0.12 · aroused **+0.77** |
| Striatum: reward 4 steps late | striatal trace learns **13× more** than a cortical-length trace |
| CA3: one trial | 15.2 vs 1.7 potentiation per cell (cortex) |

## Why the first result matters
It's **layered personality** (note 10) at the level of a single neuron: change without erasing, with
context choosing which self answers. Note 10 proposed it as an idea; here it's working machinery, using a
mechanism real neurons have (SST cells gating individual dendritic branches).

## Open
- Who sets the branch gates? Here the test set them by hand. In the brain, SST/VIP circuits driven by
  context (prefrontal top-down, acetylcholine) would. Wiring that is part of step 7.
- Clustering is weak with our rate model; spiking models show stronger effects.

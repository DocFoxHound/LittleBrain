# 13: Our neuron: the workhorse (pyramidal) template

Based on the human **pyramidal neuron**, using lessons from the fly's Kenyon cell.
Code: [`prototype/neuron.py`](../prototype/neuron.py). Demo: [`prototype/demo_single_neuron.py`](../prototype/demo_single_neuron.py).

## Why the pyramidal neuron is the template
- It's the most common neuron in the human cortex and hippocampus.
- Its **dendritic branches** act as separate compartments. That gives us the mushroom
  body's "compartments with their own rules" inside a single human-style cell.
- Every other type in the catalog (note 12) is the same design with different settings:
  core = locked down; transient = fast-only; inhibitory = negative output; hub = more branches.

## Anatomy

```
                    apical branch  (context / "which self")  ── top-down input
                          │  multiplies the cell's gain
        ┌─────────────────┴─────────────────┐
        │               SOMA                │ → output (firing rate 0..1)
        │ adaptive threshold (homeostasis)  │
        └──┬──────┬──────┬──────┬───────────┘
        basal  basal  basal  basal ...  (feedforward input)
        branch branch branch branch
          │ each branch: N synapse slots + its own nonlinearity
          │ each branch can be gated (silenced) by SST-like inhibition
```

## Parts and what each one does

### Synapse (one slot on a branch)
| Field | Meaning |
|---|---|
| `src` | which input it listens to (rewirable) |
| `fast` | instant learning; fades within minutes ("fire and forget") |
| `cascade[0..L-1]` | hidden copies at slower and slower timescales. `cascade[0]` is the surface; deeper levels hold **old values** and pull the surface back over time |
| `elig` | eligibility trace: "I was just involved." Fades over seconds |
| `tag` | "this mattered," set by emotion (dopamine × arousal). Decides what sleep keeps |
| `usage` | how often it has contributed; used for pruning |

**Effective weight = `fast + cascade[0]`.**

### Branch
- Sums its synapses' input. Produces a **dendritic spike** only if enough *clustered* input arrives at once (a threshold nonlinearity). This makes each branch a small pattern detector.
- **Gate:** inhibition can silence a branch. A silenced branch neither fires nor learns, which **protects what it stores**.

### Soma
- Adds up the branch outputs and applies the apical (context) gain.
- **Adaptive threshold:** if it fires too often, the threshold rises; too rarely, it falls. This keeps the cell sparse (the Kenyon-cell lesson). At population level, inhibitory neurons enforce the sparsity.

### Apical branch (context)
- Receives top-down signals: current goal, self-state, context.
- Multiplies the cell's response rather than adding to it, so context can **select** which learned behavior comes out. This is the hook for **deliberate restoration**: the executive raises the context that favors an older self.

## Learning (three factors, all local)
Each step:
1. `elig = elig × decay + (input × branch activity × output)`: what just happened here.
2. Modulators arrive from the emotion system:
   - **dopamine** (−1..+1): better or worse than expected → direction of change
   - **acetylcholine** (0..1): "learn this now" → how big the change is
   - **arousal / noradrenaline** (0..1): how strongly to tag it as important
3. `fast += lr × plasticity × acetylcholine × dopamine × elig`, only on open (ungated) branches.
4. `tag += arousal × |dopamine| × elig`
5. **One-shot (BTSP-like):** if dopamine × acetylcholine is very strong, the change is written straight into `cascade[0]` as well.
6. `fast` fades. The cascade flows a little between levels.

## Sleep (per neuron)
1. **Consolidate:** tagged fast weights move into `cascade[0]`; fast weights are cleared.
2. **Run the cascade:** many steps of flow between levels, so surface changes sink deeper over nights.
3. **Downscale:** shrink `cascade[0]` slightly (keeping the strongest).
4. **Prune and regrow:** synapses that are weak at every level, rarely used and untagged are reassigned to new inputs.
5. Tags fade.

## Development
- `plasticity` starts high (infancy) and falls with age toward a floor. It never reaches zero.
- Strong novelty plus arousal can briefly raise it again (reopening a critical period).

## Memory cost (as written in the prototype)
8 branches × 32 slots = 256 synapses → **9,248 bytes per neuron** (about 36 bytes per synapse,
unoptimized: float32 values and 64-bit input indices). Packed into int8 with 16-bit indices,
that becomes about 10 bytes per synapse. Optimization comes later; clarity first.

## What the demo tests
Train the neuron to like pattern A for 6 days → dislike A for 2 days (life turns harsh) →
6 days of ordinary life with no A → relearn. Compare against a "plain" neuron with no deep cascade.

## First results (2026-10-27, `python3 prototype/demo_single_neuron.py`)

| | Workhorse (3 cascade levels) | Plain (no deep memory) |
|---|---|---|
| Start | 0.14 | 0.14 |
| After childhood (A good) | 1.00 | 1.00 |
| After hard times (A bad) | 0.03 | 0.03 |
| Day 1 after | 0.03 | 0.03 |
| Day 3 after | 0.09 | 0.03 |
| Day 6 after | **0.29** | 0.03 |
| Trials to like A again | **1** | 16 |

(Response to A, 0 to 1. A brand-new neuron needs 2 trials, because A is a strong, clean pattern.)

- **Spontaneous recovery:** yes. The old liking slowly returns with time, partially, not all at once. Across seeds 1–4 the day-6 recovery was 0.10–0.29; the plain neuron stayed at 0.03.
- **Savings:** yes. 1–2 trials to relearn versus 16–17 for the plain neuron.
- **Caveat:** with seed 4 the plain neuron never relearned within 400 trials. That's a test artifact, not a cascade advantage: relearning happens within a single waking period, where fast weights hit a ceiling and the homeostatic threshold rises. It would need a night's sleep to consolidate. The test should include sleep during relearning.
- **Caveat:** a fresh neuron learns A in 2 trials, so this test is too easy to show savings against a *new* neuron. Next test: a harder pattern, and overlapping patterns (A vs a similar A′).
- Settings were tuned by a small sweep (cascade coupling 0.005, 100 sleep flow steps, threshold 6.0, soma slope 0.5). The behavior depends on them. Stronger coupling made the old self come back **within a day** (too stubborn to change); weaker coupling, barely at all.

## Next steps
- ✅ Shrink memory: done, 9,248 → 1,041 bytes per neuron (note 14). Next: a GPU kernel so computing doesn't need big temporary arrays.
- Test context (apical) restoration: can a context signal bring the old self back *on demand*, and at what effort?
- Test branch gating: store two eras on different branches; let SST-like gating choose.
- Give the threshold and plasticity proper developmental schedules.
- Build the other types from the presets and test each.
- Then: a small population (Kenyon-cell scale) with an inhibitory neuron enforcing sparsity.

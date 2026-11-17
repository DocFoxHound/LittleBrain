# 09: Sleep: the brain's maintenance and learning phase

## The theory (the user's)
AI doesn't sleep. Maybe it needs to.

- The system cycles between **wake** (acting, collecting short-term memory) and **sleep** (a lower state focused on learning). For example, ¾ wake and ¼ sleep, though the timescales will vary.
- During sleep it takes the short-term memory built up while awake and:
  - **trains its models** on it;
  - **keeps the important parts** in long-term memory, which can be used to train or reinforce the brain later;
  - **summarizes the rest** into medium- or short-term memory;
  - **throws away the junk.**
- Medium- and short-term memories can **carry over** into the next wake cycle. If they're reinforced there, they gain a higher grade and a bigger influence on learning, and can eventually be promoted to long-term memory.

## How this matches the neuroscience
- **Consolidation:** during deep (non-REM) sleep, the hippocampus replays the day's events in fast bursts, and the cortex gradually absorbs them. ●
- **Downscaling** (Tononi & Cirelli's synaptic homeostasis hypothesis): waking strengthens many connections; sleep weakens them across the board, so only the strongest survive. That frees capacity and restores the ability to learn. This is the "get rid of the junk" step. ◐
- **Emotional and reward tags pick what gets replayed:** memories tagged as emotional or rewarding are preferentially consolidated. ◐
- **REM sleep** may recombine memories in new ways and dull their emotional sting while keeping the information ("overnight therapy," Walker). ○
- **Physical cleaning:** the glymphatic system flushes waste from the brain during sleep. ◐
- **Sleep pressure** (Borbély's two-process model): need for sleep builds the longer you're awake (adenosine), on top of a daily clock (circadian rhythm). ●
- **Without sleep:** learning, emotional control and memory all get worse. ●

## How it matches AI research
These pieces exist separately:
- Hinton's **wake-sleep algorithm** (1995)
- **Prioritized experience replay** (Schaul et al. 2015): replay the most surprising experiences more often
- **Generative replay**; Dreamer's training inside its own imagined world
- **Sleep-like replay reduces catastrophic forgetting** (Tadros et al. 2022, *Nature Communications*)
- LLM agent memory tiers: MemGPT; "reflection" in Generative Agents

The combination below (graded tiers, emotion-weighted, carried across cycles, promoted when reinforced) is our own design.

## Design draft

### Memory tiers
| Tier | What it holds | How long | Brain analogy |
|---|---|---|---|
| **Short-term** | raw episodes from today, every detail | until the next sleep; most is cleared | hippocampus, recent |
| **Medium-term** | summaries and candidate memories, with a grade | a few cycles; fades unless reinforced | hippocampus, older traces |
| **Long-term (records)** | a small set of key episodes kept in detail, plus gist summaries | long; slow to fade | well-consolidated memories |
| **Long-term (in the weights)** | skills and knowledge trained into the modules | permanent-ish | cortex, cerebellum, basal ganglia |

Long-term memory comes in two forms: things you can *recall* (records) and things you just *know or can do* (in the weights). Sleep moves memories from records into weights.

### Grading a memory
Grade = a combination of:
- **emotional intensity** at the time (the amygdala tag)
- **surprise** (how wrong the prediction was)
- **reward or harm** (relevance to needs and goals)
- **reinforcement** (how often it recurred or was recalled; spaced repetition)
- **novelty**

Each sleep, every grade **fades a little** unless the memory was reinforced while awake.
Memories above a threshold are promoted. Memories below the floor are compressed to a gist or dropped.

### What happens during one sleep
1. **Deep-sleep stage: replay and train**
   - Replay graded short-term episodes, the highest grades most often.
   - **Mix in old long-term memories** so new learning doesn't overwrite old (this interleaving is the key defense against catastrophic forgetting).
   - Slow modules update their weights. Fast weights are absorbed and cleared.
2. **Downscaling: maintenance**
   - Weaken all recent changes slightly; only the strongest survive.
   - Reset dead units (note 08, hypothesis 6). Restore plasticity.
3. **Dream stage: recombine and recalibrate** (REM-like)
   - The world model generates *new* scenarios by mixing memories. This is where a diffusion-style generative model fits.
   - Train on those imagined scenarios: rehearsal, "what if," new connections.
   - Replay emotional memories at **low arousal**, so the lesson stays and the alarm fades.
4. **Sort and file**
   - Promote, summarize, compress or drop memories. Update grades. Carry medium-term memories into the next day.

### When to sleep
Not a fixed clock: **sleep pressure** builds from
- how full the short-term buffer is,
- how loaded the fast weights are,
- rising prediction errors and slower responses (fatigue),

combined with a daily rhythm. The system can also take short "naps" or replay while resting during the day.

## How this connects to earlier notes
- It **answers part of the learn-while-acting problem** (note 08). While awake, only cheap learning happens: the episodic store and fast weights. The expensive training of slow weights happens asleep.
- **Emotion grades memory, and memory grade decides training.** Emotion shapes what the system becomes (note 08, hypothesis 4).
- **Summarizing is lossy compression** (note 01): keep the gist and the surprises.
- **Self-generated training before birth** (retinal waves) and **dreaming** may be the same mechanism: training on data the brain makes itself.

## ⚠️ Priority: a robust dream system, including falling asleep
The user wants this designed carefully. Must include a **falling-asleep method**:
- **Shut down parts of the brain in stages**, not all at once. In humans:
  - the thalamus gates out the senses;
  - movement output is blocked (sleep paralysis during REM);
  - the prefrontal cortex dims;
  - different regions fall asleep at different times ("local sleep").
- **Reflect while shutting down.** The drowsy (hypnagogic) state: loose associations, the day
  replaying, half-formed images. Use this window for a **reflection pass**: review the day,
  write the autobiographical record (note 10), grade memories, notice unresolved feelings,
  *before* deep consolidation starts.
- Waking up should also be staged.
- Connects to the parked topic: self-generated training (retinal waves, the womb).

## Open questions
- What's the cost of throwing away something that matters later? Should junk be compressed rather than deleted?
- How should the dream stage be prevented from training on nonsense (runaway imagination)?
- Does the system *feel* tired? Should sleep pressure be one of its needs (hypothalamus)?
- **Testable:** deprive it of sleep and see whether it fails the way sleep-deprived brains do.

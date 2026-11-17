# 15: Neurons across VRAM, RAM and SSD: results of six experiments

Six agents each tested one strategy (2026-10-27/28). Full write-ups and runnable code are in
`prototype/experiments/<name>/README.md`; the shared brief is `prototype/experiments/BRIEF.md`.
Every agent that changed computation reran the "old self comes back" test; all matched the baseline
(day-6 recovery ≈ 0.15, relearn in 2 trials) unless noted.

## Results at a glance

| Experiment | Verdict | Headline |
|---|---|---|
| **vram-ram-paging** | promising | First GPU kernel: 1M neurons, forward 4.4 ms / modulate 7.5 ms / sleep 8.9 ms (about 1,600–1,900× NumPy, zero temporaries). 8.4M neurons (8.6 GB) run in 2.5 GB VRAM at full speed **if** co-active neurons share blocks and attention dwells. |
| **cpu-event-driven** | promising | "Axon-major" layout (synapses grouped by input) plus per-input lazy clocks: 100k neurons at 2% activity, forward 1.0 ms / modulate 4.4 ms on 3 CPU threads (about 270–640×). A plain fused CPU kernel gives 13–50× with the current layout. |
| **rolling-local-sleep** | mixed | C sleep kernel 30–87× NumPy; GPU sleep faster still. Rotating sleep doesn't add speed (sleep is only 0.7% of the day), but sleeping blocks free VRAM: **1.6–2× more neurons per GB**. Found a design flaw (see below). |
| **genome-diff** | mixed | "Genome clock": untouched synapses cost nothing, and sleep on an unused brain went from 115 ms to 0.4 ms. But under current rules **learning isn't sparse**: 39% of synapses differ on day 1. |
| **ssd-dormant** | mixed | SSD is good for **sleep and dormancy**, poor for waking. A 24 GB brain's full night: 40 s CPU / 15 s GPU; lazy night after a normal day: 3 s. With a 4 GB RAM cache, waking ran at RAM speed (98.4% hits). Recall: 0.2 ms per neuron, about 1 s per large old region. |
| **fading-precision** | mixed, one winner | Lossless freezing of idle neurons: **about 200 bytes per cold neuron** (5×) with no behavior change. Lossy fading can be made safe but saves only about 15% more, and risks the old self. |

## Findings that came up again and again

1. **Lazy time.** Most of the neuron's slow dynamics are linear (cascade flow, downscaling, fading),
   so any amount of skipped time collapses into one tiny 3×3 matrix applied when a neuron is next used.
   Five of six agents arrived at this independently. Consequences:
   - **Untouched, dormant or cold neurons cost nothing**, awake or asleep.
   - **A night costs only what the day touched.**
   - Clocks should live **per block or per input**, not population-wide.
2. **Local structure is required.** With every neuron hearing every input, nothing is ever idle and none
   of the savings apply. Neurons must be organized into **columns or assemblies** that draw from their
   own input region. That's a design requirement, not a workaround, and it matches the cortex.
3. **"Fire together, store together."** Neurons that are active together should be stored together.
   Sleep sweeps everything anyway, so it's the natural time to **reorganize storage** by co-activation
   (5× smaller working set, 5.8× faster paging).
4. **Learning isn't sparse yet, and the rules need tightening:**
   - consolidation counts *nights touched*, not *how much was learned* (fast weights saturate in about 7 trials), so a 2-day hard time spanning 3 nights can erase the old self (found by rolling-local-sleep);
   - synapses learn whenever their input is active even if the neuron didn't fire, and dopamine reaches everyone;
   - the prune rule rewires half the synapses on the first night when input is sparse;
   - tags take up a lot of the stored changes.
5. **Untagged ≠ unimportant.** Letting untagged changes fade back to the birth state erased the old
   self, because childhood memories sit in the deep levels after their tags have faded. Old selves must
   be protected as old selves (genome-diff). The fading-precision results agree: the deep level must
   fade last, if at all.
6. **Cold memories simplify on their own.** Long-unused synapses collapse to a few hundred distinct
   states, because the cascade pulls levels together. Lossless compression captures that for free.

## Recommended design: the tiered brain

| Tier | Holds | Format | Cost per neuron |
|---|---|---|---|
| **VRAM** | the attended working set (active columns + eligibility window) | packed 32-bit words, GPU kernel | ~1 KB for resident blocks; ~220 B averaged over the population |
| **Pinned RAM** | the awake brain, plus blocks changed today | packed words; per-block lazy clocks | 1,041 B |
| **RAM, cold** | idle columns | losslessly frozen (state table + zstd), time advanced lazily | ~200–300 B |
| **SSD** | dormant regions; the whole brain's master copy | compressed blocks | ~300 B on disk |

**Day:** the GPU runs attended columns. The CPU (event-driven kernel) can run the rest. Changed blocks stay
in RAM; their growing count is **sleep pressure**.
**Night:** sweep only what changed (lazy nights); consolidate; reorganize blocks by co-activation; freeze
newly idle columns; write changed blocks to the SSD. **Never write to the SSD during the day.**
**Recall:** a dormant region thaws in milliseconds to about a second ("tip of the tongue"), catching up its
missed nights in one step.

On this machine that suggests:
- **millions of neurons** running live;
- **tens of millions** held in RAM;
- **hundreds of millions** dormant on the SSD.

## Next steps (in priority order)
1. **Fix the learning rules** (finding 4): consolidate the amount learned, gate learning by postsynaptic firing or local signals, prune only when a neuron needs a new input, keep tags in a nightly-cleared side list.
2. **Columns:** give the network real local structure, with columns drawing from input regions.
3. **Merge the winners into one prototype:** GPU kernel + per-block lazy clocks + attention paging + cold freezing + SSD night sweep.
4. Then the dream system (note 09), on top of the night sweep.

## Process notes
- One agent briefly overwrote other agents' temp files in a shared scratch folder before moving to its own. No result appears affected (all reran their final suites).
- The SSD experiment wrote about 134 GB total (harmless for the drive); all large data files were deleted.
- Tools live in `~/Documents/LittleBrain/.venv` (5.8 GB). Triton needed `python3-devel`, which the user installed.

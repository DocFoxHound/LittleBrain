# Instructions for AI agents working on LittleBrain

*Written 2026-11-19 by Claude (Opus 5.5) as a handoff. The owner may have another AI (e.g. ChatGPT / Codex) continue
the work, and Claude will **validate it afterwards** against this file. Read all of it before changing anything.*

---

The original handoff below records the public baseline. The default runtime now includes the full catalog. For current setup, measurements and outstanding scientific questions, start with the full-game-brain entry in the handoff log and [the reconstruction report](notes/48-full-brain-reconstruction.md). Section 9 records the experiments that motivated this work.

## 1. What this project is

LittleBrain is a small brain built from **simulated human neuron types** (89 cell types, each modeled on a real human
cell, with researched proportions per region). It is turning into a **tool**: plug the brain into small simulated
games, and see how a character learns depending on its inputs and environment, with a desktop viewer of what the
neurons are doing and (later) tools to inspect and edit what was learned.

- It is **not** a deep-learning project. There is no backpropagation, no training set, no loss function. Cells learn
  with local, biological rules (Hebbian eligibility × a neuromodulator such as dopamine, homeostasis, sleep).
- The long-term aim (see `README.md`, "Working thesis") is a brain-like, emotional, continually learning system.
  Keep design choices compatible with that.
- **The owner is not an ML specialist.** Write notes and summaries in plain language, with brain and behaviour
  analogies. Explain any technical term the first time you use it.

## 2. Ground rules (Claude will check these)

1. **Honest measurement, always with a control.** Every claim that something "learns" or "helps" must compare against
   a control in the *same* brain: learning off, or the region silenced (`brain.ablated |= {...}`), or an untrained
   copy (`copy.deepcopy`). Report chance level. Never report a number without its control.
2. **Seeds.** Iterate with **one seed** (fast). To *claim* a result, run **3 seeds** (0, 1, 2) and report every
   seed's number, not just the mean. Say clearly when something is single-seed. Brain-to-brain variance is large.
3. **Report failures as failures.** If a change doesn't work, say so in the note, with the numbers. Several notes
   (26, 28, 32) are mostly about things that didn't work, and that's what makes the project trustworthy.
   Don't tune until a test passes and then hide the knobs.
4. **Biology first.** New cells or pathways must be based on a real human (or, where human data is missing,
   mammalian) counterpart. Name it in the cell type's `human=` string and cite a source in the note
   (author year, or a review). If you invent something with no biological basis, label it as a **stand-in**.
5. **New behaviour is opt-in until proven.** Add new mechanisms behind an environment flag or a parameter with the
   old behaviour as default (see the flags table in §6). Make it the default only after a 3-seed result, and
   say so in the note. Existing results must stay reproducible.
6. **Every change gets a note.** Add `notes/NN-short-name.md` (next number is **91**), add it to the numbered
   list in `README.md`, and update the roadmap lines there if a milestone moves. Notes follow the existing style:
   what was built, what went wrong on the way, a results table with seeds and controls, and "Next".
7. **Keep it small.** The machine is an 8 GB RTX 4070 laptop GPU plus CPU. The game brains are ~5–6k neurons and
   run on CPU. Don't grow brains unless a test needs it.
8. **Tests must still pass** (§7). Add a test for any new cell type's signature behaviour, in the style of
   `prototype/brain/test_cell_types_regions.py`.
9. **Privacy and git.**
   - The repo is **public**. Never commit personal information about the owner (health, service history, names,
     emails, home paths). The notes were scrubbed once; keep them that way.
   - Commit author email must be `86894115+DocFoxHound@users.noreply.github.com` (already set in this repo's
     git config; don't change it, never use a Gmail/Hotmail address).
   - **Never rewrite history or force-push.** Small, descriptive commits on `main` (or a branch if the owner asks).
   - Don't commit `.venv/`, `prototype/viz/.cache/`, screenshots dumped into `shots/`, or large result files.

## 3. Setup

```bash
cd LittleBrain
python3.14 -m venv .venv            # the owner's machine already has .venv (5.8 GB, with CUDA torch)
.venv/bin/pip install -r requirements.txt   # numpy, torch, numba, pygame-ce
```
- Python 3.14. Everything runs on CPU. GPU is optional (`DEVICE=gpu`, only worth it at fly scale, ~140k neurons).
- Scripts add `prototype/` and `prototype/brain/` to `sys.path` themselves; run them from the repo root with
  `.venv/bin/python path/to/script.py`.
- The viewer needs a display. Without one, use headless screenshots (§7). The owner often can't open the GUI
  (at work), so **send screenshots** of anything visual.

## 4. Repository map

```
README.md                  index of all notes, roadmap, decisions ("Clarified so far")
AGENTS.md                  this file
notes/                     the project's lab notebook, numbered; notes/regions/ = researched cell makeup per region
docs/screenshots/          viewer screenshots used in README / notes
prototype/
  neuron.py                single-neuron model and the learning rules (NeuronType, WORKHORSE_V2)
  lean_neuron.py           compact synapse storage
  games/                   GAME INTERFACE and games (know nothing about neurons)
    core.py                Game / Observation; blob_image helper
    signal_game.py         picture -> one correct button (contextual bandit)
    danger_game.py         food pictures (right button +1) and danger pictures (anything but "hide" -1)
    __init__.py            REGISTRY = {"signal": ..., "danger": ...}  <- register new games here
  brain/
    celltypes.py           all 89 cell types (CellType dataclass); each has a human counterpart
    cells.py               dynamics per family: ExcitatoryGroup (learning synapses), InhibitoryGroup, ModulatorGroup...
    regions.py             each region's researched cell makeup (percentages, sources)
    federation.py          Brain: groups + Synaptic pathways (learnable) + Pooled pathways (population-level),
                           step(), calibrate(), develop(), sleep(), homeostasis, own dopamine/arousal
    brainspec.py           brain descriptions: stage1 (cortex+thalamus), stage2 (+hippocampus),
                           stage_bg (+basal ganglia), stage_da (+dopamine cells), stage_amy (+amygdala)
    agent.py               BrainAgent: plugs a Brain into any Game (look -> choose -> act -> rest)
    diagnostics.py         honest tests: reliability (test-retest), memory with untrained control
    run_signal_game.py     signal game experiment; also prepare() = build + calibrate + develop + wake a brain
    run_danger_game.py     danger game, amygdala intact vs silenced
    run_cortex_fix.py      cortex reliability / memory configurations (note 28)
    test_*.py              cell-type and region signature tests
  viz/brainview.py         desktop viewer (pygame-ce), game-agnostic; headless screenshots
```

## 5. How the brain works (enough to work on it)

- **Map.** Every structure is laid out on a shared 8×8 tile map (`GRID=8`). `Group(key, ctype, per_tile, structure)`
  puts `per_tile` cells of a type on each tile. Same tile = same map location across structures.
- **Synaptic pathways** (`Synaptic(id, src, dst, comp, frac, gain, spread, kind)`) are learnable excitatory synapses
  into basal or apical dendrites, sampled by distance ("local") or by map location with a spread ("topo").
- **Pooled pathways** (`Pooled(id, src, dst, weight, channel, sigma)`) are population-level influences: all
  inhibition, neuromodulation, inputs like the action copy. Channels for excitatory targets: `pv` (soma), `sst`
  (apical), `dend`, `ais`, `volume`, `inh`, `exc` (additive). `sigma=0` same tile, `None` whole structure.
- **Inputs** (`spec.inputs`): groups set from outside each step: `retina` (picture), `MOT` (copy of the chosen
  action), `US` (reward/pain arriving from the body; signed).
- **A step** (`Brain.step(inputs, learn, ach, dopamine, arousal)`): phase 1 updates inhibitory/modulatory groups,
  phase 2 the excitatory ones. If the brain has dopamine cells (`VTA.DA`) their phasic firing is added to
  `dopamine`; if it has a central amygdala (`CeM`) its output raises `arousal`.
- **Learning rule** (three-factor): synapse change = learning rate × eligibility (input × cell activity, decaying)
  × dopamine × acetylcholine. D2 cells and BLA(−) cells have `dopamine_sign=-1` (learn from dips).
- **Life cycle of a brain** (`run_signal_game.prepare(seed, brain, scale)`): build → `calibrate` (cells set
  thresholds) → `develop` (2 "days" of experience with sleep) → **wake** (360 steps; without this the cortex is
  silent after sleep, note 28). Takes ~20–25 s at 5–6k neurons.
- **Agent turn** (`BrainAgent`): look at the picture 20 steps (read GPi over the last 16) → choose by softmax over
  how much each action channel's GPi brake is released → act 4 steps with the action copy on (reward arrives at
  step 3) → rest 3 steps. Action channels = vertical stripes of the 8×8 map (`brainspec.action_channel`).
- **Viewer**: watches an agent through `agent.on_step(phase, info)`. It never needs to know the game.

## 6. Environment flags (current defaults in experiment scripts)

| flag | meaning | default in run scripts / viewer |
|---|---|---|
| `HOMEO_RULE=log` | symmetric homeostasis (note 28); `quantile` = old | `log` |
| `L4_L23_GAIN` | strength of L4 → L2/3 synapses (note 28) | `3` |
| `CORTEX_LRF` | cortical learning-rate factor (note 28) | `0.05` |
| `MSN_LR`, `MSN_DECAY` | striatal learning rate and fast-weight decay (note 29) | `0.6`, `0.99995` |
| `HOMEO_SLOW=key:f,...` | per-group homeostasis speed (note 32) | `STR.SO:0.1` |
| `HOMEO_SKIP=key,...` | groups without homeostasis | none |
| `SO_WEIGHT`, `SO_MOT`, `SO_LR` | striosome → dopamine brake, action copy into striosomes, their learning rate | `4`, `1.0`, = MSN_LR |
| `AMY_US`, `AMY_DA` | reward/pain drive into BLA; BLA → dopamine strength | `2.0`, `2.0` |
| `BRAIN=...` | which brain `prepare()` builds (`run_signal_game.SPECS`): `full` (18,944-neuron reference), `human100k` (~100k, human proportions, note 70), or an earlier stage | `full` |
| `BRAIN_NEURONS` | total for `human100k` | `100000` |
| `CRITIC=average/cortex` | computed dopamine stand-in (only for `bg`) | `average` |
| `LOOK`, `READ` | look steps and steps counted (note 30) | `20`, `16` |
| `SEED`, `TURNS`, `NO_CONTROL`, `LESION` | experiment controls | |
| `DEVICE=gpu` | run on the GPU | cpu |

The flags read at import time live in `brainspec.py`; set them before importing it.

## 7. How to run and verify (baselines Claude will compare against)

```bash
# cell and region tests: must stay all-pass (36 tests, ~2 s total)
.venv/bin/python prototype/brain/test_cell_types.py          # 12/12
.venv/bin/python prototype/brain/test_cell_types_regions.py  # 19/19
.venv/bin/python prototype/brain/test_learning_by_region.py  # 5/5

# signal game (~2-3 min per seed at 400 turns)
SEED=0 .venv/bin/python prototype/brain/run_signal_game.py              # bg brain
BRAIN=da SEED=0 TURNS=300 .venv/bin/python prototype/brain/run_signal_game.py

# danger game, amygdala intact vs silenced (~3 min per seed)
SEED=0 .venv/bin/python prototype/brain/run_danger_game.py

# viewer (window), or headless screenshots
.venv/bin/python prototype/viz/brainview.py --brain amy --game danger
.venv/bin/python prototype/viz/brainview.py --brain amy --game danger --shots 5,40 --out shots/ [--hover x,y] [--scale 0.25|0.5|1|2|4]
```

Current results to reproduce (chance 0.25 on both games):

| test | result | note |
|---|---|---|
| signal game, `bg`, last 100 of 400 turns, seeds 0/1/2 | 0.73 (seed 0) vs 0.24 learning off; 0.61–0.74 over 3 seeds | 30 |
| signal game, `da` (brain-made dopamine), last 100 of 300 turns | 0.69 / 0.68 / 0.43 (earlier variant); final settings 0.63–0.68 (seeds 0, 1) | 32 |
| dopamine behaves as prediction error (`da`, seed 0) | bursts on reward shrink 0.95 → 0.85, dips on misses deepen −0.06 → −0.21 over 300 turns | 32 |
| amygdala learns danger (`amy`, seed 0, after 300 turns) | CeM 0.213 danger vs 0.150 food; extra arousal 0.091 vs 0.013 | 32 |
| danger game hurt rate, last 100 turns (random = 0.75) | intact 0.71 / 0.73, silenced 0.64 / 0.76 (seeds 0 / 1): **no avoidance learned yet** | 32 |
| cortex A L2/3 reliability (retest) | ~0 originally → 0.4–0.7 with the note 28 fixes | 28 |

## 8. Known problems and traps (learned the hard way)

- **After sleep the cortex is silent** for a few hundred steps (fast weights are wiped; thresholds were raised during
  the day). Always run the wake period before measuring (`prepare` does; `diagnostics.wake`).
- **Homeostasis can erase learning.** A cell's rate controller treats a learned, across-the-board rise as drift. That
  is why striosomes have slow homeostasis (note 32). Check this first when something learns and then un-learns.
- **Dopamine can't dip below zero firing**, so dips are shallower than bursts (realistic, but punishment learning is
  ~5x weaker than reward learning).
- **Old memory tests were a timing artifact** (note 26): never score memory with cues right after their patterns;
  use `diagnostics.memory_vs_control` (scrambled order, untrained control).
- **Viewer cache:** prepared brains are cached in `prototype/viz/.cache/<brain>_scale<s>_seed<n>.brain`. The key does
  **not** include the spec, so **delete the cache after changing a brain description**, or the viewer shows the old
  brain. (A fix would be to hash the spec into the filename; that's a welcome small task.)
- Old pickled brains lack newer attributes; code uses `getattr(self, "...", default)` for them. Keep doing that.
- `pkill -f <pattern>` can kill your own shell if the pattern appears in your command line.
- New brain inputs (`US`, `MOT`, ...) must be given every step (`agent._inputs`) or they keep their last value.

## 9. Your tasks, in order

Do them in this order. Each one ends with: a test game (registered in `games.REGISTRY`), a comparison against the
region silenced or absent, 3 seeds, a note, the README updated, a viewer territory for the new region (see
`TERRITORY` in `brainview.py`; replace its "not built yet" outline), a headless screenshot in `docs/screenshots/`,
and a commit.

### Task 1: temporal-difference dopamine and relief (unblocks the amygdala)
- **Problem (note 32):** hiding from danger produces no signal, so nothing teaches avoidance. Real brains get a
  **relief** burst when an expected danger doesn't arrive (two-factor theory). Our dopamine compares activity with a
  slow baseline, so it can't produce that.
- **Build:** dopamine cells that respond to the *change* in expectation over time: a cue that predicts reward causes
  a burst at the cue, one that predicts pain a dip, and at the outcome the error is relative to that expectation.
  Biological route: ventral striatum / striosome activity at time t vs t−1, carried by VTA GABA neurons and direct
  striosome input (Cohen et al. 2012; Eshel et al. 2015). Keep the BLA(−) → dopamine route; it should now produce
  the cue dip and, when hiding succeeds, relief.
- **Done when (3 seeds):** in the danger game, with the amygdala intact, the hurt rate on danger pictures in the last
  100 of 300 turns is **≤ 0.40** on every seed; report the silenced-amygdala numbers next to it (we expect them to be
  worse; if not, say so). The signal game with this brain must stay **≥ 0.60** (last 100 turns, every seed). Show the
  dopamine trace per turn type (food reward, food miss, danger hurt, danger hide) early vs late.

### Task 2: hypothalamus and body drives
- Give games a body: `Observation.body` already exists (e.g. `{"hunger": 0.7}`). Write a **forage game** where food
  matters more when hungry and hunger rises over time; eating lowers it.
- Wire arcuate AgRP (hunger) / POMC, lateral hypothalamus (orexin/MCH) from `regions.py` / `celltypes.py`
  (`AGRP`, `POMC`, `OREXIN`, `MCH`): body signal in → hypothalamus → modulates the value of food (LH → VTA, so food reward is bigger when
  hungry) and arousal.
- **Done when:** the character eats more when hungry than when sated (compare choice rates at high vs low hunger),
  and the effect disappears with the hypothalamus silenced. 3 seeds.

### Task 3: prefrontal cortex + mediodorsal thalamus (holding things in mind)
- Write a **delayed-cue game**: a cue shows, then a blank delay, then the choice; the right answer depends on the cue.
- Wire a prefrontal area (`MOTOR_PREFRONTAL_CORTEX` makeup in `regions.py`) with recurrent persistent activity and
  the PFC ↔ MD thalamus loop, PFC → striatum.
- **Done when:** above chance at delays where the cortex-only brain is at chance; performance falls with the
  prefrontal area silenced. 3 seeds.

### Task 4: cerebellum + inferior olive (timing)
- Write a **timing game**: act at the right moment after a cue (reward only in a time window).
- Wire granule → Purkinje → deep cerebellar nuclei, climbing fibres from the inferior olive as the error signal
  (cell types exist in `celltypes.py`: CB_GRANULE, PURKINJE, DCN_PROJ, DCN_NO, OLIVE; Purkinje learning is `ExcitatoryGroup.climbing_fiber()` in `cells.py`).
- **Done when:** timing error shrinks over training and is worse with the cerebellum silenced. 3 seeds.

### Remaining regions after that (for later; don't start unless the owner asks)
Hippocampus into the game brain (built in `stage2`, not in the game brains), CA2, cingulate / insula, pulvinar and
other higher-order thalamus, claustrum, subthalamic nucleus, substantia nigra reticulata, locus coeruleus cells
(arousal is a computed signal now), dorsal and median raphe (serotonin), nucleus basalis + medial septum
(acetylcholine, theta), pedunculopontine / laterodorsal nuclei. Then the sleep and dreaming system (note 09).

### Small, welcome tasks
- Viewer cache key: include a hash of the brain description (§8).
- Viewer: dopamine and arousal traces from the brain's own cells (`brain.last_dopamine`, `brain.last_arousal`).
- A one-command smoke test script that runs the 3 test files and a 50-turn game per brain and prints a summary.

## 10. What NOT to do
- Don't replace the biological learning with backprop, a neural-network library or an RL algorithm "to make it work".
  A computed stand-in is acceptable only if labelled as one and kept switchable (like `CRITIC=average`).
- Don't change defaults or delete flags that existing notes rely on.
- Don't rewrite or force-push history; don't commit secrets or personal data.
- Don't claim success from one seed or without a control.
- Don't make the viewer game-specific: games plug in through `games.Game`; the viewer shows only generic things
  (observation, action names, reward, the brain).

## 11. Hand back (so Claude can validate)
At the end, add a section **"Handoff log"** at the bottom of this file with, per task: what you changed (files),
the exact commands you ran, the numbers per seed with controls, what didn't work, and anything you left unfinished.
Claude will re-run your commands, compare numbers, run the tests, read the diffs, and check §2.

---

## Handoff log
*(append below)*

### Linux desktop release

The release workflow, build command and viewer save behavior are documented in [note 34](notes/34-desktop-release.md). Run `bash packaging/build-linux.sh` to build and exercise the installed program. This release uses the signal and danger viewer from the public repository.

### Store media for v0.1.0

[Note 35](notes/35-store-media.md) contains the copy and media links. The packaged viewer produced the gallery using `--game signal --shots 35` and `--game danger --shots 8`.

### Full game brain recovery

The normal game runner, external bridge and viewer use the reconstructed full specification. The desktop opens Forage. Run the CPU commands in [note 48](notes/48-full-brain-reconstruction.md) from the repository root. It records the source provenance, physiological assumptions, seeded results, silencing controls and remaining limits. The reconstruction spans `brainspec.py`, `regional_circuits.py`, `federation.py`, `association.py`, `striatal_projection.py`, the game agent, games, bridge, controls and viewer.

For repeatable behavioral comparisons, develop each seed once and start every condition from that seed's checkpoint.

```sh
export OPENBLAS_NUM_THREADS=1
for seed in 0 1 2; do
    brain="/tmp/littlebrain-seed-$seed.brain"
    python prototype/brain/check_full_games.py --seed "$seed" --prepare-only "$brain"
    python prototype/brain/check_full_runtime.py --seed "$seed" --load-brain "$brain"
    for game in signal danger forage delayed timing; do
        control=lesion
        if [ "$game" = signal ]; then control=off; fi
        for condition in learning "$control"; do
            python prototype/brain/check_full_games.py --seed "$seed" --load-brain "$brain" \
                --game "$game" --condition "$condition" --turns 300
        done
    done
done
```

The original 58 Git objects and any private learned brain state are outside this history. New verification should use the current measurements and retain original public baselines under their named comparison stages.

### Slow output verification

The phase-one update in federation.py now publishes filtered inhibition and peptide release. See note 49 for the before/after pulse measurements for seeds 0, 1 and 2. Ran the three prototype/brain/test_*.py scripts listed in section 7 and, for each developed reference checkpoint, `python prototype/brain/check_full_runtime.py --seed "$seed" --load-brain "$brain"`. All passed. No new game-learning claim.

### Region-by-region audit

See note 50 for the measured changes and controls on all three seeds. Run `OPENBLAS_NUM_THREADS=1 python prototype/brain/check_regional_audit.py --seed N` for N = 0, 1 and 2. Measurements used checkpoints freshly prepared at bfc1c6a and the optional --load-brain argument. The script compares composition against all 36 catalog entries, verifies the sampled partners, pulses one projection per runtime territory and repeats an identical stimulus schedule with each territory silenced. A successful pulse supports a local effect. Task-specific learning requires a matched behavioral experiment. Seven direct inputs were masked at this revision; see subsequent notes for fixes.

### Basal ganglia inhibition of thalamic relay cells

See [note 51](notes/51-thalamic-brakes.md) for the isolated change, matched controls and all three seeds. Run the regional audit and integrated runtime check from note 50 and section 7 on the developed reference brain.

### Restore cortical input to the pontine timing relay

See [note 52](notes/52-pontine-cortical-input.md) for the isolated change, matched controls and all three seeds. Run the regional audit and integrated runtime check from note 50 and section 7 on the developed reference brain.

### Admit cerebellar input to the mediodorsal relay

See [note 53](notes/53-cerebellar-input-md.md) for the isolated change, matched controls and all three seeds. Run the regional audit and integrated runtime check from note 50 and section 7 on the developed reference brain.

### Keep Purkinje teaching and inhibitory inputs active during timing

See [note 54](notes/54-purkinje-timing-inputs.md) for the isolated change, matched controls and all three seeds. Run the regional audit and integrated runtime check from note 50 and section 7 on the developed reference brain.

### Sign of learned mossy-fiber input

Note 55 records the post-development dentate-to-CA3 polarity failure, its matched negative-contact intervention and the network-wide scope of signed glutamatergic weights. The regional audit reproduces all three comparisons with its standard command. It leaves the examined checkpoint intact.

### Post-repair verification

Note 56 records the default CPU brain after all seven terminal repairs, including the measured signed pulse for each of the 45 territories on seeds 0, 1 and 2. Rebuilt each checkpoint using check_full_games.py --prepare-only, then reran check_regional_audit.py and check_full_runtime.py with --load-brain and the same seed. All integrated checks passed. The remaining hippocampal sign reversal is quantified in note 55.

### Connection plan comparison

Note 57 compares the 94 numbered anatomical routes in notes/regions/connections.md to actual source and target groups. The regional audit's follow-up output reproduces the EC island and deep nuclear collateral probes on seeds 0, 1 and 2. HC-6 has a receiver mismatch, the direct pontine collateral has a zero effect at reference scale, and several planned contextual and extinction pathways are absent.

### Working-memory signal diagnosis

Note 58 reports a paired prefrontal retention assay from the actual delayed game. The normal CLI can run it using --probe-held-cue. A matched motor copy during passive viewing separates loss of the visual trace from action selection. Both blank turns and the choice period are measured for all three seeds.

### Restore the maintained frontal population

The frontal holding ensemble in association.py now receives both local recurrent and MD activity, with weights drawn from the existing basal fractions in brainspec.py. Note 59 reports the before/after functional assays on all three seeds, the 300-turn matched delayed results, the frozen lesions and smoke comparisons on the other four games. All three check_full_runtime.py runs passed on the normally developed reference checkpoints.

### Dentate mossy-fiber transmission

The conductance floor in cells.py and federation.py applies to new and saved HC-7 contacts. [Note 60](notes/60-mossy-conductance.md) gives the direct and recall controls, reproduction commands and three-seed results.

### Entorhinal island-cell target

The regional specification and old full checkpoint loader now route island cells through CA1 neurogliaform cells. [Note 61](notes/61-island-feedforward-inhibition.md) records all three paired feedforward tests and the commands.

### Direct pontine input to deep cerebellar output

The reduced deep-nucleus readout in association.py admits sampled pontine drive. [Note 62](notes/62-pontine-collateral.md) records the failed spike-output blend, the two-mode contact lesion and Purkinje controls on seeds 0–2.

### Local placement budgets

[Note 63](notes/63-placement-budget.md) gives the cost of the 67 missing placements under three allocation choices, computed from the regional audit and the 64-cell game-map minimum. The anatomy decision affects reference population counts.

### Striosome thalamic input

[Note 64](notes/64-thalamic-striosome-input.md) records the failed zero-floor control, within-class plasticity in cells.py and the three-seed pulse and signal comparisons.

### Expanded regional reference brain

[Note 65](notes/65-regional-placements.md) tracks each region's tested additions to catalog_placements.py, three-seed measurements, circuit assumptions and the command for reproducing its paired input, output and terminal-lesion checks.

### Expanded reference verification

[Note 66](notes/66-expanded-reference-verification.md) reports the reference inventory, terminal lesions, emitted responses, footprint and post-development regional checks. The per-population CSV contains the individual measurements for all three seeds.

### Full-brain store page

[Note 67](notes/67-full-brain-store-page.md) links the current public-facing description and records the source of the page assets.

### Data-only checkpoints

[Note 68](notes/68-data-only-checkpoints.md) documents the save format, legacy cache behavior and end-to-end tests. Keep new persisted fields in the explicit checkpoint schema.

### Native research kernels

[Note 69](notes/69-build-time-research-kernels.md) covers the development-only kernel build step and comparisons with the earlier benchmarks.

### 100,000-neuron human-proportion brain

[Note 70](notes/70-human-proportion-100k.md): `BRAIN=human100k` (`human_scale.py`, `brainspec.full_brain_human`). Runtime check passed on seeds 0–2; signal game 0.78 / 0.95 / 0.88 learning vs 0.23 / 0.27 / 0.33 off (64 turns). The runtime anatomy check now accepts grown full brains (same 187 populations, none smaller). Still to run at 100k: regional audit, placements, and the other four games with controls.

### Vision brain (notes 71-74), in progress

`BRAIN=vision`: finer maps (`Group.grid`), a retina (`retina.py`), LGN streams, V1 tuning from retinal waves, colour
and motion games, learning routed through visual cortex (`VISION_ROUTE`), and a closed critical period after
development. Tests: `test_fine_maps.py` 6/6, `test_retina.py` 7/7; reference brain bit-identical. Colour is learned via the
colour relays (0.64 / 0.64 / 0.53) but not yet via V1 L2/3; motion waits for MT. Next steps are in note 74.

### Vision brain, hierarchy and eye movements (notes 75-76)

V2, V4, IT and MT with like-to-like wiring and a visual-cortex sharpness cap; superior colliculus and saccades
(`VISION_SACCADES`). 3 seeds: colour learned through cortex (0.39 / 0.46 / 0.41 vs ~0.24 off); motion only when MT is
read alone; shape not learned (needs temporal-trace learning). Saccades land on salient targets (0.315 → 0.086).

### Current plan

[Note 78](notes/78-minecraft-plan.md) is the working plan: phases 0-9 toward a Minecraft-trained template and shipping
blank / Minecraft templates in the app. Work the phases in order; each has a test game and "done when" criteria.
**[Note 86](notes/86-plan-status.md) is the current status** (paused 2026-11-22): phases 0, 2 and N1 done; 1 and L1
partly; phase 5 in progress with the owner's chosen next steps; open items (CA1 place cells, phase 1's chain bar).
Notes 80-85: TD credit and habenula (80), hearing (81), trace learning attempt (82), spatial maps (83), place cells
(84), V1 edge cells (85), V4 pooling and readout scale (87), feedforward-dominated visual layer 4 (88), where the shape signal is lost (89), competitive learning in IT (90).

### Phase 0 (note 79)

Saves carry a design stamp (`design.py`; `Brain.load(path, kind=...)` refuses mismatches; bump `RUNTIME_VERSION` when
simulation semantics change). Inner loops are compiled (`fastkernels.py`, numba); `LITTLEBRAIN_FAST=0` runs the exact
numpy code, which stays bit-identical to earlier results. Tests: `test_design.py`, `test_fast_kernels.py`.

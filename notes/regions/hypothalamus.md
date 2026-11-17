# Hypothalamus: functional cell types for drives and homeostasis

Brief by design: the hypothalamus has hundreds of molecularly defined types (mouse scRNA-seq atlases list >100 neuron clusters),
so this note covers only the populations that matter for **drives** (hunger, arousal, stress, social bonding) and **timing**
(circadian clock). Percentages of a whole hypothalamic nucleus are rarely known; **counts per population** are given where they exist.

Confidence: ● well established · ◐ supported, numbers vary · ○ uncertain.

---

## 1. Composition table

| Nucleus / type (marker) | Share / count | Species | Transmitter | E/I; projection | Conf. |
|---|---|---|---|---|---|
| **Arcuate AgRP/NPY** (Agrp, Npy) | **~8,000** NPY neurons (mouse, stereology); human: 82% of NPY cells contain AgRP, 87% of AgRP cells contain NPY | mouse; human | **GABA** + NPY + AgRP | inhibitory projection → PVN, LH, BNST, PVT, parabrachial; local ⊣ POMC | ● type / ◐ count |
| **Arcuate POMC** (Pomc; α-MSH, β-endorphin, CART) | **~9,000** (mouse, stereology); ~28% of POMC-labelled cells co-express Agrp in one scRNA-seq sample (45/163) | mouse | mostly **glutamate**, some GABA; α-MSH | projection → PVN (MC4R), LH, brainstem | ● type / ◐ count |
| Human infundibular (arcuate) nucleus, all neurons | **~505,000–520,000** (women, stereology); peptide-defined counts not available | human | mixed | — | ◐ |
| **LH orexin / hypocretin** (Hcrt) | **human ~50,000–83,000 (mean ~70,000)**; narcolepsy: **85–95% lost** | human | glutamate + orexin A/B (+ dynorphin) | excitatory projection → **LC, TMN (histamine), raphe, VTA, basal forebrain, PPN/LDT, cortex**, spinal cord | ● |
| **LH MCH** (Pmch; CART in a subset) | ~**2–3× orexin cells** in rat (3:1 whole hypothalamus, 2:1 within LH; 18% of MCH cells in zona incerta); human total not found | rat | **GABA** + MCH | inhibitory projection, widespread (hippocampus, cortex, NAc, LC, raphe) | ◐ |
| LH GABA (Vgat, LepR, Gal subsets) and glutamate (Vglut2) | majority of LH neurons; not quantified | mouse | GABA / glutamate | LH GABA → VTA (reward seeking, feeding); LH Vglut2 → LHb (aversion, stop feeding) | ◐ |
| **PVN parvocellular CRH** (Crh) | count not found for humans; increased in depression | human, rat | glutamate + CRH | → **median eminence** (portal blood → pituitary ACTH) = **HPA axis** | ● type / ○ count |
| **PVN magnocellular oxytocin** (Oxt) | count not found; reduced in Parkinson's, increased in depression (Purba 1996) | human | glutamate + OT | → **posterior pituitary** (blood) + axons to amygdala, NAc, VTA, brainstem, spinal cord | ● |
| **PVN/SON magnocellular vasopressin** (Avp) | count not found | human | glutamate + AVP | → posterior pituitary (water balance), + central | ● |
| PVN pre-autonomic (OT, AVP, CRH subsets) | — | rat | glutamate | → brainstem NTS/DMV, spinal sympathetic | ● |
| **SCN vasopressin (shell)** | **human 6,890 ± 520** (one side) | human | **GABA** + AVP | local + → PVN, DMH, subPVZ, thalamus PVT | ● |
| **SCN VIP (core)** | **human 1,700 ± 140** (one side); young men ~2× young women | human | **GABA** + VIP | local (synchronize the clock); retinorecipient | ● |
| SCN total | **~20,000 neurons** in mouse (bilateral); nearly all GABAergic | mouse | GABA (+ peptides: AVP, VIP, GRP, calretinin, ENK, NMS) | mostly local + hypothalamic | ● |

---

## 2. Signature behaviors to model (testable)

### Arcuate: hunger and satiety
- **AgRP neurons** are **activated by energy deficit** (ghrelin ↑, leptin/insulin ↓) and fire tonically during fasting. Their activity is felt as an *aversive* state that the animal works to reduce: on **sensory detection of food** they drop within seconds, *before* eating (Chen et al. 2015, Betley 2015). Optogenetic activation → voracious eating within minutes even when fed. Inhibit POMC both locally and at PVN.
  *Test:* rate rises with a "fasting" variable; sight of food lowers it before any calories arrive; activation produces food seeking.
- **POMC neurons** are **activated by leptin and feeding**; slow effect (hours) on reducing intake; α-MSH acts on MC4R in PVN. Also driven quickly by food cues (opposite to AgRP).
  *Test:* slow decrease in intake drive over many steps after sustained activation, versus the fast AgRP effect.
- The pair is a **push–pull integrator of energy state** with fast anticipation (sensory) and slow correction (hormones).

### Lateral hypothalamus: arousal and drive
- **Orexin neurons:** active in **active waking**, silent in NREM and REM; burst with arousal, novelty, motivated behavior; inhibited by glucose, excited by ghrelin. Excite every arousal nucleus (LC, TMN, raphe, BF, PPN/LDT) → **stabilize wakefulness** (flip-flop switch with VLPO). Loss = narcolepsy (sudden sleep/REM intrusions, cataplexy).
  *Test:* without orexin input, the modelled sleep/wake switch flips more often (fragmented states).
- **MCH neurons:** reciprocal to orexin — **active during REM sleep** (and some NREM), silent in waking; promote REM and feeding/energy storage; MCH may promote *forgetting* via hippocampal inhibition during REM (Izawa et al. 2019).
  *Test:* MCH rate high in a "REM" state and near zero awake; links to the planned dream system.

### PVN: stress and social bonding
- **CRH neurons:** respond within seconds to stressors (fast, predictive of threat); start the HPA cascade (cortisol peaks ~20–30 min later) with **negative feedback** from cortisol via glucocorticoid receptors.
  *Test:* stress input → fast CRH rise → delayed slow cortisol variable → cortisol suppresses CRH.
- **Oxytocin magnocellular cells:** pulsatile, **synchronous bursts** (milk ejection, birth) via dendritic OT release and positive feedback; central OT raises salience of social cues and supports bonding, social memory, and safety signals.
- **Vasopressin cells:** **phasic firing** (bursts of tens of seconds alternating with silence) driven by osmolality; central AVP supports social recognition, territorial/paternal behaviour.

### SCN: the clock
- Each SCN neuron has a **cell-autonomous ~24 h molecular clock** (CLOCK/BMAL1 → Per/Cry → feedback; transcription–translation loop).
- **Firing rate high by day (~5–10 Hz), low at night (~0–2 Hz)** in both diurnal and nocturnal species.
- **VIP core cells** receive **retinal input** (melanopsin ganglion cells, glutamate + PACAP) and **phase-shift** the network; VIP coupling **synchronizes** shell cells — without VIP, cells drift apart.
- **AVP shell cells** keep a robust intrinsic rhythm and send the output timing signal; human AVP-cell numbers show a daily rhythm in young adults that disappears with age.
  *Test:* uncoupled cells drift apart in phase; with VIP coupling they synchronize; a light pulse in early night delays the phase, in late night advances it (phase response curve).

---

## 3. Main connections

- **Arcuate ⇄ blood:** sits above the median eminence (leaky blood–brain barrier) — reads leptin, ghrelin, insulin, glucose directly.
- **Arcuate → PVN** (MC4R satiety node), **LH**, **PVT**, **parabrachial** (AgRP ⊣ PBN CGRP cells suppresses meal-ending signals).
- **LH orexin → arousal nuclei and VTA**; **LH GABA → VTA** (reward); **LH ⇄ NAc shell** (feeding gate).
- **PVN → pituitary** (CRH portal; OT/AVP neurohypophysis); **PVN → brainstem/spinal autonomic**; amygdala and BNST → PVN for stress.
- **SCN → subparaventricular zone → DMH → VLPO (sleep), LH orexin (wake), PVN (cortisol rhythm, melatonin via pineal)**.
- Feeds the neuromodulatory nuclei (`neuromodulatory-nuclei.md`) — hypothalamus sets *why* (need), they set *how strongly the brain cares right now*.

---

## 4. Human vs rodent

- The hypothalamus is **highly conserved**; cell types and wiring map one-to-one, so rodent function data transfer fairly well.
- Human orexin population ~70,000 vs a few thousand in rodents (rodent count not re-verified here). ◐
- Humans are **diurnal**; SCN still fires most by day, but downstream sign flips (sleep-promoting at night).
- Human SCN shows **sex and age differences** in VIP-cell number (young men ~2× young women) and loss of the AVP daily rhythm in aging.
- Human arcuate is called the **infundibular nucleus**; it has strong sex-steroid sensitivity (POMC mRNA ↓ 65% after menopause; KNDy neurons hypertrophy).

---

## 5. Sources

- Thannickal TC et al. (2000) Reduced number of hypocretin neurons in human narcolepsy. *Neuron* 27:469 (controls 50,576–83,121 Hcrt neurons; MCH section counts 734 ± 119 unchanged). https://teams.semel.ucla.edu/sleep-research/publication/reduced-number-hypocretin-neurons-human-narcolepsy
- Hofman MA, Zhou JN, Swaab DF (1996) Suprachiasmatic nucleus of the human brain: an immunocytochemical and morphometric analysis. *Anat Rec* 244:552 (VP 6,890 ± 520; VIP 1,700 ± 140). PMID 8694290
- Hofman MA, Swaab DF (1994) Alterations in circadian rhythmicity of the vasopressin-producing neurons of the human SCN with aging. *Brain Res* 651:134.
- Connectome of the suprachiasmatic nucleus: new evidence of the core-shell relationship (2018) *eNeuro* 5:ENEURO.0205-18.2018 ("~20,000 peptidergically heterogeneous neurons"). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6168316/
- Abrahamson EE, Moore RY (2001) SCN in the mouse: retinal innervation, intrinsic organization and efferent projections. *Brain Res* 916:172 (core/shell, GABA). Recalled, not re-fetched.
- Stereological NPY/POMC/orexin counts in diet-induced obese male mice (~8,000 NPY, ~9,000 POMC). https://research.monash.edu/en/publications/a-stereological-analysis-of-npy-pomc-orexin-gfap-astrocyte-and-ib/
- Lam BYH et al. (2017) Heterogeneity of hypothalamic POMC-expressing neurons revealed by single-cell RNA sequencing. *Mol Metab* 6:383. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5404100/
- Stereologic study of the human infundibular nucleus in young and older women (~520,000 vs ~505,000). https://experts.arizona.edu/en/publications/stereologic-study-of-the-hypothalamic-infundibular-nucleus-in-you/
- Human NPY/AgRP co-localization (82% / 87%) and NPY in type 2 diabetes. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3394787/
- Rat immunohistochemical comparison of MCH and hypocretin/orexin neuron distributions (MCH:H/O ≈ 3:1 hypothalamus-wide, 2:1 in LH; 18% of MCH in zona incerta). https://pmc.ncbi.nlm.nih.gov/articles/PMC2800034
- Purba JS, Hoogendijk WJ, Hofman MA, Swaab DF (1996) Increased number of vasopressin- and oxytocin-expressing neurons in the PVN in depression. *Arch Gen Psychiatry* 53:137.
- Raadsheer FC et al. (1994) Increased numbers of CRH-expressing neurons in the PVN of depressed patients. *Neuroendocrinology* 60:436.
- Chen Y, Lin YC, Kuo TW, Knight ZA (2015) Sensory detection of food rapidly modulates arcuate feeding circuits. *Cell* 160:829.
- Betley JN et al. (2015) Neurons for hunger and thirst transmit a negative-valence teaching signal. *Nature* 521:180.
- Saper CB, Scammell TE, Lu J (2005) Hypothalamic regulation of sleep and circadian rhythms. *Nature* 437:1257.
- Izawa S et al. (2019) REM sleep-active MCH neurons are involved in forgetting hippocampus-dependent memories. *Science* 365:1308.

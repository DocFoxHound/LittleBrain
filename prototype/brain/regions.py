"""
Brain regions: which neuron types each region contains, and in what proportions.

Every number comes from notes/regions/*.md (researched, with sources). Each entry records:
    (cell type, % of the subregion's neurons, species the number comes from, confidence)
Confidence: "●" well established · "◐" supported, numbers vary · "○" uncertain or our own estimate.
Human counts are per brain (both sides) unless noted.

Where human numbers don't exist we use primate, then rodent, and say so. Percentages within a
subregion sum to 100.

Run: PYTHONPATH=prototype python3 prototype/brain/regions.py      (prints every region's composition)
"""

from dataclasses import dataclass, field

import numpy as np

import celltypes as C


@dataclass
class Subregion:
    name: str
    human_neurons: float | None            # total neurons in humans (None = not known)
    cells: list                            # [(CellType, percent, species, confidence)]
    note: str = ""

    def check(self):
        total = sum(p for _, p, _, _ in self.cells)
        assert abs(total - 100) < 0.6, f"{self.name}: percentages sum to {total}"


@dataclass
class Region:
    name: str
    subregions: list = field(default_factory=list)
    role: str = ""


H, P, R, M = "human", "primate", "rat", "mouse"

# ---------------------------------------------------------------------------------------------
# Neocortex (notes/regions/neocortex.md). Interneurons are 33% of neurons in human cortex (human M1,
# Bakken 2021; mouse 16%). PV:SST:5HT3aR = 40:30:30 of interneurons (mouse; human split unknown).
# Layer 5 PT/ET cells: 2-6% of layer 5 excitatory neurons in humans (20-30% in mouse).
# Excitatory layer shares are our arithmetic from a rat S1 column count (○).
# ---------------------------------------------------------------------------------------------
_INTERNEURONS_CORTEX = [
    (C.PV, 12.2, M + " ratios x human total", "◐"),
    (C.CHANDELIER, 1.0, M, "○"),
    (C.SST, 6.0, M, "◐"),                      # Martinotti (~60% of SST, unverified)
    (C.SST_NONMARTINOTTI, 3.9, M, "○"),
    (C.VIP, 4.0, M, "●"),                      # ~12% of interneurons
    (C.LAMP5_NGF, 4.0, M, "◐"),
    (C.CCK_BASKET, 1.4, M, "○"),
    (C.ROSEHIP, 0.5, H, "◐"),                  # 10-15% of human layer-1 interneurons
]
SENSORY_CORTEX = Subregion("sensory cortex (e.g. V1, S1)", None, [
    (C.L23_IT, 26.0, R + " column, human-adjusted", "○"),
    (C.L4_STELLATE, 16.0, R, "○"),
    (C.L5_IT, 9.0, R + "/" + H, "○"),
    (C.L5_PT, 0.5, H, "◐"),
    (C.NP, 1.0, M, "○"),
    (C.L6_CT, 6.0, R, "○"),
    (C.L6_IT, 7.0, R, "○"),
    (C.L6B, 1.5, R, "○"),
] + _INTERNEURONS_CORTEX, "thick layer 4 receives the thalamus; in primate V1 layer 4 is subdivided (4A/4B/4Calpha/4Cbeta)")
MOTOR_PREFRONTAL_CORTEX = Subregion("motor / prefrontal cortex", None, [
    (C.L23_IT, 32.0, R + " column, human-adjusted", "○"),
    (C.L4_STELLATE, 3.0, H + "/" + M, "◐"),     # "L4-like cells, much sparser" in M1
    (C.L5_IT, 14.0, R + "/" + H, "○"),
    (C.L5_PT, 0.7, H, "◐"),                     # incl. Betz cells in M1
    (C.NP, 1.3, M, "○"),
    (C.L6_CT, 6.0, R, "○"),
    (C.L6_IT, 8.5, R, "○"),
    (C.L6B, 1.5, R, "○"),
] + _INTERNEURONS_CORTEX, "layer 4 thin or absent; layer 2/3 relatively larger in humans")
CINGULATE_INSULA = Subregion("anterior cingulate / frontoinsular cortex", None, [
    (C.L23_IT, 32.0, R + " column, human-adjusted", "○"),
    (C.L4_STELLATE, 3.0, H, "○"),
    (C.L5_IT, 13.0, R + "/" + H, "○"),
    (C.VON_ECONOMO, 1.0, H, "◐"),               # 0.9-1.6% frontoinsula, 0.6-0.9% ACC; 20-40% more on the right
    (C.L5_PT, 0.7, H, "◐"),
    (C.NP, 1.3, M, "○"),
    (C.L6_CT, 6.0, R, "○"),
    (C.L6_IT, 8.5, R, "○"),
    (C.L6B, 1.5, R, "○"),
] + _INTERNEURONS_CORTEX, "von Economo neurons: emotional awareness, interoception (notes/03)")
NEOCORTEX = Region("neocortex", [SENSORY_CORTEX, MOTOR_PREFRONTAL_CORTEX, CINGULATE_INSULA],
                   "perception, knowledge, planning; ~16 billion neurons in humans (Azevedo 2009)")

# ---------------------------------------------------------------------------------------------
# Thalamus (notes/regions/thalamus.md). Interneurons: 25-35% in macaque LGN, ~6% mouse dLGN,
# ~2% mouse LP; human share unknown. Core vs matrix proportions unknown (○).
# Driver synapses on relay cells: ~7% (first-order) vs ~2% (higher-order); layer 6: 30-50%.
# ---------------------------------------------------------------------------------------------
FIRST_ORDER_THALAMUS = Subregion("first-order thalamus (LGN, VPM/VPL, MGBv)", None, [
    (C.TC_CORE, 55.0, P + ", core/matrix split ○", "○"),
    (C.TC_MATRIX, 20.0, P, "○"),
    (C.TH_INTERNEURON, 25.0, P + " LGN (25-35%)", "◐"),
], "drivers from the senses (~7% of relay synapses); layer 6 modulators 30-50%")
HIGHER_ORDER_THALAMUS = Subregion("higher-order thalamus (pulvinar, MD, POm)", None, [
    (C.TC_CORE, 45.0, P, "○"),
    (C.TC_MATRIX, 40.0, P, "○"),
    (C.TH_INTERNEURON, 15.0, P, "○"),
], "drivers from cortical layer 5 PT (~2% of relay synapses); most of the primate thalamus by volume")
TRN_REGION = Subregion("thalamic reticular nucleus", None, [(C.TRN, 100.0, "all", "●")],
                       "100% GABAergic; projects only to the dorsal thalamus")
THALAMUS = Region("thalamus", [FIRST_ORDER_THALAMUS, HIGHER_ORDER_THALAMUS, TRN_REGION],
                  "relay and gate between senses, cortex and subcortex")

CLAUSTRUM = Region("claustrum", [Subregion("claustrum", None, [
    (C.CLAUSTRUM_PN, 88.0, "analogy with cortex (no unbiased count)", "○"),
    (C.PV, 3.5, M + " (22-29% of interneurons)", "◐"),
    (C.SST, 4.5, M, "◐"),
    (C.VIP, 4.0, M, "◐"),
], "drives cortical interneurons; synchronous firing triggers cortical down-states")],
    "coordinates cortex-wide state")

# ---------------------------------------------------------------------------------------------
# Hippocampal formation (notes/regions/hippocampus.md). Human counts: West & Gundersen 1990.
# Interneuron shares: rat (Bezaire & Soltesz 2013); no human interneuron census exists.
# ---------------------------------------------------------------------------------------------
ENTORHINAL = Subregion("entorhinal cortex", 7e6, [
    (C.EC_STELLATE, 7.0, R, "◐"),              # ~2/3 of MEC L2 principal cells
    (C.EC_ISLAND, 3.5, R, "◐"),
    (C.EC_L3, 33.0, R, "◐"),
    (C.EC_L5, 41.0, R, "◐"),
    (C.PV, 7.0, "rodent (assumed cortical-typical)", "○"),
    (C.SST, 4.0, "rodent", "○"),
    (C.VIP, 2.5, "rodent", "○"),
    (C.LAMP5_NGF, 2.0, "rodent", "○"),
], "human total 7-13 million (two studies disagree); grid cells ~10-30% of MEC L2/3 (○), border cells ~10%")
DENTATE_GYRUS = Subregion("dentate gyrus", 17e6, [
    (C.DG_GRANULE, 92.5, H + " (15 M granule cells)", "●"),
    (C.DG_GRANULE_YOUNG, 2.5, R + "; disputed in adult humans", "○"),
    (C.MOSSY_CELL, 3.0, R + " (~64% of hilar neurons)", "◐"),
    (C.HIPP, 1.2, R, "○"),
    (C.PV, 0.6, R, "○"),
    (C.CHANDELIER, 0.2, R, "○"),
], "human: ~700 new neurons/day by 14C dating (Spalding 2013) vs undetectable after childhood (Sorrells 2018)")
CA3 = Subregion("CA3", 2.4e6, [
    (C.CA3_PYR, 89.0, R, "◐"),
    (C.HC_PV_BASKET, 1.6, R + " (CA1 ratios applied)", "○"),
    (C.HC_AXOAXONIC, 0.4, R, "○"),
    (C.BISTRATIFIED, 1.1, R, "○"),
    (C.OLM, 1.0, R, "○"),
    (C.CCK_BASKET, 1.5, R, "○"),
    (C.IVY, 2.5, R, "○"),
    (C.HC_NGF, 1.0, R, "○"),
    (C.IS_CELL, 1.9, R, "○"),
], "recurrent collaterals: ~0.9% connection probability (Guzman 2016); human CA3+CA2 = 2.7 M")
CA2 = Subregion("CA2", 0.3e6, [
    (C.CA2_PYR, 89.0, R, "○"),
    (C.HC_PV_BASKET, 11.0, "rodent (dense PV plexus)", "○"),
], "social memory; counted with CA3 in humans")
CA1 = Subregion("CA1", 16e6, [
    (C.CA1_PYR, 89.0, R + "; human 16 M pyramids (24 M at 47 y -> 11 M at 85 y)", "●"),
    (C.HC_PV_BASKET, 1.6, R, "◐"),
    (C.HC_AXOAXONIC, 0.4, R, "◐"),
    (C.BISTRATIFIED, 1.1, R + " (incl. CCK dendrite-targeting 0.5%)", "◐"),
    (C.OLM, 1.0, R + " (incl. SST long-range 0.5%)", "◐"),
    (C.CCK_BASKET, 1.0, R, "◐"),
    (C.IVY, 2.6, R + " (most numerous: 23% of interneurons)", "◐"),
    (C.HC_NGF, 1.1, R, "◐"),
    (C.IS_CELL, 2.2, R, "◐"),
], "one-shot plateau learning (BTSP); compares CA3 (basal) with entorhinal input (apical tufts)")
SUBICULUM = Subregion("subiculum", 4.5e6, [
    (C.SUB_BURST, 58.0, R, "◐"),
    (C.SUB_REGULAR, 32.0, R, "◐"),
    (C.PV, 5.0, "rodent (assumed ~10% interneurons)", "○"),
    (C.OLM, 3.0, "rodent", "○"),
    (C.HC_NGF, 2.0, "rodent", "○"),
], "human count ~4.5 M (○, from memory of West & Gundersen); main hippocampal output")
HIPPOCAMPUS = Region("hippocampal formation", [ENTORHINAL, DENTATE_GYRUS, CA3, CA2, CA1, SUBICULUM],
                     "episodic memory, spatial maps, one-shot learning, replay in sleep")

# ---------------------------------------------------------------------------------------------
# Amygdala (notes/regions/amygdala.md). Human: 12.2 M neurons (Schumann & Amaral 2005).
# ---------------------------------------------------------------------------------------------
BLA = Subregion("basolateral amygdala (lateral + basal + accessory basal)", 8.5e6, [
    (C.BLA_POSITIVE, 41.75, R + "; valence split assumed 50/50", "○"),
    (C.BLA_NEGATIVE, 41.75, R, "○"),
    (C.PV, 7.0, R + " (~40-50% of interneurons)", "◐"),
    (C.SST, 2.5, R, "○"),
    (C.CCK_BASKET, 2.0, R, "○"),
    (C.IS_CELL, 4.5, R + " (CR/VIP: inhibit other interneurons)", "○"),
    (C.LAMP5_NGF, 0.5, "rodent", "○"),
], "~80-85% principal cells; learns what things mean (good/bad)")
ITC_REGION = Subregion("intercalated cell masses", None, [(C.ITC, 100.0, "rodent/primate", "●")],
                       "no reliable count in any species; gates BLA -> central amygdala; fear extinction")
CEA = Subregion("central amygdala", 0.36e6, [
    (C.CEL_PKCD, 33.0, M + " (~50% of CeL GABA cells)", "◐"),
    (C.CEL_SST, 25.0, M, "○"),
    (C.CEA_CRF, 10.0, "mouse/rat", "○"),
    (C.CEM_OUTPUT, 27.0, "rodent", "○"),
    (C.VTA_GLU, 5.0, "rodent (VGluT2 minority)", "○"),
], "~95% GABAergic; output to PAG (freezing), hypothalamus, brainstem (heart rate)")
AMYGDALA = Region("amygdala", [BLA, ITC_REGION, CEA], "threat, importance, emotional tagging of memory")

# ---------------------------------------------------------------------------------------------
# Basal ganglia (notes/regions/basal-ganglia.md). Human striatum: ~82 M neurons (Heinsen 1994);
# interneurons 17.1% by unbiased stereology (Lecumberri 2018); rodents ~5%.
# ---------------------------------------------------------------------------------------------
STRIATUM = Subregion("striatum (caudate, putamen, accumbens)", 82e6, [
    (C.D1_MSN, 36.5, H + " total; D1:D2 ~1:1 (mouse)", "◐"),
    (C.D2_MSN, 36.5, H + " total; D1:D2 ~1:1 (mouse)", "◐"),
    (C.STRIOSOME_MSN, 9.9, "striosomes ~10-15% of volume", "○"),
    (C.STR_CR, 8.0, H + " (calretinin most common in primates)", "○"),
    (C.STR_FSI, 4.0, H + " split unknown", "○"),
    (C.STR_LTS, 3.0, H + " split unknown", "○"),
    (C.TAN, 1.5, H + " (~1-2%)", "◐"),
    (C.STR_TH, 0.6, "mouse", "○"),
], "human interneurons 17.1% (associative 21.9%, sensorimotor 12.8%, limbic 11.1%); rodent ~5%")
GPE = Subregion("globus pallidus externa", None, [
    (C.GPE_PROTO, 73.0, R + "/" + M + " (incl. ~8% other subtypes)", "●"),
    (C.GPE_ARKY, 27.0, R + "/" + M, "●"),
], "prototypic -> STN, GPi/SNr; arkypallidal -> striatum only ('stop')")
GPI = Subregion("globus pallidus interna", None, [(C.GPI, 100.0, P, "●")], "tonic inhibition of motor thalamus")
STN_REGION = Subregion("subthalamic nucleus", 0.9e6, [(C.STN, 100.0, H + " (240-560 k per side)", "◐")],
                       "the only excitatory nucleus of the basal ganglia; hyperdirect pathway from cortex")
SNR = Subregion("substantia nigra pars reticulata", None, [
    (C.SNR, 70.0, R, "●"), (C.DA_NEURON, 30.0, R + " (displaced dopamine cells)", "◐")])
SNC = Subregion("substantia nigra pars compacta", 0.55e6, [
    (C.DA_NEURON, 71.0, R + "; human ~550 k pigmented SN neurons (Pakkenberg 1991)", "●"),
    (C.VTA_GABA, 29.0, R, "●"),
], "dopamine for movement and habits; lost in Parkinson's disease")
BASAL_GANGLIA = Region("basal ganglia", [STRIATUM, GPE, GPI, STN_REGION, SNR, SNC],
                       "action selection, habits, reward learning (direct = go, indirect = no-go, hyperdirect = stop)")

# ---------------------------------------------------------------------------------------------
# Cerebellum (notes/regions/cerebellum.md). 69 of 86 billion human neurons (Azevedo 2009).
# Per Purkinje cell: ~3,300 granule (human), ~2.7 Golgi, ~6 basket, ~17 stellate (cat).
# ---------------------------------------------------------------------------------------------
_per_pc = {"granule": 3300, "purkinje": 1, "golgi": 2.7, "basket": 6, "stellate": 17, "ubc": 0.5, "lugaro": 0.1}
_tot = sum(_per_pc.values())
CEREBELLAR_CORTEX = Subregion("cerebellar cortex", 69e9, [
    (C.CB_GRANULE, 100 * _per_pc["granule"] / _tot, H, "●"),
    (C.PURKINJE, 100 * _per_pc["purkinje"] / _tot, H + " (15-30 M)", "●"),
    (C.GOLGI, 100 * _per_pc["golgi"] / _tot, H + " (~81 M, secondary source)", "◐"),
    (C.CB_BASKET, 100 * _per_pc["basket"] / _tot, "cat", "◐"),
    (C.CB_STELLATE, 100 * _per_pc["stellate"] / _tot, "cat", "◐"),
    (C.UBC, 100 * _per_pc["ubc"] / _tot, "rodent (dense only in vestibulocerebellum)", "○"),
    (C.LUGARO, 100 * _per_pc["lugaro"] / _tot, "rat (rare)", "○"),
], "granule cells are ~80% of all neurons in the human brain; each Purkinje cell gets exactly one climbing fiber")
DCN = Subregion("deep cerebellar nuclei", None, [
    (C.DCN_PROJ, 70.0, M + " (majority; exact share not found)", "○"),
    (C.DCN_NO, 20.0, M, "○"),
    (C.GOLGI, 10.0, M + " (local GABA/glycine interneurons)", "○"),
], "cerebellar output; rebound firing after Purkinje pauses")
INFERIOR_OLIVE = Subregion("inferior olive", None, [(C.OLIVE, 100.0, "all", "●")],
                           "climbing fibers: error/teaching signal; each olive cell reaches ~14-17 Purkinje cells (human)")
CEREBELLUM = Region("cerebellum", [CEREBELLAR_CORTEX, DCN, INFERIOR_OLIVE],
                    "prediction and error correction for movement, timing, and thought")

# ---------------------------------------------------------------------------------------------
# Neuromodulatory nuclei (notes/regions/neuromodulatory-nuclei.md)
# ---------------------------------------------------------------------------------------------
VTA = Subregion("ventral tegmental area", None, [
    (C.DA_NEURON, 64.0, R + " (Nair-Roberts 2008: 40,174 DA)", "●"),
    (C.VTA_GABA, 33.5, R, "●"),
    (C.VTA_GLU, 2.5, R, "◐"),
], "reward prediction error -> accumbens, prefrontal cortex, amygdala, hippocampus")
LC = Subregion("locus coeruleus", 5e4, [
    (C.LC_NE, 95.0, H + " (45-54 k)", "●"),
    (C.VTA_GABA, 5.0, "rodent peri-LC GABA (not quantified)", "○"),
], "arousal, surprise; projects to the whole cortex")
DORSAL_RAPHE = Subregion("dorsal raphe", 4e5, [
    (C.RAPHE_5HT, 40.0, H + " count 165 k; share ○", "○"),
    (C.VTA_GABA, 40.0, "rodent", "○"),
    (C.VTA_GLU, 15.0, "rodent", "○"),
    (C.DA_NEURON, 5.0, H + " (5,600 DA cells)", "◐"),
], "serotonin to forebrain; many 5-HT cells co-release glutamate")
MEDIAN_RAPHE = Subregion("median raphe", None, [
    (C.RAPHE_5HT, 8.5, M + "; human 64 k 5-HT", "◐"),
    (C.VTA_GLU, 26.0, M + " (VGluT3)", "◐"),
    (C.VTA_GABA, 37.0, M, "◐"),
    (C.VTA_GLU, 28.5, M + " (triple-negative, likely VGluT2)", "○"),
], "-> hippocampus and septum")
NUCLEUS_BASALIS = Subregion("nucleus basalis of Meynert", 4e5, [
    (C.BF_ACH, 90.0, H + " (~200 k cholinergic per hemisphere)", "●"),
    (C.VTA_GABA, 10.0, H, "◐"),
], "acetylcholine to the entire neocortex and amygdala; lost early in Alzheimer's disease")
MEDIAL_SEPTUM = Subregion("medial septum / diagonal band", None, [
    (C.PPN_ACH, 25.0, R + " (ratio disputed)", "○"),
    (C.SEPTUM_PACEMAKER, 50.0, R, "◐"),
    (C.VTA_GLU, 25.0, R, "◐"),
], "theta pacemaker for the hippocampus; acetylcholine sets encoding mode")
PPN_LDT = Subregion("pedunculopontine / laterodorsal tegmental", None, [
    (C.PPN_ACH, 24.0, R + " (PPN 25%, LDT 22%)", "◐"),
    (C.VTA_GLU, 38.0, R, "◐"),
    (C.VTA_GABA, 38.0, R, "◐"),
], "arousal, REM sleep, locomotion")
NEUROMODULATORY = Region("neuromodulatory nuclei",
                         [VTA, LC, DORSAL_RAPHE, MEDIAN_RAPHE, NUCLEUS_BASALIS, MEDIAL_SEPTUM, PPN_LDT],
                         "the brain's chemical messengers (notes/07, notes/08)")

# ---------------------------------------------------------------------------------------------
# Hypothalamus (notes/regions/hypothalamus.md). Only the modeled functional populations are listed;
# most hypothalamic neurons are other GABA/glutamate types that aren't counted.
# ---------------------------------------------------------------------------------------------
ARCUATE = Subregion("arcuate nucleus (modeled populations)", None, [
    (C.AGRP, 47.0, M + " (~8,000 NPY)", "◐"), (C.POMC, 53.0, M + " (~9,000 POMC)", "◐")],
    "human arcuate ~510 k neurons in total; peptide counts not available")
LATERAL_HYPOTHALAMUS = Subregion("lateral hypothalamus (modeled populations)", None, [
    (C.OREXIN, 33.0, H + " ~70 k orexin; MCH 2-3x orexin (rat)", "◐"), (C.MCH, 67.0, R, "◐")],
    "orexin stabilizes wakefulness (85-95% lost in narcolepsy); MCH promotes sleep")
PVN = Subregion("paraventricular nucleus (modeled populations)", None, [
    (C.CRH, 40.0, "no human counts", "○"), (C.OXYTOCIN, 30.0, "no human counts", "○"),
    (C.VASOPRESSIN, 30.0, "no human counts", "○")], "stress axis; bonding and caregiving")
SCN = Subregion("suprachiasmatic nucleus", 1.7e4, [
    (C.SCN_CLOCK, 80.0, H + " (6,890 AVP per side)", "●"),
    (C.SCN_CLOCK, 20.0, H + " (1,700 VIP per side; light-receiving core)", "●")],
    "the master clock; nearly all GABAergic")
HYPOTHALAMUS = Region("hypothalamus", [ARCUATE, LATERAL_HYPOTHALAMUS, PVN, SCN],
                      "drives and homeostasis: hunger, sleep/wake, stress, bonding, daily rhythm")

REGIONS = [NEOCORTEX, THALAMUS, CLAUSTRUM, HIPPOCAMPUS, AMYGDALA, BASAL_GANGLIA, CEREBELLUM,
           NEUROMODULATORY, HYPOTHALAMUS]


def allocate(sub: Subregion, n: int, minimum: int = 1) -> list:
    """Split a neuron budget across a subregion's cell types by percentage. Every type gets at least
    `minimum` cells so rare types (Purkinje cells, chandeliers, von Economo neurons) still exist."""
    pct = np.array([p for _, p, _, _ in sub.cells])
    counts = np.maximum(minimum, np.floor(pct / 100 * n)).astype(int)
    extra = n - counts.sum()
    if extra > 0:
        counts[np.argmax(pct)] += extra
    return [(c, int(k)) for (c, _, _, _), k in zip(sub.cells, counts)]


def build(sub: Subregion, n: int, rng=None):
    """Instantiate a subregion's cell groups: {type name: group}."""
    from cells import make_group
    rng = rng or np.random.default_rng(0)
    groups = {}
    for ctype, k in allocate(sub, n):
        key = ctype.name if ctype.name not in groups else f"{ctype.name} ({len(groups)})"
        groups[key] = make_group(ctype, k, rng)
    return groups


def report():
    for region in REGIONS:
        print(f"\n## {region.name}: {region.role}")
        for sub in region.subregions:
            sub.check()
            hn = f"{sub.human_neurons:,.0f} neurons in humans" if sub.human_neurons else "human count unknown"
            print(f"  {sub.name}  ({hn})")
            for ctype, p, species, conf in sub.cells:
                print(f"    {p:7.3f}%  {conf}  {ctype.name:<24} {ctype.transmitter:<14} [{species}]")
            if sub.note:
                print(f"    note: {sub.note}")


if __name__ == "__main__":
    report()

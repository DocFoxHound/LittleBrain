"""
Neuron types for the brain prototype: each one modeled on a real human cell type.

A cell type is a set of parameters plus a "family" that decides which dynamics it runs
(see cells.py). Notes: notes/18-neuron-types.md (catalog) and notes/17 (wiring rules).

Rate-based model: each cell's state is a firing rate in [0, 1] updated once per step.
"""

from dataclasses import dataclass, field, replace

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuron import WORKHORSE_V2, NeuronType  # noqa: E402  (learning rules v2)


@dataclass(frozen=True)
class CellType:
    name: str
    human: str                    # the real cell it's based on
    family: str                   # which dynamics it runs; see FAMILIES in cells.py
    sign: float = 1.0             # +1 excitatory, -1 inhibitory
    layer: str = "-"
    projection: str = "local"     # "IT" | "PT" | "CT" | "local" | "TC" | "TRN"

    # Dendrites (excitatory cells only)
    n_basal: int = 8
    basal_slots: int = 16
    n_apical: int = 0             # 0 = no apical tuft (stellate, relay)
    apical_slots: int = 16
    long_range_max: float = 0.0   # max fraction of all synapse slots that come from other areas

    # Dynamics
    alpha: float = 0.5            # how fast the rate follows its drive each step (1 = instantly)
    threshold: float = 3.0
    slope: float = 1.0
    target_rate: float = 0.08     # homeostatic target (sparse)
    adapt: float = 0.0            # spike-frequency adaptation strength
    adapt_decay: float = 0.9
    apical_gain: float = 0.6      # how strongly apical (context/feedback) input multiplies output
    apical_drive: float = 0.3     # apical input also drives the cell directly, attenuated (needed for recall)
    bac_gain: float = 0.0         # extra burst when basal and apical inputs coincide (layer 5)
    burst_gain: float = 0.0       # thalamic burst mode strength (T-type calcium)
    facilitation: float = 0.0     # SST: input synapses strengthen with sustained activity
    linear: bool = False          # fast-spiking cells (PV, TRN): rate grows with drive instead of saturating
    target: str = "-"             # inhibitory cells: soma | ais | dendrite | apical | interneuron | volume
    transmitter: str = "glutamate"
    dopamine_sign: float = 1.0    # D2 cells: dopamine has the opposite effect on learning
    baseline: float = 0.0         # tonic drive: pacemaking cells fire without input
    period: float = 0.0           # oscillators: steps per cycle (clock, theta pacemaker)
    slow: float = 0.0             # neuroendocrine / volume cells: fraction of output carried over per step

    # Learning (excitatory synapses): learning rules v2
    rules: NeuronType = WORKHORSE_V2
    learns: bool = True


# --- Cortex: excitatory ------------------------------------------------------------------------

L23_IT = CellType(
    "L2/3 IT", "layer 2/3 pyramidal neuron (intratelencephalic)", "pyramidal",
    layer="2/3", projection="IT", n_apical=1, long_range_max=0.20,
    adapt=0.3, apical_gain=0.6)

L4_STELLATE = CellType(
    "L4 stellate", "layer 4 spiny stellate cell", "stellate",
    layer="4", projection="local", n_apical=0, long_range_max=0.10,  # thalamic input counts as long-range
    alpha=0.7, adapt=0.1)

L5_IT = CellType(
    "L5 IT", "layer 5A pyramidal neuron (intratelencephalic; feedback to lower areas)", "pyramidal",
    layer="5", projection="IT", n_apical=1, long_range_max=0.20, adapt=0.2, apical_gain=0.8)

L5_PT = CellType(
    "L5 PT", "layer 5B thick-tufted pyramidal neuron (pyramidal tract; output)", "pyramidal",
    layer="5", projection="PT", n_apical=1, long_range_max=0.20, adapt=0.2,
    apical_gain=0.4, bac_gain=1.5)

L6_CT = CellType(
    "L6 CT", "layer 6 corticothalamic pyramidal neuron", "pyramidal",
    layer="6", projection="CT", n_apical=0, long_range_max=0.15, alpha=0.3, adapt=0.1)

# --- Cortex: inhibitory (local) ----------------------------------------------------------------

PV = CellType(
    "PV basket", "parvalbumin fast-spiking basket cell (targets the cell body)", "pv",
    sign=-1.0, alpha=0.5, threshold=0.75, slope=4.0, learns=False, linear=True, target="soma",
    transmitter="GABA")  # alpha 0.5: GABA-A inhibition decays over ~5-10 ms; at 1.0 (instant) the cortex
                         # flip-flopped every step (notes/26)

SST = CellType(
    "SST Martinotti", "somatostatin Martinotti cell (targets dendrites and apical tufts)", "sst",
    sign=-1.0, alpha=0.2, threshold=0.75, slope=4.0, facilitation=0.15, learns=False, target="apical",
    transmitter="GABA")

VIP = CellType(
    "VIP", "vasoactive intestinal peptide interneuron (inhibits SST cells: disinhibition)", "vip",
    sign=-1.0, alpha=0.5, threshold=0.75, slope=4.0, learns=False, target="interneuron", transmitter="GABA")

# --- Thalamus ----------------------------------------------------------------------------------

TC_RELAY = CellType(
    "TC relay", "thalamocortical relay neuron (first-order thalamus)", "relay",
    layer="thalamus", projection="TC", n_basal=4, basal_slots=16, n_apical=0, long_range_max=1.0,  # all input is from outside
    alpha=0.8, threshold=1.0, slope=2.0, target_rate=0.15, burst_gain=2.0,
    rules=replace(WORKHORSE_V2, lr=0.03))  # relay synapses change slowly

TRN = CellType(
    "TRN", "thalamic reticular nucleus neuron (inhibitory gate around the thalamus)", "trn",
    sign=-1.0, layer="thalamus", projection="TRN", alpha=0.6, threshold=0.75, slope=4.0, learns=False,
    linear=True, target="soma", transmitter="GABA")

# --- Hippocampus (defined now, used later) -----------------------------------------------------

DG_GRANULE = CellType(
    "DG granule", "dentate gyrus granule cell (very sparse; pattern separation)", "granule",
    layer="DG", n_basal=4, basal_slots=16, alpha=1.0, threshold=4.0, slope=2.0, target_rate=0.02,
    rules=replace(WORKHORSE_V2, lr=0.3, encoding_learning=1.0))

DG_GRANULE_YOUNG = replace(
    DG_GRANULE, name="DG granule (young)",
    human="adult-born dentate granule cell, 4-6 weeks old (more excitable, more plastic)",
    threshold=2.5, rules=replace(WORKHORSE_V2, lr=0.6, plasticity_floor=0.8))



# ================================================================================================
# The rest of the brain (notes/regions/*.md). Each type names its real counterpart.
# ================================================================================================

def _inh(name, human, target, family="interneuron", **kw):
    """A local GABAergic interneuron defined by where it inhibits."""
    kw.setdefault("alpha", 0.6)
    return CellType(name, human, family, sign=-1.0, threshold=0.75, slope=4.0, learns=False,
                    target=target, transmitter=kw.pop("transmitter", "GABA"), **kw)


# --- Cortex: more excitatory types --------------------------------------------------------------
L6_IT = replace(L5_IT, name="L6 IT", human="layer 6 intratelencephalic pyramidal (cortico-cortical, to claustrum)",
                layer="6", n_apical=0, long_range_max=0.15)
L6B = replace(L6_CT, name="L6b", human="layer 6b subplate-remnant neuron (to L1 and L6)", projection="IT")
NP = replace(L5_IT, name="L5/6 NP", human="near-projecting pyramidal (local, few hundred micrometres)",
             projection="local", long_range_max=0.02)
VON_ECONOMO = replace(L5_PT, name="von Economo", projection="PT", alpha=0.9, adapt=0.05, bac_gain=0.5,
                      human="von Economo spindle neuron (layer 5, anterior cingulate / frontoinsula; fast long-range relay)")

# --- Cortex: more interneuron types -------------------------------------------------------------
CHANDELIER = _inh("PV chandelier", "PV axo-axonic (chandelier) cell: targets the axon initial segment", "ais",
                  alpha=1.0, linear=True)
SST_NONMARTINOTTI = _inh("SST non-Martinotti", "SST non-Martinotti cell (L4: inhibits PV cells)", "interneuron",
                         alpha=0.2, facilitation=0.15)
LAMP5_NGF = _inh("LAMP5 neurogliaform", "LAMP5/NDNF neurogliaform cell: slow volume GABA-B inhibition", "volume",
                 alpha=0.3, slow=0.9)
CCK_BASKET = _inh("CCK basket", "CCK basket cell (asynchronous, endocannabinoid-suppressed; soma)", "soma", alpha=0.4)
ROSEHIP = _inh("rosehip", "human layer 1 rosehip neuron (targets apical dendrite shafts)", "apical", alpha=0.6)

# --- Thalamus ------------------------------------------------------------------------------------
TC_CORE = replace(TC_RELAY, name="TC core", human="thalamic core relay cell (PV+; to layer 4, topographic)")
TC_MATRIX = replace(TC_RELAY, name="TC matrix", human="thalamic matrix relay cell (calbindin+; to layer 1, diffuse)")
TH_INTERNEURON = _inh("thalamic interneuron", "thalamic local interneuron (dendritic GABA release in triads)",
                      "soma", alpha=0.6)

# --- Hippocampal formation ---------------------------------------------------------------------
EC_STELLATE = replace(L4_STELLATE, name="EC L2 stellate", projection="IT", long_range_max=0.3,
                      human="medial entorhinal layer II stellate cell (grid-like; to dentate gyrus and CA3)")
EC_ISLAND = replace(L23_IT, name="EC L2 island", human="entorhinal layer II island (calbindin) pyramidal cell (to CA1)")
EC_L3 = replace(L23_IT, name="EC L3 pyramidal", human="entorhinal layer III pyramidal (temporoammonic path to CA1)")
EC_L5 = replace(L5_IT, name="EC L5/6 pyramidal", human="entorhinal deep-layer pyramidal (hippocampal output to cortex)")
MOSSY_CELL = replace(L23_IT, name="DG mossy cell", n_apical=0, layer="hilus",
                     human="hilar mossy cell (excitatory; associational loop back to granule cells)")
HIPP = _inh("HIPP", "hilar perforant-path-associated (SST) interneuron", "dendrite", alpha=0.2, facilitation=0.15)
CA3_PYR = replace(L23_IT, name="CA3 pyramidal", layer="CA3", adapt=0.15, bac_gain=0.8, long_range_max=0.4,
                  rules=replace(WORKHORSE_V2, lr=0.4, cooperativity=0.5, branch_lr=0.05,  # fast autoassociation
                                encoding_learning=1.0),
                  human="CA3 pyramidal cell (recurrent autoassociative network; bursts)")
CA2_PYR = replace(L23_IT, name="CA2 pyramidal", layer="CA2", human="CA2 pyramidal cell (social memory)")
CA1_PYR = replace(L23_IT, name="CA1 pyramidal", layer="CA1", long_range_max=0.5,
                  rules=replace(WORKHORSE_V2, oneshot_level=0.45, encoding_learning=1.0),
                  human="CA1 pyramidal cell (one-shot plateau learning, BTSP; compares CA3 vs entorhinal input)")
SUB_BURST = replace(L5_PT, name="subiculum bursting", layer="subiculum", long_range_max=0.5, bac_gain=1.2,
                    human="subicular intrinsically bursting pyramidal cell (hippocampal output)")
SUB_REGULAR = replace(L5_IT, name="subiculum regular", layer="subiculum", long_range_max=0.5,
                      human="subicular regular-spiking pyramidal cell")
HC_PV_BASKET = replace(PV, name="CA PV basket", human="hippocampal PV basket cell")
HC_AXOAXONIC = replace(CHANDELIER, name="CA axo-axonic", human="hippocampal axo-axonic cell")
BISTRATIFIED = _inh("bistratified", "hippocampal bistratified cell (PV/SST; dendrites alongside CA3 input)", "dendrite")
OLM = _inh("O-LM", "oriens-lacunosum moleculare cell (SST; gates entorhinal input on distal tufts)", "apical",
           alpha=0.2, facilitation=0.15)
IVY = _inh("ivy", "ivy cell (NPY/nNOS; slow dendritic inhibition; most numerous CA1 interneuron)", "volume",
           alpha=0.3, slow=0.8)
HC_NGF = replace(LAMP5_NGF, name="hippocampal neurogliaform", human="hippocampal neurogliaform cell (volume GABA-B)")
IS_CELL = _inh("interneuron-specific", "interneuron-specific cell (CR/VIP; inhibits other interneurons)", "interneuron")

# --- Amygdala ------------------------------------------------------------------------------------
BLA_RULES = replace(WORKHORSE_V2, lr=0.4, arousal_gated=True)  # few-trial learning, gated by noradrenaline
BLA_POSITIVE = replace(L23_IT, name="BLA principal (+)", layer="BLA", long_range_max=0.4, rules=BLA_RULES,
                       human="basolateral amygdala principal neuron, positive valence (Ppp1r1b)")
BLA_NEGATIVE = replace(L23_IT, name="BLA principal (-)", layer="BLA", long_range_max=0.4, rules=BLA_RULES,
                       human="basolateral amygdala principal neuron, negative valence (Rspo2)")
# Learning inhibitory projection cells (family "learning_inh"): GABA output, but their input synapses learn
# (notes/regions/synapse-budget.md): ITC LTP underlies extinction; CeL SST synapses potentiate in fear learning
ITC = replace(_inh("intercalated", "intercalated cell (GABA; gates BLA to central amygdala; extinction)", "soma",
                   family="learning_inh"), learns=True)
CEL_PKCD = _inh("CeL PKCδ (off)", "central amygdala PKCδ cell ('fear-off'; inhibits CeM output)", "soma",
                family="projection_inh", baseline=0.9)  # tonically active: keeps fear output off
CEL_SST = replace(_inh("CeL SST (on)", "central amygdala SST cell ('fear-on'; inhibits PKCδ cells)", "soma",
                       family="learning_inh"), learns=True)
CEA_CRF = replace(CEL_SST, name="CeA CRF", human="central amygdala CRF cell (sustained anxiety)", slow=0.9,
                  family="projection_inh", learns=False)
CEM_OUTPUT = _inh("CeM output", "central amygdala medial output cell (to PAG: freezing; to hypothalamus)", "soma",
                  family="tonic", baseline=0.9)

# --- Basal ganglia ------------------------------------------------------------------------------
D1_MSN = CellType("D1 MSN", "striatal direct-pathway medium spiny neuron (D1; to GPi/SNr)", "msn",
                  sign=-1.0, transmitter="GABA", projection="direct", n_basal=8, basal_slots=16,
                  long_range_max=0.8, threshold=4.0, slope=3.0, target_rate=0.03, dopamine_sign=+1.0,
                  rules=replace(WORKHORSE_V2, elig_decay=0.85))  # seconds-long trace: dopamine can arrive late
D2_MSN = replace(D1_MSN, name="D2 MSN", projection="indirect", dopamine_sign=-1.0,
                 human="striatal indirect-pathway medium spiny neuron (D2; to GPe)")
STRIOSOME_MSN = replace(D1_MSN, name="striosome MSN", projection="striosome",
                        human="striosomal medium spiny neuron (limbic input; projects to SNc dopamine cells)")
TAN = CellType("cholinergic TAN", "striatal cholinergic tonically active neuron (pauses on salient events)",
               "tonic", transmitter="acetylcholine", baseline=0.0, alpha=0.5, threshold=0.0, slope=4.0,
               burst_gain=1.0, learns=False)
STR_FSI = replace(PV, name="striatal PV FSI", human="striatal PV fast-spiking interneuron (feedforward to MSN somata)")
STR_LTS = _inh("striatal SST/NPY/NOS", "striatal low-threshold-spiking interneuron (SST/NPY/NOS)", "dendrite")
STR_CR = _inh("striatal calretinin", "striatal calretinin interneuron (most common in primates)", "soma")
STR_TH = _inh("striatal TH", "striatal TH interneuron", "soma")
GPE_PROTO = _inh("GPe prototypic", "globus pallidus externa prototypic neuron (PV; to STN, GPi/SNr)", "soma",
                 family="tonic", baseline=0.7)
GPE_ARKY = _inh("GPe arkypallidal", "globus pallidus externa arkypallidal neuron (FoxP2; back to striatum: 'stop')",
                "soma", family="tonic", baseline=0.3)
GPI = _inh("GPi", "globus pallidus interna output neuron (tonic inhibition of motor thalamus)", "soma",
           family="tonic", baseline=0.7)
SNR = replace(GPI, name="SNr", human="substantia nigra pars reticulata output neuron (to thalamus, superior colliculus)")
STN = CellType("STN", "subthalamic nucleus neuron (excitatory; rebound bursts; hyperdirect pathway)", "rebound",
               alpha=0.8, threshold=0.5, slope=4.0, burst_gain=2.0, baseline=0.2, learns=False)

# --- Cerebellum ---------------------------------------------------------------------------------
CB_GRANULE = replace(DG_GRANULE, name="cerebellar granule", n_basal=4, basal_slots=1, threshold=1.0,
                     target_rate=0.05, learns=False, layer="cerebellum",
                     human="cerebellar granule cell (4 short dendrites, each from one mossy fiber; expansion recoding)")
PURKINJE = CellType("Purkinje", "Purkinje cell (inhibitory output of cerebellar cortex; climbing-fiber-gated LTD)",
                    "purkinje", sign=-1.0, transmitter="GABA", projection="DCN", n_basal=8, basal_slots=32,
                    baseline=0.5, alpha=0.8, threshold=0.0, slope=3.0, learns=True,
                    rules=replace(WORKHORSE_V2, lr=0.05, post_offset=1.0))
GOLGI = _inh("Golgi", "cerebellar Golgi cell (feedback inhibition of granule cells)", "soma", alpha=0.5)
CB_BASKET = _inh("cerebellar basket", "cerebellar basket cell (pinceau around Purkinje axon initial segment)", "ais",
                 alpha=0.9)
CB_STELLATE = _inh("cerebellar stellate", "cerebellar stellate cell (Purkinje dendrites)", "dendrite", alpha=0.7)
LUGARO = _inh("Lugaro", "Lugaro cell (inhibits Golgi cells and interneurons; serotonin-driven)", "interneuron")
UBC = CellType("unipolar brush", "unipolar brush cell (excitatory; amplifies and prolongs mossy-fiber input)",
               "tonic", baseline=0.0, alpha=0.2, threshold=0.0, slope=4.0, slow=0.8, learns=False)
DCN_PROJ = CellType("DCN projection", "deep cerebellar nucleus projection neuron (glutamate; rebound after Purkinje pause)",
                    "rebound", alpha=0.8, threshold=0.5, slope=4.0, burst_gain=2.0, baseline=0.3, learns=True,
                    projection="thalamus")
DCN_NO = _inh("nucleo-olivary", "nucleo-olivary GABA neuron (silences the inferior olive for learned errors)",
              "soma", family="projection_inh")
OLIVE = CellType("inferior olive", "inferior olive neuron (climbing fiber: error/teaching signal; ~1 Hz; gap-junction coupled)",
                 "olive", alpha=1.0, threshold=0.3, slope=6.0, period=10.0, learns=False, projection="climbing fiber")

# --- Neuromodulatory nuclei ---------------------------------------------------------------------
DA_NEURON = CellType("dopamine", "midbrain dopamine neuron (VTA/SNc; reward prediction error)", "dopamine",
                     transmitter="dopamine", baseline=0.3, alpha=1.0, learns=False, projection="modulatory")
VTA_GABA = _inh("VTA GABA", "VTA GABA neuron (expectation: subtracts predicted reward from dopamine cells)", "soma")
VTA_GLU = replace(L23_IT, name="VTA glutamate", human="VTA glutamate (VGluT2) neuron", n_apical=0, learns=False)
LC_NE = replace(DA_NEURON, name="LC noradrenaline", transmitter="noradrenaline", baseline=0.2,
                human="locus coeruleus noradrenergic neuron (arousal; phasic on surprise)")
RAPHE_5HT = replace(DA_NEURON, name="raphe serotonin", transmitter="serotonin", baseline=0.3, slow=0.95,
                    human="raphe serotonergic neuron (slow; patience, average reward)")
BF_ACH = replace(DA_NEURON, name="basal forebrain ACh", transmitter="acetylcholine", baseline=0.4,
                 human="nucleus basalis cholinergic neuron (attention; encode vs recall)")
SEPTUM_PACEMAKER = CellType("septal PV pacemaker", "medial septum PV GABA neuron (theta pacemaker)", "pacemaker",
                            sign=-1.0, transmitter="GABA", period=8.0, learns=False, target="interneuron")
PPN_ACH = replace(BF_ACH, name="PPN/LDT ACh", human="pedunculopontine / laterodorsal cholinergic neuron (REM, arousal)")

# --- Hypothalamus ------------------------------------------------------------------------------
AGRP = CellType("AgRP/NPY", "arcuate AgRP/NPY neuron (hunger)", "endocrine", sign=-1.0, transmitter="GABA",
                alpha=0.05, slow=0.98, baseline=0.0, learns=False)
POMC = replace(AGRP, name="POMC", sign=1.0, transmitter="glutamate", human="arcuate POMC neuron (satiety)")
OREXIN = CellType("orexin", "lateral hypothalamus orexin/hypocretin neuron (stabilizes wakefulness)", "tonic",
                  baseline=0.5, alpha=0.1, threshold=0.0, slope=4.0, learns=False)
MCH = replace(OREXIN, name="MCH", sign=-1.0, transmitter="GABA", baseline=0.1,
              human="lateral hypothalamus MCH neuron (promotes sleep, REM)")
CRH = replace(AGRP, name="PVN CRH", sign=1.0, transmitter="glutamate", human="paraventricular CRH neuron (stress axis)")
OXYTOCIN = replace(CRH, name="PVN oxytocin", human="paraventricular oxytocin neuron (bonding, caregiving)")
VASOPRESSIN = replace(CRH, name="PVN/SON vasopressin", human="magnocellular vasopressin neuron (water balance, social)")
SCN_CLOCK = CellType("SCN clock", "suprachiasmatic clock neuron (AVP/VIP; ~24 h rhythm, entrained by light)",
                     "pacemaker", sign=-1.0, transmitter="GABA", period=2400.0, learns=False)

# --- Claustrum ---------------------------------------------------------------------------------
CLAUSTRUM_PN = replace(L6_IT, name="claustrum projection", layer="claustrum", long_range_max=0.6,
                       human="claustrum projection neuron (to cortex; drives cortical interneurons)")

# --- Types added for the full federated brain (notes/regions/connections.md, synapse-budget.md) ---------------
PONTINE = replace(TC_RELAY, name="pontine relay", layer="pons", projection="cerebellum (mossy fibers)",
                  human="pontine nucleus neuron (relays cortex L5 PT to cerebellar granule cells)", burst_gain=0.0,
                  learns=False)
LHB = replace(VTA_GLU, name="lateral habenula", layer="epithalamus", projection="RMTg",
              human="lateral habenula neuron (glutamate; fires for disappointment / punishment)", learns=True)
RMTG = _inh("RMTg GABA", "rostromedial tegmental GABA neuron (relays habenula: pauses dopamine cells)", "soma",
            family="projection_inh")
PAG = replace(_inh("PAG freezing", "ventrolateral periaqueductal gray neuron (freezing output; inhibited by CeM)",
                   "soma", family="tonic", baseline=0.2, transmitter="glutamate"), sign=1.0)
PVN_MC4R = replace(CRH, name="PVN MC4R", human="paraventricular MC4R neuron (satiety; inhibited by AgRP, excited by POMC)")
VA_VL = replace(TC_RELAY, name="VA/VL motor thalamus", human="ventral anterior/lateral motor thalamus relay "
                "(basal ganglia and cerebellar output to motor cortex)")
MD = replace(TC_RELAY, name="MD thalamus", n_apical=0, human="mediodorsal thalamus relay (prefrontal loop; working memory)",
             rules=replace(WORKHORSE_V2, lr=0.1))  # higher-order relay: more plastic than first-order
EC_L5A = replace(EC_L5, name="EC L5a pyramidal", projection="cortex",
                 human="entorhinal L5a pyramidal (hippocampal output to neocortex)")
EC_L5B = replace(EC_L5, name="EC L5b pyramidal", projection="EC superficial",
                 human="entorhinal L5b pyramidal (hippocampal output back to EC superficial layers)")

CATALOG = [v for v in list(globals().values()) if isinstance(v, CellType)]

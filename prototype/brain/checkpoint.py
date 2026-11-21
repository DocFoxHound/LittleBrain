"""Explicit records for the federated brain checkpoint."""
from dataclasses import fields

import numpy as np

import data_archive
from cells import ExcitatoryGroup, InhibitoryGroup, LeanBank, ModulatorGroup, OliveGroup, PacemakerGroup, SynapseBank
from celltypes import CellType
from federation import Brain, Group, Pooled, Spec, Synaptic, T
from neuron import NeuronType

SCHEMA = {cls: {f.name for f in fields(cls)} for cls in (NeuronType, CellType, Group, Pooled, Spec, Synaptic)}
SCHEMA[Spec].add("timed_dopamine")
SCHEMA[Brain] = set("""
_feeds _in_ref _log _posx _quiet _ravg _stats _vbar ablated adapt_rate amygdala_afferents ftile grid cue_mean cue_scale cue_integrated trace_mean cue_online critical_period_closed scale_caps trace_scale design value_afferents raw_scale value_mean
cem_base clear_context_steps clock da_base device dopamine_group dormant every food_afferents g
hold_context homeo_rule homeo_skip homeo_speed homeostasis index kernels kwta last_arousal
last_dopamine lod_frac lod_patience lod_skipped lod_total long_fraction max_scale n_neurons
n_total parallel_fiber_ltd pos r rng sensory_striatal spec structure tile
""".split())
SCHEMA[SynapseBank] = set("""
ach_suppress branch_open cascade conductance_floor elig fast gain heterosynaptic learn_mask long
paths pre src t tag thalamic_afferents transmit wave_gate
""".split())
SCHEMA[LeanBank] = set("""
M ach_suppress born branch_open branch_tags conductance_floor d def0 elig fast gain gen grace
heterosynaptic ids learn_mask long night norm path paths pending pre q qstep reset rng slot src
t tag thalamic_afferents to_rewire transmit turnover wave_gate
""".split())
SCHEMA[ExcitatoryGroup] = set("""
_spikes _xs adapt_state anorm apical avg_rate basal bnorm branch_gain branch_open branch_trace c temporal_rates post_cue
h inh_gain n nb own_threshold_rule plasticity prev_drive rate scale soma threshold
""".split())
SCHEMA[InhibitoryGroup] = set("""
c h level n rate u
""".split())
SCHEMA[ModulatorGroup] = set("""
c n rate
""".split())
SCHEMA[OliveGroup] = set("""
c n phase rate
""".split())
SCHEMA[PacemakerGroup] = set("""
c n phase rate
""".split())


def save_brain(path, brain):
    data_archive.save(path, brain, SCHEMA, "brain")


def load_brain(path):
    brain = data_archive.load(path, SCHEMA, "brain")
    try:
        _validate(brain)
    except (AttributeError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete brain checkpoint") from exc
    return brain


def _indices(values, shape, upper):
    return (isinstance(values, np.ndarray) and values.shape == shape and values.dtype.kind in "iu"
            and (not values.size or (values.min() >= 0 and values.max() < upper)))


def _validate(brain):
    if type(brain) is not Brain or type(brain.spec) is not Spec or type(brain.rng) is not np.random.Generator:
        raise ValueError("Invalid brain record")
    if brain.r.ndim != 1 or brain.n_total != brain.r.size or brain.device != "cpu":
        raise ValueError("Invalid host rate vector")
    groups = brain.spec.groups
    keys = {g.key for g in groups}
    inputs = set(brain.spec.inputs)
    if (len(groups) != len(keys) or keys != set(brain.index) or not inputs <= keys
            or keys - inputs != set(brain.g)):
        raise ValueError("Population inventory mismatch")
    offset = 0
    for spec in groups:
        key = spec.key
        gs = getattr(spec, "grid", 8)
        size = spec.per_tile * gs * gs
        index = brain.index[key]
        if (type(spec) is not Group or size <= 0 or not _indices(index, (size,), brain.n_total)
                or not np.array_equal(index, np.arange(offset, offset + size))
                or not _indices(brain.tile[key], (size,), T) or brain.pos[key].shape != (size, 2)
                or (gs != 8 and not _indices(getattr(brain, "ftile", {}).get(key), (size,), gs * gs))):
            raise ValueError("Invalid population map")
        offset += size
        if key in inputs:
            continue
        group = brain.g[key]
        if (type(group) not in (ExcitatoryGroup, InhibitoryGroup, ModulatorGroup, OliveGroup, PacemakerGroup)
                or group.n != size or group.rate.shape != (size,) or type(group.c) is not CellType):
            raise ValueError("Invalid cell population")
        if isinstance(group, ExcitatoryGroup):
            for bank in group.banks():
                if (type(bank) not in (LeanBank, SynapseBank) or bank.src.ndim != 3
                        or bank.src.shape[0] != size or not _indices(bank.src, bank.gain.shape, brain.n_total)):
                    raise ValueError("Invalid synapse array")
    if offset != brain.n_total or brain.n_neurons != sum(group.n for group in brain.g.values()):
        raise ValueError("Invalid neuron count")

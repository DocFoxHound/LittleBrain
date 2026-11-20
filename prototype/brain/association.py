"""Reduced population currents for working memory and temporal context.

The first IT cell of each PFC tile and its MD relay carry a sustained loop.
Four granule subpopulations integrate pontine input on different time scales.
These share the existing cells, map and lesion boundaries.
"""
import numpy as np

from backend import xpof
from brainspec import PFC_RECURRENT_BASAL, PFC_FROM_MD_BASAL


def target(brain, key, ordinary, channels):
    group = brain.g[key]
    xp = xpof(ordinary)
    cells = ordinary.reshape(64, -1)
    if key == "TH.core":
        driver = brain._pool("retina")
        return (0.85 * driver[:, None] + 0.15 * cells).reshape(-1)
    if key == "PFC.IT":
        relay = brain.g["MD.relay"].rate.reshape(64, -1)[:, 0]
        sensory = brain._pool("TH.core")
        # Positive feedback outlasts a stimulus, then slowly extinguishes.
        if getattr(brain, "clear_context_steps", 0):
            cells[:, 0] = 0.
        elif getattr(brain, "hold_context", False):
            # With sensory input gated out, the local recurrent and thalamic
            # afferents maintain the cue at their specified basal input shares.
            local = group.rate.reshape(64, -1)[:, 0]
            held = PFC_RECURRENT_BASAL * local + PFC_FROM_MD_BASAL * relay
            cells[:, 0] = .999 * held / (PFC_RECURRENT_BASAL + PFC_FROM_MD_BASAL)
        else:
            cells[:, 0] = xp.clip(0.985 * relay + 0.11 * sensory, 0, 1)
    elif key == "MD.relay":
        if getattr(brain, "clear_context_steps", 0):
            cells[:, 0] = 0.
            brain.clear_context_steps -= 1
        else:
            frontal = brain.g["PFC.IT"].rate.reshape(64, -1)[:, 0]
            cells[:, 0] = frontal
            if "exc" in channels and not getattr(brain, "hold_context", False):
                # Admit new cerebellar input while encoding. During the delay
                # the maintained frontal ensemble is protected from overwrite.
                drive = channels["exc"].reshape(64, -1)[:, 0]
                cells[:, 0] = (frontal + drive) / (1.0 + drive)
    elif key == "PONS.relay":
        # Preserve the sampled corticopontine current in the visual timing
        # ensemble, using the same sustained/transient mix as the sensory relay.
        cells[:, 0] = .85 * brain._pool("TH.core") + .15 * cells[:, 0]
    elif key == "CB.granule":
        if not hasattr(group, "temporal_rates"):
            group.temporal_rates = xp.zeros(4, dtype=np.float32)
        # Distributed mossy-fiber/granule response after a transient. Using a
        # continuously driven filter avoids an external clock or trial marker.
        drive = brain.g["PONS.relay"].rate.reshape(64, -1)[:, 0].max()
        group.temporal_rates += (drive - group.temporal_rates) * xp.asarray([.4, .07, .025, .009])
        cells[:, 0] = xp.repeat(group.temporal_rates, 16)
        if hasattr(brain, "parallel_fiber_ltd"):
            # A slow mossy-fiber current falls after sensory offset. Different
            # granule thresholds produce overlapping intervals without a game clock.
            group.post_cue = xp.where(drive > .35, 1., getattr(group, "post_cue", drive) * .993)
            centers = (xp.arange(8, dtype=np.float32) + .5) / 8
            phases = xp.exp(-0.5 * ((group.post_cue - centers) / .09) ** 2)
            cells[:, min(1, cells.shape[1] - 1)] = xp.repeat(phases, 8)
    elif key == "CB.Purkinje" and hasattr(brain, "parallel_fiber_ltd"):
        phase = timing_activity(brain)
        depression = xp.clip(brain.parallel_fiber_ltd @ phase, -1., 1.)
        simple = .5 - .5 * depression[xp.arange(64) % 8]
        # The timed simple-spike drive arrives on dendrites. Apply the same
        # stellate inhibition, climbing-fiber spike and axon veto as ordinary cells.
        def terminal(name):
            values = channels.get(name)
            return 0. if values is None else values.reshape(64, -1)[:, 0]

        simple /= 1.0 + 1.5 * terminal("dend")
        cells[:, 0] = group.spike_output(simple, terminal("cf"), terminal("ais"))
    elif key == "DCN.proj":
        mossy = brain.g["CB.granule"].rate.reshape(64, -1)[:, 0]
        purkinje = brain._pool("CB.Purkinje")
        if hasattr(brain, "parallel_fiber_ltd"):
            # Rows carry the interval representation, columns carry the motor
            # correction. Both pass through DCN and VA to the action circuit.
            cells[:, 0] = .5 * (1.0 - purkinje) + .5 * xp.repeat(timing_activity(brain), 8)
        else:
            cells[:, 0] = mossy / (1.0 + purkinje)
        # The sampled pontine collateral reaches the same projection cell.
        # Its suprabasal membrane drive can recruit the fraction of the
        # population output still available after the granule/Purkinje input.
        soma = group.soma.reshape(64, -1)[:, 0]
        collateral = xp.maximum(soma - group.c.baseline, 0.)
        cells[:, 0] = (cells[:, 0] + collateral) / (1.0 + collateral)
    elif key == "VA.relay":
        cells[:, 0] = brain.g["DCN.proj"].rate.reshape(64, -1)[:, 0]
    if key in ("MD.relay", "VA.relay") and "inh" in channels:
        # The projection ensemble receives pallidal/nigral inhibition even
        # when its excitatory drive is supplied by the reduced relay model.
        cells[:, 0] /= 1.0 + channels["inh"].reshape(64, -1)[:, 0]
    return ordinary


def context(brain):
    """Activity arriving in striatum from frontal, body and timing pathways."""
    xp = xpof(brain.r)
    pfc = brain.g["PFC.IT"].rate.reshape(64, -1)[:, 0]
    # Lateral contrast preserves which part of the retinotopic map is active.
    pfc = pfc - pfc.mean()
    if getattr(brain, "hold_context", False):
        # MD-supported maintenance amplifies the context reaching striatum.
        pfc = 3.0 * pfc / xp.maximum(xp.linalg.norm(pfc), .1)
    else:
        pfc = pfc / xp.maximum(xp.linalg.norm(pfc), 1.0)
    temporal = brain.g["VA.relay"].rate.reshape(4, 16, -1)[:, :, 0].mean(axis=1)
    balance = brain.g["HYP.AgRP"].rate.mean() - brain.g["HYP.POMC"].rate.mean()
    # A small signed projection lets learned food choices depend on appetite.
    fields = [0.65 * pfc, 3.0 * (temporal - temporal.mean()), xp.atleast_1d(5.0 * balance)]
    if hasattr(brain, "parallel_fiber_ltd"):
        phase = brain.g["VA.relay"].rate.reshape(8, 8, -1).mean(axis=(1, 2))
        phase = phase - phase.mean()
        fields.append(phase / xp.maximum(xp.linalg.norm(phase), .05))
    return xp.concatenate(fields)


def timing_activity(brain):
    granule = brain.g["CB.granule"].rate.reshape(8, 8, -1)
    phases = granule[:, :, min(1, granule.shape[2] - 1)].mean(axis=1)
    return phases / xpof(phases).maximum(phases.sum(), 1e-6)


def motor_stripes(brain, motor):
    """Mean efference copy in each map column, including larger per-tile populations."""
    xp = xpof(brain.r)
    stripes = brain.tile["MOT"] % 8
    return xp.bincount(stripes, weights=xp.asarray(motor), minlength=8) / xp.bincount(stripes, minlength=8)


def climbing_fiber_credit(brain, phase, motor):
    """Climbing-fiber input depresses active parallel fibers in the motor channel."""
    xp = xpof(brain.r)
    if not hasattr(brain, "parallel_fiber_ltd"):
        brain.parallel_fiber_ltd = xp.zeros((8, 8), dtype=np.float32)
        return
    if phase is None:
        return
    olive = brain.g["IO.olive"].rate.reshape(8, 8, -1).mean(axis=(0, 2))
    early, late = olive[:4].mean(), olive[4:].mean()
    cf = xp.maximum(early, late)
    selected = motor_stripes(brain, motor)
    weights = brain.parallel_fiber_ltd
    change = cf - weights @ phase
    weights += .6 * (selected * change)[:, None] * phase
    # A directional error also releases the same movement at a neighboring
    # phase, earlier for a late response and later for an early one. Equal CF
    # activity (an unsigned error) only corrects the active parallel fibers.
    later_phase = xp.zeros_like(phase)
    later_phase[:-1] = phase[1:]
    earlier_phase = xp.zeros_like(phase)
    earlier_phase[1:] = phase[:-1]
    shift = xp.maximum(early - late, 0.) * later_phase + xp.maximum(late - early, 0.) * earlier_phase
    weights -= .6 * selected[:, None] * shift
    xp.clip(weights, -1.0, 2.0, out=weights)

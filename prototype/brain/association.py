"""Reduced population currents for working memory and temporal context.

The first IT cell of each PFC tile and its MD relay carry a sustained loop.
Four granule subpopulations integrate pontine input on different time scales.
These share the existing cells, map and lesion boundaries.
"""
import numpy as np

from backend import xpof


def target(brain, key, ordinary):
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
        cells[:, 0] = xp.clip(0.985 * relay + 0.11 * sensory, 0, 1)
    elif key == "MD.relay":
        cells[:, 0] = brain.g["PFC.IT"].rate.reshape(64, -1)[:, 0]
    elif key == "PONS.relay":
        cells[:, 0] = brain._pool("TH.core")
    elif key == "CB.granule":
        if not hasattr(group, "temporal_rates"):
            group.temporal_rates = xp.zeros(4, dtype=np.float32)
        # Distributed mossy-fiber/granule response after a transient. Using a
        # continuously driven filter avoids an external clock or trial marker.
        drive = brain.g["PONS.relay"].rate.reshape(64, -1)[:, 0].max()
        group.temporal_rates += (drive - group.temporal_rates) * xp.asarray([.4, .07, .025, .009])
        cells[:, 0] = xp.repeat(group.temporal_rates, 16)
    elif key == "DCN.proj":
        mossy = brain.g["CB.granule"].rate.reshape(64, -1)[:, 0]
        purkinje = brain._pool("CB.Purkinje")
        cells[:, 0] = mossy / (1.0 + purkinje)
    elif key == "VA.relay":
        cells[:, 0] = brain.g["DCN.proj"].rate.reshape(64, -1)[:, 0]
    return ordinary


def context(brain):
    """Activity arriving in striatum from frontal, body and timing pathways."""
    xp = xpof(brain.r)
    pfc = brain.g["PFC.IT"].rate.reshape(64, -1)[:, 0]
    # Lateral contrast preserves which part of the retinotopic map is active.
    pfc = pfc - pfc.mean()
    pfc = pfc / xp.maximum(xp.linalg.norm(pfc), 1.0)
    temporal = brain.g["VA.relay"].rate.reshape(4, 16, -1)[:, :, 0].mean(axis=1)
    balance = brain.g["HYP.AgRP"].rate.mean() - brain.g["HYP.POMC"].rate.mean()
    # A small signed projection lets learned food choices depend on appetite.
    return xp.concatenate((0.65 * pfc, 3.0 * (temporal - temporal.mean()), xp.atleast_1d(5.0 * balance)))

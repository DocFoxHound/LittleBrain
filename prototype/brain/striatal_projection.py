"""Plastic, topographic sensory relay input to striatal action channels.

A pooled projection is a reduced representation of thalamostriatal convergence
(BG-2 in notes/regions/connections.md). Columns share the motor stripe map.
Eligibility is local to the recently active sensory afferents of chosen cells.
"""
import numpy as np
from backend import xpof
from association import context
from federation import GRID


def activity(brain):
    rates = brain._pool("TH.core")
    xp = xpof(rates)
    centered = rates - rates.mean()
    visual = centered / xp.maximum(xp.linalg.norm(centered), 0.1)
    return xp.concatenate((visual, context(brain)))


def transmit(brain, chan):
    if "sensory -> striatum" in brain.ablated or "TH.core" in brain.ablated:
        return
    features = activity(brain)
    if not hasattr(brain, "sensory_striatal") or brain.sensory_striatal.shape[1] != features.size:
        # Preserve the visual weights of an earlier visual-only checkpoint.
        old = getattr(brain, "sensory_striatal", None)
        brain.sensory_striatal = xpof(features).zeros((GRID, features.size), dtype=np.float32)
        if old is not None:
            brain.sensory_striatal[:, :old.shape[1]] = old
    current = brain.sensory_striatal @ features
    current -= current.mean()
    for key, sign in (("STR.D1", 1), ("STR.D2", -1)):
        chan[key]["plateau"] = sign * current[brain.tile[key] % GRID]


def reinforce(brain, eligibility, motor, dopamine):
    if "sensory -> striatum" in brain.ablated or "TH.core" in brain.ablated:
        return
    xp = xpof(brain.r)
    # Pre (cue) x post (efference copy) x phasic dopamine. Every represented
    # motor stripe stays addressable when an external game changes action count.
    stripes = brain.tile["MOT"] % GRID
    selected = xp.bincount(stripes, weights=xp.asarray(motor), minlength=GRID) / xp.bincount(stripes, minlength=GRID)
    weights = brain.sensory_striatal
    weights += 0.45 * dopamine * selected[:, None] * eligibility
    xp.clip(weights, -4.0, 4.0, out=weights)

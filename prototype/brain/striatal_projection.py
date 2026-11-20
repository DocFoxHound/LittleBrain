"""Plastic, topographic sensory relay input to striatal action channels.

A pooled projection is a reduced representation of thalamostriatal convergence
(BG-2 in notes/regions/connections.md). Columns share the motor stripe map.
Eligibility is local to the recently active sensory afferents of chosen cells.
"""
import numpy as np
from backend import like, xpof
from federation import GRID


def activity(brain):
    rates = brain._pool("TH.core")
    xp = xpof(rates)
    centered = rates - rates.mean()
    return centered / xp.maximum(xp.linalg.norm(centered), 0.1)


def transmit(brain, chan):
    if "sensory -> striatum" in brain.ablated or "TH.core" in brain.ablated:
        return
    if not hasattr(brain, "sensory_striatal"):
        brain.sensory_striatal = like(brain.r, np.zeros((GRID, GRID * GRID)))
    current = brain.sensory_striatal @ activity(brain)
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

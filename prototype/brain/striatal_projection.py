"""Plastic, topographic sensory relay input to striatal action channels.

A pooled projection is a reduced representation of thalamostriatal convergence
(BG-2 in notes/regions/connections.md). Columns share the motor stripe map.
Eligibility is local to the recently active sensory afferents of chosen cells.
"""
import numpy as np
from backend import xpof
from association import context
from federation import GRID


def visual_activity(brain):
    rates = brain._pool("TH.core")
    xp = xpof(rates)
    centered = rates - rates.mean()
    return centered / xp.maximum(xp.linalg.norm(centered), 0.1)


def activity(brain):
    return xpof(brain.r).concatenate((visual_activity(brain), context(brain)))


def amygdala_input(brain, chan):
    trace = visual_activity(brain)
    if not hasattr(brain, "amygdala_afferents"):
        brain.amygdala_afferents = xpof(trace).zeros((2, trace.size), dtype=np.float32)
    response = xpof(trace).clip(brain.amygdala_afferents @ trace, 0, 1)
    chan["BLA.pos"]["conditioned"] = response[0]
    chan["BLA.neg"]["conditioned"] = response[1]


def pair_outcome(brain, trace, outcome):
    # Separate appetitive and aversive US afferents teach the active cue.
    # Omission extinguishes an association more slowly than acquisition.
    cue = trace[:GRID * GRID]
    weights = brain.amygdala_afferents
    for row, key, us in ((0, "BLA.pos", max(0., outcome)), (1, "BLA.neg", max(0., -outcome))):
        if "BLA" in brain.ablated or key in brain.ablated:
            continue
        expectation = weights[row] @ cue
        weights[row] += (.18 if us else .035) * (us - expectation) * cue
        xpof(cue).clip(weights[row], -2., 2., out=weights[row])



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

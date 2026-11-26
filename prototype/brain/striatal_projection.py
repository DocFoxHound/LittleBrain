"""Plastic, topographic sensory relay input to striatal action channels.

A pooled projection is a reduced representation of thalamostriatal convergence
(BG-2 in notes/regions/connections.md). Columns share the motor stripe map.
Eligibility is local to the recently active sensory afferents of chosen cells.
"""
import numpy as np
from backend import to_host, xpof
from association import context, motor_stripes
from federation import GRID


# In a brain with a retina (vision.py) the learned sensory input to the striatum and amygdala comes from visual
# cortex cells, the way corticostriatal and cortex-to-amygdala projections do, instead of the 64-tile thalamic
# summary (note 74). The highest available areas are used. VISION_ROUTE=thalamus keeps the old route for comparison.
# L2/3 of the visual areas (V1 L5 still barely tracks the picture, note 74)


def route(brain):
    """thalamus (the 64-tile thalamic summary), cortex (visual cortex L2/3), or both (thalamostriatal and
    corticostriatal input together, as in the real striatum; the vision default)."""
    import os
    return os.environ.get("VISION_ROUTE", "both") if "RET.on" in brain.pos else "thalamus"


def visual_sources(brain):
    if route(brain) == "thalamus":
        return None
    import os
    if os.environ.get("VISION_SOURCE"):   # experiments: e.g. VISION_SOURCE=A.L4
        return tuple(os.environ["VISION_SOURCE"].split(","))
    # The higher visual areas feed the striatum (in primates mostly IT, V4 and MT, not V1). V1 only if nothing higher
    # exists. Reading V1's motion-free cells alongside MT halved motion learning (note 75).
    higher = tuple(f"{a}.L23" for a in ("IT", "V4", "MT", "V2") if f"{a}.L23" in brain.g)
    higher = higher or (("A.L23",) if "A.L23" in brain.g else ())
    # where you are can guide what to do: the entorhinal cortex projects to the ventral striatum (nucleus accumbens;
    # Totterdell & Meredith 1997), as do CA1/subiculum. The reconstructed hippocampus does not yet hold a stable
    # place code in CA1 (note 84; CA3 and the dentate have one), so the grid cells are read; SPATIAL_READ=CA3/CA1
    # reads those instead
    pooled = tuple(k for k in ("V4.P0", "V4.P1", "V4.P2", "V4.P3") if k in brain.pos)
    if pooled:   # large-field pooled edge cells (note 91): V4 projects to the striatum (tail of the caudate)
        higher = higher + pooled
    if "MEC.grid" in brain.pos:
        import os
        read = os.environ.get("SPATIAL_READ", "MEC")   # experiments: CA3 or CA1 (place cells, note 84)
        higher = higher + ({"CA1": ("CA1.pyr",), "CA3": ("CA3.pyr",)}.get(read, ("MEC.grid",)))
    return higher or None


def cue_size(brain):
    keys = visual_sources(brain)
    if keys is None:
        return GRID * GRID
    import os
    n = (sum(_readout_size(brain, k) for k in keys) if os.environ.get("VISION_POOL", "1") == "1"
         else sum(brain.g[k].n for k in keys))
    return n + (GRID * GRID if route(brain) == "both" else 0)


POOLED_READ = None   # pooled edge cells: regions per side the striatum reads them in (note 91); None = own map


def _pooled_q():
    import os
    q = os.environ.get("VISION_POOLED_READ", "2")
    return int(q) if q and q != "map" else None


def _readout_size(brain, k):
    q = _pooled_q() if k.startswith("V4.P") else None
    return q * q if q else brain.grid.get(k, GRID) ** 2


def _readout_pool(brain, k):
    """What the striatum reads of group k: its mean rate per tile of its own map, except the pooled edge cells, which
    are read in q x q regions (VISION_POOLED_READ, default 2: quadrants). They are already pooled over a wide area;
    reading them per tile added 256 strongly overlapping inputs that diluted colour and motion (note 91)."""
    q = _pooled_q() if k.startswith("V4.P") else None
    if not q:
        return brain._pool_fine(k)
    xp = xpof(brain.r)
    r = brain.r[brain.index[k]]
    cell = np.minimum((brain.pos[k] * q).astype(int), q - 1)
    t = cell[:, 1] * q + cell[:, 0]
    return (xp.bincount(xp.asarray(t), weights=r, minlength=q * q) / xp.asarray(np.maximum(np.bincount(t, minlength=q * q), 1))).astype(np.float32)


def _normalised(rates):
    xp = xpof(rates)
    centered = rates - rates.mean()
    return centered / xp.maximum(xp.linalg.norm(centered), 0.1)


def visual_activity(brain):
    keys = visual_sources(brain)
    thalamic = _normalised(brain._pool("TH.core"))
    if keys is None:
        return thalamic
    # Corticostriatal convergence: each striatal cell sums thousands of cortical inputs, so neighbouring cells'
    # noise averages out. Reduced here to the mean rate per tile of each source's own map (VISION_POOL=0: cells).
    import os
    if os.environ.get("VISION_POOL", "1") == "1":
        rates = xpof(brain.r).concatenate([_readout_pool(brain, k) for k in keys])
    else:
        rates = xpof(brain.r).concatenate([brain.g[k].rate for k in keys])
    # Corticostriatal synapses depress under sustained input, passing deviations from what is usually seen more than
    # the steady background shared by every picture (e.g. "a disc in the middle" in every colour). Reduced here to
    # subtracting a slow running average of the input (time constant ~500 steps).
    xp = xpof(rates)
    raw = xp.concatenate([brain._pool("TH.core"), rates])
    mean = getattr(brain, "cue_mean", None)
    if mean is None or mean.shape != raw.shape:
        mean = raw.copy()
    brain.cue_mean = mean + 0.002 * (raw - mean)
    dev = raw - brain.cue_mean
    # Scale by a slow running size rather than to unit length each step: how strongly cells fire carries information
    # (red and yellow drive the same red-green cells, red more strongly), which per-step normalisation erased
    n = GRID * GRID
    size = getattr(brain, "cue_scale", None)
    # One joint scale for the thalamic and cortical parts: a part that barely changes (brightness in a colour game)
    # stays small instead of being blown up to the size of the informative part
    # separate scales for the thalamic and cortical parts by default: motion +0.07-0.10 on 3 seeds, colour and signal
    # unchanged (note 87); VISION_SCALE=joint restores the shared scale of note 77
    if route(brain) == "both" and os.environ.get("VISION_SCALE", "separate") == "joint":
        now = xp.linalg.norm(dev) * xp.ones(2, dtype=dev.dtype)
    else:
        now = xp.stack([xp.linalg.norm(dev[:n]), xp.linalg.norm(dev[n:])])
    if os.environ.get("VISION_SCALE", "separate") == "source" and os.environ.get("VISION_POOL", "1") == "1":
        # each source area on its own scale (note 92), the pooled edge cells counted as one: a big or loud source
        # (256 pooled cells) no longer shrinks the others under one cortical scale; the cortical part keeps size ~1
        bounds, start, seen = [], n, set()
        for k in keys:
            m = _readout_size(brain, k)
            name = "V4.P" if k.startswith("V4.P") else k
            if name in seen:
                bounds[-1] = (bounds[-1][0], start + m)
            else:
                bounds.append((start, start + m))
                seen.add(name)
            start += m
        now = xp.stack([xp.linalg.norm(dev[:n])] + [xp.linalg.norm(dev[a_:b_]) for a_, b_ in bounds])
        size = now + 0.0 if size is None or size.shape != now.shape else size + 0.002 * (now - size)
        brain.cue_scale = size
        parts = [dev[a_:b_] / xp.maximum(size[i + 1], 0.05) / np.sqrt(len(bounds)) for i, (a_, b_) in enumerate(bounds)]
        brain._segments = [(0, n)] + bounds
        brain._segment_names = []
        for k in keys:
            nm = "V4.P" if k.startswith("V4.P") else k
            if nm not in brain._segment_names:
                brain._segment_names.append(nm)
        gain = getattr(brain, "source_gain", None)
        if os.environ.get("VISION_ATTEND", "0") == "input" and gain is not None and len(gain) == len(bounds) + 1:
            # learned attention on the input (note 92; worse: changing an input's size invalidates what was learned)
            parts = [p_ * float(gain[i + 1]) for i, p_ in enumerate(parts)]
            thalamic_gain = float(gain[0])
        else:
            thalamic_gain = 1.0
        thalamic, cortical = dev[:n] / xp.maximum(size[0], 0.05) * thalamic_gain, xp.concatenate(parts)
        if route(brain) == "both":
            return xp.concatenate([thalamic, cortical])
        return cortical
    size = now + 0.0 if size is None or size.shape != now.shape else size + 0.002 * (now - size)
    brain.cue_scale = size
    thalamic, cortical = dev[:n] / xp.maximum(size[0], 0.05), dev[n:] / xp.maximum(size[1], 0.05)
    if route(brain) == "both":
        return xp.concatenate([thalamic, cortical])
    return cortical


def activity(brain):
    return xpof(brain.r).concatenate((visual_activity(brain), context(brain)))


def amygdala_input(brain, chan):
    trace = visual_activity(brain)
    if not hasattr(brain, "amygdala_afferents") or brain.amygdala_afferents.shape[1] != trace.size:
        # new, or the sensory source changed (thalamic summary -> visual cortex): start the associations fresh
        brain.amygdala_afferents = xpof(trace).zeros((2, trace.size), dtype=np.float32)
    response = xpof(trace).clip(brain.amygdala_afferents @ trace, 0, 1)
    chan["BLA.pos"]["conditioned"] = response[0]
    chan["BLA.neg"]["conditioned"] = response[1]


def pair_outcome(brain, trace, outcome):
    # Separate appetitive and aversive US afferents teach the active cue.
    # Omission extinguishes an association more slowly than acquisition.
    cue = trace[:cue_size(brain)]
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
    online = getattr(brain, "cue_online", None)
    if visual_sources(brain) is not None and online is not None and online.shape == features.shape:
        features = online   # during the read window: the integrated, centred input (agent.choose, note 74)
    elif visual_sources(brain) is not None:
        # Striatal cells integrate their input over time (membrane and synaptic time constants); a single step of
        # cortical activity is too noisy to choose on (vision brains, note 74)
        held = getattr(brain, "cue_integrated", None)
        held = features if held is None or held.shape != features.shape else held + 0.25 * (features - held)
        brain.cue_integrated = held
        features = held
        # the same trial-level centering the learning uses (agent.choose): what differs from recent pictures
        m = getattr(brain, "trace_mean", None)
        if m is not None and m.shape == features.shape:
            features = features - m
    if not hasattr(brain, "sensory_striatal") or brain.sensory_striatal.shape[1] != features.size:
        # Preserve the visual weights of an earlier visual-only checkpoint.
        old = getattr(brain, "sensory_striatal", None)
        brain.sensory_striatal = xpof(features).zeros((GRID, features.size), dtype=np.float32)
        if old is not None and visual_sources(brain) is None and old.shape[1] <= features.size:
            brain.sensory_striatal[:, :old.shape[1]] = old
    current = territory_drive(brain, features)
    if current is None:
        current = brain.sensory_striatal @ features
    if hasattr(brain, "food_afferents"):
        # Revalue a learned sensory outcome from current hypothalamic firing.
        # Stripes learn food identity from contact, regardless of their names.
        xp = xpof(current)
        agrp = brain.g["HYP.AgRP"].rate.mean()
        pomc = brain.g["HYP.POMC"].rate.mean()
        drive = (agrp - pomc) / xp.maximum(agrp + pomc, .05)
        food = xp.clip(brain.food_afferents @ features[:cue_size(brain)], 0, 1)
        current = xp.tanh(current) + 5.0 * drive * food
    if hasattr(brain, "parallel_fiber_ltd"):
        # VA relays the disinhibited cerebellar correction to indirect-path MSNs.
        correction = brain.g["VA.relay"].rate.reshape(GRID, GRID, -1).mean(axis=(0, 2))
        # Saturate the associative drive before adding the cerebellar input.
        # A heavily practised movement must still be able to wait for its cue.
        current = xpof(current).tanh(current) - 10.0 * correction
    current -= current.mean()
    for key, sign in (("STR.D1", 1), ("STR.D2", -1)):
        chan[key]["plateau"] = sign * current[brain.tile[key] % GRID]


def _territories(brain, size):
    """Input territories of the striatum (STRIATUM_TERRITORIES=1, note 93): the thalamic part, each visual source
    (the pooled edge cells as one) and the rest (context). None if the option is off or the layout is unknown."""
    import os
    segs = getattr(brain, "_segments", None)
    mode = os.environ.get("STRIATUM_TERRITORIES", "0")
    if mode not in ("1", "streams") or not segs:
        return None
    end = segs[-1][1]
    segs = list(segs) + ([(end, size)] if size > end else [])
    idx = [np.arange(a_, b_) for a_, b_ in segs]
    if mode == "1":
        return idx
    # streams: the pooled edge cells, the grid cells and MT (motion was learned best from MT alone, note 75) each get
    # their own territory; everything the striatum read
    # before (thalamus, the learned visual areas, context) stays together, as colour was learned best from all of
    # them at once (note 77)
    names = ["thalamus"] + list(getattr(brain, "_segment_names", [])) + ["context"]
    alone = ("V4.P", "MEC.grid") + (("MT.L23",) if os.environ.get("STRIATUM_MT_ALONE", "1") == "1" else ())
    own = [i for i, nm in enumerate(names[:len(idx)]) if nm in alone]
    rest = [i for i in range(len(idx)) if i not in own]
    return [np.concatenate([idx[i] for i in rest])] + [idx[i] for i in own]


def territory_drive(brain, features):
    """Each territory's own action values, combined weighted by how reliable each has been (inverse of its recent
    squared prediction error; arbitration by uncertainty, Daw, Niv & Dayan 2005)."""
    terr = _territories(brain, features.size)
    if terr is None:
        return None
    xp = xpof(features)
    W = getattr(brain, "territory_w", None)
    if W is None or len(W) != len(terr) or any(w.shape[1] != len(ix) for w, ix in zip(W, terr)):
        brain.territory_w = W = [xp.zeros((GRID, len(ix)), dtype=np.float32) for ix in terr]
        brain.territory_err = np.ones(len(terr))
        brain.territory_stats = np.zeros((len(terr), 5))   # running E[q], E[d], E[q^2], E[d^2], E[qd]
    rel = territory_reliability(brain, len(terr))
    return sum(float(rel[i]) * (W[i] @ features[ix]) for i, ix in enumerate(terr))


def territory_reliability(brain, k):
    """How much to trust each territory: the running correlation between its value for the chosen action and the
    reward that followed (note 93). A territory whose input says nothing about the outcome gets ~0, however small its
    error; mean 1. Before any history: all equal. (Against the dopamine itself the correlation fades once learning
    works, as dopamine then signals only the surprise, and trust flipped at random.)"""
    st = getattr(brain, "territory_stats", None)
    if st is None or len(st) != k or not st[:, 1].any() or not st[:, 3].any():
        return np.ones(k)
    # how much more often the choice was rewarded when this territory's own favourite was chosen than when it wasn't
    agree = st[:, 0] / np.maximum(st[:, 1], 1e-6)
    disagree = st[:, 2] / np.maximum(st[:, 3], 1e-6)
    rel = np.clip(agree - disagree, 0.02, None)
    return rel / rel.mean()


def reinforce(brain, eligibility, motor, dopamine):
    if "sensory -> striatum" in brain.ablated or "TH.core" in brain.ablated:
        return
    xp = xpof(brain.r)
    terr = _territories(brain, eligibility.size)
    W = getattr(brain, "territory_w", None)
    if terr is not None and W is not None and len(W) == len(terr):
        # each territory learns on its own, with the same rule as the single readout, from its own input only, so noise
        # in one can't leak into another's weights; how well its value for the chosen action foretold the dopamine
        # sets how much it is trusted
        selected = motor_stripes(brain, motor)
        sel = float(selected.sum()) or 1.0
        st = getattr(brain, "territory_stats", None)
        if st is None or len(st) != len(terr):
            st = brain.territory_stats = np.zeros((len(terr), 5))
        d = float(dopamine)
        rw = float(getattr(brain, "last_reward", d))
        for i, ix in enumerate(terr):
            x = eligibility[ix]
            q_all = to_host(W[i] @ x)
            agreed = float(to_host(selected)[int(np.argmax(q_all))] > 0.5) if np.ptp(q_all) > 0 else 0.5
            # running sums: rewarded-and-agreed, agreed, rewarded-and-disagreed, disagreed (rate 0.01: ~100 trials)
            st[i] = 0.99 * st[i] + 0.01 * np.array([rw * agreed, agreed, rw * (1 - agreed), 1 - agreed, 0.0])
            q = float(to_host(selected) @ q_all) / sel
            brain.territory_err[i] += 0.05 * ((d - q) ** 2 - brain.territory_err[i])
            W[i] += 0.45 * d * selected[:, None] * x
            xp.clip(W[i], -4.0, 4.0, out=W[i])
        return
    # Pre (cue) x post (efference copy) x phasic dopamine. Every represented
    # motor stripe stays addressable when an external game changes action count.
    selected = motor_stripes(brain, motor)
    weights = brain.sensory_striatal
    import os
    segs = getattr(brain, "_segments", None)
    gain = getattr(brain, "source_gain", None)
    if os.environ.get("VISION_ATTEND", "0") == "1" and segs and gain is not None and len(gain) == len(segs):
        # associability (Mackintosh 1975; note 92): how fast each source is learned about, not how loud it is
        alpha = xpof(eligibility).ones_like(eligibility)
        for (a_, b_), g_ in zip(segs, gain):
            alpha[a_:b_] = float(g_)
        eligibility = eligibility * alpha
    weights += 0.45 * dopamine * selected[:, None] * eligibility
    xp.clip(weights, -4.0, 4.0, out=weights)
    if os.environ.get("VISION_ATTEND", "0") in ("1", "input") and segs:
        # Mackintosh-style attention (Mackintosh 1975; note 92): a source whose inputs the striatum has learned
        # strong, structured weights for gets more weight; one it ignores fades. Slow, mean gain 1, bounded
        use = np.array([float(xp.linalg.norm(weights[:, a_:b_])) / np.sqrt(max(b_ - a_, 1)) for a_, b_ in segs])
        target = np.clip(use / max(use.mean(), 1e-6), 0.2, 3.0)
        gain = getattr(brain, "source_gain", None)
        gain = np.ones(len(segs)) if gain is None or len(gain) != len(segs) else gain
        gain = gain + 0.02 * (target - gain)
        brain.source_gain = gain / gain.mean()


def food_outcome(brain, trace, motor, amount):
    """Pair a sensory outcome with the chosen channel and preceding visual cue."""
    xp = xpof(brain.r)
    if not hasattr(brain, "food_afferents") or brain.food_afferents.shape[1] != cue_size(brain):
        brain.food_afferents = xp.zeros((GRID, cue_size(brain)), dtype=np.float32)
    cue = trace[:cue_size(brain)]
    selected = motor_stripes(brain, motor)
    weights = brain.food_afferents
    weights += .25 * (selected * (amount - weights @ cue))[:, None] * cue
    xp.clip(weights, -2., 2., out=weights)


# ---------------------------------------------------------------- value across steps (plan phase 1, note 80)

def critic_value(brain, trace):
    """Ventral-striatum value of the current situation (a reduced readout of the striatum's sensory input), learned by
    temporal-difference errors so that it predicts reward and what the situation leads to."""
    w = getattr(brain, "value_afferents", None)
    if w is None or w.shape != trace.shape:
        return 0.0
    return float(np.clip(float(w @ trace), -1.5, 1.5))


def critic_learn(brain, trace, delta, lr=0.3):
    xp = xpof(trace)
    w = getattr(brain, "value_afferents", None)
    if w is None or w.shape != trace.shape:
        w = brain.value_afferents = xp.zeros_like(trace)
    w += lr * float(delta) * trace
    xp.clip(w, -4.0, 4.0, out=w)

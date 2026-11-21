"""
The vision brain (notes 71-77): the full reference brain with a real retina, a layered LGN, finer visual maps,
and (in later stages) motion, colour and higher visual areas. Select it with BRAIN=vision.

Stage 2 (this file's first version, note 72):
  - Retina channels (retina.py) as inputs on fine maps: midget ON/OFF and red-green on 32x32, blue-yellow,
    parasol transient and direction-selective cells on 16x16.
  - LGN streams (Livingstone & Hubel 1988; Casagrande 1994):
      TH.TC  parvocellular relay (detail and red-green) on 16x16, driven by midget cells
      TH.M   magnocellular relay (change and motion), driven by parasol and direction-selective cells
      TH.K   koniocellular relay (blue-yellow), driven by small bistratified cells
    All three receive layer-6 corticothalamic feedback and the reticular nucleus's inhibition.
  - V1 (area A) input layers on finer maps: L4 on 16x16 (all three LGN streams), L2/3 on 16x16.
"""

import os
from dataclasses import replace

import numpy as np

import celltypes as C
from federation import Group, Pooled, Synaptic
from retina import DS, KONIO, MAGNO, PARVO

FINE = 32     # retina detail channels
# parvo ON, parvo OFF, magno (transient), konio, and four direction-specific relays (note 74)
DIR_RELAYS = ("TH.D0", "TH.D1", "TH.D2", "TH.D3")
RELAYS = ("TH.TC", "TH.OFF", "TH.M", "TH.K") + DIR_RELAYS
MID = 16      # LGN, magno/konio channels, V1 input layers


def _members(v):
    return v if isinstance(v, list) else [v]


def add_retina_and_lgn(spec):
    """Retina channels, three LGN streams, and V1's input layers on finer maps."""
    spec.groups += [Group(k, None, 1, "retina", grid=FINE) for k in PARVO]
    spec.groups += [Group(k, None, 1, "retina", grid=MID) for k in KONIO + MAGNO + DS]
    spec.inputs = spec.inputs + list(PARVO + KONIO + MAGNO + DS)
    resized = {"TH.TC": (MID, 1), "A.L4": (MID, 2), "A.L23": (MID, 3)}
    spec.groups = [replace(g, grid=resized[g.key][0], per_tile=resized[g.key][1]) if g.key in resized else g
                   for g in spec.groups]
    # Critical period: V1's input layer (and optionally L2/3) learns faster than the rest of cortex during
    # development (notes 73-74); end_critical_period undoes it afterwards
    lrf = float(os.environ.get("VISION_L4_LRF", "20"))
    if lrf != 1:
        spec.groups = [replace(g, ctype=replace(g.ctype, rules=replace(g.ctype.rules, lr=g.ctype.rules.lr * lrf)))
                       if g.key in critical_period_groups() else g for g in spec.groups]
    # ON and OFF signals stay in separate relay cells (as in the LGN); TH.TC keeps its name for older pathways
    spec.groups += [Group("TH.OFF", C.TC_RELAY, 1, "TH", grid=MID),
                    Group("TH.M", C.TC_RELAY, 1, "TH", grid=MID), Group("TH.K", C.TC_RELAY, 2, "TH")]
    # Direction-selective ganglion cells keep their direction: each relay hears one direction (like the LGN shell
    # that receives them in mice, Cruz-Martin et al. 2014); a relay mixing all four would cancel direction
    spec.groups += [Group(k, C.TC_RELAY, 1, "TH", grid=MID) for k in DIR_RELAYS]
    syn = []
    for p in spec.synaptic:
        if p.id == "CT-2 retina->relay (driver)":
            p = replace(p, id="CT-2 ON midget cells->parvo ON relay (driver)", src=["RET.on", "RET.rg_on"])
        elif p.id.startswith("CT-1") and p.dst in ("A.L4", "A.L6"):
            p = replace(p, src=list(RELAYS))
            # More, weaker thalamic synapses per layer-4 cell, same total drive (env VISION_TC_FRAC, note 73):
            # real L4 cells get ~30 LGN inputs (Alonso et al. 2001), not ~6
            frac = float(os.environ.get("VISION_TC_FRAC", "0")) if p.dst == "A.L4" else 0
            if frac:
                p = replace(p, frac=frac, gain=p.gain * p.frac / frac)
        syn.append(p)
    drivers_of = [("TH.OFF", ["RET.off", "RET.rg_off"]), ("TH.M", list(MAGNO)), ("TH.K", list(KONIO))]
    drivers_of += [(r, [d]) for r, d in zip(DIR_RELAYS, DS)]
    for relay, drivers in drivers_of:
        syn += [Synaptic(f"CT-2 retina->{relay} (driver)", drivers, relay, "basal", 0.3, 3.0, 0.05),
                Synaptic(f"CT-3 L6 CT->{relay} (modulator)", "A.L6", relay, "basal", 0.7, 0.15, 0.08)]
    spec.synaptic = syn
    pooled = []
    for p in spec.pooled:
        if p.id == "TRN<-relay collaterals":
            p = replace(p, src=list(RELAYS))
        elif p.id in ("CT-5 TRN->relay", "TH.interneuron -> TH.TC"):
            p = replace(p, dst=_members(p.dst) + [r for r in RELAYS if r != "TH.TC"])
        pooled.append(p)
    spec.pooled = pooled
    return spec


VISUAL_AREAS = ("A", "MT", "V2", "V4", "IT")   # visual cortical areas (those present in a brain are used)
SLOW_HOMEOSTASIS = 0.1  # visual cortex layers balance their firing 10x more slowly (note 74)


def visual_homeostasis(brain):
    """Always-on homeostasis adapted visual cortex away from whatever it was looking at (a 20-step picture): with it
    off, V1 L2/3 identified colour pictures 75% of the time instead of 50%. Slow it for visual cortex (saved in the
    checkpoint, like HOMEO_SLOW)."""
    cap = float(os.environ.get("VISION_SCALE_CAP", "8"))
    for g in brain.spec.groups:
        if g.key.split(".")[0] in VISUAL_AREAS and g.key in brain.g:
            brain.homeo_speed[g.key] = SLOW_HOMEOSTASIS
            if cap:
                # Calibration made visual cortex cells near on/off switches (sharpness ~19), so small input
                # fluctuations flipped them and amplified noise (note 75)
                brain.scale_caps = dict(getattr(brain, "scale_caps", {}), **{g.key: cap})


def _visual_cortex_gains(p):
    """Visual cortex listens to its input (note 74): L2/3 is driven by L4 (VISION_FF, default 10x) more than by its
    own recurrent chatter (VISION_REC, default 0.3x). Without this, V1 L2/3 identified game pictures 44% of the time
    and L5 0%; with it L2/3 reached ~81% (seed 0)."""
    if p.kind != "local" or not isinstance(p.src, str) or p.src.split(".")[0] not in VISUAL_AREAS:
        return p
    s, d = p.src.split(".")[1], p.dst.split(".")[1]
    if (s, d) == ("L4", "L23"):
        return replace(p, gain=float(os.environ.get("VISION_FF", "10")))
    if (s, d, p.comp) == ("L23", "L23", "basal"):
        return replace(p, gain=float(os.environ.get("VISION_REC", "0.3")))
    return p


def add_area(spec, area, ff, spread=0.25, l4_frac=0.1, l23_frac=0.15, grid=8, scale=1.0):
    """A new visual cortical area built like V1 and B (brainspec.stage1): all layers, local wiring, PV/SST/VIP
    inhibition. ff: the groups sending feedforward input to its L4 (strong) and L2/3 (weaker), like FF A->B."""
    import brainspec as S
    S.SPREAD[area] = spread
    for g, (ct, k) in S.AREA_CELLS.items():
        spec.groups.append(Group(f"{area}.{g}", S._cortex(ct) if g in S.EXC else ct,
                                 max(1, int(round(k * scale))), area, grid=grid))
    long_ = {"L4": {"basal": [Synaptic(f"FF {'+'.join(ff)}->{area} L4", list(ff), f"{area}.L4", "basal", l4_frac, 8.0, 0.25)]},
             "L23": {"basal": [Synaptic(f"FF {'+'.join(ff)}->{area} L2/3", list(ff), f"{area}.L23", "basal", l23_frac, 3.0, 0.25)]}}
    for g in S.EXC:
        for comp in S.LOCAL[g]:
            spec.synaptic += S._with_local(area, g, comp, long_.get(g, {}).get(comp, []))
    spec.pooled += S._area_inhibition(area)
    FEEDFORWARD[area] = tuple(ff)
    return spec


FEEDFORWARD = {}   # area -> groups sending it feedforward input (for like-to-like wiring of its L4)


def add_mt(spec):
    """MT (V5): motion. L4 hears the direction-selective relays (in primates MT also gets direct thalamic motion input,
    besides V1; Sincich et al. 2004) and V1 L2/3. Direction columns develop through like-to-like wiring across the
    four direction relays (LIKE_PAIRS)."""
    return add_area(spec, "MT", ff=list(DIR_RELAYS) + ["A.L23"], spread=0.25)


def add_ventral(spec):
    """The ventral 'what' stream (Felleman & Van Essen 1991; Ungerleider & Mishkin 1982):
      V2  from V1 L2/3, plus direct koniocellular LGN input (Bullier & Kennedy 1983)
      V4  colour and form, from V2 and directly from V1 (Gattass et al. 1990)
      IT  objects regardless of position, from V4 and V2
    Each area's L4 develops like-to-like wiring from its feedforward senders (LIKE_PAIRS)."""
    add_area(spec, "V2", ff=["A.L23", "TH.K"], spread=0.25)
    add_area(spec, "V4", ff=["V2.L23", "A.L23"], spread=0.3)
    add_area(spec, "IT", ff=["V4.L23", "V2.L23"], spread=0.5)   # large receptive fields
    return spec


def vision_brain():
    from brainspec import full_brain
    spec = add_retina_and_lgn(full_brain())
    if os.environ.get("VISION_MT", "1") == "1":
        add_mt(spec)
    if os.environ.get("VISION_VENTRAL", "1") == "1":
        add_ventral(spec)
    if os.environ.get("VISION_SC", "1") == "1":
        add_colliculus(spec)
    spec.synaptic = [_visual_cortex_gains(p) for p in spec.synaptic]
    if any(g.key == "LHb.proj" for g in spec.groups):
        # Worse-than-expected value drives the lateral habenula, which silences dopamine cells through the RMTg
        # (Matsumoto & Hikosaka 2007): harm predicted across steps (plan track N1, note 80)
        spec.pooled.extend([Pooled("aversive value -> LHb", None, "LHb.proj", float(os.environ.get("LHB_GAIN", "4.0")),
                                  channel="exc", signal="value_loss"),
                           # stronger habenula -> RMTg relay: real habenula activation pauses dopamine cells almost
                           # completely; the reconstructed relay saturated at about a third of a pause (note 80)
                           Pooled("LHb -> RMTg (pause)", "LHb.proj", "RMTg.GABA", float(os.environ.get("LHB_RMTG", "3.0")),
                                  sigma=None)])
    return spec


# ---------------------------------------------------------------- visual experience for development

SIZE = 64


def render(shapes, background, size=SIZE):
    """RGB image of shapes: dicts with kind (disc | bar | square), centre, size, angle, colour."""
    yy, xx = (np.mgrid[0:size, 0:size] + 0.5) / size
    img = np.empty((size, size, 3), np.float32)
    img[:] = background
    for s in shapes:
        dx, dy = xx - s["x"], yy - s["y"]
        if s["kind"] == "disc":
            m = dx ** 2 + dy ** 2 < s["r"] ** 2
        else:
            c, n = np.cos(s["angle"]), np.sin(s["angle"])
            u, v = dx * c + dy * n, -dx * n + dy * c
            long_ = s["r"] * (3.0 if s["kind"] == "bar" else 1.0)
            wide = s["r"] * (0.35 if s["kind"] == "bar" else 1.0)
            m = (np.abs(u) < long_) & (np.abs(v) < wide)
        img[m] = s["colour"]
    return img


def random_shapes(rng, n=None):
    shapes = []
    for _ in range(n or int(rng.integers(1, 4))):
        grey = rng.random() < 0.4
        colour = np.full(3, rng.random(), np.float32) if grey else rng.random(3).astype(np.float32)
        speed = rng.choice([0.0, 0.015, 0.03])
        ang = rng.uniform(0, 2 * np.pi)
        shapes.append({"kind": rng.choice(["disc", "bar", "square"]), "x": rng.uniform(0.15, 0.85),
                       "y": rng.uniform(0.15, 0.85), "r": rng.uniform(0.05, 0.15), "angle": rng.uniform(0, np.pi),
                       "colour": colour, "vx": speed * np.cos(ang), "vy": speed * np.sin(ang)})
    return shapes


class Movie:
    """One presentation for development: shapes drifting for `hold` steps, then a plain grey screen."""

    def __init__(self, eye, rng, quiet):
        self.eye, self.quiet = eye, quiet
        self.shapes = random_shapes(rng)
        self.background = np.full(3, rng.uniform(0.3, 0.7), np.float32)

    def frame(self, s, hold):
        if s < hold:
            moved = [dict(sh, x=sh["x"] + s * sh["vx"], y=sh["y"] + s * sh["vy"]) for sh in self.shapes]
            img = render(moved, self.background)
        else:
            img = np.full((SIZE, SIZE, 3), self.background, np.float32)
        return {**self.quiet, **self.eye(img)}


def visual_experience(brain, rng):
    """make_input for Brain.calibrate / develop on a brain with a retina."""
    from retina import Retina
    eye = Retina(brain)
    quiet = {k: np.zeros(len(brain.pos[k]), np.float32) for k in brain.spec.inputs}
    return lambda: Movie(eye, rng, quiet)


# ---------------------------------------------------------------- retinal waves (prenatal development, note 73)

class Wave:
    """One spontaneous retinal wave: a band of ganglion-cell activity sweeping across the retina in a random
    direction (Meister et al. 1991; Wong 1999). In stage III waves ON cells fire before OFF cells
    (Kerschensteiner & Wong 2008), so ON and OFF bands are offset. Applied to the retinal channels directly:
    waves are activity of the retina itself, not light."""

    def __init__(self, brain, rng, quiet, width=0.07, lag=0.08):
        self.b, self.quiet, self.width, self.lag = brain, quiet, width, lag
        a = rng.uniform(0, 2 * np.pi)
        self.dir = np.array([np.cos(a), np.sin(a)])
        self.start = rng.uniform(-0.3, 0.0)

    def frame(self, s, hold):
        out = dict(self.quiet)
        if s >= hold:
            return out
        front = self.start + (s + 1) * (1.5 / hold)
        for k in ("RET.on", "RET.off", "RET.rg_on", "RET.rg_off", "RET.by", "RET.ton", "RET.toff"):
            if k not in self.b.pos:
                continue
            d = (self.b.pos[k] - 0.5) @ self.dir + 0.5
            f = front - (self.lag if k.endswith("off") else 0.0)
            act = np.exp(-((d - f) / self.width) ** 2).astype(np.float32)
            out[k] = act * (0.3 if k.startswith("RET.rg") or k == "RET.by" else 1.0)
        return out


def retinal_waves(brain, rng):
    quiet = {k: np.zeros(len(brain.pos[k]), np.float32) for k in brain.spec.inputs}
    return lambda: Wave(brain, rng, quiet)


ADULT_VISUAL_LRF = float(os.environ.get("VISION_ADULT_LRF", "0"))   # 0 = stable after development


def critical_period_groups():
    return tuple(os.environ.get("VISION_CP_GROUPS", "A.L4").split(","))


def end_critical_period(brain):
    """After development (note 74): retinogeniculate relay synapses stop learning (they are mature and stable in
    adults), and V1 layer 4 returns from its critical-period learning rate to the adult cortical rate. Without this,
    waking play kept reshaping the visual input and colour responses drifted until the striatum could not learn them."""
    from dataclasses import replace as _r
    lrf = float(os.environ.get("VISION_L4_LRF", "20"))
    for key, g in brain.g.items():
        if key in RELAYS:
            g.c = _r(g.c, learns=False)
        elif ADULT_VISUAL_LRF != 1 and key.split(".")[0] in VISUAL_AREAS[1:] and hasattr(g.c, "rules") and g.c.learns:
            # higher visual areas learn slowly in adulthood (note 77): fast learning during play made their code drift
            g.c = _r(g.c, rules=_r(g.c.rules, lr=g.c.rules.lr * ADULT_VISUAL_LRF)) if ADULT_VISUAL_LRF else _r(g.c, learns=False)
        elif key in critical_period_groups() and lrf != 1:
            g.c = _r(g.c, rules=_r(g.c.rules, lr=g.c.rules.lr / lrf))
    brain.critical_period_closed = True


# ---------------------------------------------------------------- like-to-like wiring after eye opening (note 75)

def record(brain, see, movies=150, hold=6):
    """Each cell's mean rate during each of `movies` short movies (no learning). Returns (movies, all cells)."""
    rows = []
    for _ in range(movies):
        m = see()
        acc = 0
        for s in range(hold + 3):
            brain.step(m.frame(s, hold), ach=0.5)
            if 2 <= s < hold:
                acc = acc + np.asarray(brain.r, np.float32)
        rows.append(acc / (hold - 2))
    return np.array(rows)


def like_to_like(brain, see, pairs=(("A.L4", "A.L23"),), rounds=3, reach=0.2, contacts=3, movies=150):
    """Correlation-based structural plasticity (Ko et al. 2011, 2013: like-to-like L4 -> L2/3 connectivity emerges after
    eye opening). Each receiving cell keeps, among the sending cells within `reach`, the ones whose activity best tracks
    its own, `contacts` synapses each, replacing its other synapses from that sender group. Repeated: a cell leaning
    slightly towards one feature keeps more inputs of that feature and leans further. Reduced to a bulk step between
    bouts of visual experience, like night-time rewiring. Returns the mean correlation of kept partners per round."""
    history = []
    for _ in range(rounds):
        R = record(brain, see, movies)
        means = []
        for src, dst in pairs:
            srcs = (src,) if isinstance(src, str) else tuple(src)
            si, di = np.concatenate([brain.index[s] for s in srcs]), brain.index[dst]
            # what each cell prefers: its response per movie, minus its layer's response to that movie (so the shared
            # "something is on the screen" does not count as similarity), correlated across movies
            Z = {}
            for name, idx in (("s", si), ("d", di)):
                X = R[:, idx] - R[:, idx].mean(1, keepdims=True)
                Z[name] = (X - X.mean(0)) / (X.std(0) + 1e-6)
            C = (Z["d"].T @ Z["s"]) / len(R)                               # (receivers, senders)
            bank = brain.g[dst].basal
            member = np.zeros(brain.n_total, bool)
            member[si] = True
            flat = bank.src.reshape(bank.src.shape[0], -1)
            P, Q = brain.pos[dst], np.concatenate([brain.pos[s] for s in srcs])
            cells, br, sl, new, kept = [], [], [], [], []
            for i in range(len(di)):
                slots = np.flatnonzero(member[flat[i]])
                if not slots.size:
                    continue
                near = np.flatnonzero(np.linalg.norm(Q - P[i], axis=1) < reach)
                if not near.size:
                    continue
                k = max(1, int(np.ceil(slots.size / contacts)))
                best = near[np.argsort(-C[i, near])[:k]]
                choice = np.resize(best, slots.size)                       # `contacts` synapses per kept partner
                b_, s_ = np.unravel_index(slots, bank.src.shape[1:])
                cells += [i] * slots.size
                br += list(b_)
                sl += list(s_)
                new += list(si[choice])
                kept.append(C[i, best].mean())
            where = (np.array(cells), np.array(br), np.array(sl))
            if hasattr(bank, "rewire"):
                changed = bank.src[where] != np.array(new)
                w = tuple(x[changed] for x in where)
                bank.rewire(w, np.array(new)[changed], np.zeros(int(changed.sum()), np.uint8))
            else:
                bank.src[where] = np.array(new)
            means.append(float(np.mean(kept)))
        history.append(means)
    return history


def LIKE_PAIRS(brain):
    """Feedforward pairs that get like-to-like wiring: L4 -> L2/3 in every visual area present, and each area's L4 from
    its feedforward senders (MT: across the four direction relays)."""
    pairs = []
    for a in ("MT", "V2", "V4", "IT"):   # feedforward order: senders are wired before their receivers
        if f"{a}.L4" in brain.g:
            ff = FEEDFORWARD.get(a) or {"MT": DIR_RELAYS}.get(a)
            if ff:
                pairs.append((tuple(ff), f"{a}.L4"))
    pairs += [(f"{a}.L4", f"{a}.L23") for a in VISUAL_AREAS if f"{a}.L4" in brain.g and f"{a}.L23" in brain.g]
    return tuple(pairs)


# ---------------------------------------------------------------- eye movements (note 76)

SC_VISUAL = replace(C.L4_STELLATE, name="SC visual", layer="SC superficial", long_range_max=1.0, adapt=0.3,
                    human="superior colliculus superficial-layer visual neuron (direct retinal input; salience)")
SC_MOTOR = replace(C.L4_STELLATE, name="SC saccade", layer="SC intermediate", long_range_max=1.0, adapt=0.2,
                   human="superior colliculus intermediate-layer build-up / burst neuron (saccade motor map; "
                         "Munoz & Wurtz 1995)")


def add_colliculus(spec):
    """Superior colliculus: a retinotopic salience map (superficial layer, direct retinotectal input from transient
    and contrast cells, plus V1 layer 5 corticotectal input) and a saccade motor map (intermediate layer) where one
    target wins (fast competition, kWTA). The basal ganglia output (SNr) holds it back, the classic saccade gate
    (Hikosaka et al. 2000). The superficial layer also reaches the pulvinar, which relays to visual cortex."""
    spec.groups += [Group("SC.sup", SC_VISUAL, 1, "SC", grid=MID),
                    Group("SC.deep", SC_MOTOR, 1, "SC", grid=MID, kwta=0.02)]
    spec.synaptic += [
        Synaptic("retinotectal (transient + contrast cells)", ["RET.ton", "RET.toff", "RET.on", "RET.off"],
                 "SC.sup", "basal", 0.8, 3.0, 0.06),
        Synaptic("corticotectal V1 L5 -> SC", "A.L5PT", "SC.sup", "basal", 0.2, 2.0, 0.08),
        Synaptic("SC superficial -> intermediate", "SC.sup", "SC.deep", "basal", 0.8, 3.0, 0.08),
        Synaptic("frontal L5 -> SC (top-down)", "PFC.L5PT" if any(g.key == "PFC.L5PT" for g in spec.groups)
                 else "A.L5PT", "SC.deep", "basal", 0.2, 1.0, 0.5)]
    spec.pooled += [Pooled("SNr -| SC (saccade gate)", "SNr.out", "SC.deep", 0.8, channel="inh", sigma=None)]
    if any(g.key == "PUL.core" for g in spec.groups):
        spec.pooled += [Pooled("SC superficial -> pulvinar", "SC.sup", "PUL.core", 0.3, channel="exc", sigma=1.0)]
    return spec


def saccade_target(brain, retina, min_rate=0.15, contrast=3.0):
    """Where the colliculus' motor map points, if it is bursting: its most active cells (top 2%) must fire at least
    `contrast` times the map's mean and above `min_rate` (a saccade is triggered by the burst, not at a fixed time).
    Returns the rate-weighted centre of those cells, converted from map position to a visual-field position through
    the fovea warp, or None."""
    r = np.asarray(brain.g["SC.deep"].rate, np.float64)
    top = np.argsort(-r)[: max(1, int(0.02 * r.size))]
    if r[top].mean() < max(min_rate, contrast * r.mean()):
        return None
    p = (brain.pos["SC.deep"][top] * r[top, None]).sum(0) / r[top].sum()
    c = 2 * p - 1
    rad = min(np.linalg.norm(c), 1.0)
    from retina import GAMMA
    offset = c * rad ** (GAMMA - 1)
    return retina.gaze + offset * retina.extent / 2

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
    spec.synaptic = [_visual_cortex_gains(p) for p in syn]
    pooled = []
    for p in spec.pooled:
        if p.id == "TRN<-relay collaterals":
            p = replace(p, src=list(RELAYS))
        elif p.id in ("CT-5 TRN->relay", "TH.interneuron -> TH.TC"):
            p = replace(p, dst=_members(p.dst) + [r for r in RELAYS if r != "TH.TC"])
        pooled.append(p)
    spec.pooled = pooled
    return spec


VISUAL_AREAS = ("A",)   # grows as higher visual areas are added (note 75)
SLOW_HOMEOSTASIS = 0.1  # visual cortex layers balance their firing 10x more slowly (note 74)


def visual_homeostasis(brain):
    """Always-on homeostasis adapted visual cortex away from whatever it was looking at (a 20-step picture): with it
    off, V1 L2/3 identified colour pictures 75% of the time instead of 50%. Slow it for visual cortex (saved in the
    checkpoint, like HOMEO_SLOW)."""
    for g in brain.spec.groups:
        if g.key.split(".")[0] in VISUAL_AREAS and g.key in brain.g:
            brain.homeo_speed[g.key] = SLOW_HOMEOSTASIS


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


def vision_brain():
    from brainspec import full_brain
    return add_retina_and_lgn(full_brain())


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
        elif key in critical_period_groups() and lrf != 1:
            g.c = _r(g.c, rules=_r(g.c.rules, lr=g.c.rules.lr / lrf))
    brain.critical_period_closed = True

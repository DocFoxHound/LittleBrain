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

from dataclasses import replace

import numpy as np

import celltypes as C
from federation import Group, Pooled, Synaptic
from retina import DS, KONIO, MAGNO, PARVO

FINE = 32     # retina detail channels
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
    spec.groups += [Group("TH.M", C.TC_RELAY, 1, "TH", grid=MID), Group("TH.K", C.TC_RELAY, 2, "TH")]
    syn = []
    for p in spec.synaptic:
        if p.id == "CT-2 retina->relay (driver)":
            p = replace(p, id="CT-2 midget cells->parvo relay (driver)", src=list(PARVO))
        elif p.id.startswith("CT-1") and p.dst in ("A.L4", "A.L6"):
            p = replace(p, src=["TH.TC", "TH.M", "TH.K"])
        syn.append(p)
    for relay, drivers in (("TH.M", list(MAGNO + DS)), ("TH.K", list(KONIO))):
        syn += [Synaptic(f"CT-2 retina->{relay} (driver)", drivers, relay, "basal", 0.3, 3.0, 0.05),
                Synaptic(f"CT-3 L6 CT->{relay} (modulator)", "A.L6", relay, "basal", 0.7, 0.15, 0.08)]
    spec.synaptic = syn
    pooled = []
    for p in spec.pooled:
        if p.id == "TRN<-relay collaterals":
            p = replace(p, src=["TH.TC", "TH.M", "TH.K"])
        elif p.id in ("CT-5 TRN->relay", "TH.interneuron -> TH.TC"):
            p = replace(p, dst=_members(p.dst) + ["TH.M", "TH.K"])
        pooled.append(p)
    spec.pooled = pooled
    return spec


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

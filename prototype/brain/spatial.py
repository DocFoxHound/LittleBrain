"""
Knowing where you are (plan phase 2, note 83): head-direction cells, grid cells and the hippocampus they feed.

Self-motion is integrated into a sense of heading and position (path integration), the way the vestibular system,
proprioception and the motor copy do in animals. **Reduced model**: in the brain, path integration is done by
attractor networks (a ring of head-direction cells in the anterior thalamus/postsubiculum; continuous-attractor sheets
of grid cells in the medial entorhinal cortex; McNaughton et al. 2006, Burak & Fiete 2009). Here a small integrator
keeps heading and position and the cells fire from it, as the retina and the ear are filter banks:

  ATN.hd     head-direction cells: von Mises tuning to heading (Taube, Muller & Ranck 1990), preferred directions
             spread over the map
  MEC.grid   grid cells: three modules whose spacing grows along the map's y axis (dorsal small, ventral large, as
             along the real dorsoventral axis; Stensola et al. 2012); each cell a hexagonal pattern, sum of three
             cosines 60 degrees apart (Solstad, Moser & Einevoll 2006), with a phase set by its x position and band row

The integrator drifts (noise per step) and is reset by a landmark fix: when the game says the character is at home
(obs.body["home_scent"] or a path starting there, obs.body["from_home"]), position goes back to the origin. Place
cells are not built: they form in the hippocampus (DG/CA3/CA1) from grid and visual input.

Observations report self-motion as obs.motion: a list of (turn in radians, distance) steps since the last observation.
"""

import numpy as np

KAPPA = 4.0                 # head-direction tuning sharpness (half-width about 35 degrees)
SPACINGS = (2.0, 3.0, 4.5)  # grid spacing of the three modules, in world units (game cells)
SHARP = 3.0                 # grid field sharpness
DRIFT = 0.03                # path-integration noise per unit distance


class SelfMotion:
    def __init__(self, brain, seed=0):
        self.b = brain
        self.rng = np.random.default_rng(seed)
        self.heading, self.pos = 0.0, np.zeros(2)
        self.hd_pref, self.grid = None, None
        if "ATN.hd" in brain.pos:
            p = brain.pos["ATN.hd"]
            self.hd_pref = 2 * np.pi * (p[:, 0] + p[:, 1] * 0.125 % 1.0) % (2 * np.pi)
        if "MEC.grid" in brain.pos:
            p = brain.pos["MEC.grid"]
            module = np.minimum((p[:, 1] * len(SPACINGS)).astype(int), len(SPACINGS) - 1)
            within = p[:, 1] * len(SPACINGS) - module          # row inside the module's band, 0..1
            lam = np.array(SPACINGS)[module]
            orient = np.array([0.0, 0.3, 0.7])[module]          # modules differ a little in orientation
            # phase: x and the row inside the band tile one unit rhombus of the module's lattice
            a1 = np.stack([np.cos(orient), np.sin(orient)], 1)
            a2 = np.stack([np.cos(orient + np.pi / 3), np.sin(orient + np.pi / 3)], 1)
            phase = lam[:, None] * ((p[:, 0] * 4 % 1.0)[:, None] * a1 + within[:, None] * a2)
            ks = []
            for j in range(3):
                th = orient + j * np.pi / 3 + np.pi / 6
                ks.append(4 * np.pi / (np.sqrt(3) * lam)[:, None] * np.stack([np.cos(th), np.sin(th)], 1))
            self.grid = (np.stack(ks, 1), phase)                # (cells, 3, 2), (cells, 2)

    def reset(self):
        self.heading, self.pos = 0.0, np.zeros(2)

    def move(self, motion=None, body=None):
        """Integrate this observation's self-motion; landmark fixes reset the estimate."""
        body = body or {}
        if body.get("from_home"):
            self.reset()
        for turn, dist in (motion or []):
            self.heading = (self.heading + float(turn)) % (2 * np.pi)
            d = float(dist) * (1 + DRIFT * self.rng.standard_normal())
            self.pos = self.pos + d * np.array([np.cos(self.heading), np.sin(self.heading)]) \
                + DRIFT * abs(float(dist)) * self.rng.standard_normal(2)
        if body.get("home_scent"):
            self.pos = np.zeros(2)

    def channels(self):
        out = {}
        if self.hd_pref is not None:
            out["ATN.hd"] = np.exp(KAPPA * (np.cos(self.heading - self.hd_pref) - 1)).astype(np.float32)
        if self.grid is not None:
            k, phase = self.grid
            arg = np.einsum("cjd,cd->cj", k, self.pos[None, :] - phase)
            g = (np.cos(arg).sum(1) + 1.5) / 4.5                # 0..1, peaks on the lattice
            out["MEC.grid"] = (g ** SHARP).astype(np.float32)
        return out


def add_spatial(spec):
    """Head-direction and grid cells as inputs to the hippocampal formation (needs the hippocampal loop)."""
    from federation import Group, Synaptic
    if not any(g.key == "CA1.pyr" for g in spec.groups):
        return spec
    spec.groups += [Group("ATN.hd", None, 1, "ATN"), Group("MEC.grid", None, 1, "MEC", grid=16)]
    spec.inputs = spec.inputs + ["ATN.hd", "MEC.grid"]
    S = Synaptic
    spec.synaptic += [
        S("grid cells -> DG (perforant path)", "MEC.grid", "DG.gc", "basal", 0.4, 10.0, 0.4),
        S("grid cells -> CA3 (perforant path)", "MEC.grid", "CA3.pyr", "basal", 0.1, 5.0, 0.4),
        S("grid cells -> CA1 tuft", "MEC.grid", "CA1.pyr", "apical", 0.4, 1.0, 0.4),
        S("head direction -> EC (via postsubiculum)", "ATN.hd", "EC.stellate", "basal", 0.3, 1.0, 0.4)]
    # Place cells (note 84). The grid input must be strong enough to steer the dentate and CA3 (gains above), and
    # must not be rewritten on every visit: learning fast, the dentate's winners (which change from step to step)
    # each learned every place and the fields smeared away (CA3 place-like cells 0% vs 22% at 1/20 the speed).
    # Homeostasis in the sparse principal cells is slow, as real homeostatic plasticity is (hours, not seconds).
    spec.plasticity = dict(getattr(spec, "plasticity", {}), **{p: 0.05 for p in (
        "grid cells -> DG (perforant path)", "grid cells -> CA3 (perforant path)", "grid cells -> CA1 tuft")})
    spec.homeo_speed = dict(getattr(spec, "homeo_speed", {}),
                            **{k: 0.02 for k in ("DG.gc", "CA3.pyr", "CA1.pyr", "CA2.pyr")})
    return spec

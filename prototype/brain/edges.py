"""
Edge cells (note 85): V1's orientation-tuned cells as a fixed filter stage, like the retina, ear and grid cells.

**Reduced model.** In animals V1 simple cells sum ON and OFF LGN cells from side-by-side strips (Hubel & Wiesel 1962;
Reid & Alonso 1995), and complex cells pool simple cells of one orientation over phase (Movshon et al. 1978). That
tuning develops before and just after eye opening. In LittleBrain, growing it from waves and experience reached only
about half of real V1 (note 73) and later regressed to the measurement's floor (note 85), too weak for the areas
above to learn shapes. So the stage is built in, from the retina's ON and OFF midget cells on their full 32x32 map,
and learning starts above it (V2, V4, IT).

  A.E0..A.E3   complex cells for edges at 0, 45, 90 and 135 degrees: phase-invariant energy (even and odd Gabor
               filters over ON minus OFF), divided by the summed energy of all orientations nearby (cross-orientation
               normalisation; Heeger 1992), 16x16 map each

The filters work in the brain's map coordinates, so like V1 they are finer in the fovea (cortical magnification).
"""

import numpy as np

ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
EDGE_GROUPS = tuple(f"A.E{i}" for i in range(len(ANGLES)))
WAVELENGTH = 4.0     # in retina map pixels (32 across the field): ~1/8 of the field
SIGMA = 1.5
GAIN = 3.0
SEMI = 0.15          # semi-saturation of the normalisation


def _kernels():
    r = np.arange(-3, 4)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    out = []
    for a in ANGLES:
        u = xx * np.cos(a) + yy * np.sin(a)            # across the edge
        env = np.exp(-(xx ** 2 + yy ** 2) / (2 * SIGMA ** 2))
        even = env * np.cos(2 * np.pi * u / WAVELENGTH)
        odd = env * np.sin(2 * np.pi * u / WAVELENGTH)
        even -= even.mean()
        out.append((even / np.abs(even).sum(), odd / np.abs(odd).sum()))
    return out


KERNELS = _kernels()


def _filter(img, k):
    h, w = img.shape
    p = np.pad(img, 3, mode="edge")
    out = np.zeros_like(img)
    for dy in range(7):
        for dx in range(7):
            if k[dy, dx]:
                out += k[dy, dx] * p[dy:dy + h, dx:dx + w]
    return out


class EdgeCells:
    def __init__(self, brain):
        self.keys = [k for k in EDGE_GROUPS if k in brain.pos]
        self.n = 32
        on = brain.pos.get("RET.on")
        self.ok = bool(self.keys) and on is not None and "RET.off" in brain.pos
        if not self.ok:
            return
        self.n = int(round(np.sqrt(len(on))))
        self.on_px = np.minimum((on * self.n).astype(int), self.n - 1)
        off = brain.pos["RET.off"]
        self.off_px = np.minimum((off * self.n).astype(int), self.n - 1)
        self.at = {k: np.minimum((brain.pos[k] * self.n).astype(int), self.n - 1) for k in self.keys}

    def __call__(self, channels):
        if not self.ok or "RET.on" not in channels:
            return {}
        img = np.zeros((self.n, self.n), np.float32)
        img[self.on_px[:, 1], self.on_px[:, 0]] += channels["RET.on"]
        img[self.off_px[:, 1], self.off_px[:, 0]] -= channels["RET.off"]
        energy = []
        for even, odd in KERNELS:
            e, o = _filter(img, even), _filter(img, odd)
            energy.append(np.sqrt(e * e + o * o))
        energy = np.array(energy)
        pool = _filter(energy.mean(0), np.full((7, 7), 1 / 49.0))
        out = {}
        for i, k in enumerate(self.keys):
            xy = self.at[k]
            v = energy[i][xy[:, 1], xy[:, 0]] / (SEMI + pool[xy[:, 1], xy[:, 0]])
            out[k] = np.tanh(GAIN * np.maximum(v - 1.0, 0.0)).astype(np.float32)
        return out


def orientation_columns(pos, rng, columns=2.0, waves=8):
    """Which of the 4 orientations each cell's edge input comes from: a smooth pinwheel map (half the angle of a sum
    of plane waves; Swindale 1996), about `columns` hypercolumns across the map, so neighbouring cells share one."""
    z = np.zeros(len(pos), complex)
    for j in range(waves):
        a = np.pi * j / waves
        k = 2 * np.pi * columns * np.array([np.cos(a), np.sin(a)])
        z += np.exp(1j * (pos @ k + rng.uniform(0, 2 * np.pi)))
    return (((np.angle(z) + np.pi) / (2 * np.pi)) * len(ANGLES)).astype(int) % len(ANGLES)


def feature_wiring(brain):
    """Feature-specific convergence (note 87): in the groups listed in spec.feature_wiring, each cell's synapses from
    the edge cells all come from one orientation, set by an orientation-column map; where on the map they come from
    (the receptive field) is unchanged. A cell then sums one kind of edge over a large region, so it answers that edge
    anywhere in its field (V4: Desimone & Schein 1987; feature-specific V1 -> V2/V4 projections: Sincich & Horton
    2005). Run once, when a brain is built; the weights learn as usual."""
    from backend import like, to_host
    every = np.arange(len(to_host(brain.r)))
    ids = [every[brain.index[k]] for k in EDGE_GROUPS]
    pos = [brain.pos[k] for k in EDGE_GROUPS]
    for dst in getattr(brain.spec, "feature_wiring", ()):
        if dst not in brain.g:
            continue
        pref = orientation_columns(brain.pos[dst], brain.rng)
        for bank in brain.g[dst].banks():
            src = to_host(bank.src).copy()
            for j in range(len(EDGE_GROUPS)):
                cell, b_, s_ = np.nonzero(np.isin(src, ids[j]))
                if not len(cell):
                    continue
                q = pos[j][np.searchsorted(ids[j], src[cell, b_, s_])]
                for t in range(len(EDGE_GROUPS)):
                    m = pref[cell] == t
                    if t == j or not m.any():
                        continue
                    nearest = np.argmin(((q[m][:, None, :] - pos[t][None]) ** 2).sum(2), axis=1)
                    src[cell[m], b_[m], s_[m]] = ids[t][nearest]
            bank.src = like(bank.src, src, dtype=src.dtype)


def add_edges(spec):
    """Edge-cell input groups. V2 hears them alongside V1 layer 2/3. V4 takes its V1 input from them instead of V1
    layer 2/3 (which carries almost no shape, note 85), over a wider field, each cell from one orientation
    (feature_wiring)."""
    from dataclasses import replace
    from federation import Group
    spec.groups += [Group(k, None, 1, "A", grid=16) for k in EDGE_GROUPS]
    spec.inputs = spec.inputs + list(EDGE_GROUPS)
    import os
    v4_feature = os.environ.get("VISION_V4_FEATURE", "0") == "1"   # opt-in: no clear 3-seed gain yet (note 87)
    syn = []
    for p in spec.synaptic:
        if p.dst in ("V2.L4", "V2.L23", "V4.L4", "V4.L23") and p.id.startswith("FF "):
            src = (p.src if isinstance(p.src, list) else [p.src])
            if v4_feature and p.dst.startswith("V4"):
                src = [s_ for s_ in src if s_ != "A.L23"]
                p = replace(p, spread=max(p.spread, 0.35))
            p = replace(p, src=src + list(EDGE_GROUPS))
        syn.append(p)
    spec.synaptic = syn
    if v4_feature:
        spec.feature_wiring = ("V4.L4", "V4.L23")
    from vision import FEEDFORWARD
    for area in ("V2", "V4"):
        if area in FEEDFORWARD:
            ff = tuple(FEEDFORWARD[area])
            if v4_feature and area == "V4":
                ff = tuple(s_ for s_ in ff if s_ != "A.L23")
            FEEDFORWARD[area] = ff + EDGE_GROUPS
    return spec

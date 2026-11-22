"""
A retina for LittleBrain (note 72): the eye as a filter bank, the way the real retina pre-processes light.

The brain's eye used to be 256 grey point samples. A real retina sends ~20 kinds of ganglion-cell output. This
models the main ones as functional filters (computed, like the old retina input, not simulated cell by cell):

  RET.on / RET.off       midget (parvocellular) ON and OFF centre-surround cells: fine contrast detail.
                         Difference of Gaussians, surround 3x the centre (Enroth-Cugell & Robson 1966).
  RET.rg_on / RET.rg_off midget red-green opponent cells: L-cone centre against M-cone surround (Dacey 2000).
  RET.by                 small bistratified blue-ON cells: S cones against L+M (Dacey & Lee 1994).
  RET.ton / RET.toff     parasol (magnocellular) transient cells: respond to change between frames; larger fields.
  RET.ds0..3             ON-OFF direction-selective cells, 4 directions (Barlow & Levick 1965), as correlation
                         detectors on the parasol signal (Reichardt-style).
  retina                 the old 8x8 light-level input, kept for older pathways (e.g. light to the SCN clock,
                         like intrinsically photosensitive ganglion cells).

Fovea (cortical magnification): every channel sits on a uniform map, but the map is warped onto the visual field
so the centre is sampled densely and sharply and the periphery sparsely and blurrily: a map point at distance r
(0..1) from the map centre looks at visual-field radius r^GAMMA, and its receptive field grows with eccentricity.
Downstream visual areas inherit the magnification because they sit on the same map.

Contrast is divided by local mean light (Weber adaptation) and squashed to firing rates in [0, 1].
"""

import numpy as np

GAMMA = 2.0            # foveal magnification: field radius = map radius ** GAMMA
RF_SCALE = 0.6         # centre sigma relative to the local spacing of cells
PARASOL_SCALE = 2.0    # parasol cells have ~2x larger fields than midget cells
GAIN = 4.0             # contrast -> rate
MIN_SIGMA_PX = 0.6     # smallest centre field, in pixels of the incoming picture
DS_GAIN = 160.0        # direction-selective cells (was 16: too weak for textures such as random dots, note 74)

PARVO = ("RET.on", "RET.off", "RET.rg_on", "RET.rg_off")
KONIO = ("RET.by",)
MAGNO = ("RET.ton", "RET.toff")
DS = ("RET.ds0", "RET.ds1", "RET.ds2", "RET.ds3")   # preferred directions: right, down, left, up
CHANNELS = PARVO + KONIO + MAGNO + DS
DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], np.float64)


def cones(image):
    """RGB in [0,1] (or grey) -> L, M, S cone images (a simple linear approximation)."""
    img = np.asarray(image, np.float32)
    if img.ndim == 2:
        return img, img, img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.6 * r + 0.4 * g, 0.3 * r + 0.7 * g, b


def _blur(img, sigma):
    """Gaussian blur with sigma in pixels (FFT, wrap-free by padding)."""
    if sigma < 0.3:
        return img
    h, w = img.shape
    pad = int(min(3 * sigma, max(h, w)))
    p = np.pad(img, pad, mode="edge")
    H, W = p.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(p) * g, s=p.shape)
    return out[pad:pad + h, pad:pad + w].astype(np.float32)


class Retina:
    """Samples images for one brain. gaze: where the fovea points, in image coordinates [0,1]^2.
    extent: how much of the image the eye covers (1 = the whole image when looking at its centre)."""

    def __init__(self, brain, gaze=(0.5, 0.5), extent=1.0):
        self.b = brain
        self.gaze, self.extent = np.array(gaze, np.float64), float(extent)
        self.keys = [k for k in CHANNELS if k in brain.pos]
        self.prev = None
        from edges import EdgeCells
        self.edges = EdgeCells(brain)   # V1 edge cells, if the brain has them (note 85)
        self.geo = {}
        for k in self.keys + (["retina"] if "retina" in brain.pos else []):
            gs = getattr(brain, "grid", {}).get(k, 8)
            c = 2 * brain.pos[k] - 1                          # map position, centred, in [-1, 1]
            r = np.minimum(np.linalg.norm(c, axis=1), 1.0)
            spacing = GAMMA * np.maximum(r, 1.0 / gs) ** (GAMMA - 1) * (2.0 / gs)   # field units per map tile
            scale = PARASOL_SCALE if (k in MAGNO or k in DS) else 1.0
            self.geo[k] = (c * r[:, None] ** (GAMMA - 1), RF_SCALE * scale * spacing / 2)  # offset, sigma (half-field units)

    def look_at(self, gaze):
        self.gaze = np.clip(np.asarray(gaze, np.float64), 0, 1)

    def _where(self, k, shape, shift=None):
        off, sig = self.geo[k]
        h, w = shape
        xy = self.gaze + off * self.extent / 2
        if shift is not None:
            xy = xy + shift
        # sigma in pixels, never below ~half a pixel: the picture's own resolution limits the sharpest field
        # (like the eye's optics limiting the fovea)
        return xy, np.maximum(sig * self.extent / 2 * max(h, w), MIN_SIGMA_PX)

    @staticmethod
    def _sample(img, xy):
        """Bilinear sample of img at image coordinates xy (N,2) in [0,1]; outside the image = edge value."""
        h, w = img.shape
        x = np.clip(xy[:, 0] * w - 0.5, 0, w - 1)
        y = np.clip(xy[:, 1] * h - 0.5, 0, h - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
                + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)

    def _bank(self, img, sigmas):
        """Blurred copies of img at a few scales; each cell reads the copy nearest its own field size."""
        levels = np.array([0.0, 0.7, 1.4, 2.8, 5.6, 11.2, 22.4])
        want = np.unique(np.abs(levels[:, None] - np.concatenate([s.ravel() for s in sigmas])[None]).argmin(0))
        return levels, {int(i): _blur(img, levels[i]) for i in want}

    def _read(self, bank, xy, sigma):
        levels, imgs = bank
        idx = np.abs(levels[:, None] - sigma[None]).argmin(0)
        out = np.empty(len(xy), np.float32)
        for i in np.unique(idx):
            m = idx == i
            out[m] = self._sample(imgs[int(i)], xy[m])
        return out

    def __call__(self, image, blank=False):
        """All channels for one frame: {group key: rates}. Keeps the previous frame for transient and motion cells."""
        image = np.asarray(image, np.float32)
        L, M, S = cones(image)
        shape = L.shape
        lum = 0.5 * (L + M)
        mean_light = float(lum.mean()) + 0.1
        out, sig_all = {}, []
        pos = {k: self._where(k, shape) for k in self.geo}
        for k in self.geo:
            sig_all += [pos[k][1], 3 * pos[k][1]]
        bL, bM, bS, blum = (self._bank(x, sig_all) for x in (L, M, S, lum))
        f = lambda v: np.tanh(GAIN * np.maximum(v, 0) / mean_light).astype(np.float32)
        for k in self.keys:
            xy, s = pos[k]
            if k in ("RET.on", "RET.off"):
                d = self._read(blum, xy, s) - self._read(blum, xy, 3 * s)
                out[k] = f(d if k == "RET.on" else -d)
            elif k in ("RET.rg_on", "RET.rg_off"):
                d = self._read(bL, xy, s) - self._read(bM, xy, 3 * s)
                out[k] = f(d if k == "RET.rg_on" else -d)
            elif k == "RET.by":
                out[k] = f(self._read(bS, xy, s) - 0.5 * (self._read(bL, xy, s) + self._read(bM, xy, s)))
        # parasol signal now, and at a small displacement in each preferred direction (for direction selectivity)
        now = {}
        for k in [k for k in self.keys if k in MAGNO or k in DS]:
            xy, s = pos[k]
            now[k] = self._read(blum, xy, s) - self._read(blum, xy, 3 * s)
            if k in DS:
                d = DIRS[DS.index(k)] * (s / max(shape))[:, None] * 2.0
                now[k + "@"] = self._read(blum, xy - d, s) - self._read(blum, xy - d, 3 * s)
        prev = self.prev if self.prev is not None else now
        for k in self.keys:
            if k in MAGNO:
                d = now[k] - prev[k]
                out[k] = f(d if k == "RET.ton" else -d)
            elif k in DS:
                # Reichardt correlator: motion along the preferred direction moves the earlier signal from the
                # displaced point to here; the mirror term cancels flicker and the opposite direction
                corr = now[k] * prev[k + "@"] - now[k + "@"] * prev[k]
                out[k] = np.tanh(DS_GAIN * np.maximum(corr, 0) / mean_light ** 2).astype(np.float32)
        self.prev = now
        if "retina" in self.geo:   # the old light-level input: plain samples, as before
            xy, _ = pos["retina"]
            out["retina"] = self._sample(lum, xy).astype(np.float32)
        if self.edges.ok:
            out.update(self.edges(out))
        return out


class Sight:
    """What the agent passes around for one picture on a brain with a retina: each brain step asks for this
    step's channels, so transient and motion cells see the picture appear, change and vanish."""

    def __init__(self, retina, image):
        """image: one picture, or a list of pictures (a movie, one per brain step, looping)."""
        self.retina = retina
        if isinstance(image, (list, tuple)):
            self.frames = [np.asarray(f, np.float32) for f in image]
        else:
            self.frames = [np.asarray(image, np.float32)]
        self.image, self.t = self.frames[0], 0

    def channels(self):
        img = self.frames[self.t % len(self.frames)]
        self.t += 1
        return self.retina(img)

    def __mul__(self, k):            # 0 * sight = darkness (the agent's rest phase)
        return Sight(self.retina, self.frames[0] * k)

    __rmul__ = __mul__

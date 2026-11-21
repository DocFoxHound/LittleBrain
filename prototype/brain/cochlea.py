"""
An ear for LittleBrain (plan track L1, note 81): sound -> auditory-nerve firing, the way the retina turns light into
ganglion-cell firing (retina.py).

The cochlea splits sound into frequency bands (a "cochleagram": energy per band over time). Each band is read by
auditory-nerve fibres with staggered thresholds, so loudness is coded by how many fibres fire (dynamic-range
fractionation; Liberman 1978), plus onset fibres that fire when a band's energy rises (Rhode & Smith 1986 onset
responses). On the brain's map: frequency along x (the tonotopic axis), fibre threshold along y.

  EAR.sus   sustained fibres: tanh(gain * (energy - threshold_y)) for band x
  EAR.on    onset fibres: rise in energy since the previous frame

Sources of sound:
  cochleagram(wave, rate)   real audio -> bands x frames (log-spaced filter bank over an FFT, ~10 ms frames)
  text_sound(text)          text -> the same kind of cochleagram: each letter gets a fixed, distinct spectral
                            pattern (two "formant" bands plus a little noise for consonants), one frame per letter,
                            a silent frame between words. A stand-in for speech: words read and words heard reach
                            the brain the same way (plan track L).
"""

import numpy as np

BANDS = 16          # frequency bands (map x)
LEVELS = 16         # fibre thresholds per band (map y)
GAIN = 4.0


def _letter_table():
    """A fixed spectral pattern per letter: vowels get two formant bands (vowel-like); consonants a band of noise-like
    energy (high bands for s/f/sh-like letters, low bands for m/n/b-like)."""
    rng = np.random.default_rng(20261122)
    table = {}
    vowels = "aeiouy"
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        spec = np.zeros(BANDS, np.float32)
        if ch in vowels:
            f1, f2 = 2 + (i % 5), 7 + (i * 3) % 8
            spec[f1] += 1.0
            spec[min(BANDS - 1, f2)] += 0.8
        else:
            centre = int(rng.integers(0, BANDS))
            width = int(rng.integers(1, 4))
            spec[max(0, centre - width):centre + width] += 0.6
        spec += 0.05 * rng.random(BANDS).astype(np.float32)
        table[ch] = spec / spec.max()
    return table


LETTERS = _letter_table()


def text_sound(text, level=0.8):
    """Text -> cochleagram (bands x frames): one frame per letter, a silent frame between words."""
    frames = []
    for word in str(text).lower().split():
        for ch in word:
            if ch in LETTERS:
                frames.append(level * LETTERS[ch])
        frames.append(np.zeros(BANDS, np.float32))
    return np.array(frames, np.float32).T if frames else np.zeros((BANDS, 1), np.float32)


def cochleagram(wave, rate, frame_ms=10.0, fmin=100.0, fmax=8000.0):
    """Real audio -> bands x frames: magnitude spectrum in log-spaced bands, compressed (loudness ~ cube root)."""
    wave = np.asarray(wave, np.float32)
    hop = max(1, int(rate * frame_ms / 1000))
    win = 2 * hop
    n = max(1, (len(wave) - win) // hop + 1)
    edges = np.geomspace(fmin, min(fmax, rate / 2 - 1), BANDS + 1)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    out = np.zeros((BANDS, n), np.float32)
    w = np.hanning(win).astype(np.float32)
    for t in range(n):
        seg = wave[t * hop:t * hop + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * w))
        for b in range(BANDS):
            m = (freqs >= edges[b]) & (freqs < edges[b + 1])
            out[b, t] = spec[m].mean() if m.any() else 0.0
    out = np.cbrt(out)
    return out / (out.max() + 1e-6)


class Ear:
    """Turns cochleagram frames into auditory-nerve firing for one brain (groups EAR.sus, EAR.on)."""

    def __init__(self, brain):
        self.b = brain
        self.keys = [k for k in ("EAR.sus", "EAR.on") if k in brain.pos]
        self.band, self.level = {}, {}
        for k in self.keys:
            p = brain.pos[k]
            self.band[k] = np.minimum((p[:, 0] * BANDS).astype(int), BANDS - 1)
            self.level[k] = p[:, 1]              # fibre threshold, 0..1 (low-threshold fibres at the top)
        self.prev = np.zeros(BANDS, np.float32)

    def __call__(self, frame):
        """frame: energy per band (length BANDS) for this brain step. Returns {group key: rates}."""
        e = np.asarray(frame, np.float32)
        out = {}
        if "EAR.sus" in self.keys:
            th = 0.05 + 0.85 * self.level["EAR.sus"]
            out["EAR.sus"] = np.tanh(GAIN * np.maximum(e[self.band["EAR.sus"]] - th, 0)).astype(np.float32)
        if "EAR.on" in self.keys:
            rise = np.maximum(e - self.prev, 0)
            th = 0.02 + 0.5 * self.level["EAR.on"]
            out["EAR.on"] = np.tanh(GAIN * np.maximum(rise[self.band["EAR.on"]] - th, 0)).astype(np.float32)
        self.prev = e
        return out


class Hearing:
    """What the agent passes around for one turn's sound: each brain step asks for its frame (frames run once, then
    silence)."""

    def __init__(self, ear, cochleagram_):
        self.ear, self.c, self.t = ear, np.asarray(cochleagram_, np.float32), 0

    def channels(self):
        frame = self.c[:, self.t] if self.t < self.c.shape[1] else np.zeros(BANDS, np.float32)
        self.t += 1
        return self.ear(frame)

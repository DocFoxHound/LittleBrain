"""
A small piece of brain for the first wiring test (roadmap step 4).

    retina -> thalamus (TC relay + TRN) -> area A (lower, "posterior") -> area B (higher, "anterior")
                       ^-- A layer 6 (CT) --'      ^------- feedback: B deep layers -> A apical tufts

Wiring rules (notes/17-connectivity-and-waves.md):
  - local connections fall off with distance; long-range ones land at the matching map location;
  - feedforward comes from layer 2/3 and lands on basal dendrites (layer 4 first);
  - feedback comes from deep layers (5 IT) and lands on apical tufts;
  - higher areas see larger regions, and feedback spreads wider than feedforward;
  - every cell type has a cap on the fraction of its synapses that come from other structures.
Inhibition (PV, SST, VIP, TRN) is pooled per tile. Neuromodulators are functional models.

Waves (optional): a traveling wave along the back-to-front axis modulates excitability and gates
long-range synapses ("communication through coherence"): a forward wave favors feedforward
transmission, a reverse wave favors feedback.
"""

from dataclasses import dataclass

import numpy as np

from cells import ExcitatoryGroup, InhibitoryGroup, Neuromodulators
from celltypes import L4_STELLATE, L5_IT, L5_PT, L6_CT, L23_IT, PV, SST, TC_RELAY, TRN, VIP

GRID = 8                     # tiles per side in every structure (shared map layout)
PER_TILE = {"L4": (L4_STELLATE, 6), "L23": (L23_IT, 10), "L5IT": (L5_IT, 3), "L5PT": (L5_PT, 3),
            "L6": (L6_CT, 5), "PV": (PV, 4), "SST": (SST, 3), "VIP": (VIP, 2)}
EXC = ["L4", "L23", "L5IT", "L5PT", "L6"]
AXIS = {"retina": -0.6, "TH": -0.5, "A": 0.0, "B": 1.5}   # position on the back-to-front axis

# Incoming excitatory synapses: compartment -> [(source, fraction of slots, kind, strength)]
# kind "local": same area, distance-weighted.  kind "topo": other structure, same map location.
LOCAL = {
    "L4":   {"basal": [("L4", 0.45), ("L6", 0.30), ("L23", 0.25)]},
    "L23":  {"basal": [("L4", 0.35), ("L23", 0.50), ("L5IT", 0.15)], "apical": [("L23", 0.5), ("L5IT", 0.5)]},
    "L5IT": {"basal": [("L23", 0.5), ("L5IT", 0.2), ("L5PT", 0.1), ("L4", 0.2)], "apical": [("L23", 1.0)]},
    "L5PT": {"basal": [("L23", 0.5), ("L5IT", 0.2), ("L5PT", 0.1), ("L4", 0.2)], "apical": [("L23", 1.0)]},
    "L6":   {"basal": [("L4", 0.3), ("L6", 0.3), ("L5IT", 0.2), ("L23", 0.2)]},
}
# Long-range inputs (fraction, strength, map spread). FF lands basal, FB lands apical.
LONG = {
    # Thalamic synapses are few (~10%) but fire in sync, so they're effectively strong (Bruno & Sakmann 2006)
    ("A", "L4"):   {"basal": [("TH.TC", 0.10, 8.0, 0.06)]},
    ("A", "L6"):   {"basal": [("TH.TC", 0.10, 5.0, 0.06)]},
    ("A", "L23"):  {"apical": [("B.L5IT", 0.6, 1.0, 0.30)]},           # feedback: wide spread
    ("A", "L5IT"): {"apical": [("B.L5IT", 0.6, 1.0, 0.30)]},
    ("A", "L5PT"): {"apical": [("B.L5IT", 0.6, 1.0, 0.30)]},
    ("B", "L4"):   {"basal": [("A.L23", 0.10, 8.0, 0.25)]},            # feedforward: larger receptive fields
    ("B", "L23"):  {"basal": [("A.L23", 0.15, 3.0, 0.25)]},
}
LOCAL_SPREAD = {"A": 0.12, "B": 0.25}  # higher areas integrate over larger regions


@dataclass
class Wave:
    amplitude: float = 0.3       # excitability modulation
    period: float = 12.0         # steps per cycle
    k: float = np.pi / 3         # spatial frequency along the back-to-front axis
    direction: int = 1           # +1 back-to-front ("encode"), -1 front-to-back ("recall")


class Brain:
    def __init__(self, seed=0, lr_scale=1.0, remove_feedback=False):
        self.rng = rng = np.random.default_rng(seed)
        self.mod = Neuromodulators()
        T = GRID * GRID
        self.groups, self.inh, self.index, self.pos, self.tile, self.X = {}, {}, {}, {}, {}, {}
        offset = 0

        def place(name, n_per_tile, structure):
            nonlocal offset
            tiles = np.repeat(np.arange(T), n_per_tile)
            xy = (np.stack([tiles % GRID, tiles // GRID], 1) + rng.random((tiles.size, 2))) / GRID
            self.index[name] = np.arange(offset, offset + tiles.size)
            self.pos[name], self.tile[name] = xy, tiles
            self.X[name] = AXIS[structure] + (0.3 if structure in ("retina", "TH") else 1.0) * xy[:, 0]
            offset += tiles.size
            return tiles.size

        place("retina", 4, "retina")
        self.groups["TH.TC"] = ExcitatoryGroup(TC_RELAY, place("TH.TC", 4, "TH"), rng)
        self.inh["TH.TRN"] = InhibitoryGroup(TRN, place("TH.TRN", 1, "TH"))
        for area in ("A", "B"):
            for g, (ctype, k) in PER_TILE.items():
                n = place(f"{area}.{g}", k, area)
                if g in EXC:
                    self.groups[f"{area}.{g}"] = ExcitatoryGroup(ctype, n, rng)
                else:
                    self.inh[f"{area}.{g}"] = InhibitoryGroup(ctype, n)
        self.n_total = offset
        self.r = np.zeros(offset, np.float32)
        d = np.arange(T)
        dist2 = ((d[:, None] % GRID - d[None] % GRID) ** 2 + (d[:, None] // GRID - d[None] // GRID) ** 2)
        self.K = np.exp(-dist2 / 2.0)
        self.K /= self.K.sum(1, keepdims=True)
        self._wire(lr_scale, remove_feedback)
        self.wave = None
        self.ablated = set()
        self.long_fraction = {name: float(np.mean(np.concatenate(
            [b.long.ravel() for b in g.banks()]))) for name, g in self.groups.items()}

    # ---------- wiring ----------

    def _sample(self, recv, src, k, spread, kind):
        """For each receiving cell, pick k source cells: distance-weighted (local) or map-matched (topo)."""
        d = np.linalg.norm(self.pos[recv][:, None, :] - self.pos[src][None, :, :], axis=2)
        w = np.exp(-d / spread) if kind == "local" else np.exp(-d ** 2 / (2 * spread ** 2))
        cdf = np.cumsum(w, axis=1)
        u = self.rng.random((w.shape[0], k)) * cdf[:, -1:]
        return self.index[src][np.minimum(np.array([np.searchsorted(c, uu) for c, uu in zip(cdf, u)]),
                                          w.shape[1] - 1)]

    def _wire_bank(self, recv, bank, local_spec, long_spec, area, lr_scale):
        n, slots = bank.src.shape[0], bank.src.shape[1] * bank.src.shape[2]
        cols, longs, gains = [], [], []
        long_total = 0.0
        for src, frac, gain, spread in long_spec:
            k = int(round(frac * lr_scale * slots))
            if k:
                cols.append(self._sample(recv, src, k, spread, "topo"))
                longs.append(np.ones((n, k), bool))
                gains.append(np.full((n, k), gain, np.float32))
            long_total += k
        remaining = slots - int(long_total)
        if not local_spec:  # e.g. thalamic relay cells: every input comes from outside
            assert remaining == 0, f"{recv}: long-range fractions must fill all slots"
        local_total = sum(f for _, f in local_spec) or 1.0
        counts = [int(round(f / local_total * remaining)) for _, f in local_spec]
        if counts:
            counts[0] += remaining - sum(counts)
        for (src, _), k in zip(local_spec, counts):
            if k:
                cols.append(self._sample(recv, f"{area}.{src}", k, LOCAL_SPREAD.get(area, 0.12), "local"))
                longs.append(np.zeros((n, k), bool))
                gains.append(np.ones((n, k), np.float32))
        src_all, long_all, gain_all = (np.concatenate(x, 1) for x in (cols, longs, gains))
        perm = self.rng.permuted(np.tile(np.arange(slots), (n, 1)), axis=1)  # mix inputs across branches
        take = lambda a: np.take_along_axis(a, perm, 1).reshape(bank.src.shape)
        bank.src[:], bank.long[:], bank.gain[:] = take(src_all), take(long_all), take(gain_all)

    def _wire(self, lr_scale, remove_feedback):
        # Thalamic relay cells: retina (drivers: few, strong) + layer 6 of area A (modulators: many, weak)
        tc = self.groups["TH.TC"].basal
        self._wire_bank("TH.TC", tc, [], [("retina", 0.3, 3.0, 0.05), ("A.L6", 0.7, 0.15, 0.08)], "TH", 1.0)
        for area in ("A", "B"):
            for g in EXC:
                name, grp = f"{area}.{g}", self.groups[f"{area}.{g}"]
                spec = LONG.get((area, g), {})
                for comp, bank in (("basal", grp.basal), ("apical", grp.apical)):
                    if bank is None:
                        continue
                    long_spec = [s for s in spec.get(comp, []) if not (remove_feedback and comp == "apical")]
                    scale = 1.0 if long_spec and long_spec[0][0].startswith("TH") else lr_scale
                    self._wire_bank(name, bank, LOCAL[g][comp], long_spec, area, scale)

    # ---------- waves ----------

    def set_wave(self, wave: Wave | None):
        """Install a wave: gates every synapse by phase coherence for the wave's direction."""
        self.wave = wave
        Xg = np.zeros(self.n_total, np.float32)
        for name, idx in self.index.items():
            Xg[idx] = self.X[name]
        for name, g in self.groups.items():
            Xd = self.X[name]
            for b in g.banks():
                if wave is None:
                    b.wave_gate = None
                    continue
                Xs = Xg[b.src]
                D = np.abs(Xd[:, None, None] - Xs)
                # Arrival phase minus local phase, with conduction speed matched to the wave speed
                diff = wave.k * ((Xs - Xd[:, None, None]) + wave.direction * D)
                b.wave_gate = (0.5 * (1 + np.cos(diff))).astype(np.float32)

    def ablate(self, *what):
        """Lesions for experiments (applied after development):
          "feedback"    B -> A apical synapses silenced
          "topdown"     B -> A top-down drive to VIP cells removed
          "B"           area B silenced entirely
          "learn_A_local" / "learn_long"   those synapses can't learn"""
        for w in what:
            self.ablated.add(w)
            for name, g in self.groups.items():
                for b in g.banks():
                    if w == "feedback" and name.startswith("A.") and b is g.apical:
                        b.gain = np.where(b.long, 0.0, b.gain).astype(np.float32)
                    if w == "learn_A_local" and name.startswith("A."):
                        b.learn_mask = b.long.astype(np.float32)
                    if w == "learn_long":
                        b.learn_mask = (~b.long).astype(np.float32)

    def wave_gain(self, name, t):
        w = self.wave
        if w is None:
            return 1.0
        return 1.0 + w.amplitude * np.cos(w.k * self.X[name] - w.direction * 2 * np.pi / w.period * t)

    def crest(self, name, t, window=0.3):
        """Cells near the wave crest at step t (for wave-scheduled updates)."""
        w = self.wave
        return (np.cos(w.k * self.X[name] - w.direction * 2 * np.pi / w.period * t) > window).astype(np.float32)

    # ---------- running ----------

    def _pool(self, name, rate):
        T = GRID * GRID
        return np.bincount(self.tile[name], weights=rate, minlength=T) / np.bincount(self.tile[name], minlength=T)

    def step(self, retina, t, learn=False, ach=0.5, dopamine=0.0, arousal=0.0, scheduled=False):
        r = self.r
        r[self.index["retina"]] = retina
        apical_mode = 1.0 - 0.5 * (ach - 0.5)  # Hasselmo: ACh suppresses feedback during encoding
        new = {}
        inhib = {}
        for area in ("A", "B"):
            E = np.mean([self._pool(f"{area}.{g}", self.groups[f"{area}.{g}"].rate) for g in EXC], axis=0)
            vip_pool = self._pool(f"{area}.VIP", self.inh[f"{area}.VIP"].rate)
            topdown = (self._pool("B.L5IT", self.groups["B.L5IT"].rate)
                       if area == "A" and "topdown" not in self.ablated else 0.0)
            KE = self.K @ E
            pv = self.inh[f"{area}.PV"].step((6.0 * KE)[self.tile[f"{area}.PV"]])
            sst = self.inh[f"{area}.SST"].step((4.0 * KE - 2.0 * (self.K @ vip_pool))[self.tile[f"{area}.SST"]])
            vip = self.inh[f"{area}.VIP"].step((1.0 * ach + 2.0 * topdown + 0 * KE)[self.tile[f"{area}.VIP"]])
            inhib[area] = (self.K @ self._pool(f"{area}.PV", pv), self.K @ self._pool(f"{area}.SST", sst))
            for nm, rate in ((f"{area}.PV", pv), (f"{area}.SST", sst), (f"{area}.VIP", vip)):
                new[nm] = rate
        trn_drive = 3.0 * self._pool("A.L6", self.groups["A.L6"].rate) + 2.0 * self._pool("TH.TC", self.groups["TH.TC"].rate)
        trn = self.inh["TH.TRN"].step(trn_drive[self.tile["TH.TRN"]])
        new["TH.TRN"] = trn
        trn_in = (self.K @ self._pool("TH.TRN", trn))[self.tile["TH.TC"]]

        self.updated = {}
        for name, g in self.groups.items():
            if "B" in self.ablated and name.startswith("B."):
                g.rate[:] = 0.0
                new[name] = g.rate
                continue
            gain = self.wave_gain(name, t)
            if name == "TH.TC":
                target = g.compute(r, trn_in=trn_in, gain=gain, allow_burst=ach < 0.3)
            else:
                area = name[0]
                pv_in, sst_in = (x[self.tile[name]] for x in inhib[area])
                target = g.compute(r, pv_in=1.5 * pv_in, sst_in=sst_in, gain=gain, apical_mode=apical_mode)
                g.inhibitory_plasticity(1.5 * pv_in, eta=0.05)  # PV inhibition tracks each cell's excess firing
            mask = self.crest(name, t) if (scheduled and self.wave is not None) else None
            g.update(target, learn=learn, mask=mask)
            if learn:
                g.modulate(dopamine, ach, arousal)
            g.tick()
            new[name] = g.rate
            self.updated[name] = 1.0 if mask is None else float(mask.mean())
        for name, rate in new.items():
            r[self.index[name]] = rate

    def sleep(self):
        for g in self.groups.values():
            g.sleep()

    def experience(self, n_blobs=3, radius=0.13):
        """One ordinary sight: a few blobs at random places on the retina."""
        c = self.rng.random((n_blobs, 2))
        d = np.linalg.norm(self.pos["retina"][:, None, :] - c[None], axis=2).min(axis=1)
        return (d < radius).astype(np.float32)

    def develop(self, days=4, presentations=60, hold=6, blank=3):
        """Development: ordinary experience (random patterns) with learning on, and a night's sleep after
        each day. Turns an unselective, randomly wired cortex into a selective one."""
        n_ret, t = self.index["retina"].size, 0
        for _ in range(days):
            for _ in range(presentations):
                x = self.experience()
                for s in range(hold + blank):
                    self.step(x if s < hold else 0 * x, t, learn=True, ach=0.6,
                              dopamine=0.5 if s == 2 else 0.0, arousal=0.3)
                    t += 1
            self.sleep()

    def calibrate(self, steps=900, density=0.2, slope=6.0, hold=6, blank=3, eta=0.2):
        """Development: each cell tunes its own threshold and sharpness, closed-loop, until it fires
        for about its target fraction of the time in the running network (fast homeostasis)."""
        n_ret = self.index["retina"].size
        mean = {k: 0.0 for k in self.groups}
        var = {k: 1.0 for k in self.groups}
        warm = {k: [] for k in self.groups}
        for t in range(steps):
            if t % (hold + blank) == 0:  # task-like: a pattern held for a few steps, then a gap
                pattern = (self.rng.random(n_ret) < density).astype(np.float32)
            self.step(pattern if t % (hold + blank) < hold else 0 * pattern, t)
            if t < 45:  # first guess: thresholds from the first few presentations
                for k, g in self.groups.items():
                    warm[k].append(g.soma.copy())
                if t == 44:
                    for k, g in self.groups.items():
                        h = np.array(warm[k])
                        g.threshold = np.quantile(h, 1 - g.c.target_rate, axis=0).astype(np.float32)
                        mean[k], var[k] = h.mean(0), h.var(0)
                continue
            for k, g in self.groups.items():
                mean[k] = 0.98 * mean[k] + 0.02 * g.soma
                var[k] = 0.98 * var[k] + 0.02 * (g.soma - mean[k]) ** 2
                std = np.sqrt(var[k]) + 1e-3
                g.scale = (slope / g.c.slope / std).astype(np.float32)
                g.threshold += (eta * (std + 0.05) * ((g.soma > g.threshold) - g.c.target_rate)).astype(np.float32)
        for g in self.groups.values():
            g.rate[:] = 0
            g.adapt_state[:] = 0
        for g in self.inh.values():
            g.rate[:] = 0
        self.r[:] = 0

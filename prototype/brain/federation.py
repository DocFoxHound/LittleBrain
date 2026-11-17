"""
The federated brain: regions built from data (groups + pathways) instead of hand-written code.
Roadmap step 7. Pathway IDs refer to notes/regions/connections.md.

A brain is a list of GROUPS (cell type x structure, placed on an 8x8 tile map shared by all
structures, so "same map location" means the same thing everywhere) and two kinds of connection:

  Synaptic pathway   learnable excitatory synapses into a group's basal or apical dendrites, sampled
                     either "local" (same structure, distance-weighted) or "topo" (another structure,
                     same map location, with a spread). Long-range = from another structure.
  Pooled pathway     population-level influence (all inhibition, inhibitory projections, tonic
                     systems): the target receives the source's rate pooled per tile and blurred by a
                     kernel. Into excitatory cells it arrives on a channel: pv (cell body), sst (apical),
                     dend (basal branches), ais (axon initial segment), volume (slow GABA-B),
                     inh (generic subtractive, e.g. Purkinje -> DCN, TRN -> relay), exc (additive).
Global signals (acetylcholine, dopamine, arousal, reward) are passed to step() each time.
"""

import os
from dataclasses import dataclass, field

import numpy as np

import backend
from backend import like, to_host, xpof
from cells import ExcitatoryGroup, InhibitoryGroup, ModulatorGroup, make_group
from lean_neuron import _hash_uniform

GRID = 8
CHOICES = 64   # seeded choice: candidates offered per rewiring slot (6 bits stored)
LOG_MAX = 512  # moments that mattered, kept per day for value-guided rewiring
CONTACTS = 4   # synaptic contacts per connected pair (cortical pyramidal pairs: ~1-8, typically ~4-5)
T = GRID * GRID


@dataclass
class Group:
    key: str          # "A.L23", "CA1.pyr", ...
    ctype: object
    per_tile: int
    structure: str    # cells in the same structure connect "locally"
    kwta: float = 0.0 # if set: only this top fraction may fire (stands in for very strong, fast
                      # feedback inhibition that a rate model can't resolve, e.g. dentate gyrus ~2%)
    every: int = 1    # clock: update every N steps (slow regions: hypothalamus, hormones, mood)


@dataclass
class Synaptic:
    id: str           # pathway ID from connections.md, or "local"
    src: str | list   # group key(s)
    dst: str
    comp: str         # "basal" | "apical"
    frac: float       # fraction of the destination compartment's slots
    gain: float = 1.0
    spread: float = 0.12
    kind: str = "topo"   # "topo" | "local"
    ach_suppress: float = 0.0   # acetylcholine turns this pathway down by this much per unit ACh (Hasselmo:
                                # while learning, recurrent/associational input is muted so new input writes)


@dataclass
class Pooled:
    id: str
    src: str | list | None   # group key(s); None = a global signal named in `signal`
    dst: str | list
    weight: float
    channel: str = "drive"   # for excitatory targets: pv | sst | dend | ais | volume | inh | exc
    sigma: float = 1.0       # kernel width in tiles; 0 = same tile only; None = whole structure
    signal: str = ""         # global signal name (e.g. "ach") when src is None


@dataclass
class Spec:
    groups: list
    synaptic: list
    pooled: list
    inputs: list = field(default_factory=list)   # group keys whose rates are set from outside


def _kernel(sigma):
    d = np.arange(T)
    dist2 = (d[:, None] % GRID - d[None] % GRID) ** 2 + (d[:, None] // GRID - d[None] // GRID) ** 2
    if sigma is None:
        K = np.ones((T, T))
    elif sigma == 0:
        K = np.eye(T)
    else:
        K = np.exp(-dist2 / (2.0 * sigma ** 2))
    return (K / K.sum(1, keepdims=True)).astype(np.float32)


def _rowwise_search(cdf, u):
    """searchsorted for each row of cdf with that row's u, in one call (works on CPU and GPU)."""
    xp = xpof(cdf)
    n, m = cdf.shape
    off = xp.arange(n, dtype=xp.float64)[:, None]
    total = cdf[:, -1:].astype(xp.float64)
    idx = xp.searchsorted((cdf / total + off).ravel(), (u / total + off).ravel()).reshape(u.shape)
    return xp.minimum(idx - (off * m).astype(idx.dtype), m - 1)


class Brain:
    def __init__(self, spec: Spec, seed=0, lean=None, suite=None, scale=None):
        """lean / suite: see cells.ExcitatoryGroup (lean synapse storage; biological sizes per suite).
        scale: cap each cell's synapses at CONTACTS x the distinct partners within its reach, so synapse
        counts scale with the size of the simulated brain (default: env SCALE=1)."""
        # Defaults (notes/24-25): the Biological Synapse Count Scale with the partner cap. SUITE=none gives the
        # reference layout (float unless LEAN=1), for comparisons with earlier results.
        suite = suite or os.environ.get("SUITE", "biological")
        suite = None if suite == "none" else suite
        scale = os.environ.get("SCALE", "1") == "1" if scale is None else scale
        self.spec, self.rng = spec, np.random.default_rng(seed)
        rng = self.rng
        self.g, self.index, self.pos, self.tile = {}, {}, {}, {}
        off = 0
        for gr in spec.groups:
            tiles = np.repeat(np.arange(T), gr.per_tile)
            self.pos[gr.key] = (np.stack([tiles % GRID, tiles // GRID], 1) + rng.random((tiles.size, 2))) / GRID
            self.tile[gr.key] = tiles
            self.index[gr.key] = np.arange(off, off + tiles.size)
            off += tiles.size
        caps = self._partner_caps() if scale else {}
        self.n_neurons = sum(self.tile[gr.key].size for gr in spec.groups if gr.key not in spec.inputs)
        for gr in spec.groups:
            if gr.key not in spec.inputs:
                c = caps.get(gr.key)
                self.g[gr.key] = make_group(gr.ctype, self.tile[gr.key].size, rng, lean=lean, suite=suite,
                                            caps=None if c is None else (c.get("basal", 1e9), c.get("apical", 1e9)),
                                            brain_neurons=self.n_neurons)
        self.structure = {gr.key: gr.structure for gr in spec.groups}
        self.kwta = {gr.key: gr.kwta for gr in spec.groups}
        self.every = {gr.key: gr.every for gr in spec.groups}
        self.clock = 0
        # Level of detail: a region whose incoming activity drops below lod_frac of its usual level
        # goes dormant (not updated, rates held) until input returns. 0 = off.
        self.lod_frac, self.lod_patience = 0.0, 3
        self.dormant, self._quiet, self._in_ref = set(), {}, {}
        self.lod_skipped = self.lod_total = 0
        self._log = []
        self._vbar = {}
        self.adapt_rate = float(os.environ.get("KWTA_ADAPT", "0.02"))  # competition adapts to each cell's usual drive
        self.n_total = off
        self.r = np.zeros(off, np.float32)
        self.kernels = {}
        self.ablated = set()
        self.homeostasis = 0.02      # always-on intrinsic plasticity (0 = off)
        self._stats = {}
        self._wire()
        self.device = "cpu"
        if backend.DEVICE == "gpu":
            self.to("gpu")

    # ---------- device and saving ----------

    def to(self, device):
        """Move the whole brain to "gpu" or "cpu" (positions stay on the host for the tasks; a device copy
        is kept for night-time rewiring)."""
        backend.move(self, device, skip=("pos",))
        self._posx = {k: backend.to_device(v, device) for k, v in self.pos.items()}
        for grp in self.g.values():
            backend.move(grp, device)
            if isinstance(grp, ExcitatoryGroup):
                for b in grp.banks():
                    backend.move(b, device, skip=("M",))
        self.device = device
        return self

    def save(self, path):
        """Save the whole brain (host copy, compressed). Load on either device with Brain.load."""
        import pickle
        import zlib
        device = self.device
        self.to("cpu")
        try:
            with open(path, "wb") as f:
                f.write(zlib.compress(pickle.dumps(self, protocol=5), 1))
        finally:
            self.to(device)

    @staticmethod
    def load(path, device=None):
        import pickle
        import zlib
        with open(path, "rb") as f:
            b = pickle.loads(zlib.decompress(f.read()))
        return b.to(device or backend.DEVICE)

    # ---------- wiring ----------

    def _partner_caps(self, sample=64):
        """Per destination compartment: CONTACTS x the effective number of distinct partners within reach,
        summed over its pathways. Effective partners of one cell = (sum w)^2 / sum w^2 of its sampling
        weights (the participation ratio: all reachable cells if equal, fewer if the kernel is narrow)."""
        caps = {}
        for p in self.spec.synaptic:
            recv = self.pos[p.dst][:: max(1, len(self.pos[p.dst]) // sample)]
            for s_ in (p.src if isinstance(p.src, list) else [p.src]):
                d = np.linalg.norm(recv[:, None, :] - self.pos[s_][None, :, :], axis=2)
                w = np.exp(-d / p.spread) if p.kind == "local" else np.exp(-d ** 2 / (2 * p.spread ** 2))
                pr = float(np.mean(w.sum(1) ** 2 / (w ** 2).sum(1)))
                comp = caps.setdefault(p.dst, {})
                comp[p.comp] = comp.get(p.comp, 0.0) + CONTACTS * pr
        return caps

    def _sample(self, recv, src, k, spread, kind, chunk=512):
        out = []
        P = self.pos[recv]
        for i in range(0, len(P), chunk):  # in chunks: at fly scale a full distance matrix is gigabytes
            d = np.linalg.norm(P[i:i + chunk, None, :] - self.pos[src][None, :, :], axis=2)
            w = np.exp(-d / spread) if kind == "local" else np.exp(-d ** 2 / (2 * spread ** 2))
            cdf = np.cumsum(w, axis=1)
            u = self.rng.random((w.shape[0], k)) * cdf[:, -1:]
            out.append(_rowwise_search(cdf, u))
        idx = np.concatenate(out)
        return self.index[src][np.minimum(idx, len(self.pos[src]) - 1)]

    def _wire(self):
        by_dst = {}
        for p in self.spec.synaptic:
            by_dst.setdefault((p.dst, p.comp), []).append(p)
        self.long_fraction = {}
        for (dst, comp), paths in by_dst.items():
            grp = self.g[dst]
            bank = grp.basal if comp == "basal" else grp.apical
            if bank is None:
                raise ValueError(f"{dst} has no {comp} dendrites")
            n, slots = bank.src.shape[0], bank.src.shape[1] * bank.src.shape[2]
            total = sum(p.frac for p in paths)
            counts = [int(round(p.frac / total * slots)) for p in paths]
            counts[int(np.argmax([p.frac for p in paths]))] += slots - sum(counts)
            cols, longs, gains, pids, sups = [], [], [], [], []
            bank.paths = []
            for p, k in zip(paths, counts):
                if k <= 0:
                    continue
                srcs = p.src if isinstance(p.src, list) else [p.src]
                share = np.array_split(np.arange(k), len(srcs))
                for s, part in zip(srcs, share):
                    if part.size == 0:
                        continue
                    cols.append(self._sample(dst, s, part.size, p.spread, p.kind))
                    is_long = self.structure[s] != self.structure[dst]
                    longs.append(np.full((n, part.size), is_long))
                    gains.append(np.full((n, part.size), p.gain, np.float32))
                    pids.append(np.full((n, part.size), len(bank.paths), np.uint8))
                    sups.append(np.full((n, part.size), p.ach_suppress, np.float32))
                    bank.paths.append((s, p.spread, p.kind))
            src_all, long_all, gain_all, pid_all = (np.concatenate(x, 1) for x in (cols, longs, gains, pids))
            perm = self.rng.permuted(np.tile(np.arange(slots), (n, 1)), axis=1)
            take = lambda a: np.take_along_axis(a, perm, 1).reshape(bank.src.shape)
            bank.src[:], bank.long[:], bank.gain[:] = take(src_all), take(long_all), take(gain_all)
            if hasattr(bank, "path"):
                bank.path[:] = take(pid_all)
            sup_all = np.concatenate(sups, 1)
            bank.ach_suppress = take(sup_all) if sup_all.any() else None
        for key, grp in self.g.items():
            if isinstance(grp, ExcitatoryGroup):
                lf = np.mean(np.concatenate([b.long.ravel() for b in grp.banks()]))
                self.long_fraction[key] = float(lf)
                if lf > grp.c.long_range_max + 0.02:
                    print(f"warning: {key} long-range fraction {lf:.2f} > cap {grp.c.long_range_max}")

    # ---------- running ----------

    def _pool(self, key):
        rate = self.r[self.index[key]]
        return (np.bincount(self.tile[key], weights=rate, minlength=T) / np.maximum(
            np.bincount(self.tile[key], minlength=T), 1)).astype(np.float32)

    def _K(self, sigma):
        if sigma not in self.kernels:
            self.kernels[sigma] = like(self.r, _kernel(sigma))
        return self.kernels[sigma]

    def step(self, inputs: dict, learn=False, ach=0.5, dopamine=0.0, arousal=0.0, signals=None):
        """inputs: {group key: rates} for input groups. signals: extra global scalars by name."""
        sig = {"ach": ach, "dopamine": dopamine, "arousal": arousal, **(signals or {})}
        for key, x in inputs.items():
            self.r[self.index[key]] = like(self.r, x)
        if learn and dopamine:  # day log of moments that mattered, for value-guided rewiring at night
            self._log.append((float(dopamine), self.r.astype(np.float16)))
            if len(self._log) > LOG_MAX:
                self._log.pop(int(self.rng.integers(len(self._log))))
        self.clock += 1
        if self.lod_frac:
            self._level_of_detail()
        # Phase 1: inhibitory and other pooled cells respond to the previous step's activity.
        # Phase 2: excitatory cells respond, feeling this step's inhibition (as in notes/19's circuit;
        # a one-step lag in feedback inhibition destabilizes the network).
        new = {}
        for phase in (1, 2):
            chan = self._channels(sig, excitatory=(phase == 2))
            for key, grp in self.g.items():
                if isinstance(grp, ExcitatoryGroup) != (phase == 2):
                    continue
                self.lod_total += 1
                if self.clock % self.every[key] or key.split(".")[0] in self.dormant:
                    self.lod_skipped += 1
                    continue
                new[key] = self._update(key, grp, chan[key], phase, learn, ach, dopamine, arousal)
                if phase == 1:
                    self.r[self.index[key]] = new[key]
        for key, rate in new.items():
            self.r[self.index[key]] = rate

    def _channels(self, sig, excitatory):
        chan = {key: {} for key in self.g}
        for p in self.spec.pooled:
            if p.id in self.ablated:
                continue
            dsts = [d for d in (p.dst if isinstance(p.dst, list) else [p.dst])
                    if d in self.g and isinstance(self.g[d], ExcitatoryGroup) == excitatory]
            if not dsts:
                continue
            if p.src is None:
                field_ = xpof(self.r).full(T, sig.get(p.signal, 0.0), np.float32)
            else:
                srcs = p.src if isinstance(p.src, list) else [p.src]
                field_ = self._K(p.sigma) @ xpof(self.r).stack([self._pool(s) for s in srcs]).mean(axis=0)
            for dst in dsts:
                c = chan[dst].setdefault(p.channel, xpof(self.r).zeros(self.g[dst].n, np.float32))
                c += p.weight * field_[self.tile[dst]]
        return chan

    def _update(self, key, grp, ch, phase, learn, ach, dopamine, arousal):
        if key.split(".")[0] in self.ablated or key in self.ablated:
            grp.rate[:] = 0.0
            return grp.rate
        n = self.every[key]
        if phase == 1:
            if isinstance(grp, InhibitoryGroup):
                grp.step(ch.get("drive", xpof(self.r).zeros(grp.n, np.float32)), steps=n)
            elif isinstance(grp, ModulatorGroup):
                grp.step(float(np.mean(ch.get("drive", 0.0))), steps=n)
            else:  # olive / pacemaker groups take a single drive
                grp.step(float(np.mean(ch.get("drive", 0.0))))
            return grp.rate
        for b in grp.banks():
            if b.ach_suppress is not None:
                b.transmit = 1.0 - b.ach_suppress * ach
        target = grp.compute(self.r, pv_in=ch.get("pv", 0.0), sst_in=ch.get("sst", 0.0),
                             trn_in=ch.get("inh", 0.0), dend_in=ch.get("dend", 0.0),
                             ais_in=ch.get("ais", 0.0), volume_in=ch.get("volume", 0.0),
                             external=ch.get("exc", 0.0), apical_mode=1.0 - 0.5 * (ach - 0.5),
                             allow_burst=ach < 0.3, cf=ch.get("cf", 0.0))
        if self.kwta[key]:
            # Rank by the membrane drive, not the output: saturated outputs tie at 1.0, and a cut on the output
            # then lets every saturated cell through (CA3 ran at 27% active under a "5%" rule, notes/26)
            # The winners fire: the inhibition sets the bar at the k-th cell, so a winner fires by how far it
            # clears that bar (scale-free), even when every cell is below its own threshold (the dentate gyrus
            # went silent otherwise)
            # Cells compete on how unusual their drive is now, relative to their own usual level and swing
            # (fast adaptation and gain control). The entorhinal input is nearly the same for every pattern,
            # so ranking on the raw drive let the same cells win every time (notes/26)
            v = grp.soma - grp.threshold
            m, var = self._vbar.setdefault(key, (v.copy(), xpof(v).ones_like(v)))
            m += self.adapt_rate * (v - m)
            var += self.adapt_rate * ((v - m) ** 2 - var)
            v = (v - m) / np.sqrt(var + 1e-6)
            k = max(1, int(round(self.kwta[key] * v.size)))
            cut = xpof(v).partition(v, v.size - k)[v.size - k]
            over = 1.0 / (1.0 + np.exp(-(4.0 * (v - cut) / (v.std() + 1e-6) + 1.0)))
            target = np.where(v >= cut, np.maximum(target, over), 0.05 * target)
        if "pv" in ch:
            grp.inhibitory_plasticity(ch["pv"], eta=0.05)
        grp.update(target, learn=learn)
        if self.homeostasis:
            self._homeostasis(key, grp, self.homeostasis)
        if learn:
            grp.modulate(dopamine, ach, arousal)
        grp.tick(n)
        return grp.rate

    def _level_of_detail(self):
        """Per region: mean activity arriving from other regions vs its long-run average."""
        if not hasattr(self, "_feeds"):
            self._feeds = {}
            for p in list(self.spec.synaptic) + [q for q in self.spec.pooled if q.src is not None]:
                for d in (p.dst if isinstance(p.dst, list) else [p.dst]):
                    for s_ in (p.src if isinstance(p.src, list) else [p.src]):
                        if s_.split(".")[0] != d.split(".")[0]:
                            self._feeds.setdefault(d.split(".")[0], set()).add(s_)
        for region, srcs in self._feeds.items():
            now = float(np.mean([float(self.r[self.index[s_]].mean()) for s_ in srcs]))
            ref = self._in_ref.get(region, now)
            self._in_ref[region] = 0.999 * ref + 0.001 * now
            if self.lod_frac and now < self.lod_frac * ref:
                self._quiet[region] = self._quiet.get(region, 0) + 1
                if self._quiet[region] >= self.lod_patience and region not in self.dormant:
                    self.dormant.add(region)  # going dormant = falling silent, not freezing mid-thought
                    for key in self.g:
                        if key.split(".")[0] == region:
                            self.g[key].rate[:] = 0.0
                            self.r[self.index[key]] = 0.0
            else:
                self._quiet[region] = 0
                self.dormant.discard(region)

    def _homeostasis(self, key, g, eta, slope=6.0):
        """Intrinsic plasticity: each cell keeps adjusting its threshold and sharpness so it fires about
        its target fraction of the time, whatever learning does to its inputs."""
        mean, var = self._stats.get(key, (g.soma.copy(), np.ones_like(g.soma)))
        mean = 0.98 * mean + 0.02 * g.soma
        var = 0.98 * var + 0.02 * (g.soma - mean) ** 2
        self._stats[key] = (mean, var)
        std = np.sqrt(var) + 1e-3
        g.scale = (slope / g.c.slope / std).astype(np.float32)
        g.threshold += (eta * (std + 0.05) * ((g.soma > g.threshold) - g.c.target_rate)).astype(np.float32)

    def sleep(self):
        for key, grp in self.g.items():
            if isinstance(grp, ExcitatoryGroup):
                grp.sleep()
                for bank in grp.banks():
                    if getattr(bank, "to_rewire", None) is not None:
                        self._seeded_choice(key, bank)
        self._log = []

    def _seeded_choice(self, key, bank, chunk=256):
        """Value-guided seeded choice (notes/24): each rewiring slot is offered CHOICES candidates by the seed,
        drawn from its own pathway's source group with the wiring's distance rule. It takes the one that fired
        together with its branch's strongest input when it mattered today (signed by dopamine)."""
        cells, br, sl = bank.to_rewire
        xp = xpof(bank.src)
        pos = getattr(self, "_posx", self.pos)
        c0 = bank.cascade[0]
        anchor = bank.src[cells, br, np.argmax(c0[cells, br], axis=1)]
        if self._log:
            da = like(bank.src, [m[0] for m in self._log])
            R = xp.stack([m[1] for m in self._log]).astype(np.float32)          # (moments, all cells)
        new_src = bank.src[cells, br, sl].copy()
        choice = xp.zeros(len(cells), np.uint8)
        pid = bank.path[cells, br, sl]
        for p in [int(v) for v in to_host(np.unique(pid))]:
            src, spread, kind = bank.paths[p]
            sel = np.flatnonzero(pid == p)
            for i in range(0, len(sel), chunk):
                rows = sel[i:i + chunk]
                d = np.linalg.norm(pos[key][cells[rows], None, :] - pos[src][None, :, :], axis=2)
                w = np.exp(-d / spread) if kind == "local" else np.exp(-d ** 2 / (2 * spread ** 2))
                cdf = np.cumsum(w, axis=1)
                k = xp.arange(CHOICES, dtype=np.uint64)[None, :]
                gen = (bank.gen[cells[rows], br[rows], sl[rows]].astype(np.uint64) + np.uint64(1))[:, None]
                u = _hash_uniform(bank.ids[cells[rows], 0, 0][:, None], bank.slot[0][br[rows], sl[rows]][:, None],
                                  gen * np.uint64(256) + k) * cdf[:, -1:]
                cand = self.index[src][_rowwise_search(cdf, u)]                          # (rows, CHOICES)
                if self._log:
                    a = da[:, None] * R[:, anchor[rows]]                                  # (moments, rows)
                    X = R[:, cand.ravel()].reshape(len(da), len(rows), CHOICES)
                    score = np.einsum("mr,mrc->rc", a, X) / (X.sum(axis=0) + 1.0)
                    best = np.argmax(score, axis=1)
                else:
                    best = xp.zeros(len(rows), np.int64)  # nothing mattered today: the seed's first offer
                new_src[rows] = cand[np.arange(len(rows)), best]
                choice[rows] = best
        bank.rewire((cells, br, sl), new_src, choice)

    def storage(self):
        """Synapses, and bytes of learnable synapse state: as float arrays now, packed (32 bits each),
        and lean long-term storage (only for lean banks)."""
        syn = lean_bits = 0
        for g in self.g.values():
            if isinstance(g, ExcitatoryGroup):
                for b in g.banks():
                    syn += b.src.size
                    lean_bits += b.storage_bits() if hasattr(b, "storage_bits") else 0
        return {"synapses": syn, "float MB": syn * 37 / 1e6, "packed MB": syn * 4 / 1e6,
                "lean MB": lean_bits / 8 / 1e6}

    def excitatory(self):
        return {k: g for k, g in self.g.items() if isinstance(g, ExcitatoryGroup)}

    def calibrate(self, make_input, steps=900, slope=6.0, hold=6, blank=3, eta=0.2):
        """Development, part 1: each cell tunes its threshold and sharpness closed-loop (fast homeostasis),
        as a warm start for the always-on homeostasis in step()."""
        live, self.homeostasis = self.homeostasis, 0.0
        E = self.excitatory()
        mean, var, warm = {k: 0.0 for k in E}, {k: 1.0 for k in E}, {k: [] for k in E}
        for t in range(steps):
            if t % (hold + blank) == 0:
                x = make_input()
            self.step({k: (v if t % (hold + blank) < hold else 0 * v) for k, v in x.items()})
            if t < 45:
                for k, g in E.items():
                    warm[k].append(g.soma.copy())
                if t == 44:
                    for k, g in E.items():
                        h = xpof(g.soma).stack(warm[k])
                        g.threshold = np.quantile(h, 1 - g.c.target_rate, axis=0).astype(np.float32)
                        mean[k], var[k] = h.mean(0), h.var(0)
                continue
            for k, g in E.items():
                mean[k] = 0.98 * mean[k] + 0.02 * g.soma
                var[k] = 0.98 * var[k] + 0.02 * (g.soma - mean[k]) ** 2
                std = np.sqrt(var[k]) + 1e-3
                g.scale = (slope / g.c.slope / std).astype(np.float32)
                g.threshold += (eta * (std + 0.05) * ((g.soma > g.threshold) - g.c.target_rate)).astype(np.float32)
        self._stats = {k: (mean[k], var[k]) for k in E}
        self.homeostasis = live
        self.reset()

    def develop(self, make_input, days=4, presentations=60, hold=6, blank=3):
        """Development, part 2: ordinary experience with learning on, and sleep each night."""
        t = 0
        for _ in range(days):
            for _ in range(presentations):
                x = make_input()
                for s in range(hold + blank):
                    self.step({k: (v if s < hold else 0 * v) for k, v in x.items()}, learn=True, ach=0.6,
                              dopamine=0.5 if s == 2 else 0.0, arousal=0.3)
                    t += 1
            self.sleep()

    def reset(self):
        for g in self.g.values():
            g.rate[:] = 0
            if isinstance(g, ExcitatoryGroup):
                g.adapt_state[:] = 0
        self.r[:] = 0

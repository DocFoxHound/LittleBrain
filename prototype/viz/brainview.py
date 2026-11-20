"""
LittleBrain desktop visualizer (notes/31): watch a character's brain while it plays a game.

  .venv/bin/python prototype/viz/brainview.py [--scale 1.0] [--seed 0]
  headless screenshots: --shots 5,60,200 --out shots/   (renders without a window, saves PNGs at those turns)

Left: a side view of a human brain (facing left). Cortical areas are drawn as layered ribbons at the surface
(L2/3 outermost, L6 innermost), deep nuclei (thalamus, striatum, pallidum) inside, the eye in front. Every dot
is a small group of neighbouring neurons of one type, placed where those neurons sit on the brain's map, and
glows with their current firing. A bigger brain gets more and larger dots up to a cap (level of detail), so it
never turns into thousands of specks; the legend says how many neurons one dot stands for.
Right: what the brain sees, its action channels (how much each action is being released, and which it chose),
dopamine and reward. The viewer knows nothing about any game: it watches a BrainAgent that is plugged into any
games.Game (reset/step). Use it from code:  watch(agent, game)  or from here with --game.

Keys: Space pause | Right/Left faster/slower | L learning on/off | S sleep | [ / ] smaller/bigger brain |
      T labels | P pathways | F12 screenshot | Esc quit.  Hover a dot for what it is.
"""

import argparse
import copy
import math
import os
import sys
import threading
import time
from pathlib import Path

for k, v in {"HOMEO_RULE": "log", "L4_L23_GAIN": "3", "CORTEX_LRF": "0.05", "HOMEO_SLOW": "STR.SO:0.1"}.items():
    os.environ.setdefault(k, v)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brain"))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

np.seterr(over="ignore")

# ------------------------------------------------------------------ anatomy (normalized brain-panel coordinates)

OUTLINE = [(0.10, 0.50), (0.11, 0.37), (0.17, 0.25), (0.28, 0.16), (0.42, 0.11), (0.57, 0.11), (0.70, 0.16),
           (0.80, 0.24), (0.87, 0.34), (0.90, 0.46), (0.87, 0.56), (0.79, 0.61), (0.67, 0.62), (0.60, 0.68),
           (0.49, 0.73), (0.37, 0.73), (0.27, 0.68), (0.21, 0.61), (0.14, 0.58)]
EYE = ((0.05, 0.50), 0.042)


def ell(cx, cy, rx, ry, n=48, a0=0.0, a1=2 * math.pi):
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(a0, a1, n)]


def crescent(cx, cy, rx, ry, r_in, r_out, a0, a1, n=40):
    A = np.linspace(a0, a1, n)
    outer = [(cx + rx * r_out * math.cos(a), cy + ry * r_out * math.sin(a)) for a in A]
    inner = [(cx + rx * r_in * math.cos(a), cy + ry * r_in * math.sin(a)) for a in A[::-1]]
    return outer + inner


TH_SHAPE = (0.565, 0.475, 0.065, 0.042)
# Where each simulated region lives: territory polygons (cortex is clipped to the cerebrum's outline), seen from
# the side. Cortex is a sheet covering the surface, so a cortical area's neurons fill its lobe face-on.
TERRITORY = {
    "A": ("cortex", [[(0.785, 0.08), (1.0, 0.08), (1.0, 0.75), (0.745, 0.64), (0.762, 0.40)]],
          "visual cortex (V1, occipital lobe)"),
    "B": ("cortex", [[(0.515, 0.05), (0.80, 0.05), (0.775, 0.36), (0.70, 0.40), (0.60, 0.415), (0.505, 0.375),
                      (0.488, 0.24)]], "parietal association cortex"),
    "MOT": ("cortex", [[(0.43, 0.05), (0.512, 0.05), (0.487, 0.24), (0.463, 0.39), (0.405, 0.42), (0.428, 0.24)]],
            "motor cortex (copy of the chosen action)"),
    "EC": ("deep", [ell(0.44, 0.665, 0.036, 0.018)], "entorhinal cortex"),
    "TH": ("deep", [ell(*TH_SHAPE)], "thalamus"),
    "TH.TRN": ("deep", [crescent(*TH_SHAPE, 1.04, 1.22, math.radians(115), math.radians(300))], "reticular nucleus"),
    "BLA": ("deep", [ell(0.398, 0.614, 0.024, 0.019)], "basolateral amygdala"),
    "CeA": ("deep", [ell(0.437, 0.604, 0.020, 0.016)], "central amygdala"),
    "VTA": ("deep", [ell(0.588, 0.645, 0.022, 0.013)], "dopamine cells (VTA / SNc)"),
    "STR": ("deep", [ell(0.365, 0.385, 0.05, 0.04),                                             # caudate head
                     [(0.39, 0.352), (0.47, 0.338), (0.55, 0.352), (0.605, 0.378), (0.598, 0.398), (0.545, 0.377),
                      (0.47, 0.367), (0.40, 0.388)],                                             # caudate body
                     ell(0.418, 0.468, 0.056, 0.034)], "striatum (caudate + putamen)"),          # putamen
    "GPe": ("deep", [ell(0.472, 0.518, 0.030, 0.019)], "globus pallidus ext."),
    "GPi": ("deep", [ell(0.508, 0.540, 0.022, 0.015)], "globus pallidus int."),
    "DG": ("deep", [ell(0.50, 0.648, 0.025, 0.014)], "dentate gyrus"),
    "CA3": ("deep", [ell(0.535, 0.637, 0.022, 0.014)], "CA3"),
    "CA1": ("deep", [ell(0.567, 0.622, 0.022, 0.014)], "CA1"),
    "SUB": ("deep", [ell(0.597, 0.610, 0.020, 0.013)], "subiculum"),
    "retina": ("eye", [crescent(EYE[0][0], EYE[0][1], EYE[1], EYE[1], 0.74, 0.94, math.radians(-70),
                                math.radians(70))], "retina (back of the eye)"),
}
# Regions designed (cell types, percentages) but not wired into this brain yet: drawn as faint outlines
NOT_BUILT = {"amygdala": [ell(0.405, 0.612, 0.026, 0.020)], "hypothalamus": [ell(0.478, 0.588, 0.030, 0.017)],
             "claustrum": [[(0.372, 0.43), (0.378, 0.43), (0.385, 0.51), (0.379, 0.51)]]}
LOBE_LABELS = [((0.155, 0.33), "frontal lobe · not simulated"), ((0.25, 0.68), "temporal lobe"),
               ((0.66, 0.70), "cerebellum · not built yet"), ((0.55, 0.88), "brainstem · noradrenaline, serotonin: not built yet")]

BG, INK, DIM = (14, 16, 22), (225, 228, 235), (120, 126, 140)
TISSUE, TISSUE_EDGE = (34, 30, 38), (92, 78, 92)
EXC_LO, EXC_HI = np.array([70, 32, 28]), np.array([255, 214, 120])
INH_LO, INH_HI = np.array([24, 40, 78]), np.array([110, 210, 255])
INP_LO, INP_HI = np.array([60, 60, 66]), np.array([245, 245, 245])
GOOD, BAD, ACCENT = (110, 220, 140), (240, 110, 110), (255, 200, 90)


def catmull(points, n=24):
    P = np.array(points + points[:3])
    out = []
    for i in range(len(points)):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.roll(np.array(out), 24, axis=0)


def inside(poly, pts):
    """Ray casting, vectorized: which of pts (N,2) are inside the polygon."""
    P = np.asarray(poly)
    x, y = pts[:, 0], pts[:, 1]
    res = np.zeros(len(pts), bool)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi, xj, yj = P[i, 0], P[i, 1], P[j, 0], P[j, 1]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        res ^= cross
        j = i
    return res


class Territory:
    """A region's area, filled evenly: sample points uniformly inside it, then give each point coordinates (qx, qy)
    in [0,1]^2 by rank (left-right, then top-bottom within its column). A neuron at map position (u, v) goes to the
    free point nearest (u, v): an even, order-keeping stretch of the brain's square map over the real shape."""

    def __init__(self, polys, clip=None, n=6000, seed=0):
        rng = np.random.default_rng(seed)
        allp = np.concatenate([np.asarray(p) for p in polys])
        lo, hi = allp.min(0), allp.max(0)
        pts = np.zeros((0, 2))
        while len(pts) < n:
            c = lo + rng.random((n * 3, 2)) * (hi - lo)
            ok = np.zeros(len(c), bool)
            for p in polys:
                ok |= inside(p, c)
            if clip is not None:
                ok &= inside(clip, c)
            pts = np.concatenate([pts, c[ok]])
        self.pts = pts[:n]
        q = np.zeros_like(self.pts)
        order = np.argsort(self.pts[:, 0])
        q[order, 0] = (np.arange(n) + 0.5) / n
        bins = np.minimum((q[:, 0] * 24).astype(int), 23)
        for k in range(24):
            idx = np.flatnonzero(bins == k)
            o = idx[np.argsort(self.pts[idx, 1])]
            q[o, 1] = (np.arange(len(o)) + 0.5) / max(len(o), 1)
        self.q = q
        self.free = np.ones(n, bool)
        self.polys, self.clip = polys, clip

    def take(self, uv):
        d = np.sum((self.q - uv) ** 2, 1)
        d[~self.free] = np.inf
        i = int(np.argmin(d))
        self.free[i] = False
        return self.pts[i]


class Anatomy:
    def __init__(self):
        self.outline = catmull(OUTLINE)


# ------------------------------------------------------------------ dots: level of detail per cell group

class GroupDots:
    """One cell group drawn as up to `cap` dots; each dot = a contiguous run of neurons (neurons are stored tile by
    tile, so a run is a small patch of the map)."""

    def __init__(self, key, idx, pos, kind, ctype):
        n = len(idx)
        self.key, self.kind, self.ctype, self.n = key, kind, ctype, n
        dots = int(min(n, max(4, 9 * math.sqrt(n)), 200))
        bounds = np.linspace(0, n, dots + 1).astype(int)
        self.starts = idx[0] + bounds[:-1]
        self.sizes = np.diff(bounds)
        self.mpos = np.array([pos[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
        self.per_dot = n / dots
        self.radius = 1.6 + 0.4 * math.log2(1 + self.per_dot)
        self.xy = None
        self.norm = 0.05

    def rates(self, r):
        sums = np.add.reduceat(r, self.starts) if len(self.starts) else np.zeros(0)
        last = self.starts[-1]
        sums[-1] = r[last:last + self.sizes[-1]].sum()
        return sums / self.sizes


def kind_of(b, key):
    from cells import ExcitatoryGroup, ModulatorGroup
    if key in b.spec.inputs:
        return "input"
    g = b.g[key]
    if isinstance(g, ExcitatoryGroup):
        return "exc" if g.c.sign > 0 else "inh"   # MSNs are 'excitatory-style' cells that release GABA
    if isinstance(g, ModulatorGroup):
        return "mod"
    return "inh"


def describe(b, key):
    ct = next((gr.ctype for gr in b.spec.groups if gr.key == key), None)
    if ct is None:
        return key, {"retina": "light-sensitive input (the picture)", "MOT": "copy of the chosen action",
                     "US": "reward arriving from the body (lateral hypothalamus / brainstem)"}.get(key, "input")
    return ct.name, ct.human


def territory_of(key):
    if key == "CeM" or key.startswith("CeL."):
        return "CeA"
    return key if key in TERRITORY else key.partition(".")[0]


class Layout:
    def __init__(self, b, rect):
        self.b, self.rect = b, rect
        self.anat = Anatomy()
        rng = np.random.default_rng(0)
        self.groups = []
        for gr in b.spec.groups:
            self.groups.append(GroupDots(gr.key, b.index[gr.key], b.pos[gr.key], kind_of(b, gr.key), gr.ctype))
        self.terr = {}
        for g in self.groups:
            tk = territory_of(g.key)
            if tk not in TERRITORY:
                continue
            if tk not in self.terr:
                kind, polys, _ = TERRITORY[tk]
                n = 3 * sum(x.n and len(x.starts) for x in self.groups if territory_of(x.key) == tk) + 400
                self.terr[tk] = Territory(polys, clip=self.anat.outline if kind == "cortex" else None, n=n)
        # place dots: random order across groups so no group gets the leftovers; jitter within its map tile
        jobs = [(g, i) for g in self.groups if territory_of(g.key) in self.terr for i in range(len(g.mpos))]
        for g in self.groups:
            g.npts = np.zeros((len(g.mpos), 2))
        for k in rng.permutation(len(jobs)):
            g, i = jobs[k]
            uv = np.clip(g.mpos[i] + rng.uniform(-0.5, 0.5, 2) / 8 * (g.per_dot ** 0 * 0.9), 0, 1)
            if g.key == "TH.TRN":
                uv = uv[::-1]   # the shell runs bottom-front to top: spread along it
            g.npts[i] = self.terr[territory_of(g.key)].take(uv)
        self.groups = [g for g in self.groups if territory_of(g.key) in self.terr]
        self.edges = self._edges()
        self.place()

    def _edges(self):
        e = {}
        for p in list(self.b.spec.synaptic) + [q for q in self.b.spec.pooled if q.src is not None]:
            for s in (p.src if isinstance(p.src, list) else [p.src]):
                for d in (p.dst if isinstance(p.dst, list) else [p.dst]):
                    rs, rd = s.partition(".")[0], d.partition(".")[0]
                    if rs != rd:
                        e.setdefault((rs, rd), s)
        return e

    def to_screen(self, p):
        x0, y0, w, h = self.rect   # one scale for both axes: true proportions
        return (x0 + p[0] * w, y0 + p[1] * w)

    def place(self):
        for g in self.groups:
            g.xy = np.array([self.to_screen(p) for p in g.npts])
        self.center = {}
        for g in self.groups:
            self.center.setdefault(g.key.partition(".")[0], []).append(g.xy.mean(0))
        self.center = {k: np.mean(v, 0) for k, v in self.center.items()}


# ------------------------------------------------------------------ the app

class App:
    def __init__(self, make_game, scale=1.0, seed=0, headless=False, shots=(), out="shots", turns=None, brain=None):
        """make_game(): returns any games.Game. brain: an already prepared Brain to watch (else one is grown)."""
        import pygame
        self.pg = pygame
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.W, self.H = 1600, 900
        self.screen = pygame.display.set_mode((self.W, self.H)) if not headless else pygame.Surface((self.W, self.H))
        if not headless:
            pygame.display.set_caption("LittleBrain")
        self.font = pygame.font.SysFont("dejavusans,liberationsans,arial", 15)
        self.small = pygame.font.SysFont("dejavusans,liberationsans,arial", 12)
        self.big = pygame.font.SysFont("dejavusans,liberationsans,arial", 22, bold=True)
        self.headless, self.shots, self.out, self.max_turns = headless, sorted(shots), Path(out), turns
        self.save_root = os.environ.get("SLOP_SAVE_DIR")
        if self.save_root:
            self.out = Path(self.save_root) / "shots"
        self.scale, self.seed, self.make_game, self.given = scale, seed, make_game, brain
        self.paused, self.speed, self.labels, self.paths = False, 3, True, True
        self.status = ""
        self.brain = self.agent = self.layout = None
        self.loading = None
        self.rewards, self.dopa = [], []
        self.phase, self.info, self.seen = "look", {}, None
        self.quit = False
        self.clock = pygame.time.Clock()
        self.frame = 0
        self.hover = None

    # ---------- brain lifecycle ----------

    def build(self, scale):
        """Prepare a brain of this size (cached on disk), in a background thread."""
        def work():
            try:
                build_brain()
            except Exception:
                import traceback
                self.loading = ("error", traceback.format_exc(), time.time())

        def build_brain():
            from federation import Brain
            from run_signal_game import prepare
            cache = HERE / ".cache" / f"{os.environ.get('BRAIN', 'bg')}_scale{scale:g}_seed{self.seed}.brain"
            t0 = time.time()
            if self.save_root or getattr(sys, "frozen", False):
                # A distributed viewer develops its brain locally without loading pickle files.
                b = prepare(self.seed, scale=scale)
            elif cache.exists():
                b = Brain.load(cache)
            else:
                b = prepare(self.seed, scale=scale)
                cache.parent.mkdir(exist_ok=True)
                b.save(cache)
            self.loading = ("done", b, time.time() - t0)
        self.loading = ("busy", scale, time.time())
        if self.headless:
            work()          # no window to keep responsive: build right here
        else:
            threading.Thread(target=work, daemon=True).start()

    def adopt(self, b):
        from agent import BrainAgent
        self.brain = b
        self.game = self.make_game()
        self.game_name = type(self.game).__name__
        n_channels = len(set(np.asarray(__import__("brainspec").action_channel(b.tile["GPi"], self.game.n_actions)).tolist()))
        self.agent = BrainAgent(b, self.game.n_actions, seed=self.seed)
        assert n_channels == self.game.n_actions, "the brain's map has fewer stripes than the game has actions"
        self.agent.on_step = self.on_step
        self.layout = Layout(b, (40, -36, 920, 920))
        self._mask = None
        self.rewards, self.dopa = [], []
        syn = b.storage()["synapses"]
        self.size_text = f"{b.n_neurons:,} neurons · {syn / 1e6:.1f} M synapses · scale {self.scale:g}"

    # ---------- per-step hook ----------

    def on_step(self, phase, info):
        self.phase, self.info = phase, info
        if "image" in info:
            self.seen = info["image"]
        self.dopa.append(info.get("dopamine", 0.0))
        self.dopa = self.dopa[-400:]
        self.frame += 1
        draw_now = self.headless and self._want_shot() or (not self.headless and self.frame % max(1, self.speed) == 0)
        if self.headless:
            if draw_now:
                self.draw()
                self.save_shot()
            return
        while True:
            self.events()
            if self.quit or self.loading:
                raise StopIteration
            if draw_now or self.paused:
                self.draw()
                self.pg.display.flip()
                self.clock.tick(60 if self.paused else 120)
            if not self.paused:
                break

    def _want_shot(self):
        t = self.agent.turn + 1
        return self.shots and t >= self.shots[0] and self.phase == "act" and self.info.get("step") == 1

    def save_shot(self):
        self.out.mkdir(parents=True, exist_ok=True)
        t = self.shots.pop(0)
        path = self.out / f"littlebrain_scale{self.scale:g}_turn{t}.png"
        self.pg.image.save(self.screen, str(path))
        print("saved", path, flush=True)

    # ---------- input ----------

    def events(self):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                self.quit = True
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pg.K_RIGHT:
                    self.speed = min(self.speed * 2, 64)
                elif e.key == pg.K_LEFT:
                    self.speed = max(self.speed // 2, 1)
                elif e.key == pg.K_l and self.agent:
                    self.agent.learn = not self.agent.learn
                elif e.key == pg.K_s and self.brain:
                    self.brain.sleep()
                    self.status = "slept: the day's learning was consolidated"
                elif e.key == pg.K_t:
                    self.labels = not self.labels
                elif e.key == pg.K_p:
                    self.paths = not self.paths
                elif e.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    steps = [0.25, 0.5, 1.0, 2.0, 4.0]
                    i = steps.index(self.scale) if self.scale in steps else 2
                    i = max(0, i - 1) if e.key == pg.K_LEFTBRACKET else min(len(steps) - 1, i + 1)
                    if steps[i] != self.scale:
                        self.scale = steps[i]
                        self.build(self.scale)
                elif e.key == pg.K_F12:
                    self.shots = [self.agent.turn + 1] if self.agent else []
                    self.draw()
                    self.save_shot()
            elif e.type == pg.MOUSEMOTION:
                self.hover = e.pos

    # ---------- drawing ----------

    def color(self, kind, a):
        lo, hi = {"exc": (EXC_LO, EXC_HI), "inh": (INH_LO, INH_HI), "mod": (INH_LO, GOOD)}.get(kind, (INP_LO, INP_HI))
        a = np.clip(a, 0, 1)[:, None]
        return (lo + (hi - lo) * a).astype(int)

    def draw_brain(self, r):
        pg, s, L = self.pg, self.screen, self.layout
        S = L.to_screen
        P = lambda poly: [S(p) for p in poly]
        out = P(L.anat.outline)
        # cerebellum and brainstem behind the cerebrum
        stem = P([(0.55, 0.62), (0.635, 0.62), (0.64, 0.74), (0.625, 0.86), (0.585, 0.86), (0.57, 0.74)])
        pg.draw.polygon(s, (30, 27, 34), stem)
        pg.draw.polygon(s, TISSUE_EDGE, stem, 1)
        cx, cy = S((0.745, 0.665))
        rx, ry = 0.125 * L.rect[2], 0.068 * L.rect[2]
        pg.draw.ellipse(s, (32, 28, 36), (cx - rx, cy - ry, 2 * rx, 2 * ry))
        for k in range(-3, 4):   # folia: fine curved grooves across the cerebellum
            yy = cy + k * ry / 4.2
            half = rx * math.sqrt(max(1 - (k / 4.2) ** 2, 0)) * 0.9
            xs = np.linspace(cx - half, cx + half, 24)
            pg.draw.lines(s, (62, 54, 66), False,
                          [(float(x), float(yy + 5 * (1 - ((x - cx) / max(half, 1)) ** 2))) for x in xs], 1)
        pg.draw.ellipse(s, TISSUE_EDGE, (cx - rx, cy - ry, 2 * rx, 2 * ry), 1)
        pg.draw.polygon(s, TISSUE, out)
        # cortical territories (the sheet seen face-on), lightly tinted
        for tk, T in L.terr.items():
            if TERRITORY[tk][0] == "cortex":
                tint = pg.Surface((self.W, self.H), pg.SRCALPHA)
                for poly in T.polys:
                    pg.draw.polygon(tint, (120, 90, 120, 22), P(poly))
                    pg.draw.polygon(tint, (170, 140, 175, 90), P(poly), 1)   # area border
                tint.blit(self._mask_outside(out), (0, 0), special_flags=pg.BLEND_RGBA_MULT)
                s.blit(tint, (0, 0))
        # sulci
        for line in ([(0.52, 0.115), (0.50, 0.24), (0.47, 0.36)], [(0.27, 0.57), (0.40, 0.52), (0.56, 0.49), (0.66, 0.44)],
                     [(0.80, 0.20), (0.77, 0.36)], [(0.20, 0.27), (0.30, 0.22), (0.40, 0.21)],
                     [(0.22, 0.43), (0.31, 0.38), (0.39, 0.37)], [(0.62, 0.20), (0.70, 0.27), (0.74, 0.36)],
                     [(0.33, 0.64), (0.45, 0.62), (0.55, 0.59)]):
            pg.draw.lines(s, (62, 52, 64), False, P(line), 2)
        pg.draw.polygon(s, TISSUE_EDGE, out, 2)
        # deep structures: darker, outlined, seen through the surface
        for tk, T in L.terr.items():
            if TERRITORY[tk][0] == "deep":
                for poly in T.polys:
                    pg.draw.polygon(s, (22, 19, 27), P(poly))
                for poly in T.polys:
                    pg.draw.polygon(s, (84, 72, 92), P(poly), 1)
        built = {"amygdala": "BLA"}
        for name, polys in NOT_BUILT.items():
            if built.get(name) in L.terr:
                continue
            for poly in polys:
                pg.draw.polygon(s, (70, 64, 78), P(poly), 1)
        # eye
        (ex, ey), er = EYE
        X, Y = S((ex, ey))
        pg.draw.circle(s, (40, 40, 48), (X, Y), er * L.rect[2])
        pg.draw.circle(s, (100, 100, 112), (X, Y), er * L.rect[2], 1)
        pg.draw.circle(s, (20, 20, 26), (X - er * L.rect[2] * 0.95, Y), er * L.rect[2] * 0.28)   # pupil / lens side
        # pathways
        if self.paths:
            for (rs, rd), src_key in L.edges.items():
                if rs not in L.center or rd not in L.center:
                    continue
                a, bpt = L.center[rs], L.center[rd]
                act = float(r[self.brain.index[src_key]].mean())
                mid = (a + bpt) / 2 + np.array([0, -40 if a[0] < bpt[0] else 40])
                pts = [tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * bpt) for t in np.linspace(0, 1, 18)]
                v = int(36 + 120 * min(act * 6, 1))
                pg.draw.lines(s, (v, v, v + 24), False, pts, 1)
        # dots
        for g in L.groups:
            a = g.rates(r)
            g.norm = max(0.97 * g.norm + 0.03 * float(np.percentile(a, 95)), 0.03)
            cols = self.color(g.kind, a / g.norm)
            rad = g.radius * (0.8 if g.kind == "inh" and TERRITORY[territory_of(g.key)][0] == "cortex" else 1.0)
            for (x, y), c in zip(g.xy, cols):
                pg.draw.circle(s, c.tolist(), (float(x), float(y)), rad)
        # labels
        if self.labels:
            for (pos, txt) in LOBE_LABELS:
                s.blit(self.small.render(txt, True, (82, 84, 98)), S(pos))
            for name, polys in NOT_BUILT.items():
                if built.get(name) in L.terr:
                    continue
                at = {"amygdala": (0.215, 0.60), "hypothalamus": (0.455, 0.612), "claustrum": (0.205, 0.468)}[name]
                s.blit(self.small.render(f"{name} · not built yet", True, (82, 84, 98)), S(at))
            spots = {"A": (0.79, 0.07), "B": (0.56, 0.085), "MOT": (0.27, 0.075), "TH": (0.60, 0.41), "TH.TRN": (0.62, 0.53),
                     "STR": (0.19, 0.40), "GPe": (0.30, 0.53), "GPi": (0.53, 0.555), "retina": (0.005, 0.57),
                     "EC": (0.36, 0.69), "VTA": (0.62, 0.655), "BLA": (0.215, 0.60), "CeA": (0.44, 0.582), "DG": (0.47, 0.67), "CA3": (0.52, 0.66), "CA1": (0.56, 0.645), "SUB": (0.60, 0.635)}
            for tk in L.terr:
                txt = TERRITORY[tk][2]
                s.blit(self.small.render(txt, True, (170, 172, 186)), S(spots.get(tk, (0.5, 0.95))))
            d1 = next((g for g in L.groups if g.key == "STR.D1"), None)
            if d1 is not None:
                n = self.agent.n
                ch = np.minimum((d1.mpos[:, 0] * n).astype(int), n - 1)
                top = d1.xy[:, 1].min() - 18
                for k in range(n):
                    x = float(np.median(d1.xy[ch == k, 0]))
                    s.blit(self.small.render(str(k), True, (150, 150, 165)), (x - 3, top))
                s.blit(self.small.render("action channels", True, (110, 110, 125)), (float(d1.xy[:, 0].min()), top - 15))

    def _mask_outside(self, out):
        """Alpha mask that keeps only what lies inside the cerebrum's outline (cached)."""
        if getattr(self, "_mask", None) is None:
            m = self.pg.Surface((self.W, self.H), self.pg.SRCALPHA)
            m.fill((255, 255, 255, 0))
            self.pg.draw.polygon(m, (255, 255, 255, 255), out)
            self._mask = m
        return self._mask

    def draw_hover(self, r):
        if not self.hover or not self.layout:
            return
        hx, hy = self.hover
        best, bd = None, 12 ** 2
        for g in self.layout.groups:
            d = ((g.xy[:, 0] - hx) ** 2 + (g.xy[:, 1] - hy) ** 2)
            i = int(np.argmin(d))
            if d[i] < bd:
                best, bd = (g, i), d[i]
        if not best:
            return
        g, i = best
        name, human = describe(self.brain, g.key)
        rate = float(g.rates(r)[i])
        lines = [f"{name}  ({g.key})", human, f"{g.sizes[i]} neuron{'s' if g.sizes[i] > 1 else ''} in this dot · "
                 f"{g.n:,} in the group", f"firing now: {rate:.3f}  (0 = silent, 1 = maximum)"]
        surf = [self.small.render(t, True, INK) for t in lines]
        w = max(x.get_width() for x in surf) + 16
        h = 18 * len(surf) + 10
        x, y = min(hx + 16, self.W - w - 4), min(hy + 16, self.H - h - 4)
        self.pg.draw.rect(self.screen, (28, 30, 40), (x, y, w, h), border_radius=6)
        self.pg.draw.rect(self.screen, (90, 96, 120), (x, y, w, h), 1, border_radius=6)
        for k, t in enumerate(surf):
            self.screen.blit(t, (x + 8, y + 6 + 18 * k))
        self.pg.draw.circle(self.screen, ACCENT, g.xy[i], g.radius + 3, 1)

    def plot(self, rect, ys, lo, hi, color, ref=None, label=""):
        pg, s = self.pg, self.screen
        x0, y0, w, h = rect
        pg.draw.rect(s, (24, 26, 34), rect, border_radius=4)
        if ref is not None:
            yy = y0 + h - (ref - lo) / (hi - lo) * h
            pg.draw.line(s, (80, 80, 96), (x0, yy), (x0 + w, yy), 1)
        if len(ys) > 1:
            pts = [(x0 + i / (len(ys) - 1) * w, y0 + h - (np.clip(v, lo, hi) - lo) / (hi - lo) * h) for i, v in enumerate(ys)]
            pg.draw.lines(s, color, False, pts, 2)
        s.blit(self.small.render(label, True, DIM), (x0, y0 - 16))

    def draw_panel(self):
        pg, s = self.pg, self.screen
        X = 1030
        s.blit(self.big.render("LittleBrain", True, INK), (X, 24))
        s.blit(self.font.render(self.size_text, True, DIM), (X, 56))
        a = self.agent
        # what the brain sees (any game's observation image)
        img = self.seen if (self.seen is not None and self.phase != "rest") else None
        if img is not None:
            surf = pg.surfarray.make_surface(np.repeat((np.clip(img.T, 0, 1)[:, :, None] * 235 + 12).astype(np.uint8), 3, 2))
            s.blit(pg.transform.scale(surf, (176, 176)), (X, 100))
        else:
            pg.draw.rect(s, (12, 13, 18), (X, 100, 176, 176))
        pg.draw.rect(s, (90, 96, 120), (X, 100, 176, 176), 1)
        s.blit(self.small.render("what it sees", True, DIM), (X, 280))
        phase = {"look": "looking…", "act": "acting", "rest": "resting"}[self.phase]
        s.blit(self.font.render(f"{self.game_name} · turn {a.turn + 1} · {phase}", True, INK), (X + 200, 100))
        s.blit(self.font.render(f"learning {'ON' if a.learn else 'OFF'}" + ("   ⏸ paused" if self.paused else ""),
                                True, GOOD if a.learn else BAD), (X + 200, 124))
        # action channels: bar = how far the GPi brake is lifted; number = chance of choosing; outline = chosen
        n = a.n
        names = list(getattr(self.game, "action_names", ()) or [f"action {k}" for k in range(n)])
        w = min(80, 340 // max(n, 1))
        chosen = self.info.get("action") if self.phase == "act" else None
        s.blit(self.small.render("actions: bar = brake released · number = odds of choosing", True, DIM), (X + 200, 150))
        rel, p = a.last.get("release"), a.last.get("p")
        for k in range(n):
            bx = X + 200 + k * w
            pg.draw.rect(s, (60, 64, 80), (bx + 6, 170, w - 14, 60), 1)
            if rel is not None:
                lo, hi = rel.min() - 0.01, rel.max() + 0.01
                h = float(60 * (rel[k] - lo) / (hi - lo))
                pg.draw.rect(s, ACCENT, (bx + 6, 230 - h, w - 14, h))
                s.blit(self.small.render(f"{p[k]:.2f}", True, INK), (bx + 8, 234))
            if chosen == k:
                pg.draw.rect(s, INK, (bx + 2, 166, w - 6, 86), 2, border_radius=5)
            s.blit(self.small.render(names[k][:10], True, DIM), (bx + 4, 252))
        if chosen is not None:
            rwd = self.info.get("reward", 0.0)
            s.blit(self.font.render(f"chose {names[chosen]} · reward {rwd:+.2f}", True, GOOD if rwd > 0 else (BAD if rwd < 0 else DIM)),
                   (X + 200, 274))
        # dopamine and reward
        self.plot((X, 330, 500, 80), self.dopa, -1, 1, (120, 220, 160), ref=0,
                  label="dopamine: surprise about reward (up = better than expected)")
        rr = np.array(self.rewards)
        roll = [rr[max(0, i - 49):i + 1].mean() for i in range(len(rr))]
        lo, hi = (min(0.0, min(roll)), max(1.0, max(roll))) if roll else (0, 1)
        chance = getattr(self.game, "chance", None)
        self.plot((X, 450, 500, 120), roll[-400:], lo, hi, ACCENT, ref=chance,
                  label=f"reward per turn, average of last 50: {roll[-1] if roll else 0:.2f}"
                        + (f"   (line = chance {chance:.2f})" if chance is not None else ""))
        # legend
        y = 610
        s.blit(self.font.render("Legend", True, INK), (X, y))
        for k, (kind, name) in enumerate((("exc", "excitatory cells (pyramidal, stellate, relay, striatal)"),
                                         ("inh", "inhibitory cells (interneurons, pallidum)"),
                                         ("input", "inputs (eye, action copy)"))):
            for j, v in enumerate((0.1, 0.5, 1.0)):
                pg.draw.circle(s, self.color(kind, np.array([v]))[0], (X + 10 + 14 * j, y + 32 + 22 * k), 5)
            s.blit(self.small.render(name, True, DIM), (X + 56, y + 25 + 22 * k))
        per = np.median([g.per_dot for g in self.layout.groups])
        s.blit(self.small.render("dim → bright = quiet → firing (each group scaled to its own usual range)", True, DIM), (X, y + 96))
        s.blit(self.small.render(f"one dot ≈ {per:.0f} neuron{'s' if per >= 1.5 else ''} · bigger brains get more and larger "
                                 "dots, up to a cap", True, DIM), (X, y + 116))
        keys = "Space pause · ←/→ speed · L learn · S sleep · [ ] size · T labels · P paths · hover"
        s.blit(self.small.render(keys, True, (100, 104, 120)), (X, 860))
        if self.status:
            s.blit(self.small.render(self.status, True, ACCENT), (X, 836))

    def draw(self):
        s = self.screen
        s.fill(BG)
        r = np.asarray(self.brain.r if not hasattr(self.brain.r, "get") else self.brain.r.get())
        self.draw_brain(r)
        self.screen.blit(self.small.render("side view, facing left · deep structures shown through the surface", True,
                                           (90, 94, 110)), (30, 20))
        self.draw_panel()
        self.draw_hover(r)

    def draw_loading(self):
        s = self.screen
        s.fill(BG)
        _, scale, t0 = self.loading
        msg = f"growing a brain at scale {scale:g}… calibrating and 2 days of development ({time.time() - t0:.0f} s)"
        s.blit(self.big.render(msg, True, INK), (60, 420))
        s.blit(self.font.render("(saved to disk afterwards, so next time this size opens instantly)", True, DIM), (60, 460))

    # ---------- main loop ----------

    def run(self):
        if self.given is not None:
            self.loading = ("done", self.given, 0.0)
        else:
            self.build(self.scale)
        while not self.quit:
            if self.loading and self.loading[0] == "error":
                print(self.loading[1], flush=True)
                if self.headless:
                    break
                self.status = "building the brain failed: " + self.loading[1].strip().splitlines()[-1][:120]
                self.loading = None
                if self.brain is None:
                    break
                continue
            if self.loading and self.loading[0] == "busy":
                if self.headless:
                    time.sleep(0.2)
                else:
                    self.events()
                    self.draw_loading()
                    self.pg.display.flip()
                    self.clock.tick(10)
                continue
            if self.loading and self.loading[0] == "done":
                self.adopt(self.loading[1])
                self.status = f"brain ready ({self.loading[2]:.0f} s)"
                self.loading = None
            try:
                self.rewards += self.agent.play(self.game, 1)   # one turn of whatever game is plugged in
            except StopIteration:
                pass
            if self.headless and (not self.shots or (self.max_turns and self.agent.turn + 1 >= self.max_turns)):
                break
        self.pg.quit()



def watch(agent_brain, make_game, **kw):
    """Open the viewer on a prepared Brain playing games from make_game() (any games.Game)."""
    App(make_game, brain=agent_brain, **kw).run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="brain size factor (0.25 .. 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game", default="signal", help="which game to plug in (games.REGISTRY)")
    ap.add_argument("--brain", default="amy", help="which brain: bg | da (dopamine cells) | amy (+ amygdala)")
    ap.add_argument("--shots", default="", help="headless: comma-separated turns to screenshot")
    ap.add_argument("--out", default="shots")
    ap.add_argument("--hover", default="", help="headless: pretend the mouse is at x,y (shows the info box)")
    a = ap.parse_args()
    shots = [int(t) for t in a.shots.split(",") if t]
    os.environ["BRAIN"] = a.brain
    import games
    make = lambda: games.REGISTRY[a.game](seed=a.seed)
    app = App(make, a.scale, a.seed, headless=bool(shots), shots=shots, out=a.out)
    if a.hover:
        app.hover = tuple(int(v) for v in a.hover.split(","))
    app.run()


if __name__ == "__main__":
    main()

"""
Competitive learning in IT (note 90): meant to make IT cells specialise on recurring objects, wherever they appear.
**Not adopted**: on the vision brain IT specialised on where objects were and how bright they were, not on shape
(note 90). Kept as a tool; not part of development.

**Reduced model of a slow developmental process.** In animals, IT neurons become selective for object features over
long visual experience, through Hebbian learning under strong lateral inhibition (competitive learning; Rumelhart &
Zipser 1985), with temporal continuity tying together the views of one object (Foldiak 1991; Wallis & Rolls 1997),
and neighbouring cells sharing preferences (feature columns; Tanaka 1996). Here that is compressed into a step of
development, like the like-to-like wiring (vision.like_to_like):

1. The brain watches movies of one object sliding across the view (random shape, size, brightness and path). V4's
   response is averaged over each movie, so the object's different positions blur together (the temporal trace).
2. IT's input cells compete for each movie: the best-matching few (k-winners-take-all) and their neighbours on IT's
   map move their preferences toward that movie's V4 pattern (a self-organising map; Kohonen 1982). No labels are
   used, only which views came together in time.
3. Each IT cell's synapses from V4 are then re-assigned to the V4 cells it prefers most (same number of synapses,
   same strengths): structural plasticity, as in like_to_like.
"""

import numpy as np

from backend import like, to_host


def _rates(brain, src):
    """Rates of the source groups (cells or input maps), concatenated."""
    r = to_host(brain.r)
    return np.concatenate([np.asarray(r[brain.index[k]]) for k in src]).copy()


def object_movies(brain, rng, n, frames=12, blank=4, src=("V4.L23",)):
    """V4 L2/3's response to n object movies, frame by frame: (movies, frames, cells)."""
    from games.vision_games import ShapeGame
    from retina import Retina
    eye = Retina(brain)
    quiet = {k: np.zeros(len(brain.pos[k]), np.float32) for k in brain.spec.inputs}
    grey = np.full((64, 64), 0.5, np.float32)
    sg = ShapeGame(0)
    out = []
    for _ in range(n):
        shape = sg.SHAPES[int(rng.integers(len(sg.SHAPES)))]
        r, level = rng.uniform(0.12, 0.22), rng.choice([0.1, 0.9])
        a = rng.uniform(0, 2 * np.pi)
        start = 0.5 - 0.25 * np.array([np.cos(a), np.sin(a)])
        step = 0.5 * np.array([np.cos(a), np.sin(a)]) / frames
        for _ in range(blank):
            brain.step({**quiet, **eye(grey)}, learn=False, ach=0.6)
        movie = []
        for f in range(frames):
            x, y = np.clip(start + f * step, 0.15, 0.85)
            brain.step({**quiet, **eye(sg.picture(shape, x, y, r, level))}, learn=False, ach=0.6)
            movie.append(_rates(brain, src))
        out.append(movie)
    return np.array(out, np.float32)


def competitive_it(brain, rng, movies=400, epochs=8, k=0.05, eta=0.1, sigma0=0.3, sigma1=0.06, trace=0.8,
                   src="V4.L23", dst="IT.L4"):
    src = (src,) if isinstance(src, str) else tuple(src)
    if any(k not in brain.index for k in src) or dst not in brain.g:
        return None
    M = object_movies(brain, rng, movies, src=src)      # (movies, frames, cells)
    M = M - M.reshape(-1, M.shape[2]).mean(0)            # what differs between views, not "something is there"
    M /= np.linalg.norm(M, axis=2, keepdims=True) + 1e-9
    X = M.reshape(-1, M.shape[2])
    every = np.arange(len(to_host(brain.r)))
    src_ids = np.concatenate([every[brain.index[k]] for k in src])
    n_dst = brain.g[dst].n
    pos = brain.pos[dst]
    G = rng.standard_normal((n_dst, X.shape[1])) * 0.1
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(2)
    kk = max(1, int(round(k * n_dst)))
    for e in range(epochs):
        sigma = sigma0 * (sigma1 / sigma0) ** (e / max(epochs - 1, 1))   # the neighbourhood shrinks (SOM schedule)
        for m in rng.permutation(len(M)):
            # the trace rule (Foldiak 1991): activity is smoothed over the movie's frames, so the cells that win early
            # keep winning while the object moves, and learn each of its positions
            ybar = np.zeros(n_dst)
            for x in M[m]:
                ybar = trace * ybar + (1 - trace) * (G @ x)
                win = np.argpartition(ybar, -kk)[-kk:]
                h = np.exp(-d2[:, win] / (2 * sigma ** 2)).max(1)
                G += eta * h[:, None] * (x[None, :] - G)
                G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    brain._it_template = G   # the learned preferences, kept on the object for analysis only (not saved)
    # structural step: each IT cell's V4 synapses go to the V4 cells it prefers most
    for bank in brain.g[dst].banks():
        s = to_host(bank.src).copy()
        for i in range(n_dst):
            slots = np.nonzero(np.isin(s[i], src_ids))
            if not len(slots[0]):
                continue
            top = np.argsort(-G[i])[:len(slots[0])]
            s[i][slots] = src_ids[top]
        bank.src = like(bank.src, s, dtype=s.dtype)
    # how specialised IT became: how often each cell wins, and how different the winners are between movies
    wins = np.zeros(n_dst)
    for x in X:
        wins[np.argpartition(G @ x, -kk)[-kk:]] += 1
    return {"cells that ever win": float((wins > 0).mean()), "max share of wins": float(wins.max() / len(X))}

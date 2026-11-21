"""
Games that need real vision (notes 74-76). Each is built so that the old brightness-only route can't solve it.

  ColourGame   one shape, same place, in one of 4 colours of equal brightness -> the matching button.
  MotionGame   a field of random dots drifting in one of 4 directions (new random positions every turn)
               -> the matching button. Sends a movie (Observation.frames).
"""

import numpy as np

from .core import Game, Observation

SIZE = 64
# Equal brightness (0.45 R + 0.55 G = 0.5, the retina model's L+M luminance) on a 0.5 grey background
COLOURS = {"red": (1.0, 0.09, 0.3), "green": (0.0, 0.91, 0.3), "blue": (0.4, 0.58, 1.0), "yellow": (0.6, 0.42, 0.0)}


def _xy(size=SIZE):
    return (np.mgrid[0:size, 0:size] + 0.5) / size


class ColourGame(Game):
    def __init__(self, seed=0, n_actions=4):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.names = list(COLOURS)[:n_actions]
        self.answer = self.rng.permutation(n_actions)
        self.action_names = tuple(f"button {i}" for i in range(n_actions))
        self.chance = 1 / n_actions
        self.signal = 0

    def picture(self, colour):
        yy, xx = _xy()
        img = np.full((SIZE, SIZE, 3), 0.5, np.float32)
        img[(xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.25 ** 2] = COLOURS[colour]
        return img

    def reset(self):
        self.signal = int(self.rng.integers(self.n_actions))
        return Observation(self.picture(self.names[self.signal]))

    def step(self, action):
        correct = int(action) == int(self.answer[self.signal])
        info = {"signal": self.signal, "colour": self.names[self.signal], "correct": correct}
        return self.reset(), float(correct), False, info


class MotionGame(Game):
    DIRS = {"right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}

    def __init__(self, seed=0, n_actions=4, dots=60, speed=0.02, frames=20):
        self.rng = np.random.default_rng(seed)
        self.n_actions, self.dots, self.speed, self.n_frames = n_actions, dots, speed, frames
        self.names = list(self.DIRS)[:n_actions]
        self.answer = self.rng.permutation(n_actions)
        self.action_names = tuple(f"button {i}" for i in range(n_actions))
        self.chance = 1 / n_actions
        self.signal = 0

    def movie(self, direction):
        d = np.array(self.DIRS[direction], np.float64)
        p = self.rng.random((self.dots, 2))
        frames = []
        for t in range(self.n_frames):
            q = (p + t * self.speed * d) % 1.0
            img = np.full((SIZE, SIZE), 0.5, np.float32)
            ix, iy = (q * SIZE).astype(int).T
            for dx in (0, 1):
                for dy in (0, 1):
                    img[(iy + dy) % SIZE, (ix + dx) % SIZE] = 0.05
            frames.append(img)
        return frames

    def reset(self):
        self.signal = int(self.rng.integers(self.n_actions))
        frames = self.movie(self.names[self.signal])
        return Observation(frames[0], frames=frames)

    def step(self, action):
        correct = int(action) == int(self.answer[self.signal])
        info = {"signal": self.signal, "direction": self.names[self.signal], "correct": correct}
        return self.reset(), float(correct), False, info


class ShapeGame(Game):
    """One of 4 shapes (disc, square, horizontal bar, vertical bar) at a random place and size, random grey level ->
    the matching button. Where it is and how bright it is don't predict the answer; only its shape does (IT's job)."""

    SHAPES = ("disc", "square", "hbar", "vbar")

    def __init__(self, seed=0, n_actions=4):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.answer = self.rng.permutation(n_actions)
        self.action_names = tuple(f"button {i}" for i in range(n_actions))
        self.chance = 1 / n_actions
        self.signal = 0

    def picture(self, shape, x=0.5, y=0.5, r=0.18, level=0.9):
        yy, xx = _xy()
        img = np.full((SIZE, SIZE), 0.5, np.float32)
        dx, dy = np.abs(xx - x), np.abs(yy - y)
        m = {"disc": dx ** 2 + dy ** 2 < r * r, "square": (dx < r * .85) & (dy < r * .85),
             "hbar": (dx < r * 1.4) & (dy < r * .35), "vbar": (dx < r * .35) & (dy < r * 1.4)}[shape]
        img[m] = level
        return img

    def reset(self):
        self.signal = int(self.rng.integers(self.n_actions))
        r = self.rng.uniform(0.12, 0.22)
        x, y = self.rng.uniform(0.25, 0.75, 2)
        level = self.rng.choice([0.1, 0.9])
        return Observation(self.picture(self.SHAPES[self.signal], x, y, r, level))

    def step(self, action):
        correct = int(action) == int(self.answer[self.signal])
        info = {"signal": self.signal, "shape": self.SHAPES[self.signal], "correct": correct}
        return self.reset(), float(correct), False, info


class PeripheralColourGame(ColourGame):
    """A small coloured disc somewhere away from the centre, where the eye is blurry: its colour -> the matching button.
    Tests eye movements (note 76): looking at it should make the colour easier to see. `target` records where it was."""

    def picture(self, colour, x=0.5, y=0.5, r=0.06):
        """A dark outline makes the target salient (the colliculus is largely colour-blind: it sees brightness edges);
        its identity is only in its colour, at the background's brightness."""
        yy, xx = _xy()
        img = np.full((SIZE, SIZE, 3), 0.5, np.float32)
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        img[d2 < (r + 0.02) ** 2] = 0.1
        img[d2 < r * r] = COLOURS[colour]
        return img

    def reset(self):
        self.signal = int(self.rng.integers(self.n_actions))
        a, d = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.25, 0.38)
        self.target = (0.5 + d * np.cos(a), 0.5 + d * np.sin(a))
        return Observation(self.picture(self.names[self.signal], *self.target))

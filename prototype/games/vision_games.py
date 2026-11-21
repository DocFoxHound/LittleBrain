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

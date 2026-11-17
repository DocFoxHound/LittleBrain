"""
Danger game: food pictures and danger pictures. Food: the right button (one of the first n-1) earns +1.
Danger: anything except "hide" (the last action) hurts, -1; hiding is safe (0). Hiding at food earns nothing.
Tests learning from punishment, the amygdala's job: how fast does the character learn to hide?
"""

import numpy as np

from .core import Game, Observation, blob_image


class DangerGame(Game):
    def __init__(self, n_food=3, n_danger=3, n_actions=4, seed=0, blobs=3, size=16):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.action_names = tuple(f"button {i}" for i in range(n_actions - 1)) + ("hide",)
        n = n_food + n_danger
        self.images = [blob_image(rng.uniform(0.1, 0.9, (blobs, 2)), size) for _ in range(n)]
        self.danger = np.array([False] * n_food + [True] * n_danger)
        self.answer = rng.integers(n_actions - 1, size=n)        # food: the right button
        self.rng, self.signal = rng, 0
        self.chance = None

    def reset(self):
        self.signal = int(self.rng.integers(len(self.images)))
        return Observation(self.images[self.signal])

    def step(self, action):
        hide = int(action) == self.n_actions - 1
        if self.danger[self.signal]:
            reward = 0.0 if hide else -1.0
        else:
            reward = 1.0 if int(action) == int(self.answer[self.signal]) else 0.0
        info = {"signal": self.signal, "danger": bool(self.danger[self.signal]), "hid": hide,
                "correct": reward > 0 or (self.danger[self.signal] and hide)}
        return self.reset(), reward, False, info

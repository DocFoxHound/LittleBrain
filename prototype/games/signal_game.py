"""
Signal game: one of a few pictures appears; each picture has one correct button. Reward 1 for the right
button, 0 otherwise. The mapping is random per game seed, so it must be learned, not built in.
(A contextual bandit: the classic test of striatal, dopamine-taught action learning.)
"""

import numpy as np

from .core import Game, Observation, blob_image


class SignalGame(Game):
    def __init__(self, n_signals=4, n_actions=4, seed=0, blobs=3, size=16):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.action_names = tuple(f"button {i}" for i in range(n_actions))
        self.images = [blob_image(rng.uniform(0.1, 0.9, (blobs, 2)), size) for _ in range(n_signals)]
        self.answer = rng.integers(n_actions, size=n_signals)
        self.rng, self.signal = rng, 0

    def reset(self):
        self.signal = int(self.rng.integers(len(self.images)))
        return Observation(self.images[self.signal])

    def step(self, action):
        correct = int(action) == int(self.answer[self.signal])
        info = {"signal": self.signal, "correct": correct}
        return self.reset(), float(correct), False, info

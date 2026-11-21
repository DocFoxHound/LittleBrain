"""
Games that need credit across steps (plan phase 1, track N1; note 80). Each turn is one decision; an episode spans
several turns and `done` marks its end.

  KeyDoorGame     a chain of k situations (distinct pictures). The right action in each moves to the next; only the
                  last one pays (+1) and ends the episode. A wrong action stays where it is (every episode finishes).
                  A learner that only credits immediate rewards can't learn the early steps.
  SlowPoisonGame  a berry: eating pays a little now (+0.3) but makes the character sick k turns later (-1); skipping
                  pays nothing and costs nothing. A learner that only credits immediate rewards keeps eating.
"""

import numpy as np

from .core import Game, Observation, blob_image


def _pictures(rng, n, size=16):
    return [blob_image(rng.uniform(0.15, 0.85, (3, 2)), size) for _ in range(n)]


class KeyDoorGame(Game):
    def __init__(self, seed=0, k=3, n_actions=4):
        rng = np.random.default_rng(seed)
        self.k, self.n_actions = k, n_actions
        self.images = _pictures(rng, k)
        self.answer = rng.integers(n_actions, size=k)
        self.action_names = tuple(f"button {i}" for i in range(n_actions))
        self.chance = 1 / n_actions
        self.state = 0

    def reset(self):
        self.state = 0
        return Observation(self.images[0])

    def step(self, action):
        s = self.state
        correct = int(action) == int(self.answer[s])
        info = {"state": s, "correct": correct}
        if not correct:
            return Observation(self.images[s]), 0.0, False, info
        if s == self.k - 1:
            self.state = 0
            info["completed"] = True
            return Observation(self.images[0]), 1.0, True, info
        self.state = s + 1
        return Observation(self.images[s + 1]), 0.0, False, info


class SlowPoisonGame(Game):
    """State 'berry' (eat = action 0, anything else = skip), then k turns of 'after eating' or 'after skipping'
    pictures where actions do nothing, then the consequence."""

    def __init__(self, seed=0, k=2, gain=0.3, cost=1.0, n_actions=4):
        rng = np.random.default_rng(seed)
        self.k, self.gain, self.cost, self.n_actions = k, gain, cost, n_actions
        self.berry, self.ate_pic, self.skip_pic = _pictures(rng, 3)
        self.action_names = ("eat",) + tuple(f"skip {i}" for i in range(1, n_actions))
        self.chance = None
        self.phase, self.ate, self.left = "berry", False, 0

    def reset(self):
        self.phase, self.ate, self.left = "berry", False, 0
        return Observation(self.berry)

    def step(self, action):
        if self.phase == "berry":
            self.ate = int(action) == 0
            self.phase, self.left = "after", self.k
            info = {"state": "berry", "ate": self.ate, "correct": not self.ate}
            return Observation(self.ate_pic if self.ate else self.skip_pic), (self.gain if self.ate else 0.0), False, info
        self.left -= 1
        info = {"state": "after", "ate": self.ate}
        if self.left > 0:
            return Observation(self.ate_pic if self.ate else self.skip_pic), 0.0, False, info
        reward = -self.cost if self.ate else 0.0
        info["sick"] = self.ate
        return self.reset(), reward, True, info

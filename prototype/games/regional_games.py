"""Small, seeded experiments for body state, working memory and timing."""
import numpy as np

from .core import Game, Observation, blob_image


class ForageGame(Game):
    n_actions = 2
    action_names = ("eat", "wait")

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.image = blob_image([(0.5, 0.5)])

    def observe(self):
        return Observation(self.image, {"hunger": self.hunger})

    def reset(self):
        self.hunger = float(self.rng.uniform())
        return self.observe()

    def step(self, action):
        ate = action == 0
        hunger = self.hunger
        self.hunger = float(np.clip(hunger + 0.06 - 0.35 * ate, 0, 1))
        return self.observe(), float(ate), False, {"food_contact": ate, "food_reward": float(ate), "hunger": hunger}


class DelayedCueGame(Game):
    n_actions = 2
    action_names = ("left", "right")

    def __init__(self, seed=0, delay=2):
        if delay < 0:
            raise ValueError("delay must be nonnegative")
        self.rng = np.random.default_rng(seed)
        self.delay = delay
        self.cues = [blob_image([(x, 0.5)]) for x in (0.25, 0.75)]
        self.blank = np.zeros_like(self.cues[0])
        self.go = blob_image([(0.5, 0.5)])

    def reset(self):
        self.side = int(self.rng.integers(2))
        self.phase = 0
        return Observation(self.cues[self.side])

    def step(self, action):
        if self.phase == self.delay + 1:
            correct = action == self.side
            return Observation(self.blank), float(correct), True, {"choice": True, "correct": correct}
        self.phase += 1
        image = self.go if self.phase == self.delay + 1 else self.blank
        return Observation(image), 0.0, False, {"choice": False, "feedback": False}


class TimingGame(Game):
    n_actions = 2
    action_names = ("wait", "act")

    def __init__(self, seed=0, interval=3):
        if interval < 1:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.cue = blob_image([(0.5, 0.5)])
        self.blank = np.zeros_like(self.cue)

    def reset(self):
        self.elapsed = 0
        return Observation(self.cue)

    def step(self, action):
        done = action == 1 or self.elapsed >= 2 * self.interval
        error = min(1.0, abs(self.elapsed - self.interval) / self.interval) if done else 0.0
        self.elapsed += 1
        return Observation(self.blank), float(done and error == 0), done, {"timing_error": error, "ended": done, "feedback": done}

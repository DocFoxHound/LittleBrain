"""
A LittleBrain character: a federated brain with a basal ganglia action loop (brainspec.stage_bg) playing a
game (prototype/games). Each turn:

  1. look:    the picture is shown for `look` steps; the brain's GPi output per action channel is read over the
              last 3 steps. Lower GPi = that action is released (disinhibited) more.
  2. choose:  softmax over the channels' release (temperature = how much neural noise drives exploration).
  3. act:     the efference copy of the chosen action excites its striatal channel for `act` steps, the game
              answers, and dopamine (reward minus expected reward) arrives while the eligibility traces are hot.
  4. rest:    a short blank.

Dopamine here is a stand-in critic: one running average of reward (no cue-specific expectation yet).
"""

import numpy as np

from backend import to_host
from brainspec import action_channel


class BrainAgent:
    def __init__(self, brain, n_actions, look=6, act=4, rest=3, temperature=0.02, ach=0.6, seed=0,
                 learn=True, dopamine_gain=1.0):
        self.b, self.n = brain, n_actions
        self.look, self.act, self.rest = look, act, rest
        self.temperature, self.ach, self.learn, self.dopamine_gain = temperature, ach, learn, dopamine_gain
        self.rng = np.random.default_rng(seed)
        self.expected = 0.0
        self.chan = {k: action_channel(brain.tile[k], n_actions) for k in ("MOT", "GPi", "STR.D1", "STR.D2")}
        P = brain.pos["retina"]
        self._px = P
        self.last = {}

    def _retina(self, image):
        h, w = image.shape
        ix = np.minimum((self._px[:, 0] * w).astype(int), w - 1)
        iy = np.minimum((self._px[:, 1] * h).astype(int), h - 1)
        return image[iy, ix].astype(np.float32)

    def _mot(self, action=None):
        m = np.zeros(len(self.chan["MOT"]), np.float32)
        if action is not None:
            m[self.chan["MOT"] == action] = 1.0
        return m

    def _per_channel(self, key):
        r = to_host(self.b.g[key].rate)
        return np.array([r[self.chan[key] == a].mean() for a in range(self.n)])

    def choose(self, image):
        x, gpi = self._retina(image), []
        for s in range(self.look):
            self.b.step({"retina": x, "MOT": self._mot()}, learn=self.learn, ach=self.ach)
            if s >= self.look - 3:
                gpi.append(self._per_channel("GPi"))
        release = -np.mean(gpi, 0)
        p = np.exp((release - release.max()) / self.temperature)
        p /= p.sum()
        action = int(self.rng.choice(self.n, p=p))
        self.last = {"release": release, "p": p, "D1": self._per_channel("STR.D1"), "D2": self._per_channel("STR.D2")}
        return action, x

    def outcome(self, x, action, reward):
        rpe = float(np.clip(reward - self.expected, -1, 1)) * self.dopamine_gain
        self.expected += 0.05 * (reward - self.expected)
        for s in range(self.act):
            self.b.step({"retina": x, "MOT": self._mot(action)}, learn=self.learn, ach=self.ach,
                        dopamine=rpe if (self.learn and s == 2) else 0.0, arousal=0.3)
        for _ in range(self.rest):
            self.b.step({"retina": 0 * x, "MOT": self._mot()}, ach=self.ach)
        return rpe

    def play(self, game, turns, sleep_every=0, log=None):
        """Play `turns` turns; returns the list of rewards. sleep_every: sleep after this many turns (0 = never)."""
        obs, rewards = game.reset(), []
        for t in range(turns):
            action, x = self.choose(obs.image)
            obs_next, reward, done, info = game.step(action)
            self.outcome(x, action, reward)
            rewards.append(reward)
            if log is not None:
                log.append({"turn": t, "action": action, "reward": reward, **info, **self.last})
            if sleep_every and (t + 1) % sleep_every == 0:
                self.b.sleep()
            obs = game.reset() if done else obs_next
        return rewards

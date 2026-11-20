"""
A LittleBrain character: a federated brain with a basal ganglia action loop (brainspec.stage_bg) playing a
game (prototype/games). Each turn:

  1. look:    the picture is shown for `look` steps; the brain's GPi output per action channel is read over the
              last 3 steps. Lower GPi = that action is released (disinhibited) more.
  2. choose:  softmax over the channels' release (temperature = how much neural noise drives exploration).
  3. act:     the efference copy of the chosen action excites its striatal channel for `act` steps, the game
              answers, and dopamine (reward minus expected reward) arrives while the eligibility traces are hot.
  4. rest:    a short blank.

Dopamine = reward - expected reward. critic="average": one running average of reward. critic="cortex": the
expectation is read out of the cortical activity during the look phase (a linear readout learned by the delta
rule), a stand-in for striosomes / ventral striatum, which learn what each situation is worth (notes/30).
"""

import numpy as np

from backend import to_host
from brainspec import action_channel


class BrainAgent:
    def __init__(self, brain, n_actions, look=20, read=16, act=4, rest=3, temperature=0.02, ach=0.6, seed=0,
                 learn=True, dopamine_gain=1.0, critic="average", critic_lr=0.3):
        self.b, self.n = brain, n_actions
        self.look, self.act, self.rest, self.read = look, act, rest, read   # read: last steps of the look that count
        self.temperature, self.ach, self.learn, self.dopamine_gain = temperature, ach, learn, dopamine_gain
        self.rng = np.random.default_rng(seed)
        self.timed = getattr(brain.spec, "timed_dopamine", False)
        self.cue_value = 0.0
        self.expected = 0.0
        self.critic, self.critic_lr, self.w_value, self.features = critic, critic_lr, None, None
        self.value_groups = ("A.L23", "A.L5IT", "A.L5PT")
        self.chan = {k: action_channel(brain.tile[k], n_actions) for k in ("MOT", "GPi", "STR.D1", "STR.D2")}
        P = brain.pos["retina"]
        self._px = P
        self.last = {}
        # Dopamine from the brain's own dopamine cells if it has them (reward goes in through the US input);
        # otherwise the stand-in critic below computes it
        self.own_dopamine = "US" in brain.pos and "VTA.DA" in brain.g
        # Called after every brain step as on_step(phase, info): phase is "look", "act" or "rest". The visualizer
        # uses it to draw (and to pause the brain between steps).
        self.on_step = None
        self.turn = self.turns_played = 0

    def _tick(self, phase, **info):
        if self.on_step is not None:
            self.on_step(phase, info)

    def _retina(self, image):
        h, w = image.shape
        ix = np.minimum((self._px[:, 0] * w).astype(int), w - 1)
        iy = np.minimum((self._px[:, 1] * h).astype(int), h - 1)
        return image[iy, ix].astype(np.float32)

    def _inputs(self, x, action=None, us=0.0):
        d = {"retina": x, "MOT": self._mot(action)}
        if self.own_dopamine:
            d["US"] = np.full(len(self.b.pos["US"]), us, np.float32)
        return d

    def _mot(self, action=None):
        m = np.zeros(len(self.chan["MOT"]), np.float32)
        if action is not None:
            m[self.chan["MOT"] == action] = 1.0
        return m

    def _per_channel(self, key):
        r = to_host(self.b.g[key].rate)
        return np.array([r[self.chan[key] == a].mean() for a in range(self.n)])

    def choose(self, image):
        x, gpi, feats = self._retina(image), [], []
        for s in range(self.look):
            signals = {}
            if self.timed and s == self.look - 1:
                rates = self.b.g
                value = (float(to_host(rates["STR.SO"].rate).mean())
                         + 4 * float(to_host(rates["BLA.pos"].rate - rates["BLA.neg"].rate).mean()))
                self.cue_value = float(np.clip(value, -1, 1))
                signals["value_change"] = self.cue_value
            self.b.step(self._inputs(x), learn=self.learn, ach=self.ach, signals=signals)
            self._tick("look", image=image, step=s)
            if s >= self.look - self.read:
                gpi.append(self._per_channel("GPi"))
                feats.append(np.concatenate([to_host(self.b.g[k].rate) for k in self.value_groups]))
        self.features = np.mean(feats, 0)
        release = -np.mean(gpi, 0)
        p = np.exp((release - release.max()) / self.temperature)
        p /= p.sum()
        action = int(self.rng.choice(self.n, p=p))
        self.last = {"release": release, "p": p, "D1": self._per_channel("STR.D1"), "D2": self._per_channel("STR.D2")}
        return action, x

    def outcome(self, x, action, reward):
        if self.critic == "cortex":
            f = self.features
            if self.w_value is None:
                self.w_value = np.zeros_like(f)
            expected = float(self.expected + self.w_value @ f)
            delta = reward - expected
            if self.learn:  # normalized delta rule; the bias term is the running average
                self.w_value += self.critic_lr * delta * f / (f @ f + 1.0)
                self.expected += 0.05 * delta
        else:
            expected = self.expected
            self.expected += 0.05 * (reward - self.expected)
        rpe = float(np.clip(reward - expected, -1, 1)) * self.dopamine_gain
        if self.own_dopamine:
            rpe = 0.0
            for s in range(self.act):
                self.b.step(self._inputs(x, action, us=reward if s == 2 else 0.0), learn=self.learn, ach=self.ach,
                            arousal=0.3, signals={"value_change": -self.cue_value if self.timed and s == 2 else 0.0})
                rpe += self.b.last_dopamine if s == 2 else 0.0
                self._tick("act", action=action, reward=reward, dopamine=self.b.last_dopamine, step=s)
        else:
            for s in range(self.act):
                self.b.step(self._inputs(x, action), learn=self.learn, ach=self.ach,
                            dopamine=rpe if (self.learn and s == 2) else 0.0, arousal=0.3)
                self._tick("act", action=action, reward=reward, dopamine=rpe if s == 2 else 0.0, step=s)
        for s in range(self.rest):
            self.b.step(self._inputs(0 * x), ach=self.ach)
            self._tick("rest", step=s)
        return rpe

    def play(self, game, turns, sleep_every=0, log=None):
        """Play `turns` turns; returns the list of rewards. sleep_every: sleep after this many turns (0 = never)."""
        obs, rewards = game.reset(), []
        for t in range(turns):
            self.turn = self.turns_played
            self.turns_played += 1
            action, x = self.choose(obs.image)
            obs_next, reward, done, info = game.step(action)
            self.last_info = info
            self.last_rpe = self.outcome(x, action, reward)
            rewards.append(reward)
            if log is not None:
                log.append({"turn": t, "action": action, "reward": reward, "dopamine": self.last_rpe, **info, **self.last})
            if sleep_every and (t + 1) % sleep_every == 0:
                self.b.sleep()
            obs = game.reset() if done else obs_next
        return rewards

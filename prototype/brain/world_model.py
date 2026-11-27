"""
A world model and look-ahead (plan phase 3, note 96).

**Reduced model.** In animals the hippocampus holds a predictive map of where situations lead, replays and sweeps
through it ahead of choices (Johnson & Redish 2007; Pfeiffer & Foster 2013), and orbitofrontal / ventromedial
prefrontal cortex score the imagined outcomes. Here, on the place code (the grid cells' firing, spatial.py):

  T_a   for each action, where it leads: phi' ~ T_a phi        (learned by a delta rule from experience)
  R_a   what it pays: r ~ R_a . phi                            (delta rule)
  E_a   whether it ends the episode: end ~ E_a . phi           (delta rule)
  v     how good a place is: V(phi) = v . phi, learned from real steps **and from replay**: remembered places are
        replayed through the model, each updated toward its best imagined action (Dyna; Sutton 1990), so a newly
        found wall or a newly valuable place reshapes values within a few steps, without walking every path again.

Before each choice the brain looks two steps ahead through the model (Q of each action); the result biases the basal
ganglia's choice (agent.choose), which still has the last word. Silencing the hippocampal formation (EC, DG, CA3,
CA1, SUB or MEC) switches the look-ahead off, leaving the habit learner (the striatum).
"""

import numpy as np

HIPPOCAMPAL = {"EC", "DG", "CA3", "CA1", "SUB", "MEC"}


class WorldModel:
    def __init__(self, n_features, n_actions, gamma=0.9, lr=0.5, replay=20, memory=400, seed=0):
        self.n, self.k, self.gamma, self.lr, self.replay = n_features, n_actions, gamma, lr, replay
        self.T = np.stack([np.eye(n_features, dtype=np.float32) for _ in range(n_actions)])   # start: "nothing moves"
        self.R = np.zeros((n_actions, n_features), np.float32)
        self.E = np.zeros((n_actions, n_features), np.float32)
        self.v = np.zeros(n_features, np.float32)
        self.memory, self.mem_size = [], memory
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _norm(phi):
        return float(phi @ phi) + 1e-6

    def q(self, phi, depth=2):
        """Value of each action from phi, looking `depth` steps ahead through the model."""
        out = np.zeros(self.k)
        for a in range(self.k):
            nxt = self.T[a] @ phi
            end = float(np.clip(self.E[a] @ phi, 0, 1))
            later = (self.q(nxt, depth - 1).max() if depth > 1 else float(self.v @ nxt))
            out[a] = float(self.R[a] @ phi) + self.gamma * (1 - end) * later
        return out

    def observe(self, phi, a, reward, nxt=None):
        """One real step: phi --a--> nxt (None if the episode ended), with its reward."""
        n = self._norm(phi)
        done = nxt is None
        if not done:
            self.T[a] += self.lr * np.outer(nxt - self.T[a] @ phi, phi) / n
        self.R[a] += self.lr * (reward - self.R[a] @ phi) * phi / n
        self.E[a] += self.lr * (float(done) - self.E[a] @ phi) * phi / n
        target = reward + (0.0 if done else self.gamma * float(self.v @ nxt))
        self.v += self.lr * (target - self.v @ phi) * phi / n
        # a memory of places, not of moments: a place already remembered is not stored again, so the layout learned
        # while exploring is not pushed out by many visits to the same few places (note 96)
        if self.memory:
            M = np.array(self.memory)
            sim = (M @ phi) / (np.linalg.norm(M, axis=1) * np.linalg.norm(phi) + 1e-9)
            new = sim.max() < 0.97
        else:
            new = True
        if new:
            self.memory.append(phi.copy())
            if len(self.memory) > self.mem_size:
                self.memory.pop(int(self.rng.integers(len(self.memory))))
        self.dream(self.replay)

    def dream(self, steps):
        """Replay remembered places through the model: each place's value moves toward its best imagined action."""
        if not self.memory:
            return
        for i in self.rng.integers(len(self.memory), size=steps):
            phi = self.memory[i]
            target = self.q(phi, depth=1).max()
            self.v += 0.5 * self.lr * (target - self.v @ phi) * phi / self._norm(phi)

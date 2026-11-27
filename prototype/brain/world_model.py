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
    """Outcomes (note 97): besides plain reward, the model can predict *which* outcome each action brings (O_a: e.g.
    food A, food B; the orbitofrontal cortex's outcome identity, Schoenbaum et al. 2009) and keep, for each outcome,
    how much of it lies ahead (successor features, Barreto et al. 2017). Their value is set at the moment of choice by
    what the body wants now (`want`), so a food that became worthless loses its pull at once, without relearning."""

    def __init__(self, n_features, n_actions, gamma=0.9, lr=0.5, replay=20, memory=400, seed=0, n_outcomes=0):
        self.n, self.k, self.gamma, self.lr, self.replay = n_features, n_actions, gamma, lr, replay
        self.n_out = n_outcomes
        self.O = np.zeros((n_actions, n_outcomes, n_features), np.float32)
        self.V_out = np.zeros((n_outcomes, n_features), np.float32)
        self.want = np.ones(n_outcomes, np.float32)
        self.rewarded = False   # has anything worth having happened yet (food, or a reward beyond a step's cost)?
        self.counts = []        # visits to each remembered place (curiosity, note 99)
        import os
        self.curiosity = float(os.environ.get("WM_CURIOSITY", "0.2"))
        self.novel_sim = float(os.environ.get("WM_NOVEL_SIM", "0.8"))        # imagined places are blurry: match loosely
        self.curiosity_fade = float(os.environ.get("WM_CURIOSITY_FADE", "0.1"))   # after the first reward
        self.T = np.stack([np.eye(n_features, dtype=np.float32) for _ in range(n_actions)])   # start: "nothing moves"
        self.R = np.zeros((n_actions, n_features), np.float32)
        self.E = np.zeros((n_actions, n_features), np.float32)
        self.v = np.zeros(n_features, np.float32)
        self.memory, self.mem_size = [], memory
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _norm(phi):
        return float(phi @ phi) + 1e-6

    def value(self, phi):
        """How good a place is now: plain value plus each outcome still ahead, weighted by how much it is wanted."""
        return float(self.v @ phi) + (float(self.want @ (self.V_out @ phi)) if self.n_out else 0.0)

    def _nearest(self, phi):
        if not self.memory:
            return -1, 0.0
        M = np.array(self.memory)
        sim = (M @ phi) / (np.linalg.norm(M, axis=1) * np.linalg.norm(phi) + 1e-9)
        i = int(np.argmax(sim))
        return i, float(sim[i])

    def novelty(self, phi):
        """How new a place is: 1 if never visited, fading as 1/sqrt(1 + visits) (count-based curiosity; the
        hippocampus's novelty signal drives dopamine, Lisman & Grace 2005)."""
        i, sim = self._nearest(phi)
        return 1.0 if sim < self.novel_sim else 1.0 / np.sqrt(1.0 + self.counts[i])

    def q(self, phi, depth=2, curious=True):
        """Value of each action from phi, looking `depth` steps ahead through the model; when `curious`, each imagined
        place also counts for how new it is (WM_CURIOSITY), which steers the look-ahead but is never learned as value."""
        out = np.zeros(self.k)
        for a in range(self.k):
            nxt = self.T[a] @ phi
            end = float(np.clip(self.E[a] @ phi, 0, 1))
            later = (self.q(nxt, depth - 1, curious).max() if depth > 1 else self.value(nxt))
            now = float(self.R[a] @ phi) + (float(self.want @ (self.O[a] @ phi)) if self.n_out else 0.0)
            if curious and self.curiosity:
                # curiosity is strongest before anything rewarding is known; after that it fades to a tenth, so a
                # new place no longer outbids known food a few steps away
                now += self.curiosity * (self.curiosity_fade if self.rewarded else 1.0) * self.novelty(nxt)
            out[a] = now + self.gamma * (1 - end) * later
        return out

    def observe(self, phi, a, reward, nxt=None, outcome=None):
        """One real step: phi --a--> nxt (None if the episode ended), with its reward; `outcome`: which outcomes
        happened (one entry per outcome), whose wanted value is then taken out of `reward`."""
        n = self._norm(phi)
        done = nxt is None
        if abs(reward) > 0.1 or (outcome is not None and any(outcome)):
            self.rewarded = True
        if not done:
            self.T[a] += self.lr * np.outer(nxt - self.T[a] @ phi, phi) / n
        if self.n_out and outcome is not None:
            o = np.asarray(outcome, np.float32)
            reward = reward - float(self.want @ o)            # the rest (e.g. the cost of a step) is plain reward
            self.O[a] += self.lr * np.outer(o - self.O[a] @ phi, phi) / n
            later = 0.0 if done else self.gamma * (self.V_out @ nxt)
            self.V_out += self.lr * np.outer(o + later - self.V_out @ phi, phi) / n
        self.R[a] += self.lr * (reward - self.R[a] @ phi) * phi / n
        self.E[a] += self.lr * (float(done) - self.E[a] @ phi) * phi / n
        target = reward + (0.0 if done else self.gamma * float(self.v @ nxt))
        self.v += self.lr * (target - self.v @ phi) * phi / n
        # a memory of places, not of moments: a place already remembered is not stored again, so the layout learned
        # while exploring is not pushed out by many visits to the same few places (note 96)
        i, sim = self._nearest(phi)
        if sim < 0.97:
            self.memory.append(phi.copy())
            self.counts.append(1)
            if len(self.memory) > self.mem_size:
                j = int(self.rng.integers(len(self.memory)))
                self.memory.pop(j)
                self.counts.pop(j)
        else:
            self.counts[i] += 1
        self.dream(self.replay)

    def dream(self, steps):
        """Replay remembered places through the model: each place's value moves toward its best imagined action."""
        if not self.memory:
            return
        for i in self.rng.integers(len(self.memory), size=steps):
            phi = self.memory[i]
            qs = self.q(phi, depth=1, curious=False)
            a = int(np.argmax(qs))
            nxt = self.T[a] @ phi
            end = float(np.clip(self.E[a] @ phi, 0, 1))
            n = self._norm(phi)
            target = float(self.R[a] @ phi) + self.gamma * (1 - end) * float(self.v @ nxt)
            self.v += 0.5 * self.lr * (target - self.v @ phi) * phi / n
            if self.n_out:
                tgt = self.O[a] @ phi + self.gamma * (1 - end) * (self.V_out @ nxt)
                self.V_out += 0.5 * self.lr * np.outer(tgt - self.V_out @ phi, phi) / n

"""
Planning games (plan phase 3, note 96).

DetourGame (Tolman 1948): a 7x7 room split by a wall down the middle with two gaps: in the middle (the short route)
and at the top (a longer one). The character starts on the left (x=1, y=3); food is on the right (x=5, y=3),
unmarked: nothing in the view shows where it is. The first `explore` episodes have no food (it can learn the room's
layout; latent learning). Then food is there. At episode `switch` the middle gap closes: the short route is blocked
and the known longer one is the detour. A planner that has the layout re-routes within a few episodes; a habit keeps
pushing into the old gap until it unlearns it.
Food +1 (ends the episode), each step -0.02, at most 24 steps. Self-motion arrives as obs.motion; each episode's
path starts at the start (obs.body["from_home"]).
"""
import numpy as np

from .core import Game, Observation
from .spatial_games import CELL, HEADINGS, MOVES, WALL

N = 7
START, FOOD = (1, 3), (5, 3)


class DetourGame(Game):
    n_actions = 4
    action_names = ("north", "east", "south", "west")

    def __init__(self, seed=0, switch=40, limit=24, explore=10):
        self.rng = np.random.default_rng(seed)
        self.switch, self.limit, self.explore = switch, limit, explore
        self.episode = -1
        self.pos, self.heading, self.t = START, 0.0, 0

    def gaps(self):
        return (0, 3) if self.episode < self.switch else (0,)

    def blocked(self, x, y):
        if not (0 <= x < N and 0 <= y < N):
            return True
        return x == 3 and y not in self.gaps()

    def _view(self):
        img = np.full((5 * CELL, 5 * CELL, 3), 0.5, np.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = self.pos[0] + dx, self.pos[1] + dy
                side = "n" if y < 0 else "s" if y >= N else "w" if x < 0 else "e" if x >= N else None
                colour = WALL[side] if side else ((0.15, 0.15, 0.15) if self.blocked(x, y) else None)
                if colour is not None:
                    img[(dy + 2) * CELL:(dy + 3) * CELL, (dx + 2) * CELL:(dx + 3) * CELL] = colour
        return img

    def reset(self):
        self.episode += 1
        self.pos, self.heading, self.t = START, 0.0, 0
        return Observation(self._view(), body={"from_home": 1.0}, motion=[])

    def step(self, action):
        x, y = self.pos[0] + MOVES[action][0], self.pos[1] + MOVES[action][1]
        turn = (HEADINGS[action] - self.heading + np.pi) % (2 * np.pi) - np.pi
        self.heading = HEADINGS[action]
        moved = not self.blocked(x, y)
        if moved:
            self.pos = (x, y)
        self.t += 1
        fed = self.pos == FOOD and self.episode >= self.explore
        done = fed or self.t >= self.limit
        info = {"episode": self.episode, "after_switch": self.episode >= self.switch, "fed": fed, "done": done,
                "steps": self.t if done else None, "bumped": not moved, "timeout": done and not fed}
        return (Observation(self._view(), motion=[(turn, 1.0 if moved else 0.0)]), 1.0 if fed else -0.02, done, info)


class RevaluationGame(Game):
    """Outcome revaluation (Adams & Dickinson 1981; Balleine & Dickinson 1998). An open 7x7 room with two unmarked
    foods: A near the start, B farther away. Each is worth what the character wants it (obs.body["want"] = [A, B]).
    With a = switch // 3:
      episodes [0, a)        only B is there      (the way to B is learned)
      episodes [a, 2a)       only A is there      (then a habit of going to A, nearer and more recent)
      episodes [2a, switch)  both, both wanted; the episode ends at the first food eaten (A, nearer, should win)
      episodes >= switch     the character is full of A (want [0, 1]); the episode ends when B is eaten
    A planner that knows which food lies where goes to B from the first test episode; a habit still heads for A.
    Each step -0.02, at most 24 steps. info["outcome"] = [ate A, ate B] this step."""
    n_actions = 4
    action_names = ("north", "east", "south", "west")
    START, FOODS = (3, 5), ((1, 3), (5, 1))

    def __init__(self, seed=0, switch=30, limit=24):
        self.rng = np.random.default_rng(seed)
        self.switch, self.limit = switch, limit
        self.episode = -1
        self.eaten = [False, False]

    def phase(self):
        a = self.switch // 3
        return "B" if self.episode < a else "A" if self.episode < 2 * a else "choice" if self.episode < self.switch else "test"

    def present(self):
        return {"A": [True, False], "B": [False, True]}.get(self.phase(), [True, True])

    def want(self):
        """What each food is worth now (nothing once eaten this episode: the body knows it has just eaten)."""
        base = [0.0, 1.0] if self.phase() == "test" else [1.0, 1.0]
        return [0.0 if self.eaten[k] else base[k] for k in range(2)]

    def _view(self):
        img = np.full((5 * CELL, 5 * CELL, 3), 0.5, np.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = self.pos[0] + dx, self.pos[1] + dy
                side = "n" if y < 0 else "s" if y >= N else "w" if x < 0 else "e" if x >= N else None
                if side:
                    img[(dy + 2) * CELL:(dy + 3) * CELL, (dx + 2) * CELL:(dx + 3) * CELL] = WALL[side]
        return img

    def reset(self):
        self.episode += 1
        self.pos, self.heading, self.t, self.eaten, self.order = self.START, 0.0, 0, [False, False], []
        return Observation(self._view(), body={"from_home": 1.0, "want": self.want()}, motion=[])

    def step(self, action):
        x, y = self.pos[0] + MOVES[action][0], self.pos[1] + MOVES[action][1]
        turn = (HEADINGS[action] - self.heading + np.pi) % (2 * np.pi) - np.pi
        self.heading = HEADINGS[action]
        moved = 0 <= x < N and 0 <= y < N
        if moved:
            self.pos = (x, y)
        self.t += 1
        want = self.want()   # before this step's eating
        outcome = [0.0, 0.0]
        for k, f in enumerate(self.FOODS):
            if self.pos == f and not self.eaten[k] and self.present()[k]:
                self.eaten[k], outcome[k] = True, 1.0
                self.order.append(k)
        reward = float(np.dot(want, outcome)) - 0.02
        ph = self.phase()
        ate_any = any(outcome)
        if ph == "test":
            done = self.eaten[1]
        elif ph == "choice":
            done = ate_any
        else:
            done = ate_any
        timeout = not done and self.t >= self.limit
        done = done or timeout
        info = {"episode": self.episode, "phase": ph, "outcome": outcome, "done": done, "timeout": timeout,
                "steps": self.t if done else None, "order": list(self.order) if done else None,
                "got_wanted": self.eaten[1] if ph == "test" else bool(self.order)}
        return (Observation(self._view(), body={"want": self.want()}, motion=[(turn, 1.0 if moved else 0.0)]), reward,
                done, info)

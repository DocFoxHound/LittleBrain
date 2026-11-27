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
                "steps": self.t if done else None, "bumped": not moved}
        return (Observation(self._view(), motion=[(turn, 1.0 if moved else 0.0)]), 1.0 if fed else -0.02, done, info)

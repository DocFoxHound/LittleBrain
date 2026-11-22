"""
Spatial games (plan phase 2, note 83).

HomingGame: a 7x7 room with coloured walls (north red, east green, south blue, west yellow). Home is the middle cell,
and nothing marks it: the floor looks the same everywhere, and the centre's 3x3 neighbourhood sees no wall at all.
Each episode the character is carried a few random steps away from home (the path arrives as self-motion,
obs.motion, starting at home: obs.body["from_home"]); then it must walk back. Actions move one cell north/east/south/
west (bumping into a wall does nothing). Home pays +1 and ends the episode; each step costs 0.02; after 8 steps the
episode ends. The view is a top-down 5x5 patch around the character (allocentric, north up).
Knowing the way home needs path integration or a place memory: the view alone is ambiguous near home.
"""
import numpy as np

from .core import Game, Observation

N = 7
HOME = (3, 3)
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))          # north, east, south, west (y grows southwards)
HEADINGS = (-np.pi / 2, 0.0, np.pi / 2, np.pi)      # world angle of each move
WALL = {"n": (0.9, 0.15, 0.15), "e": (0.15, 0.8, 0.2), "s": (0.2, 0.3, 0.95), "w": (0.95, 0.9, 0.15)}
CELL = 4


class HomingGame(Game):
    n_actions = 4
    action_names = ("north", "east", "south", "west")

    def __init__(self, seed=0, out=(2, 4), limit=8):
        self.rng = np.random.default_rng(seed)
        self.out, self.limit = out, limit
        self.pos, self.heading, self.t = HOME, 0.0, 0

    def _view(self):
        img = np.full((5 * CELL, 5 * CELL, 3), 0.5, np.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = self.pos[0] + dx, self.pos[1] + dy
                side = "n" if y < 0 else "s" if y >= N else "w" if x < 0 else "e" if x >= N else None
                if side:
                    img[(dy + 2) * CELL:(dy + 3) * CELL, (dx + 2) * CELL:(dx + 3) * CELL] = WALL[side]
        return img

    def _walk(self, a):
        """Move; returns the self-motion step (turn, distance)."""
        x, y = self.pos[0] + MOVES[a][0], self.pos[1] + MOVES[a][1]
        turn = (HEADINGS[a] - self.heading + np.pi) % (2 * np.pi) - np.pi
        self.heading = HEADINGS[a]
        if 0 <= x < N and 0 <= y < N:
            self.pos = (x, y)
            return turn, 1.0
        return turn, 0.0

    def dist(self):
        return abs(self.pos[0] - HOME[0]) + abs(self.pos[1] - HOME[1])

    def reset(self):
        self.pos, self.heading, self.t = HOME, 0.0, 0
        path = []
        for _ in range(100):
            if len(path) >= self.rng.integers(self.out[0], self.out[1] + 1) and self.pos != HOME:
                break
            path.append(self._walk(int(self.rng.integers(4))))
        self.start = self.dist()
        return Observation(self._view(), body={"from_home": 1.0}, motion=path)

    def step(self, action):
        before = self.dist()
        motion = [self._walk(int(action))]
        self.t += 1
        home = self.pos == HOME
        done = home or self.t >= self.limit
        reward = 1.0 if home else -0.02
        info = {"state": before, "closer": self.dist() < before, "home": home, "done": done,
                "start": self.start if done else None}
        body = {"home_scent": 1.0} if home else {}
        return Observation(self._view(), body=body, motion=motion), reward, done, info

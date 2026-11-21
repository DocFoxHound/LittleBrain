"""
The game interface: anything a LittleBrain character can live in.

A game shows the character a picture and takes one action per turn, like a tiny gym environment:

    obs = game.reset()                      # Observation
    obs, reward, done, info = game.step(a)  # a in range(game.n_actions)

Observation.image is a 2D array in [0, 1] (what the character sees; any resolution: the agent samples it
at its retina cells' positions). Observation.body holds optional internal signals (hunger, pain, ...) for
later games. Games know nothing about neurons, so the same game can test any brain, or a baseline agent.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    image: np.ndarray
    body: dict = field(default_factory=dict)
    frames: list = None   # optional movie for this turn (motion): one picture per brain step, looping; `image` is
                          # its first frame, for brains and viewers that take a still picture
    sound: object = None  # optional sound for this turn: a cochleagram (bands x frames, cochlea.py), one frame per
                          # brain step (note 81)
    text: str = None      # optional text heard/read this turn; reaches the brain as sound (cochlea.text_sound)


class Game:
    n_actions: int = 4
    action_names: tuple = ()

    def reset(self) -> Observation:
        raise NotImplementedError

    def step(self, action: int):
        """Returns (observation, reward, done, info)."""
        raise NotImplementedError


def blob_image(centers, size=16, radius=0.13):
    """An image with soft-edged discs at the given (x, y) centers in [0, 1]."""
    y, x = (np.mgrid[0:size, 0:size] + 0.5) / size
    img = np.zeros((size, size), np.float32)
    for cx, cy in centers:
        img = np.maximum(img, ((x - cx) ** 2 + (y - cy) ** 2 < radius ** 2).astype(np.float32))
    return img

"""Small games for LittleBrain characters (notes/29). See core.Game for the interface."""
from .core import Game, Observation  # noqa: F401
from .signal_game import SignalGame  # noqa: F401
from .danger_game import DangerGame  # noqa: F401

# Games the tools can load by name (viewer --game, experiment runners). Add yours here.
from .vision_games import ColourGame, MotionGame  # noqa: F401,E402
REGISTRY = {"colour": ColourGame, "motion": MotionGame, "signal": SignalGame, "danger": DangerGame}

from .regional_games import DelayedCueGame, ForageGame, TimingGame

REGISTRY.update(forage=ForageGame, delayed=DelayedCueGame, timing=TimingGame)

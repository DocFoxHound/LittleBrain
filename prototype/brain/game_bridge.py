"""Local JSON Lines interface for one persistent LittleBrain character."""
import argparse
from contextlib import redirect_stdout
import json
import math
import sys

import numpy as np
from run_signal_game import Brain, BrainAgent, prepare
from games.core import Observation


def image_array(message):
    raw = np.asarray(message["image"])
    fmt = message.get("format", "gray")
    if raw.dtype.kind not in "iuf" or not raw.size or not np.isfinite(raw).all():
        raise ValueError("expected a finite numeric image")
    if fmt == "rgb8" and raw.ndim == 3 and raw.shape[2] == 3:
        pixels = raw @ np.array([0.299, 0.587, 0.114])
    elif fmt in {"gray", "gray8"} and raw.ndim == 2:
        pixels = raw
    else:
        raise ValueError("invalid image format or shape")
    limit = 1 if fmt == "gray" else 255
    if raw.min() < 0 or raw.max() > limit:
        raise ValueError("pixel outside permitted range")
    return pixels.astype(np.float32) / limit


def number(value, low=0, high=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("invalid numeric field")
    return float(value)


class Session:
    def __init__(self, brain, actions, seed, learn=True, sleep_every=0):
        if not 1 <= len(actions) <= 8 or len(set(actions)) != len(actions) or not all(actions):
            raise ValueError("supply one to eight distinct nonempty actions")
        self.actions = list(actions)
        self.agent = BrainAgent(brain, len(actions), seed=seed, learn=learn)
        self.sleep_every = sleep_every

    def request(self, msg):
        if not isinstance(msg, dict):
            raise ValueError("expected an object")
        agent = self.agent
        if msg.get("type") == "observe":
            raw_body = msg.get("body", {})
            if not isinstance(raw_body, dict) or set(raw_body) - {"hunger", "thirst", "social_contact"}:
                raise ValueError("invalid body field")
            body = {k: number(v) for k, v in raw_body.items()}
            index = agent.begin_turn(Observation(image_array(msg), body))
            return {"type": "action", "turn": agent.turn + 1, "index": index, "action": self.actions[index]}
        if msg.get("type") == "outcome":
            turn = msg.get("turn")
            if type(turn) is not int or turn != agent.turn + 1:
                raise ValueError("stale or missing turn")
            info = {k: number(msg[k]) for k in ("food_reward", "timing_error") if k in msg}
            if "food_contact" in msg:
                if type(msg["food_contact"]) != bool:
                    raise ValueError("food_contact must be boolean")
                info["food_contact"] = msg["food_contact"]
            agent.finish_turn(number(msg["reward"], -1, 1), info, self.sleep_every)
            return {"type": "applied", "turn": turn}
        raise ValueError("unknown request")


def send(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-learn", action="store_true")
    parser.add_argument("--load-brain")
    parser.add_argument("--save-brain")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        brain = Brain.load(args.load_brain) if args.load_brain else prepare(args.seed)
        session = Session(brain, args.actions, args.seed, not args.no_learn)
    send({"type": "ready", "version": 1, "actions": session.actions, "neurons": brain.n_neurons})
    for line in sys.stdin:
        try:
            with redirect_stdout(sys.stderr):
                reply = session.request(json.loads(line))
        except (ValueError, TypeError, KeyError) as error:
            reply = {"type": "error", "message": str(error)}
        send(reply)
    if session.agent.pending is not None:
        parser.exit(1, "incomplete turn at shutdown\n")
    if args.save_brain:
        brain.save(args.save_brain)


if __name__ == "__main__":
    main()

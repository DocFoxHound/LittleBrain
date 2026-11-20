"""Measure the default game brain and matched controls from a common checkpoint."""
import argparse
import json
import numpy as np

from run_signal_game import Brain, BrainAgent, prepare
from games import REGISTRY

LESIONS = {"signal": {"sensory -> striatum"}, "danger": {"BLA", "CeL", "CeM"},
           "forage": {"HYP"}, "delayed": {"PFC", "MD"}, "timing": {"CB", "DCN", "IO"}}


def mean(events):
    values = list(events)
    return round(float(np.mean(values)), 4) if values else None


def metrics(game, block):
    if game == "signal":
        return dict(accuracy=mean(e["correct"] for e in block))
    if game == "danger":
        groups = {"food_reward": (False, True), "food_miss": (False, False),
                  "danger_hide": (True, True), "danger_hurt": (True, False)}
        return dict(hurt=mean(e["reward"] < 0 for e in block if e["danger"]),
                    fed=mean(e["reward"] > 0 for e in block if not e["danger"]),
                    dopamine={k: mean(e["dopamine"] for e in block if (e["danger"],e["correct"]) == v)
                              for k,v in groups.items()})
    if game == "forage":
        return dict(hungry_eat=mean(e["food_contact"] for e in block if e["hunger"] >= .65),
                    sated_eat=mean(e["food_contact"] for e in block if e["hunger"] <= .35))
    if game == "delayed":
        return dict(accuracy=mean(e["correct"] for e in block if e["choice"]),
                    choices=sum(e["choice"] for e in block))
    return dict(error=mean(e["timing_error"] for e in block if e["ended"]),
                completions=sum(e["ended"] for e in block))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game", choices=REGISTRY, default="signal")
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--condition", choices=["learning", "off", "lesion"], default="learning")
    parser.add_argument("--load-brain", help="trusted local checkpoint")
    parser.add_argument("--prepare-only", metavar="PATH")
    args = parser.parse_args()
    b = Brain.load(args.load_brain) if args.load_brain else prepare(args.seed)
    if args.prepare_only:
        b.save(args.prepare_only)
        return
    g = REGISTRY[args.game](seed=args.seed)
    if args.condition == "lesion":
        b.ablated.update(LESIONS[args.game])
    a = BrainAgent(b, g.n_actions, seed=args.seed, learn=args.condition != "off")
    log = []
    for start in range(0, args.turns, 50):
        a.play(g, min(50, args.turns - start), log=log)
        print(json.dumps(dict(seed=args.seed, game=args.game, condition=args.condition, turns=len(log),
                              **metrics(args.game, log[-100:]))), flush=True)


if __name__ == "__main__":
    main()

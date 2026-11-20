"""Check a newly reconstructed stage before enabling the next factory."""
import argparse
import copy
import numpy as np
from run_signal_game import Brain, S


def check(factory, source, target, seed):
    brain = Brain(getattr(S, factory)(), seed=seed)
    quiet = {k: np.zeros(len(brain.pos[k]), np.float32) for k in brain.spec.inputs}
    silent = copy.deepcopy(brain)
    active_input = {**quiet, source: np.ones(len(brain.pos[source]), np.float32)}
    silent_input = {**quiet, source: np.zeros(len(brain.pos[source]), np.float32)}
    for _ in range(6):
        brain.step(active_input)
        silent.step(silent_input)
    assert np.isfinite(brain.r).all()
    response = float(np.max(np.abs(brain.g[target].rate - silent.g[target].rate)))
    assert response > 1e-6, (source, target, response)
    brain.ablated.add(target)
    brain.step(quiet)
    assert np.count_nonzero(brain.g[target].rate) == 0
    print(f"{factory} seed {seed}: {brain.n_neurons} neurons; {target} active and silenced", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("factory")
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    for seed in (0, 1, 2):
        check(args.factory, args.source, args.target, seed)

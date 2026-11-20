"""Audit each reconstructed terminal against its silenced copy before advancing."""
import numpy as np
from run_signal_game import Brain, S
from cells import ExcitatoryGroup
from regional_circuits import CIRCUITS, extend

for circuit in CIRCUITS:
    for seed in (0, 1, 2):
        b = Brain(extend(S.stage_insula(), circuit.key), seed=seed)
        key, target = circuit.key, circuit.target
        b.r[b.index[key]] = 1.
        exc = isinstance(b.g[target], ExcitatoryGroup)
        def terminal():
            return b._channels({}, exc)[target].get(circuit.terminal, 0.)
        intact = terminal()
        b.ablated.add(f"{key} -> {target}")
        delta = np.mean(intact - terminal())
        assert np.isclose(delta, circuit.weight, atol=1e-5), (key, seed, delta)
    print(key, b.n_neurons, "3/3 terminal signs and amplitudes verified", flush=True)

"""Pulse each new population's afferents and efferent in a matched CPU brain."""
import argparse
import copy
import json
import numpy as np
from run_signal_game import Brain, S
from catalog_placements import PLACEMENTS
from check_regional_audit import composition, projection, territory, TERRITORIES
from cells import ExcitatoryGroup


def paired(brain, source, target, sign, channel):
    # Use membrane drive for dendritic currents and emitted activity for an
    # axon veto. A driven receiver can reveal inhibition below firing threshold.
    soma = isinstance(brain.g[target], ExcitatoryGroup) and channel != "ais"
    for bias in (0., .8, 2.):
        delta = projection(brain, source, target, soma=soma, drive_bias=bias)
        if sign * delta > 1e-7:
            return round(delta, 7), bias
    raise AssertionError((source, target, sign, channel, delta))


def check(brain, name):
    inventory = composition(brain)
    rows = {}
    for c in PLACEMENTS[name]:
        sub = TERRITORIES[territory(c.key)]
        assert not inventory[sub.name]["missing_types"], sub.name
        assert brain.g[c.key].c.name == c.cell.name
        inputs = [c.source] + ([c.apical] if c.apical else [])
        afferents = [paired(brain, src, c.key, np.sign(c.input_weight), "drive") for src in inputs]
        sign = -1 if c.terminal in ("pv", "sst", "dend", "volume", "ais", "inh") else np.sign(c.weight)
        out, bias = paired(brain, c.key, c.target, sign, c.terminal)
        lesion = copy.copy(brain)
        lesion.ablated = brain.ablated | {f"{c.key} -> {c.target}"}
        # A fixed presynaptic pulse must disappear when just its terminal is cut.
        soma = isinstance(brain.g[c.target], ExcitatoryGroup) and c.terminal != "ais"
        muted = projection(lesion, c.key, c.target, soma=soma, drive_bias=bias)
        assert abs(muted) < 1e-7, (c.key, c.target, muted)
        rows[c.key] = dict(inputs=afferents, output=out, bias=bias, muted=muted)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=list(PLACEMENTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-brain", help="trusted checkpoint of the current reference brain")
    args = parser.parse_args()
    brain = Brain.load(args.load_brain) if args.load_brain else Brain(S.full_brain(), seed=args.seed)
    if not args.load_brain:
        for step in range(12):
            brain.step({"retina": (brain.tile["retina"] % 8 < 4) * (step < 8)})
    names = [args.region] if args.region else PLACEMENTS
    print(json.dumps({name: check(brain, name) for name in names}))

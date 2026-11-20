"""Integration checks on a developed full brain. Pass --load-brain for a trusted checkpoint."""
import argparse
from pathlib import Path
import sys
import tempfile
import numpy as np

from run_signal_game import Brain, BrainAgent, prepare
from games import SignalGame
import celltypes as C

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype/viz"))
sys.path.insert(0, str(ROOT / "integrations/python"))
from brainview import TERRITORY, territory_of
from littlebrain_client import GameClient


def anatomy(b):
    catalog = {v.name for v in vars(C).values() if isinstance(v, C.CellType)}
    assert {grp.c.name for grp in b.g.values()} == catalog
    assert all(territory_of(k) in TERRITORY for k in b.g)
    incoming, outgoing = set(), set()
    for path in [*b.spec.synaptic, *b.spec.pooled]:
        incoming.update(path.dst if isinstance(path.dst, list) else [path.dst])
        outgoing.update(path.src if isinstance(path.src, list) else [path.src])
    # VTA supplies phasic dopamine to plasticity. MCH regulates replay ACh.
    assert set(b.g) <= incoming
    assert set(b.g) <= outgoing | {"VTA.DA", "LH.MCH"}
    print(f"{b.n_neurons} neurons, {b.storage()['synapses']} sampled synapses, "
          f"{len(b.g)} mapped populations, {len(catalog)} catalog types")
    assert b.n_neurons == 14656


def night(b, seed):
    banks = [bank for grp in b.excitatory().values() for bank in grp.banks()]
    before = [bank.src.copy() for bank in banks]
    n = banks[0].night
    a = BrainAgent(b, 4, seed=seed)
    # Exercise the scheduled boundary in one turn on the normally developed brain.
    a.turns_played = 63
    a.play(SignalGame(seed=seed), 1)
    changed = sum(np.count_nonzero(bank.src != prev) for bank, prev in zip(banks, before))
    assert banks[0].night == n + 1 and not b._log and changed > 0
    assert np.isfinite(b.r).all()
    print(f"Scheduled night completed. {changed} sampled partners changed.")


def bridge(b, seed):
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        initial, saved = directory / "input.brain", directory / "output.brain"
        b.save(initial)
        command = [sys.executable, str(ROOT / "prototype/brain/game_bridge.py"), "--load-brain", str(initial),
                   "--save-brain", str(saved), "--seed", str(seed), "--actions", "left", "right"]
        with GameClient(command) as client:
            assert client.ready["neurons"] == 14656
            action = client.observe([[10, 240], [128, 0]], format="gray8", body={"thirst": .8})
            assert action in ("left", "right")
            try:
                client.request(type="outcome", turn=2, reward=1.)
            except ValueError:
                pass
            else:
                raise AssertionError("bridge applied a stale outcome")
            client.outcome(1.)
        checkpoint = Brain.load(saved)
        assert np.isfinite(checkpoint.r).all()
        assert not np.array_equal(b.sensory_striatal, checkpoint.sensory_striatal)
        print("External observe/outcome, stale-turn retry and learned checkpoint round trip passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-brain")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    b = Brain.load(args.load_brain) if args.load_brain else prepare(args.seed)
    anatomy(b)
    night(b, args.seed)
    bridge(b, args.seed)


if __name__ == "__main__":
    main()

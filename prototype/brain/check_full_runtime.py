"""Integration checks on a developed full brain. Pass --load-brain for a trusted checkpoint."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
import numpy as np

from run_signal_game import Brain, BrainAgent, prepare
from games import SignalGame
from game_controls import Controls
import celltypes as C
from cells import InhibitoryGroup

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


def regional_responses(b):
    paths = [("CA3.pyr", "CA2.pyr"), ("EC.L3", "CA2.pyr"), ("CA2.pyr", "CA1.pyr"),
             ("CA1.pyr", "SUB.pyr"), ("INS.IT", "ACC.IT"), ("ACC.IT", "STR.D1"),
             ("PUL.matrix", "A.L23"), ("CLA.pyr", "A.PV"), ("CB.Purkinje", "DCN.proj")]
    quiet = {key: np.zeros(len(b.pos[key]), np.float32) for key in b.spec.inputs}
    for source, target in paths:
        response = []
        for pulse in (0., 1.):
            trial = copy.deepcopy(b)
            for _ in range(4):
                trial.step({**quiet, source: np.full(len(b.pos[source]), pulse, np.float32)})
            response.append(trial.g[target].rate.copy())
        assert np.max(np.abs(response[1] - response[0])) > 1e-6, (source, target)
    print(f"{len(paths)} paired regional activity probes passed.")



def slow_terminals(b):
    trial = copy.deepcopy(b)
    groups = {k: g for k, g in trial.g.items() if isinstance(g, InhibitoryGroup) and g.c.slow}
    assert groups
    trial.step({"retina": 1., "MOT": 0., "US": 0.}, signals={"hunger": 1., "social_contact": 1.})
    for key, group in groups.items():
        np.testing.assert_allclose(trial.r[trial.index[key]], group.level, err_msg=key)
    assert not np.allclose(trial.g["HYP.AgRP"].level, trial.g["HYP.AgRP"].rate)
    trial.ablated.update(groups)
    trial.step({})
    assert all(np.all(trial.r[trial.index[key]] == 0) for key in groups)
    print(f"{len(groups)} slow outputs transmitted and silenced.")


def thalamic_brakes(b):
    from check_regional_audit import projection
    for source, target, driver in (("GPi", "VA.relay", "DCN.proj"), ("SNr.out", "MD.relay", "PFC.IT")):
        delta = projection(b, source, target, {driver: .5})
        assert delta < -1e-6, (source, target, delta)
    print("Both basal ganglia brakes reduced thalamic firing.")


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


def game_controls():
    keys = Controls({"controls": [{"name": "walk", "kind": "key"},
                                  {"name": "turn", "kind": "axis", "values": [-1, 0, 1]}]})
    assert keys.select(1) == [{"kind": "key", "name": "walk", "value": "hold"}]
    assert keys.select(3) == [{"kind": "axis", "name": "turn", "value": -1}]
    assert keys.reset() == [{"kind": "key", "name": "walk", "value": "release"},
                            {"kind": "axis", "name": "turn", "value": 0}]
    for value in (0, 65, 0xd7ff, 0xe000, 0x1f43a, 0x10ffff):
        text = Controls({"controls": [{"name": "chat", "kind": "text"}]})
        index = value if value < 0xd800 else value - 0x800
        events = text.select(0)
        while not events:
            slot = next(i for i, item in enumerate(text.stack[-1])
                        if item["scalar_range"][0] <= index < item["scalar_range"][1])
            events = text.select(slot)
        assert events == [{"kind": "text", "name": "chat", "value": chr(value)}]
        assert text.panel()["depth"] == 0
    print("Held inputs, axis reset and Unicode scalar entry passed.")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-brain")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    b = Brain.load(args.load_brain) if args.load_brain else prepare(args.seed)
    anatomy(b)
    slow_terminals(b)
    thalamic_brakes(b)
    regional_responses(b)
    night(b, args.seed)
    bridge(b, args.seed)
    game_controls()


if __name__ == "__main__":
    main()

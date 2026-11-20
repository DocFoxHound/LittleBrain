# Game Bridge v1

Run the bridge as a child process from the repository root. It exchanges one UTF-8 JSON object per line over standard input and output. Preparation messages go to standard error. The first reply is a `ready` object with the protocol version, neuron count and action names. The process owns one developing brain for the whole session.

```sh
python prototype/brain/game_bridge.py --actions left right wait
```

Send an observation, apply the returned action in the game, then send its outcome. The example turn below has a two-by-two grayscale camera. Use the turn number in the action reply.

```jsonl
{"type":"observe","format":"gray","image":[[0,1],[1,0]],"body":{"hunger":0.8}}
{"type":"outcome","turn":1,"reward":0.5}
```

The camera accepts two-dimensional gray values from zero to one, gray8 from zero to 255, or RGB triplets in rgb8 format. The latter two use the same 0–255 range. Body fields are optional and use values from zero to one. Signed reward is in the range −1 to +1. For ingestion, pass `food_contact` and the food component `food_reward` separately from injury or other feedback. The optional body fields are `hunger`, `thirst` and `social_contact`. Intermediate frames with no outcome can set `feedback` to false and send zero reward. A host that ignores a motor command during a passive observation phase can report `action_applied` as false. The PFC/MD loop then holds the observed context until a command is applied. A timing task can supply a normalized `timing_error`. After an actual movement it may also give a signed `timing_offset` from −1 (early) to +1 (late). Both are outcome fields. Each accepted outcome returns `applied`. Invalid input yields an error reply, and the host can correct and retry the pending turn.

EOF after a complete turn closes the process. Use `--save-brain PATH` and `--load-brain PATH` for neural checkpoints. Only load trusted local files, as checkpoints use Python pickle. A checkpoint contains the neural state. The host retains its own game state, turn counters and controls.

[Python client](../integrations/python/littlebrain_client.py) · [Game controls](46-game-controls.md)

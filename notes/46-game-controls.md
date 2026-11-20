# Game control layouts

A game with one to eight discrete actions can supply their names with `--actions`. For a larger control surface, pass `--controls layout.json`. The host defines the vocabulary of physical controls and implements the events returned by the bridge.

```json
{"controls":[{"name":"forward","kind":"key"},{"name":"jump","kind":"button"},{"name":"turn","kind":"axis","values":[-1,0,1]},{"name":"cursor","kind":"pointer","values":[[-1,0],[1,0]]},{"name":"chat","kind":"text"}]}
```

The overlay adds a visual strip containing stable patterns for the current menu. Seven motor slots select entries and an eighth goes back. The returned panel describes the menu depth, available slots, held inputs and axis states. Key and button operations include a press, hold and release. An axis retains its chosen value. The host interprets pointer samples in its own coordinates. Text entry traverses Unicode scalar ranges, skipping the surrogate range. The host sends the resulting character to the focused field. Reading, writing and vocabulary are learning tasks.

Apply returned events in order. Call `reset_controls` between completed turns when resetting the game or closing a session, and apply its release and neutral-axis events. On unexpected disconnection the host must release any inputs it still owns.

[Game Bridge v1](45-game-bridge.md) · [Client](../integrations/python/littlebrain_client.py)

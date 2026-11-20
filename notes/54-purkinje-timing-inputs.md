# Keep Purkinje teaching and inhibitory inputs active during timing

Once timing learning initialized its parallel-fiber weights, the learned simple-spike rate replaced the whole Purkinje response. An olive pulse could still update the timing weights but could no longer trigger a complex spike in the default one-cell-per-tile population. Basket and stellate cells also lost their influence on it.

The timing ensemble now applies the conventional dendritic inhibition to its learned simple-spike drive. It then uses the same climbing-fiber complex-spike and axon-initial-segment veto calculation as an ordinary Purkinje cell. These roles correspond to CB-5 and CB-6 in the [connection plan](regions/connections.md).

| Pulse to timing-mode Purkinje cells | Pre-fix, all seeds | Seed 0 after | Seed 1 after | Seed 2 after |
|---|---:|---:|---:|---:|
| Olive, climbing fiber | 0 | +0.3813 | +0.3771 | +0.3798 |
| Basket, axon inhibition | 0 | −0.2313 | −0.2308 | −0.2314 |
| Stellate, dendritic inhibition | 0 | −0.1380 | −0.1362 | −0.1375 |

Each change is relative to silencing that source with the same initial state and timing weights. The measurements isolate direct spike transmission from the separate weight update.

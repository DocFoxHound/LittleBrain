
**forward: milliseconds per step, 100,000 neurons (25.6 M synapses), 4,096 inputs**

| input activity | elig. union | dendrite 1T | axon 1T | bitslice 1T | dendrite 3T | axon 3T | bitslice 3T |
|---|---|---|---|---|---|---|---|
| 1% | 4% | 34.5 | 2.1 | 8.6 | 12.6 | 0.9 | 3.6 |
| 2% | 8% | 34.6 | 2.4 | 11.3 | 12.5 | 1.0 | 5.1 |
| 5% | 19% | 34.1 | 4.4 | 12.6 | 12.4 | 1.7 | 4.8 |
| 10% | 35% | 34.2 | 6.4 | 12.6 | 12.3 | 2.6 | 5.5 |
| 20% | 58% | 34.3 | 10.8 | 12.7 | 12.4 | 4.4 | 5.5 |
| 50% | 94% | 33.3 | 21.7 | 12.6 | 12.5 | 8.6 | 4.9 |
| 100% | 100% | 32.7 | 36.8 | 12.3 | 12.4 | 13.8 | 4.7 |

NumPy baseline, same scenario: 844 ms (1 thread).

**modulate (tick first): milliseconds per step, 100,000 neurons (25.6 M synapses), 4,096 inputs**

| input activity | elig. union | dendrite 1T | axon 1T | bitslice 1T | dendrite 3T | axon 3T | bitslice 3T |
|---|---|---|---|---|---|---|---|
| 1% | 4% | 94.2 | 5.9 | 53.0 | 33.2 | 3.0 | 19.1 |
| 2% | 8% | 93.9 | 9.5 | 52.6 | 33.1 | 4.4 | 19.2 |
| 5% | 19% | 93.0 | 19.8 | 51.8 | 32.7 | 8.3 | 19.0 |
| 10% | 35% | 93.2 | 35.6 | 52.0 | 33.0 | 13.4 | 19.1 |
| 20% | 58% | 92.3 | 56.5 | 51.6 | 32.8 | 21.0 | 19.0 |
| 50% | 94% | 90.8 | 87.4 | 50.8 | 32.4 | 32.3 | 19.0 |
| 100% | 100% | 90.0 | 92.0 | 50.7 | 32.1 | 33.9 | 19.4 |

NumPy baseline, same scenario: 1954 ms (1 thread).

**Throughput, 2% activity (synapses of the whole population per second; events/s for axon-major)**

| layout | step | 1 thread | 3 threads | speedup vs NumPy (1T / 3T) |
|---|---|---|---|---|
| dendrite | forward | 0.74 G/s | 2.05 G/s | 24x / 67x |
| axon | forward | 10.71 G/s | 24.91 G/s (0.21 / 0.50 G events/s) | 353x / 821x |
| bitslice | forward | 2.27 G/s | 5.04 G/s | 75x / 166x |
| dendrite | modulate | 0.27 G/s | 0.77 G/s | 21x / 59x |
| axon | modulate | 2.70 G/s | 5.85 G/s (0.21 / 0.45 G events/s) | 206x / 446x |
| bitslice | modulate | 0.49 G/s | 1.33 G/s | 37x / 102x |

**Steps that touch every synapse (ms)**

| layout | catch-up 1T | catch-up 3T | sleep 1T | sleep 3T |
|---|---|---|---|---|
| dendrite | 57 | 21 | 132 | 47 |
| axon | 57 | 21 | 163 | 60 |
| bitslice | 69 | 26 | 110 | 40 |

NumPy baseline sleep: 2551 ms (catch-up is part of it).

Axon-major forward with ZERO active inputs (the per-neuron floor): 1.23 ms (1T), 0.49 ms (3T).

**Memory**

| layout | bytes/neuron (stored) | peak temp. forward | modulate | sleep |
|---|---|---|---|---|
| dendrite | 1041 | 0.6 MiB | 2.2 MiB | 0.4 MiB |
| axon | 1565 | 0.5 MiB | 2.8 MiB | 12.1 MiB |
| bitslice | 1041 | 0.4 MiB | 0.6 MiB | 0.4 MiB |
| NumPy baseline | 1041 | 684 MiB | 1175 MiB | 1294 MiB |

**Cache: ns per synapse vs population size (1 thread, 10% activity)**

| neurons | synapse MiB | stream_read | dendrite_fwd | axon_fwd | bitslice_fwd | dendrite_cu | bitslice_cu |
|---|---|---|---|---|---|---|---|
| 2,000 | 2 | 0.07 | 1.28 | 0.27 | 0.52 | 2.21 | 2.74 |
| 8,000 | 8 | 0.13 | 1.26 | 0.27 | 0.52 | 2.24 | 2.74 |
| 23,000 | 22 | 0.21 | 1.37 | 0.28 | 0.55 | 2.45 | 3.00 |
| 64,000 | 62 | 0.19 | 1.42 | 0.26 | 0.55 | 2.44 | 2.97 |
| 200,000 | 195 | 0.19 | 1.38 | 0.26 | 0.52 | 2.34 | 2.86 |

**Axon-major block (tile) size, 100,000 neurons, 1 thread (ms)**

| block (neurons) | accumulator | index bytes/neuron | fwd 2% | mod 2% | fwd 20% | mod 20% |
|---|---|---|---|---|---|---|
| 256 | 10 KiB | 192 | 6.47 | 16.1 | 17.13 | 81.2 |
| 1024 | 40 KiB | 48 | 3.50 | 11.0 | 13.42 | 63.2 |
| 4096 | 160 KiB | 12 | 2.78 | 9.1 | 10.87 | 59.2 |
| 8192 | 320 KiB | 6 | 2.65 | 8.5 | 10.75 | 59.5 |

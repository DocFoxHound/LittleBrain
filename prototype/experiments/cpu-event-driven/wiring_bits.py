"""
Can axon-major wiring live in the 4 freed gen bits? (analysis, no kernels)

Axon-major stores synapses grouped by input, so the PRE side is free (it is the storage position),
but each synapse still needs its POST side: which (neuron, branch) of its block it sits on. Explicit
16-bit ids cost 2 bytes per synapse. Alternative: keep each bucket sorted by post id and store only
the GAP to the previous synapse in the 4 free bits (0..15). A gap larger than 15 needs extra
"spacer" words (a dead synapse that only advances the position).

The mean gap is (inputs the block draws from) / (slots per branch), so this needs LOCALITY:
blocks must draw from a small region of the input space.

Run: .venv/bin/python prototype/experiments/cpu-event-driven/wiring_bits.py
"""

import common

import json

import numpy as np

BLK, B, S = 4096, 8, 32           # one block: 4,096 neurons x 8 branches x 32 slots
POSTS = BLK * B


def spacer_overhead(region, bits=4, seed=0):
    """Random wiring of one block onto `region` inputs; extra words per synapse for `bits`-bit gaps."""
    rng = np.random.default_rng(seed)
    post = np.repeat(np.arange(POSTS, dtype=np.int64), S)
    pre = rng.integers(0, region, post.size)
    order = np.lexsort((post, pre))
    pre, post = pre[order], post[order]
    first = np.r_[True, pre[1:] != pre[:-1]]
    gap = np.where(first, post, post - np.r_[0, post[:-1]])
    step = (1 << bits) - 1
    spacers = np.maximum(0, -(-gap // step) - 1)      # ceil(gap/step) - 1
    return float(spacers.sum() / post.size), float(gap.mean())


if __name__ == "__main__":
    out = []
    print(f"block of {BLK} neurons x {B} branches x {S} slots; 4-bit gap code in the freed gen bits")
    print(f"{'inputs per block':>18}{'mean gap':>10}{'spacers/syn':>13}{'bytes/syn':>11}   (explicit 16-bit post: 6.00)")
    for region in (64, 128, 256, 512, 1024, 4096):
        sp, mg = spacer_overhead(region)
        out.append(dict(region=region, mean_gap=mg, spacers_per_synapse=sp, bytes_per_synapse=4 * (1 + sp)))
        print(f"{region:>18}{mg:>10.1f}{sp:>13.3f}{4 * (1 + sp):>11.2f}")
    with open(f"{common.HERE}/results_wiring_bits.json", "w") as f:
        json.dump(out, f, indent=1)

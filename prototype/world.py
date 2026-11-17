"""
A world with structure (notes/24): things are built from shared, recurring parts.

Real experience is not a stream of unrelated random patterns. An animal has ears, a coat colour, a call;
a face has eyes, a nose, a mouth; a word has syllables. The same parts recur across many things, and a
part's features always arrive together. That co-occurrence is what lets inputs cluster on a dendrite.

  part:   a fixed group of inputs that always fire together (e.g. "pointed ears", "red", "a bark")
  object: a few parts + a few details of its own (e.g. "the neighbour's dog")
  seeing an object: its parts and details, with some features missing (variation) and a little noise

Defaults give about 5% of inputs active per object, like test_lean_capacity's random patterns.
"""

import numpy as np


class PartsWorld:
    def __init__(self, rng, n_inputs=2048, n_parts=32, part_size=24, parts_per_object=3, details=30,
                 dropout=0.1, noise=4):
        self.rng, self.n_inputs, self.dropout, self.noise = rng, n_inputs, dropout, noise
        order = rng.permutation(n_inputs)
        self.parts = order[:n_parts * part_size].reshape(n_parts, part_size)
        self.detail_pool = order[n_parts * part_size:]
        self.parts_per_object, self.details = parts_per_object, details

    def make_objects(self, n):
        """Each object: which parts it has, and its own details. Returns (part ids, input sets)."""
        parts = np.array([self.rng.choice(len(self.parts), self.parts_per_object, replace=False) for _ in range(n)])
        sets = [np.concatenate([self.parts[p].ravel(), self.rng.choice(self.detail_pool, self.details, replace=False)])
                for p in parts]
        return parts, sets

    def see(self, inputs, exact=False):
        x = np.zeros(self.n_inputs, np.float32)
        if exact:
            x[inputs] = 1.0
            return x
        x[inputs[self.rng.random(len(inputs)) >= self.dropout]] = 1.0
        x[self.rng.choice(self.n_inputs, self.noise, replace=False)] = 1.0
        return x

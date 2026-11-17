"""
A scenario where sparsity is real (see the "Watch out" note in BRIEF.md).

* Neurons are grouped in COLUMNS of `npc` (128) neurons. Every neuron of a column draws all its
  synapses from the column's own `col_size` (64) inputs, via the same hash wiring as the
  prototype (target = column_base + hash(id, slot, gen) % col_size). Input space = N/2 inputs
  (4 million for 8M neurons).
* Columns belong to ASSEMBLIES of 8 columns (1,024 neurons) that switch on together: an object,
  a place, a word.
* ATTENTION keeps A = active_frac x n_assemblies assemblies on. Each step, each attended assembly
  is replaced by a random other one with probability `churn` (dwell time ~ 1/churn steps).
  Inputs of an active column are on with probability p_in (0.2), all others are exactly 0.
* LAYOUT: where each column sits in storage. "clustered": an assembly's 8 columns fill exactly one
  1,024-neuron storage block. "scattered": columns placed at random. A storage block is part of
  the working set if any of its columns is active.
"""

import numpy as np


class AssemblyScenario:
    def __init__(self, n_neurons, *, npc=128, col_size=64, cols_per_asm=8, block_size=1024, active_frac=0.05,
                 churn=0.02, noise_frac=0.0, seed=0, col_slot=None):
        self.npc, self.col_size, self.bs = npc, col_size, block_size
        self.n_cols = n_neurons // npc
        self.n_asm = self.n_cols // cols_per_asm
        self.asm_cols = np.arange(self.n_asm * cols_per_asm).reshape(self.n_asm, cols_per_asm)
        self.rng = np.random.default_rng(seed)
        self.A = max(1, int(round(active_frac * self.n_asm)))
        self.churn = churn
        self.noise = int(round(noise_frac * self.A * cols_per_asm))
        self.active = self.rng.choice(self.n_asm, self.A, replace=False)
        self.is_on = np.zeros(self.n_asm, bool)
        self.is_on[self.active] = True
        self.set_layout(np.arange(self.n_cols) if col_slot is None else col_slot)

    def set_layout(self, col_slot):
        """col_slot[c] = storage position (in columns) of column c."""
        self.col_slot = np.asarray(col_slot, np.int64)
        self.col_block = (self.col_slot * self.npc) // self.bs

    def storage_ids(self):
        """ids[p] = neuron id stored at storage position p (the GPU population needs this)."""
        inv = np.empty(self.n_cols, np.int64)
        inv[self.col_slot] = np.arange(self.n_cols)
        p = np.arange(self.n_cols * self.npc)
        return (inv[p // self.npc] * self.npc + p % self.npc).astype(np.uint32)

    def step(self):
        """Advance attention one step. Returns (active columns, active storage blocks)."""
        swap = np.nonzero(self.rng.random(self.A) < self.churn)[0] if self.A < self.n_asm else []
        for i in swap:
            while True:
                a = int(self.rng.integers(self.n_asm))
                if not self.is_on[a]:
                    break
            self.is_on[self.active[i]] = False
            self.is_on[a] = True
            self.active[i] = a
        cols = self.asm_cols[self.active].ravel()
        if self.noise:
            cols = np.concatenate([cols, self.rng.integers(0, self.n_cols, self.noise)])
        cols = np.unique(cols)
        return cols, np.unique(self.col_block[cols])

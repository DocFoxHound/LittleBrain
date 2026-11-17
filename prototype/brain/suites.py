"""
Two neuron suites (notes/24), sized per cell type from biology (synapse_budget.py, notes/regions/synapse-budget.md).

  standard:  baseline 4096 synapses x 16 levels (a human cortical pyramidal cell, ~7,000-10,000 synapses).
             Other types scale with their biological synapse count; high-memory types get 32 levels.
  low:       the experimental "low-synapse suite": the same types at 1/16 the synapses (baseline 256 x 16),
             to find out whether a brain of small cells is functionally different or similar.
  biological: the "Biological Synapse Count Scale", the official testing scale. Each type's ceiling is the
             power of two nearest its biological (upper) synapse count, capped at 32,768 (Purkinje exempt:
             524,288; L2/3 IT ~30,000 -> 32,768). Every type then gets the same fraction of its ceiling,
             set by the size of the simulated brain and calibrated on real animals (human_fraction):
             a worm-sized brain gets ~0.25%, fly ~3.6%, mouse ~33%, human (86 billion) 100%.

In a brain, every suite can be scaled to the network (federation.Brain(scale=True)): a cell gets no more
synapses than about 4 contacts per partner within its reach, so a small simulated brain uses a fraction of
the ceiling, and cells grow toward it as the simulated brain grows.

Both use the lean neuron with the settings adopted in notes/24:
  chemical (per-branch) tags, gentle surface-only sleep reset, constant spine turnover with seeded-choice
  rewiring (64 candidates, value-guided), 3-night grace for newborn synapses.

A cell type not yet in the budget falls back to the suite's baseline.
"""

from dataclasses import replace

import numpy as np

from celltypes import CATALOG
from synapse_budget import BUDGET
from lean_neuron import LeanPopulation

BASELINE = {"standard": 4096, "low": 256, "biological": 32768}
BRANCH_SLOTS = {"standard": 128, "low": 32, "biological": 256}   # synapses per dendritic branch
BIO_CAP = 32768
BIO_CAP_EXCEPT = {"Purkinje": 2 ** 19}   # ~300,000-500,000 parallel-fiber synapses in humans: 524,288
Q = {16: 7, 32: 15}                            # levels -> signed range (-q..q)
OPTIONS = dict(chem=True, rewiring="seeded", guide="value", turnover=True, choices=64)
GRACE, SLEEP_RESET = 3, -1


# synapse_budget.py anchors 4096 to a 7,000-10,000-synapse pyramid, which puts human L2/3 IT (~30,000) at
# 8192. The user's standard is L2/3 IT = 4096, so every type is halved; relative sizes are unchanged.
ANCHOR = 0.5


# Synapses per neuron vs. brain size in real animals, as a fraction of human (notes/24):
# worm 302 neurons ~25 synapses/neuron (White 1986); fly ~140,000, ~360 (FlyWire 2024, ~50M synapses);
# mouse ~70M, cortical pyramidal ~10,000; human ~86 billion, L2/3 ~30,000 (Azevedo 2009; Eyal 2018).
# Rough: good to a factor of 2-3. Interpolated log-log; extrapolated below the worm with its slope.
ANIMALS = np.array([[302, 0.0025], [1.4e5, 0.036], [7e7, 0.33], [8.6e10, 1.0]])


def human_fraction(n_neurons):
    x, y = np.log10(ANIMALS[:, 0]), np.log10(ANIMALS[:, 1])
    lx = np.log10(max(n_neurons, 1))
    if lx < x[0]:
        return float(10 ** (y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (lx - x[0])))
    return float(10 ** np.interp(lx, x, y))


def synapses(ctype, suite="standard", brain_neurons=None):
    """brain_neurons (biological suite): size of the simulated brain; None = full human scale."""
    name = ctype if isinstance(ctype, str) else ctype.name
    b = BUDGET.get(name)
    if b is None:
        return BASELINE[suite]
    if suite == "biological":
        ceiling = int(min(BIO_CAP_EXCEPT.get(name, BIO_CAP), max(4, 2 ** round(np.log2(b[1])))))
        if brain_neurons is None:
            return ceiling
        n = ceiling * human_fraction(brain_neurons)
        return int(max(4, min(ceiling, 32 * round(n / 32) if n >= 64 else 2 ** round(np.log2(max(n, 4))))))
    standard = max(4, int(b[2] * ANCHOR))
    return standard if suite == "standard" else max(4, standard // 16)


def levels(ctype):
    b = BUDGET.get(ctype if isinstance(ctype, str) else ctype.name)
    return 16 if b is None else b[4]   # None = synapses don't learn (fixed)


def lean_rules(ctype, suite="standard"):
    """The cell type's learning rules with its suite's size. Thresholds scale with size, so one synapse
    means the same in every type (same scaling as test_lean_capacity)."""
    n = synapses(ctype, suite)
    slots = min(BRANCH_SLOTS[suite], n)
    branches = max(1, n // slots)
    r = ctype.rules
    f, g = branches * slots / (r.n_branches * r.slots), slots / r.slots
    return replace(r, n_branches=branches, slots=slots, threshold_init=r.threshold_init * f,
                   threshold_lr=r.threshold_lr * f, soma_slope=r.soma_slope / f,
                   branch_threshold=r.branch_threshold * g, spike_bonus=r.spike_bonus * g)


def population(ctype, n, n_inputs, rng, suite="standard", first_id=0):
    lv = levels(ctype)
    pop = LeanPopulation(lean_rules(ctype, suite), n, n_inputs, rng, first_id=first_id, q=Q[lv or 16],
                         **OPTIONS)
    pop.grace, pop.anchor = GRACE, True
    pop.fixed = lv is None
    return pop


def report():
    rows = []
    for name in BUDGET:
        lv = levels(name)
        rows.append(f"  {name:<28}{synapses(name, 'biological'):>7}{synapses(name, 'standard'):>7}"
                    f"{synapses(name, 'low'):>7}  {'fixed' if lv is None else f'{lv} lv':>6}")
    return "  type                           bio  standard  low  precision\n" + "\n".join(rows)


if __name__ == "__main__":
    print(report())

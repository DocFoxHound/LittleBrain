"""
Step 1: realistic packed states.

A small "brain" of 20 columns x 1,024 workhorse neurons (20,480 neurons, PackedPopulation) lives
through 16 simulated days and nights with sparse patterns, rewards, punishments and rest.

Scenario (stated on purpose, see BRIEF "Watch out"): LOCAL wiring. Each column owns its own region
of 64 inputs in a 1,280-input space, and a synapse's hashed target is drawn only from its own
column's region. One column is therefore exactly the demo neuron's world (8 of 64 inputs per
pattern, 40 trials a day), and a column that gets no input really is idle. With 5-10 of 20 regions
active, about 3-6% of all inputs are active at a time.

Life story (days 1..16), per group of columns:
  A  cols  0-5   days 1-6 pattern rewarded, days 7-8 punished (hard times), then idle (cold)
  B  cols  6-9   days 1-8 one pattern rewarded, another punished, then idle (cold)
  C  cols 10-14  idle (never used) days 1-8, then a new life days 9-16 (reward + punishment)
  D  cols 15-17  idle until day 12, then rewarded days 13-16 (recent learning)
  E  cols 18-19  never used (spare reserve)
Days follow the demo exactly: an active day is 40 trials (80 awake steps) then sleep; an idle
day is a rest of 200 steps then sleep (the demo's "time passes" day).

Outputs (in data/): pop_day8.lbdata (groups A and B just as they go cold, for the population
behavior test) and pop_day16.lbdata (every group, with nights-unused per neuron, for compression).
"""

from dataclasses import fields

from common import DATA, RESULTS, np, save_json, time  # noqa: F401

import data_archive
from neuron import NeuronType, WORKHORSE
from packed_neuron import PackedPopulation, _get, _targets

COL = 1024            # neurons per column
REGION = 64           # inputs per column region (the demo neuron's world)
K_ACTIVE = 8          # active inputs per pattern (as in the demo)
N_COLS = 20
N_INPUTS = N_COLS * REGION
TRIALS = 40           # rewarded/punished trials per active day (each = 2 awake steps), as in the demo
IDLE_REST = 200       # rest steps on an idle day (as in the demo's "time passes")

GROUPS = {"A": range(0, 6), "B": range(6, 10), "C": range(10, 15), "D": range(15, 18), "E": range(18, 20)}


class LocalPopulation(PackedPopulation):
    """PackedPopulation whose synapses only reach their column's region of the input space."""

    def __init__(self, ntype, cols, rng, region=REGION, n_inputs=N_INPUTS, col_size=COL):
        cols = np.asarray(list(cols))
        super().__init__(ntype, len(cols) * col_size, n_inputs, rng, first_id=int(cols[0]) * col_size)
        self.cols = cols
        self.region = region
        self.col_of = np.repeat(cols, col_size)
        self.region_base = (self.col_of * region).astype(np.intp)[:, None, None]

    def _src(self, w):
        return self.region_base + _targets(self._ids(), self.slot, _get(w, "gen"), self.region)


def column_patterns(seed=0):
    """Two fixed patterns per column ('main' and 'second'), each 8 of the column's 64 inputs."""
    r = np.random.default_rng(seed)
    pats = np.zeros((2, N_COLS, N_INPUTS), np.float32)
    for c in range(N_COLS):
        for p in range(2):
            pats[p, c, c * REGION + r.choice(REGION, K_ACTIVE, replace=False)] = 1.0
    return pats


def group_input(pats, cols, which):
    return pats[which, list(cols)].sum(axis=0)


def background(rng, cols):
    x = np.zeros(N_INPUTS, np.float32)
    for c in cols:
        x[c * REGION + rng.choice(REGION, K_ACTIVE, replace=False)] = 1.0
    return x


def probe(pop, pats, which=0):
    """Mean response of each column to its own pattern."""
    r = pop.forward(group_input(pats, pop.cols, which), learn_trace=False)
    return r.reshape(len(pop.cols), -1).mean(axis=1)


def plan(group, day):
    """What a group does on a given day: list of (pattern index, dopamine) alternating per trial, or None."""
    if group == "A":
        return [(0, +1.0)] if day <= 6 else [(0, -1.0)] if day <= 8 else None
    if group == "B":
        return [(0, +1.0), (1, -1.0)] if day <= 8 else None
    if group == "C":
        return [(0, +1.0), (1, -1.0)] if day >= 9 else None
    if group == "D":
        return [(0, +1.0)] if day >= 13 else None
    return None


def live_day(pop, rng, pats, schedule, trials=TRIALS):
    for i in range(trials):
        which, da = schedule[i % len(schedule)]
        pop.forward(group_input(pats, pop.cols, which))
        pop.modulate(dopamine=da, ach=0.8, arousal=0.8)
        pop.tick()
        pop.forward(background(rng, pop.cols))
        pop.tick()


def any_used(pop):
    return (_get(pop.words, "used") != 0).any(axis=(1, 2))


def simulate(days=16, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    pats = column_patterns(seed)
    pops = {g: LocalPopulation(WORKHORSE, cols, np.random.default_rng(seed * 100 + i))
            for i, (g, cols) in enumerate(GROUPS.items())}
    unused = {g: np.zeros(p.n, np.int32) for g, p in pops.items()}   # consecutive nights unused
    history = []
    t0 = time.time()
    for day in range(1, days + 1):
        for g, pop in pops.items():
            sched = plan(g, day)
            if sched:
                live_day(pop, rng, pats, sched)
            else:
                pop.rest(IDLE_REST)
            used = any_used(pop)
            unused[g] = np.where(used, 0, unused[g] + 1)
            pop.sleep()
        row = {"day": day, **{g: [round(float(v), 3) for v in probe(p, pats)] for g, p in pops.items()}}
        history.append(row)
        if verbose:
            means = "  ".join(f"{g}={np.mean(row[g]):.2f}" for g in pops)
            print(f"  day {day:2d}  response to main pattern: {means}   ({time.time() - t0:.0f}s)")
        if day == 8:
            data_archive.save(DATA / "pop_day8.lbdata", {
                "A": pops["A"], "B": pops["B"], "pats": pats,
                "unused": {g: unused[g].copy() for g in ("A", "B")},
            }, SNAPSHOT_SCHEMA, "day8")
    data_archive.save(DATA / f"pop_day{days}.lbdata",
                      {"pops": pops, "pats": pats, "unused": unused}, SNAPSHOT_SCHEMA, f"day{days}")
    save_json("brainsim_history.json", history)
    return pops, unused


SNAPSHOT_SCHEMA = {
    NeuronType: {f.name for f in fields(NeuronType)},
    LocalPopulation: set("""
        t n n_inputs rng first_id words gate_bits threshold plasticity post_hist age avg_rate
        pre_hist elig_weights slot pending fast_scale cols region col_of region_base
    """.split()),
}


def load_day16():
    return data_archive.load(DATA / "pop_day16.lbdata", SNAPSHOT_SCHEMA, "day16")


def load_day8():
    return data_archive.load(DATA / "pop_day8.lbdata", SNAPSHOT_SCHEMA, "day8")


if __name__ == "__main__":
    simulate()

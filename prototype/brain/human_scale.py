"""
Grow the full brain toward a neuron total using human proportions (docs/scaling.md steps 1-3; note 70).

1. Each catalog subregion's budget = its share of human neurons. Catalog totals are used where they exist
   (regions.py, with sources there). Where the catalog has no total, ASSUMED_TOTALS below records the value
   used and where it comes from.
2. Each subregion's budget is split across its cell types by the catalog percentages. Maps in the running brain
   that share one catalog entry (e.g. areas A and B are both "sensory cortex") split it evenly.
3. No population shrinks below its current, verified size: the human budget only adds cells where it exceeds
   what the reference brain already has. Groups without a catalog entry (relays, inputs) keep their size.
The human-proportion scale is raised until the brain reaches the requested total.

At ~100,000 neurons almost every addition goes to the cerebellar granule layer (~80% of human neurons) and the
neocortex (~19%); every other subregion's human share is below its existing minimum of one 64-cell map per type.
"""

from collections import defaultdict

GRID_CELLS = 64   # one 8x8 map

# Subregions with no human total in regions.py: value used here and its basis.
# Only the cortex values matter at ~100k neurons; the rest stay far below the one-map minimum.
CORTEX_TOTAL = 16.34e9   # human cerebral cortex neurons (Azevedo et al. 2009)
ASSUMED_TOTALS = {
    "sensory cortex (e.g. V1, S1)": (CORTEX_TOTAL * 0.86,
        "cerebral cortex 16.34e9 (Azevedo 2009) minus the prefrontal and cingulate/insula shares below; areas A "
        "and B stand for all sensory and association cortex"),
    "motor / prefrontal cortex": (CORTEX_TOTAL * 0.08,
        "~8% of cortical neurons are prefrontal (Gabi et al. 2016, ~1.3e9)"),
    "anterior cingulate / frontoinsular cortex": (CORTEX_TOTAL * 0.06, "assumed ~6% of cortical neurons"),
    "first-order thalamus (LGN, VPM/VPL, MGBv)": (6e6, "assumed order of magnitude (LGN ~1e6 per side)"),
    "higher-order thalamus (pulvinar, MD, POm)": (2e7, "assumed order of magnitude (MD ~6e6 bilaterally)"),
    "thalamic reticular nucleus": (1e6, "assumed order of magnitude"),
    "claustrum": (4e6, "assumed order of magnitude"),
    "intercalated cell masses": (2e5, "assumed order of magnitude"),
    "globus pallidus externa": (5.4e5, "Thörner et al. 1975 (~540k)"),
    "globus pallidus interna": (1.7e5, "Thörner et al. 1975 (~170k)"),
    "substantia nigra pars reticulata": (3e5, "assumed order of magnitude"),
    "deep cerebellar nuclei": (5e6, "assumed order of magnitude (dentate nucleus ~5e6)"),
    "inferior olive": (1.5e6, "assumed ~0.75e6 per side"),
    "ventral tegmental area": (6e5, "assumed order of magnitude"),
    "median raphe": (1.5e5, "assumed order of magnitude"),
    "medial septum / diagonal band": (2e5, "assumed order of magnitude"),
    "pedunculopontine / laterodorsal tegmental": (2e5, "assumed order of magnitude"),
    "arcuate nucleus (modeled populations)": (4e5, "assumed order of magnitude"),
    "lateral hypothalamus (modeled populations)": (1e5, "assumed order of magnitude (orexin ~7e4)"),
    "paraventricular nucleus (modeled populations)": (1e5, "assumed order of magnitude"),
}


def human_totals():
    """Human neurons per catalog subregion name, and which were assumed."""
    import regions as R
    totals, assumed = {}, {}
    for region in R.REGIONS:
        for sub in region.subregions:
            if sub.human_neurons:
                totals[sub.name] = float(sub.human_neurons)
            else:
                totals[sub.name], assumed[sub.name] = ASSUMED_TOTALS[sub.name]
    return totals, assumed


def _catalog_of(spec):
    """For each non-input group with a catalog entry: (subregion name, its type's catalog percent, number of maps
    in this brain sharing that subregion and type)."""
    from check_regional_audit import TERRITORIES, territory
    entry = {}
    for g in spec.groups:
        if g.key in spec.inputs or g.ctype is None:
            continue
        sub = TERRITORIES.get(territory(g.key))
        pct = sum(p for c, p, _, _ in sub.cells if c.name == g.ctype.name) if sub is not None else 0
        if pct > 0:
            entry[g.key] = (sub.name, pct, g.ctype.name)
    sharing = defaultdict(int)
    for sub, _, name in entry.values():
        sharing[sub, name] += 1
    return {k: (sub, pct, sharing[sub, name]) for k, (sub, pct, name) in entry.items()}


def _per_tile(spec, cat, totals, human_scale):
    """per_tile for each group at a given human-proportion scale (neurons per human neuron)."""
    out = {}
    for g in spec.groups:
        if g.key not in cat:
            out[g.key] = g.per_tile
            continue
        sub, pct, k = cat[g.key]
        target = human_scale * totals[sub] * pct / 100 / k
        out[g.key] = max(g.per_tile, int(round(target / GRID_CELLS)))
    return out


def human_sized(spec, n_neurons=100_000):
    """The brain grown to about n_neurons with human proportions (never shrinking a population).
    Returns (spec, report)."""
    import copy
    spec = copy.deepcopy(spec)
    totals, assumed = human_totals()
    cat = _catalog_of(spec)

    def size(scale):
        pt = _per_tile(spec, cat, totals, scale)
        return sum(v for k, v in pt.items() if k not in spec.inputs) * GRID_CELLS, pt

    lo, hi = 0.0, 1e-3
    while size(hi)[0] < n_neurons:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if size(mid)[0] < n_neurons else (lo, mid)
    total, pt = size(hi)
    before = {g.key: g.per_tile for g in spec.groups}
    for g in spec.groups:
        g.per_tile = pt[g.key]
    grown = {k: (before[k] * GRID_CELLS, pt[k] * GRID_CELLS) for k in pt if pt[k] != before[k]}
    report = {"neurons": total, "human_scale": hi, "grown": grown, "assumed": assumed}
    return spec, report


def describe(report):
    lines = [f"{report['neurons']:,} neurons; human-proportion scale {report['human_scale']:.3g} "
             f"model neurons per human neuron"]
    for k, (a, b) in sorted(report["grown"].items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {k:<22} {a:>7,} -> {b:>7,}")
    return "\n".join(lines)

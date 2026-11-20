# Scaling LittleBrain

1. Pick the target cell budget for each modeled structure. Keep the rationale with the specification. Use the human totals, uncertainty and within-region percentages in the regional catalog. Where a total is unknown, record the assumed budget. Small game brains retain rare functional populations even where a literal human ratio would round to zero.
2. Allocate each subregion across its cell types by catalog percentage. The minimum in `regions.allocate` defaults to one representative per type. That minimum can make the realized total exceed a small requested budget. Report actual counts.
3. The federated game maps use 8 × 8 tiles. A layer with one cell per tile has 64 cells. Scale the per-tile counts, round and retain at least one per represented group. Count externally supplied inputs separately from neurons. Compare actual regional proportions after rounding before accepting a new size.
4. Use the biological synapse suite and partner cap for official comparisons. The suite interpolates the fraction of the human synapse budget in log-log space at worm, fly, mouse and human sizes. Each cell type has its own ceiling. Reachable partners constrain the result further, at four contacts per effective partner. Use the actual connected brain for memory measurements.
5. Report the scale, seed, realized cell and synapse counts, and device. Compare behavior against a matched brain at the same scale.

[Regional allocation](../prototype/brain/regions.py) · [Suite constants and rounding](../prototype/brain/suites.py) · [Partner cap and input accounting](../prototype/brain/federation.py)

The current reference budget is 18,944 neurons. The regional expansion adds one 64-cell map for each of 67 missing catalog placements and preserves the earlier population counts. The full audit reports actual and planned shares for each subregion. The game runner’s scale multiplier rounds the chosen map counts. Moving toward human proportions requires choosing new regional budgets using steps 1–3. [Expanded reference measurements](../notes/66-expanded-reference-verification.md)

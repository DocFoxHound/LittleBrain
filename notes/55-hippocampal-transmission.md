# Dentate input to CA3 after development

After all seven routing repairs, three brains were built from scratch at the default scale and developed with the standard two-day schedule. The dentate granule cells' mossy fibers are supposed to excite CA3 pyramidal cells. The reference connection is HC-7, a small number of strong glutamatergic inputs. [Hippocampal connection plan](regions/connections.md)

In all three brains, raising dentate output from zero to one **lowered** the mean membrane drive in CA3. This was measurable before the sparse competition that selects which CA3 cells fire. The sampled partners were in the correct brain region. The learned weights had changed the sign of most of the 2,560 mossy-fiber contacts.

| Seed | Negative mossy contacts | Mean effective mossy weight | CA3 drive change, intact | CA3 drive change, negative mossy contacts muted |
|---:|---:|---:|---:|---:|
| 0 | 2,454 / 2,560 | −0.252 | −0.171 | +0.297 |
| 1 | 2,502 / 2,560 | −0.281 | −0.251 | +0.153 |
| 2 | 2,522 / 2,560 | −0.298 | −0.304 | +0.188 |

For the matched control, a copy of each developed brain retained the same contacts, weights and cellular state. Only transmission from its negative mossy-fiber contacts was muted. In each condition the receiver was updated once with dentate output clamped to zero or one. The direction of the membrane response reversed on every seed. Sparse competition still makes spike counts and membrane drive distinct readouts.

The shared synapse model adds a signed fast weight to a long-term component during transmission. Depression can take their sum below zero. Across all sampled routes from cells explicitly catalogued as glutamatergic with positive pathway gain, the audit counted the following contacts.

| Seed | Negative / total contacts | Routes with at least one negative contact / inspected |
|---:|---:|---:|
| 0 | 31,167 / 1,519,104 | 87 / 132 |
| 1 | 32,651 / 1,519,040 | 85 / 132 |
| 2 | 32,499 / 1,519,104 | 84 / 132 |

A few negative contacts do not establish the net effect of a route. The mossy-fiber intervention establishes a sign reversal for this particular projection. A general change to effective conductance would also affect the cortex, striatum and other learned circuits. That change needs a separate controlled plasticity experiment, including sensitivity to stored checkpoints and memory retention. The pulse intervention above operates only on a disposable diagnostic copy.

Run `OPENBLAS_NUM_THREADS=1 python prototype/brain/check_regional_audit.py --seed N` for N equal to 0, 1 and 2. The `glutamate` result reports both counts and the paired intervention.

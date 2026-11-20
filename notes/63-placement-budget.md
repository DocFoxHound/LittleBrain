# Reference-scale placement budget

There are 98 implemented cell types. The current comparison covers 167 distinct combinations of catalog subregion and cell type. The reference game brain represents 100 of those combinations. A cell type can occur elsewhere in the same brain and still be missing from its expected location. The largest gaps are the 14 entries in the combined cingulate/insula catalog, 12 in prefrontal cortex and seven in CA3. [Regional composition and missing types](50-regional-audit.md)

Each population covers an 8 × 8 map. The smallest new population therefore costs 64 neurons. The calculations below keep every existing named population, count catalog entries the same way as the regional audit and exclude external inputs. They are lower bounds on representation. Learning performance and regional proportions need separate measurements. [Scaling rules](../docs/scaling.md)

| Allocation at the present map resolution | Consequence |
|---|---|
| Preserve every existing population count | Add 67 maps, giving **18,944 neurons**. Recheck regional proportions before selecting any larger scale. |
| Redistribute inside each affected subregion first | At most 20 of the 67 maps can come from multi-map populations locally. Seventeen subregions still need 47 maps between them. Bringing those regions up to their minima would give **17,664 neurons** if their remaining cells are added. |
| Keep the global budget of **14,656 neurons** | Redistribute the 20 maps locally and move another 47 maps, or 3,008 neurons, across catalog subregions. This changes the chosen size of other regions and the functional circuits that use them. |

For each subregion, count its existing maps and named populations. Reserve one map per existing population. For a subregion with missing types, the remaining maps up to its missing-type count are the maximum that can be reassigned locally. The other missing entries are its shortfall. A one-map allocation for every rare type also sets a lower bound on distortion at small scale. For example CA3 currently has four maps of pyramidal cells and one basket map. Allocating its seven missing interneuron types by first taking three pyramidal maps would leave one pyramidal map among nine, despite pyramids being the principal cells.

The catalog intentionally combines some game maps. Sensory areas A and B each match 11 of 16 types despite matching 14 in aggregate. MD uses a specialized relay, and the CeL/CeM split places distinct amygdala outputs on different maps. The DCN catalog uses the Golgi model as an approximation for a local inhibitory cell. A literal type-name match is insufficient to choose which map should receive a population. [Regional catalog](../prototype/brain/regions.py)

For a staged implementation, start with a single well-specified local projection, such as a CA1 CCK basket population inhibiting pyramidal somata. Select its source and population budget from the regional notes. Compare the existing circuit, the circuit with the new population, and a copy with only its output silenced. Test the receiver at fixed input and repeat the relevant behavior before moving to the next region. Notes 60–62 use the same regional budgets as the preceding reference brain.

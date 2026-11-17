# Research brief: neuron types by brain region, with percentages

Project: a brain-inspired AI in `~/Documents/MemoryPhilosophy` (read `README.md` and `notes/18-neuron-types.md`
for context). We're about to organize our model into brain regions. The user wants to be
**exceptionally specific about which neuron types each region contains, and in what proportions.**

For each region assigned to you, write `notes/regions/<region>.md` containing:

1. **Composition table**, one row per neuron type:
   - name (and common marker, e.g. PV, SST, D1-MSN);
   - **percentage of the region's neurons** (give a range when sources disagree);
   - **species the number comes from** (human preferred; then non-human primate; then rodent, clearly labeled);
   - absolute counts if known (especially human);
   - transmitter (glutamate / GABA / dopamine / …);
   - excitatory or inhibitory; local or projection, and where its axon goes;
   - a confidence mark: ● well established · ◐ supported, numbers vary · ○ uncertain.
2. **Subregions** where they matter (e.g. CA1 vs CA3; BLA vs central amygdala; striosome vs matrix), each with its own table.
3. **Signature behaviors to model** for each type: what makes it *that* cell (firing pattern, dendritic properties, plasticity rule, what it targets). Be concrete enough that someone could write a test for it, like the tests in `prototype/brain/test_cell_types.py`.
4. **Main connections** in and out of the region (which types talk to which, and to other regions).
5. **Human vs rodent differences** that matter (e.g. primates have more cortical/striatal interneurons).
6. **Sources:** papers, reviews or atlases, with links. **Verify numbers with web searches. Don't invent figures.** If you can't find a number, say so and give the best qualitative statement.

Keep each file focused and well organized (tables, short bullets). Prefer authoritative reviews and stereological
counts (e.g. Azevedo et al. 2009; West & Gundersen 1990; Rudy et al. 2011; Nair-Roberts et al. 2008; Allen Brain
Cell Atlas / BICCN cell censuses).

Don't edit any files other than your own region files. Final message to the coordinator: under 300 words,
listing the files written and any numbers you're unsure of.

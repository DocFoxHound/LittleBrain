# 71: Finer maps for some structures

*2026-11-21. Vision step 2 of 6 (owner: "do it all"). Every structure used to share one 8×8 map, which caps the
spatial detail of local processing (pooled inhibition, neuromodulation, any population-level connection).*

- `Group.grid` (default 8): a structure can use a 16×16 or 32×32 map; `per_tile` counts cells per tile of its own map.
- Fine groups get fine tiles (`Brain.ftile`) and positions on their own map. **Every group still has its coarse 8×8
  tile**, so all existing code (striatal projection, association currents, audits) works unchanged.
- Population-level connections that involve a fine map use resampling kernels between maps (`Brain._Kx`); kernel
  widths stay in 8×8-map units, so a connection covers the same physical area at every resolution; `sigma=0` means
  "the same place" (about one fine tile).
- Checkpoint schema and validation include fine tiles; size counting (`sized`, `human_scale`) uses each group's map.
- **The reference brain is bit-identical** (60-step fixed-seed run with learning, compared array-for-array).
- `test_fine_maps.py` 6/6: sizes, tiles agree with positions, kernels, a spot of light on a 32×32 retina drives the
  16×16 relay cells at the same place (0.550 inside vs 0.005 elsewhere), checkpoint round trip.

# 72: A real retina, LGN streams and the vision brain

*2026-11-21. Vision step 1 of 6. The old eye was 256 grey point samples feeding a 256-cell thalamic relay, so a
bigger cortex could not see more. `BRAIN=vision` (`vision.py`, `retina.py`).*

## The retina (`retina.py`): a functional filter bank, like the old eye input (computed, not simulated cells)
| channel | real cell | map |
|---|---|---|
| `RET.on`, `RET.off` | midget ON/OFF centre-surround (difference of Gaussians, surround 3x; Enroth-Cugell & Robson 1966) | 32×32 |
| `RET.rg_on`, `RET.rg_off` | midget red-green opponent: L centre vs M surround (Dacey 2000) | 32×32 |
| `RET.by` | small bistratified blue-ON: S vs L+M (Dacey & Lee 1994) | 16×16 |
| `RET.ton`, `RET.toff` | parasol transient cells: change between frames; 2x larger fields | 16×16 |
| `RET.ds0..3` | ON-OFF direction-selective cells, 4 directions (Barlow & Levick 1965), Reichardt-style correlation | 16×16 |
| `retina` | the old 8x8 light-level input, kept for older pathways (e.g. light to the SCN clock) | 8×8 |

- **Fovea:** each channel's map is warped onto the visual field (field radius = map radius²), so the centre is
  sampled densely and sharply and the periphery sparsely and blurrily; fields grow with eccentricity. Visual areas
  on the same map inherit this magnification. A minimum field of ~0.6 pixel of the incoming picture stands in for the
  eye's optics (the fovea out-resolves a 64-pixel picture otherwise).
- Cone images from RGB (linear approximation), contrast divided by mean light (Weber), squashed to [0, 1].
- `Retina.look_at(gaze)` moves the fovea (for eye movements, step 6).
- `test_retina.py` **7/7**: a small bright spot drives ON cells centred on it (0.74) with OFF silent (0.000), the
  reverse for a dark spot; red drives red-green ON not OFF; blue drives blue-yellow cells; transient cells fire at
  onset (0.91) and not while the picture is held; each direction-selective type prefers its direction by >5x
  (~0.065 vs ~0.003); central fields are >3x smaller than peripheral. A large even area gives centre-surround cells
  nothing in its interior: the retina reports edges, not areas (correct, and why the first versions of two tests failed).

## LGN streams and V1 (`vision.add_retina_and_lgn`)
- `TH.TC` becomes the **parvocellular** relay on 16×16, driven by midget cells; new **magnocellular** `TH.M` (16×16,
  parasol + direction-selective) and **koniocellular** `TH.K` (blue-yellow). All get layer-6 feedback and reticular
  inhibition (Livingstone & Hubel 1988; Casagrande 1994).
- V1 (area A) L4 and L2/3 on 16×16 (512 and 768 cells, from 384 and 640); L4 is driven by all three streams.
- 19,584 neurons + 6,272 retina inputs; ~10 ms per retina frame, ~27 ms per brain step.

## Development, agent and games
- A brain with a retina grows up on **moving coloured shapes** (`vision.Movie`: discs, bars at all angles, squares,
  drifting): `Brain.calibrate/develop` accept a "movie" stimulus that supplies each step's input (plain stimuli unchanged).
- The agent gives such brains a real eye (`retina.Sight`): channels are recomputed every step, so transient and motion
  cells see pictures appear, change and vanish. The game bridge keeps colour (`rgb8`); older brains use brightness.

## First measurements (seed 0, before stage 3)
- Grow + develop: ~1 min. Signal game 64 turns: **0.89 learning vs 0.28 off** (still through the old thalamic
  shortcut to the striatum, note 73 / task "route learning through cortex").
- Drifting gratings, 4 orientations: every LGN and V1 cell responds; orientation selectivity (OSI) median: LGN
  0.12–0.13 (noise floor of this measurement), **V1 L4 0.26, L2/3 0.26**, area B 0.14. Real V1: ~0.3–0.5.
  Measured with the earlier, weaker ON/OFF gain; re-measured in stage 3.

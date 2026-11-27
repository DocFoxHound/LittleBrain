# 95: Shape recognition becomes part of the vision brain (phase 5 closes, nearly met)

*2026-11-26. Owner's decision after note 94: make the new setup the vision brain's default.*

## New defaults (vision brain)
| switch | default | what |
|---|---|---|
| `VISION_POOLED` | **1** | pooled edge cells V4.P0-P3 (note 91) |
| `VISION_POOLED_IT` | **0** | ...read only by the striatum, not IT (note 94) |
| `VISION_POOLED_READ` | **map** | ...per tile of their map |
| `VISION_SCALE` | **source** | each striatal input source on its own scale (note 92) |
| `STRIATUM_TERRITORIES` | **streams** | separate striatal territories: pooled cells, grid cells and MT each alone, the rest together; trust by how often following each paid off (note 93) |
Earlier behaviour: `VISION_POOLED=0 VISION_SCALE=separate STRIATUM_TERRITORIES=0`. The reference (non-vision) brain is
unaffected (bit-identical in exact mode). Brains saved before this lack the pooled cells; they still load and play.

## Evidence (3 seeds, same brains, old readout → new; note 94)
shape 0.19 / 0.29 / 0.17 → **0.54 / 0.51 / 0.66**; colour −0.02, motion −0.04, key-door +0.05, homing −0.03, poison
within noise; **signal −0.07** (seed 0 −0.14): open item.

## Phase 5 against its bar
"Shape at any position learned ≥ 0.6 on every seed": 0.51-0.66, **met on one seed of three**; "IT identifies shapes
across positions": IT 0.52-0.62 when it heard the pooled cells (note 91), but the default no longer feeds them to IT.
Recorded as **nearly met**. What was tried and why is in notes 82-94.

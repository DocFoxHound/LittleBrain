# 31: The desktop viewer

*2026-11-17. `prototype/viz/brainview.py` (pygame-ce). Screenshots: `docs/screenshots/`.*

![viewer](../docs/screenshots/viewer.png)

## What it shows
- **A side view of a human brain** (facing left). Each simulated region's neurons **fill its real territory**:
  cortex is a sheet covering the surface, so area A fills the occipital lobe (V1), area B the parietal lobe and the
  action copy the precentral (motor) strip, seen face-on. Deep structures are drawn darker, seen through the surface:
  thalamus (an egg, with the reticular nucleus as a front shell), striatum (caudate head and body + putamen),
  globus pallidus behind/below the putamen, hippocampal loop in the temporal lobe (when wired). The retina is the cup
  at the back of the eye.
- **Placement keeps the brain's own map:** each territory is filled evenly and the square 8×8 map is stretched over
  it in order (front/back, up/down), so neighbours on the map stay neighbours in the picture; the action channels
  stay in order across the striatum.
- **Regions designed but not wired yet** (amygdala, hypothalamus, claustrum, cerebellum, frontal lobe, brainstem
  nuclei) are faint outlines with a "not built yet" label.
- Dots glow with their neurons' current firing (warm = excitatory, cool = inhibitory, grey = inputs), each group
  against its own usual range.
- **Brain size without clutter (level of detail):** a group gets at most ~9·√n dots (capped at 200). Small brains
  show one dot per neuron; bigger brains show more and larger dots, each standing for more neurons. The legend
  says how many neurons a dot stands for. Grow/shrink live with `[` `]` (0.25× – 4×; 1,728 – 22,528 neurons);
  a new size is grown once (calibration + development) and cached in `prototype/viz/.cache/`.
- **Hover a dot:** cell type, its human counterpart, which group, how many neurons, firing now.
- **Pathways** between regions brighten with their source's activity (`P` toggles).
- **Side panel, game-agnostic:** what the brain sees, its action channels (how far each brake is released, odds
  of choosing, the chosen one), dopamine, and reward per turn. Action names and an optional chance line come from
  the game; the viewer knows nothing else about it.

## Plugging in games
The viewer watches a `BrainAgent` playing any `games.Game` (`reset()` / `step(action)`; picture in, action out,
reward back). Register a game in `games.REGISTRY` and run `--game name`, or from code:
`watch(prepared_brain, make_game)`. The signal game is only the first test game.

## Run
    .venv/bin/python prototype/viz/brainview.py [--scale 1] [--seed 0] [--game signal]
    headless screenshots:  --shots 5,60,250 --out shots/  [--hover x,y]
Keys: Space pause · ←/→ speed · L learning on/off · S sleep · [ ] brain size · T labels · P pathways · F12 screenshot.

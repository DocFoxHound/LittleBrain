# 90: Competitive learning in IT (phase 5 research): IT learns where and how bright, not what

*2026-11-22. The owner's choice after note 89: make IT cells specialise on objects through competitive learning, the
route real IT is thought to take. Tool: `it_learning.competitive_it` (opt-in, not part of development).*

## Built
A developmental step, like the like-to-like wiring: the brain watches 400 movies of one object sliding across the
view (random shape, size, brightness and path); IT's input cells compete for each view (k-winners-take-all, 5%) and
the winners and their neighbours on IT's map move their preferences toward V4's pattern (competitive Hebbian
learning with a shrinking neighbourhood, a self-organising map; Rumelhart & Zipser 1985, Kohonen 1982); each IT cell
then re-points its V4 synapses at the V4 cells it prefers most. No labels. Two versions of the temporal link:
V4 averaged over each movie, and the frame-by-frame trace rule (Földiák 1991, activity smoothed over frames, 0.8).

## Results (seed 0, tiled V4 columns; linear decoder, 200 train / 200 test)
| | shape (chance 0.25) | brightness (0.5) | quadrant of the view (0.25) |
|---|---|---|---|
| V4 L2/3 | 0.69-0.70 | 0.72 | 0.85 |
| IT as learned, if implemented exactly | **0.34-0.37** | 0.66 | **0.77** |
| IT, per map tile (what the striatum reads) | 0.33-0.34 | 0.69 | 0.79 |
| the real IT after rewiring (movie-averaged version) | 0.47 (IT L4), 0.37 (IT L2/3) | | |
Both versions alike. **IT specialised on where the object was and how bright it was.**

## Why
Unsupervised competition specialises on what varies most between views. In V4, a map of the visual field, that is
position, then brightness; shape is a small part of the variance. The trace rule ties together the views of one
movie, but along a movie the position keeps changing (the object slides), so position still dominates. To learn
"same object, different place", the input has to be partly pooled over position already, and pooling over position
is what the learned areas don't do (notes 85-89): a chicken-and-egg problem.

## Where shape stands
Shape is decodable from V4 (0.70) and reaches the striatum (0.65), but no learned stage turns it into a small,
position-tolerant code, and the striatum can't learn from the spread-out version in a few hundred trials. The one
route shown to work is a fixed per-region pooling of edge energy (0.57 in the game, note 85).

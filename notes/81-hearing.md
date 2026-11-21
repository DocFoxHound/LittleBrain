# 81: Hearing (plan track L1)

*2026-11-22. The first step of the communication track: an ear, an auditory thalamus and an auditory cortex in the
vision brain (now 33,152 neurons). `HEARING=0` leaves them out.*

## Built
- **The ear** (`cochlea.py`): sound → cochleagram (energy in 16 frequency bands per brain step) → auditory-nerve fibres
  on a 16×16 map (frequency along x, fibre threshold along y): sustained fibres with staggered thresholds code loudness
  by how many fire (dynamic-range fractionation; Liberman 1978), onset fibres fire when a band's energy rises.
  Real audio: `cochleagram(wave, rate)` (log-spaced bands, ~10 ms frames, cube-root loudness). **Text**:
  `text_sound(text)`: each letter a fixed spectral pattern (vowels two formant-like bands, consonants a noise-like
  band), one frame per letter, silence between words. Words read and words heard reach the brain the same way.
- **MGB.v** (ventral medial geniculate, TC relay cells) driven by the nerve, layer-6 feedback from A1, reticular
  inhibition shared with the visual relays; **AUD** = primary auditory cortex, all layers (`vision.add_area`), narrow
  feedforward reach to keep the frequency map; the visual-cortex treatment (gains, slow homeostasis, sharpness cap,
  like-to-like wiring, stable adulthood) applies to it too.
- Observations can carry `sound` (a cochleagram) or `text`; the agent's ear plays one frame per brain step.
- Development: a growing brain hears random babble-like syllables (1-3 CV syllables) along with its moving shapes.
- `test_cochlea.py` 6/6: one frame per letter; letters differ; a tone drives only its band; louder recruits more
  fibres; onsets fire at the start only; real audio: a 200 Hz tone peaks in a lower band (2) than a 4 kHz tone (13).

## Results (`check_hearing.py`, 3 seeds)
**Syllables** (6 syllables × 5, nearest centroid, chance 0.17):
| seed | ear | MGB | A1 L4 | A1 L2/3 |
|---|---|---|---|---|
| 0 | 1.00 | 0.73 | 0.87 | 0.77 |
| 1 | 1.00 | 0.90 | 1.00 | 0.83 |
| 2 | 1.00 | 0.97 | 0.93 | 0.83 |

**Statistical learning** (Saffran et al. 1996): 300 words of a continuous stream (tupiro golabu bidaku padoti), with
learning on; a twin copy hears the same syllables in random order. Then words vs part-words (spanning word
boundaries), learning off; A1 L2/3 response to the 2nd and 3rd syllables. Effect = part-words − words:
| seed | exposed to words | control |
|---|---|---|
| 0 | +0.0039 | +0.0027 |
| 1 | +0.0073 | −0.0011 |
| 2 | +0.0109 | +0.0051 |
Every exposed brain responds less to words than to part-words and by more than its control, the predicted direction,
but the effect is small (~3-10% of the response, no error estimate yet): **a hint, not a result**.

## Next (track L)
- A stronger statistical-learning test (more exposure, trial-level statistics, a belt area that predicts the next syllable).
- L2: babbling and imitation with auditory feedback; the bridge accepting `text`/`sound` fields.

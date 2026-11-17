# 01: Storing information

## True / false / unknown
- **Two bits, one state unused:** simple, slightly wasteful. Same idea as SQL `NULL` and nullable booleans.
- **"Known" bit + "value" bit:** three-valued (Kleene) logic becomes fast bitwise operations.
- **Packed:** one three-way value holds log₂(3) ≈ 1.585 bits. Since 3⁵ = 243 ≤ 256, five fit in one byte, about 99% efficient.
- **Ternary hardware:** the Soviet Setun computer (1958). Binary won because transistors are reliable at on/off, yet TLC flash already stores 8 voltage levels per cell.
- In decision theory, "unknown" becomes a **belief**: a probability over what might be true (see POMDPs, note 2).

## The smallest way to store something
- **Shannon entropy:** the average bits needed, given how predictable the data is.
- **Kolmogorov complexity:** the length of the shortest program that reproduces the data. The true minimum, but impossible to calculate in general.
- Compression = **predict well, then store only the surprises.**

| | Raw | Everyday tools | Best known |
|---|---|---|---|
| Text | 8 bits/char | gzip about 2.5–3 bits/char | under 1 bit/char (prediction models; the Hutter Prize) |
| Images | 24 bits/pixel | PNG 2–3× (lossless) | JPEG/WebP/AVIF 10–50× (lossy); a text description plus a generative model (semantic) |

## Ideas worth keeping
- How small something can get depends on **what counts as "the same" when you get it back** (every bit, looks the same to a person, or same meaning) and on **how much the reader already knows**.
- Human memory is at the lossy, semantic end: it keeps the gist plus a few surprising details and rebuilds the rest each time you remember.
- A memory system must tell "I know it's false" apart from "I never stored that."

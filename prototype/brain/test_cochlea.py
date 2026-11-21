"""The ear (cochlea.py, note 81): text and audio become cochleagrams; fibres code loudness and onsets."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import cochlea  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


class FakeBrain:
    """Just the ear's input maps (16x16: frequency along x, fibre threshold along y)."""
    def __init__(self):
        g = (np.stack(np.meshgrid(np.arange(16), np.arange(16)), -1).reshape(-1, 2) + 0.5) / 16
        self.pos = {"EAR.sus": g, "EAR.on": g}


def main():
    a, b = cochlea.text_sound("ta"), cochlea.text_sound("ki")
    check("text becomes one frame per letter plus a silent frame between words",
          cochlea.text_sound("ta ki").shape == (cochlea.BANDS, 6))
    check("different letters sound different", float(np.abs(a[:, 0] - b[:, 0]).sum()) > 0.5)
    ear = cochlea.Ear(FakeBrain())
    quiet = ear(np.zeros(cochlea.BANDS))
    loud = np.zeros(cochlea.BANDS, np.float32)
    loud[5] = 1.0
    on = ear(loud)
    held = ear(loud)
    soft = np.zeros(cochlea.BANDS, np.float32)
    soft[5] = 0.3
    ear(np.zeros(cochlea.BANDS))
    s = ear(soft)
    band5 = ear.band["EAR.sus"] == 5
    check("a tone drives only fibres of its own band", on["EAR.sus"][band5].sum() > 0 and on["EAR.sus"][~band5].sum() == 0)
    check("louder sound recruits more fibres (threshold ladder)",
          (on["EAR.sus"][band5] > 0.05).sum() > (s["EAR.sus"][band5] > 0.05).sum())
    check("onset fibres fire when a sound starts, not while it is held",
          on["EAR.on"].sum() > 0 and held["EAR.on"].sum() == 0 and quiet["EAR.on"].sum() == 0)
    rate = 16000
    t = np.arange(rate // 2) / rate
    low = cochlea.cochleagram(np.sin(2 * np.pi * 200 * t), rate)
    high = cochlea.cochleagram(np.sin(2 * np.pi * 4000 * t), rate)
    check("real audio: a low tone peaks in a lower band than a high tone",
          int(low.mean(1).argmax()) < int(high.mean(1).argmax()),
          f"bands {int(low.mean(1).argmax())} vs {int(high.mean(1).argmax())}")
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

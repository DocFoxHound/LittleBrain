"""Design stamps on saved brains (note 79): what loads, what is refused, what warns."""
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
import design  # noqa: E402
from federation import Brain  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


def roundtrip(b, d, name, **kw):
    path = os.path.join(d, name)
    b.save(path)
    try:
        return Brain.load(path, quiet=True, **kw), None
    except design.DesignError as e:
        return None, str(e)


def main():
    with tempfile.TemporaryDirectory() as d:
        b = design.stamp(Brain(S.stage_bg(), seed=0), "bg")
        c, err = roundtrip(b, d, "ok.brain", kind="bg")
        check("a stamped brain loads as its own kind, with no warnings",
              c is not None and not design.check(c, "bg"), err or design.describe(c))
        _, err = roundtrip(b, d, "kind.brain", kind="vision")
        check("loading it as a different kind is refused", err is not None and "expected" in err, err)
        b2 = design.stamp(Brain(S.stage_bg(), seed=0), "bg")
        b2.design["runtime"] = design.RUNTIME_VERSION - 1
        _, err = roundtrip(b2, d, "runtime.brain")
        check("a different runtime version is refused", err is not None and "runtime version" in err, err)
        b3 = design.stamp(Brain(S.stage_bg(), seed=0), "bg")
        b3.spec.synaptic[0] = replace(b3.spec.synaptic[0], gain=b3.spec.synaptic[0].gain * 2)
        _, err = roundtrip(b3, d, "edited.brain")
        check("a design that no longer matches its fingerprint is refused", err is not None and "fingerprint" in err, err)
        b4 = Brain(S.stage_bg(), seed=0)
        c, err = roundtrip(b4, d, "legacy.brain")
        w = design.check(c) if c is not None else []
        check("an unstamped (older) save loads with a warning", c is not None and any("before design" in x for x in w), w)
        b5 = Brain(S.stage_bg(), seed=0)
        b5.spec.synaptic[0] = replace(b5.spec.synaptic[0], gain=b5.spec.synaptic[0].gain * 2)
        design.stamp(b5, "bg")   # stamped, but with a design the current code no longer builds
        c, err = roundtrip(b5, d, "old-design.brain", kind="bg")
        w = design.check(c, "bg") if c is not None else []
        check("a brain from an older design loads with a 'design has changed' warning",
              c is not None and any("has changed" in x for x in w), w)
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

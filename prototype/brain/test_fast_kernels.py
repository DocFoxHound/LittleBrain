"""The compiled inner loops (fastkernels.py, note 79) match the numpy code they replace."""
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
np.seterr(over="ignore")
import brainspec as S  # noqa: E402
import fastkernels  # noqa: E402
from cells import ExcitatoryGroup, SynapseBank  # noqa: E402
from federation import Brain  # noqa: E402

passed = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail else ""))
    passed.append(ok)


def main():
    if not fastkernels.ENABLED:
        print("compiled kernels are switched off (LITTLEBRAIN_FAST=0 or no numba): nothing to compare")
        return True
    b = Brain(S.stage_amy(), seed=0)
    rng = np.random.default_rng(0)
    x = (rng.random(b.n_total) < 0.2).astype(np.float32)
    worst = {"drive": 0.0, "fast": 0.0, "elig": 0.0, "tag": 0.0}
    for g in b.g.values():
        if not isinstance(g, ExcitatoryGroup):
            continue
        for bank in g.banks():
            if not hasattr(bank, "_fast_drive"):
                continue
            bank.fast = rng.normal(0, .1, bank.fast.shape).astype(np.float32)
            bank.elig = (rng.random(bank.elig.shape) * .2).astype(np.float32)
            ref = copy.deepcopy(bank)
            d1, _ = bank.drive(x)
            d0, _ = SynapseBank.drive(ref, x)
            worst["drive"] = max(worst["drive"], float(np.abs(d1 - d0 * ref.norm).max() / (np.abs(d0).max() + 1e-9)))
            xs = (rng.random(bank.src.shape) < .2).astype(np.float32)
            post = rng.random(g.n).astype(np.float32)
            pl = (0.5 + rng.random(g.n)).astype(np.float32)
            bank.trace(xs, post)
            SynapseBank.trace(ref, xs, post)
            bank.modulate(pl, 0.7, 0.6, 0.4)
            SynapseBank.modulate(ref, pl, 0.7, 0.6, 0.4)
            for k in ("fast", "elig", "tag"):
                a, c = getattr(bank, k), getattr(ref, k)
                worst[k] = max(worst[k], float(np.abs(a - c).max() / (np.abs(c).max() + 1e-9)))
    check("compiled drive, traces and learning match numpy (relative difference < 1e-5)",
          all(v < 1e-5 for v in worst.values()), str({k: f"{v:.1e}" for k, v in worst.items()}))
    # the same brain run for 40 steps both ways stays essentially the same
    traces = {}
    for mode in ("1", "0"):
        code = (f"import sys,numpy as np; sys.path.insert(0,'{Path(__file__).resolve().parent}'); "
                f"sys.path.insert(0,'{Path(__file__).resolve().parent.parent}'); np.seterr(over='ignore'); "
                "import brainspec as S; from federation import Brain; b=Brain(S.stage_amy(),seed=0); "
                "r=np.random.default_rng(1); q={k:np.zeros(len(b.pos[k]),np.float32) for k in b.spec.inputs}\n"
                "for t in range(40): q['retina']=(r.random(len(b.pos['retina']))<.2).astype(np.float32); "
                "b.step(q, learn=True, dopamine=0.5 if t%9==2 else 0.0, ach=0.6, arousal=0.3)\n"
                "np.save(sys.argv[1], np.asarray(b.r))")
        out = Path(os.environ.get("TMPDIR", "/tmp")) / f"lb_fast_{mode}_{os.getpid()}.npy"
        subprocess.run([sys.executable, "-c", code, str(out)], check=True, env=dict(os.environ, LITTLEBRAIN_FAST=mode),
                       capture_output=True)
        traces[mode] = np.load(out)
        out.unlink()
    a, c = traces["1"], traces["0"]
    r = float(np.corrcoef(a, c)[0, 1])
    check("40 steps with learning: compiled and numpy runs agree (correlation > 0.999)", r > 0.999,
          f"correlation {r:.6f}, largest rate difference {float(np.abs(a - c).max()):.4f}")
    print(f"\n{sum(passed)}/{len(passed)} passed")
    return all(passed)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

"""
Fading precision: run everything (about 15 minutes; never more than 3 busy threads).

  python run.py            all steps, then delete the big intermediate files in data/
  python run.py --keep     same, but keep data/ (the simulated brain snapshots, ~35 MB)
  python run.py <step>     one step: sim | lossless | behavior | fringe | coldstore (coldstore needs fringe first)

Results go to results/ (JSON + text logs); README.md quotes them.
"""

import shutil
import subprocess
import sys

from common import DATA, HERE, RESULTS

PY = sys.executable


def sh(*args, log=None):
    print(f"\n=== {' '.join(args)}", flush=True)
    out = subprocess.run([PY, *args], cwd=HERE, capture_output=True, text=True)
    text = out.stdout + out.stderr
    print(text, flush=True)
    if log:
        (RESULTS / log).write_text(text)
    if out.returncode:
        raise SystemExit(f"step failed: {args}")


def step(name):
    if name == "sim":
        sh("brainsim.py", log="brainsim_log.txt")
    elif name == "lossless":
        sh("lossless.py", log="lossless.txt")
    elif name == "behavior":
        sh("behavior.py", "named", log="behavior_named.txt")
        sh("behavior.py", "grid", log="behavior_grid_eager_prune.txt")
        sh("behavior.py", "gridkeep", log="behavior_grid_eager_keepwiring.txt")
        sh("behavior.py", "gridlazy", log="behavior_grid_lazy_keepwiring.txt")
        sh("behavior.py", "ablation", log="behavior_ablation.txt")
    elif name == "coldstore":
        sh("coldstore.py", "sizes", log="coldstore_sizes.txt")
        sh("coldstore.py", "speed", log="coldstore_speed.txt")
        sh("coldstore.py", "pop", log="coldstore_pop.txt")
    elif name == "fringe":
        sh("fringe.py", log="fringe.txt")
        sh("behavior.py", "fringe", log="behavior_fringe.txt")
    else:
        raise SystemExit(f"unknown step {name}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        for a in args:
            step(a)
    else:
        if not (DATA / "pop_day16.pkl").exists():
            step("sim")
        for s in ("lossless", "behavior", "fringe", "coldstore"):   # coldstore uses fringe's codebooks
            step(s)
        if "--keep" not in sys.argv:
            shutil.rmtree(DATA, ignore_errors=True)
            print("deleted", DATA)

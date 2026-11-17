"""
SSD-dormant experiment: one command runs everything.

    .venv/bin/python prototype/experiments/ssd-dormant/run.py                 # all parts, then delete data
    .venv/bin/python prototype/experiments/ssd-dormant/run.py drive wake      # selected parts
    .venv/bin/python prototype/experiments/ssd-dormant/run.py all --keep      # keep the 24 GiB brain file

Parts: create (24 GiB brain, 26 GB written), drive (8 GB written), span, wake, day (3 GB),
night (77 GB: two full nights + 4 GiB variants + lazy nights), dormant (1 GB), behavior, clean.
Results go to results/*.json and *.log and are summarized in README.md. Needs ~26 GB free in this
folder; a full run takes ~25 minutes and writes ~115 GB (the WD SN770 2 TB is rated for
1,200 TBW, so that is ~0.01% of its life).
"""

import shutil
import sys

import common  # noqa: F401  (sets thread caps before numba loads)

PARTS = ["create", "drive", "span", "wake", "day", "night", "dormant", "behavior"]


def clean():
    for p in common.DATA.iterdir():
        if p.is_file():
            p.unlink()
    print(f"deleted data files in {common.DATA}")


def main(argv):
    keep = "--keep" in argv
    parts = [a for a in argv if not a.startswith("--")] or ["all"]
    if "all" in parts:
        parts = PARTS + ([] if keep else ["clean"])
    for p in parts:
        print(f"\n######## {p} ########", flush=True)
        if p == "create":
            import brain
            brain.ensure()
        elif p == "clean":
            clean()
        else:
            if p in ("drive", "span", "wake", "day", "night", "dormant"):
                import brain
                brain.ensure()
            mod = __import__(p)
            mod.main()
    print(f"\nSSD bytes written by this experiment (ledger): {common.LEDGER.total / 1e9:.1f} GB")
    print(f"free space here: {shutil.disk_usage(common.HERE).free / 1e12:.2f} TB")


if __name__ == "__main__":
    main(sys.argv[1:])

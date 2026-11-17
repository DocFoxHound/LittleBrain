"""
Run every experiment of the vram-ram-paging study, one subprocess per step (each under ~2 min),
saving each step's output to results/<step>.txt and the benchmark rows to results/bench.jsonl.

  .venv/bin/python prototype/experiments/vram-ram-paging/run.py           # everything (~15-20 min)
  .venv/bin/python prototype/experiments/vram-ram-paging/run.py --quick   # checks + 1M bench + reorder (~5 min)

Caps (BRIEF.md): 3 CPU threads, VRAM pool <= 2.05 GB (+ CUDA context, < 2.5 GB total),
RAM <= 10 GB (the 8M-neuron runs pin 8.6 GB of host RAM; peak RSS 8.7 GB).
"""

import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M1, M8 = "1048576", "8388608"


def bench(*args):
    return ["bench.py", *args]


FULL = [
    ("verify_kernel", ["verify_kernel.py"]),
    ("test_paging", ["test_paging.py"]),
    ("behavior", ["behavior.py"]),
    ("pcie_micro", ["pcie_micro.py"]),
    ("bench_A_1M", bench("--n", M1, "--modes", "paged,zerocopy,resident", "--cache", "256",
                         "--frac", "0.001,0.01,0.05,0.2,1.0", "--churn", "0.02,1.0", "--steps", "10",
                         "--tag", "A_1M", "--sleep")),
    ("bench_B_8M", bench("--n", M8, "--modes", "paged,zerocopy", "--cache", "1536", "--frac", "0.001,0.01,0.05",
                         "--churn", "0.02,1.0", "--steps", "10", "--tag", "B_8M")),
    ("bench_C_8M", bench("--n", M8, "--modes", "paged,zerocopy", "--cache", "1536", "--frac", "0.2",
                         "--churn", "0.02,1.0", "--rounds", "2", "--steps", "6", "--warmup", "4", "--tag", "C_8M",
                         "--sleep")),
    ("bench_D_churn", bench("--n", M8, "--modes", "paged,zerocopy", "--cache", "1536", "--frac", "0.05",
                            "--churn", "0,0.005,0.05,0.2,0.5", "--rounds", "2", "--steps", "8", "--tag", "D_churn")),
    ("bench_E_cache", bench("--n", M8, "--modes", "paged", "--cache", "1536,1024,512,448,384,256,128",
                            "--frac", "0.05", "--churn", "0.02,0.2", "--rounds", "2", "--steps", "8", "--warmup", "8",
                            "--tag", "E_cache")),
    ("bench_F_managed", bench("--n", M8, "--modes", "managed", "--cache", "1536", "--frac", "0.05",
                              "--churn", "0.02,0.2,1.0", "--rounds", "2", "--steps", "8", "--tag", "F_managed",
                              "--sleep")),
    ("bench_F_managed_remote", bench("--n", M8, "--modes", "managed", "--cache", "0", "--frac", "0.05",
                                     "--churn", "0.02,1.0", "--rounds", "2", "--steps", "6", "--warmup", "4",
                                     "--tag", "F_managed_remote")),
    ("bench_G_lazy", bench("--n", M1, "--modes", "managed_lazy", "--cache", "256", "--frac", "0.05",
                           "--churn", "0.02,0.2,1.0", "--rounds", "2", "--steps", "8", "--tag", "G_lazy")),
    ("bench_G_managed1M", bench("--n", M1, "--modes", "managed", "--cache", "256", "--frac", "0.05",
                                "--churn", "0.02,0.2,1.0", "--rounds", "2", "--steps", "8", "--tag", "G_managed1M")),
    ("bench_G_paged1M", bench("--n", M1, "--modes", "paged", "--cache", "256", "--frac", "0.05",
                              "--churn", "0.2", "--rounds", "2", "--steps", "8", "--tag", "G_paged1M")),
    ("cpu_forward", ["cpu_forward.py"]),
    ("reorder", ["reorder.py"]),
    ("summarize", ["summarize.py"]),
]
QUICK = {"verify_kernel", "test_paging", "behavior", "pcie_micro", "bench_A_1M", "reorder", "summarize"}


def main():
    quick = "--quick" in sys.argv
    os.makedirs(RES, exist_ok=True)
    jl = os.path.join(RES, "bench.jsonl")
    if os.path.exists(jl):
        shutil.move(jl, os.path.join(RES, "bench.prev.jsonl"))
    env = dict(os.environ, OMP_NUM_THREADS="3", NUMBA_NUM_THREADS="3", MKL_NUM_THREADS="3",
               OPENBLAS_NUM_THREADS="3", CUPY_GPU_MEMORY_LIMIT="2050000000", PYTHONWARNINGS="ignore")
    for name, args in FULL:
        if quick and name not in QUICK:
            continue
        t0 = time.perf_counter()
        print(f"\n######## {name}: {' '.join(args)}", flush=True)
        p = subprocess.run([sys.executable, *args], cwd=HERE, env=env, capture_output=True, text=True)
        out = p.stdout + ("\n" + p.stderr if p.returncode else "")
        with open(os.path.join(RES, f"{name}.txt"), "w") as fh:
            fh.write(out)
        print(out.rstrip())
        print(f"######## {name}: {'OK' if p.returncode == 0 else 'FAILED'} in {time.perf_counter() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()

"""
Run every experiment of the cpu-event-driven strategy, in order (about 4 minutes):

  validate.py       correctness of every kernel against prototype/packed_neuron.py
  behavior.py       the childhood -> hard times -> relearn story, seeds 1-8, all implementations
  bench_numpy.py    the NumPy baseline in this session
  bench_layouts.py  dendrite-major vs axon-major vs bit-sliced at 1-100% input activity, 1 and 3 threads
  bench_cache.py    population size across the 24 MB L3; axon-major block (tile) size
  wiring_bits.py    can axon-major wiring fit in the 4 freed gen bits?
  summarize.py      results_tables.md

Run: .venv/bin/python prototype/experiments/cpu-event-driven/run.py
Each script also runs on its own. Threads are capped at 3, memory stays under ~2.5 GB.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["validate.py", "behavior.py", "bench_numpy.py", "bench_layouts.py", "bench_cache.py",
         "wiring_bits.py", "summarize.py"]

if __name__ == "__main__":
    env = dict(os.environ, NUMBA_NUM_THREADS="3", OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3",
               KMP_BLOCKTIME="0", OMP_WAIT_POLICY="PASSIVE")
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, os.path.join(HERE, step)], check=True, env=env)
        print(f"({time.time() - t0:.0f} s)", flush=True)

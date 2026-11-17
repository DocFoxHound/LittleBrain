"""
Rolling local sleep: run every experiment, then print the summary tables used in README.md.

  .venv/bin/python prototype/experiments/rolling-local-sleep/run.py            # everything (~6 min)
  .venv/bin/python prototype/experiments/rolling-local-sleep/run.py --summary  # tables only

Each step is a separate process and stays under ~2 minutes:
  1. bench_sleep.py     CPU sleep kernel: correctness + speed (NumPy / C / Numba / GPU), PCIe, GPU awake kernels
  2. bench_schedule.py  sync-gpu / sync-cpu / rolling (several M, S) at 1,008,000 neurons
  3. bench_schedule.py  micro-sleep and two-hemisphere rolling
  4. bench_schedule.py  the same with a 3x heavier night
  5. behavior.py        demo adapter check + population story under each schedule
  6. phase_check.py     why the night's timing matters (NumPy prototype vs kernels, cascade trace)
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
R = HERE / "results"

STEPS = [
    ["bench_sleep.py"],
    ["bench_schedule.py", "--only", "sync-gpu,sync-cpu,rolling M=8 S=1 +spare,rolling M=8 S=2 +spare,"
     "rolling M=8 S=4 +spare,rolling M=8 S=4 nospare", "--out", "schedules_m8.json"],
    ["bench_schedule.py", "--only", "micro-gpu,rolling M=2 S=1 +spare", "--out", "schedules_other.json"],
    ["bench_schedule.py", "--only", "sync-gpu,rolling M=8 S=2 +spare,rolling M=8 S=4 +spare", "--reps", "3",
     "--repeats", "2", "--out", "schedules_heavy3.json"],
    ["behavior.py"],
    ["phase_check.py"],
]


def summary():
    d = json.loads((R / "sleep_kernel.json").read_text())
    print("\n== Sleep kernel (median wall s; peak temporary MiB) ==")
    for n, r in d["speed"].items():
        row = {k: f"{v['s']:.4f}s/{v.get('peak_temp_MiB', 0):.1f}MiB" for k, v in r.items()
               if isinstance(v, dict) and "s" in v}
        print(f"  {int(n):>9,} neurons: {row}")
        print(f"  {'':>9}  PCIe: D2H {r['pcie']['d2h_GBps']:.1f} GB/s, H2D {r['pcie']['h2d_GBps']:.1f} GB/s")
    c = d["correctness"]
    print("  C == Numba == GPU (bit for bit):", c["bit_identical_pending0"], c["bit_identical_pending37"])
    print("  vs NumPy, same state:", json.dumps(c["vs_numpy_same_state"]))
    print("\n== GPU awake kernels ==")
    for n, r in d["gpu_awake"].items():
        print(f"  {int(n):>9,}: " + ", ".join(f"{k} {v['s'] * 1e3:.2f} ms ({v['neuron_steps_per_s']:.3g} neuron-steps/s)"
                                            for k, v in r.items() if isinstance(v, dict)))

    print("\n== Schedules (1,008,000 neurons, D = 84 awake steps per day; medians of repeats) ==")
    cols = ["wall_per_day_s", "throughput_vs_sync_gpu", "gpu_awake_busy_frac", "gpu_sleep_busy_frac",
            "gpu_idle_frac", "cpu_sleep_worker_busy_frac", "process_cpu_util_of_3_threads",
            "vram_bytes_per_neuron", "pinned_ram_bytes_per_neuron", "cpu_night_s_median", "night_latency_s_median"]
    for f in ("schedules_m8.json", "schedules_other.json", "schedules_heavy3.json"):
        p = R / f
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        print(f"  -- {f} (night passes = {s['reps']})")
        base = None
        for name, v in s["summary"].items():
            base = base or (v["wall_per_day_s"] if name == "sync-gpu" else None)
        for name, v in s["summary"].items():
            vals = " ".join(f"{k.split('_frac')[0][:14]}={v[k]:.3f}" for k in cols if isinstance(v.get(k), float))
            print(f"  {name:26s} {vals}  runs={v['wall_per_day_runs']}")

    b = json.loads((R / "behavior.json").read_text())
    print("\n== Behavior: demo adapter (seeds 1-16) ==")
    for k, v in b["adapter"].items():
        d6 = v["time passes: day 6"]
        print(f"  {k:36s} day6 {d6['mean']:.3f} [{d6['min']:.2f}-{d6['max']:.2f}]  "
              f"day3 {v['time passes: day 3']['mean']:.3f}  relearn median {v['relearn']['median']:.0f}")
    print("\n== Behavior: population story (5 blocks x 200 neurons, seeds 1-8) ==")
    for k, v in b["population"].items():
        print(f"  {k:30s} day3 {v['day 3']['mean']:.3f} day6 {v['day 6']['mean']:.3f} "
              f"(per block {v['day 6']['per_block_mean']}, frac>0.06 {v['day 6']['frac_neurons_above_0.06']:.2f})  "
              f"relearn median {v['relearn']['median']:.0f} mean {v['relearn']['mean']:.2f}  "
              f"events seen {v['frac_world_events_seen_by_some_block']:.2f}")


def main():
    if "--summary" not in sys.argv:
        for step in STEPS:
            print(f"\n>>> {' '.join(step)}", flush=True)
            subprocess.check_call([PY, str(HERE / step[0]), *step[1:]])
    summary()


if __name__ == "__main__":
    main()

"""
Print the result tables (markdown) from results/bench.jsonl. Times are the best round's median
(other processes time-slice the shared GPU and only ever add time); all rounds are in the JSON.

Run: .venv/bin/python prototype/experiments/vram-ram-paging/summarize.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = [json.loads(line) for line in open(os.path.join(HERE, "results", "bench.jsonl"))]
    for r in rows:
        if "fwd_ms_rounds" in r and "fwd_ms_best" not in r:
            r["fwd_ms_best"] = min(r["fwd_ms_rounds"])
            r["mod_ms_best"] = min(r["mod_ms_rounds"]) if r.get("mod_ms_rounds") else None
    return rows


def step_table(rows, tag, title):
    rs = [r for r in rows if r.get("tag") == tag and r.get("op") != "sleep"]
    if not rs:
        return
    n = rs[0]["n"]
    print(f"\n### {title}\n")
    print("| mode | cache (blocks) | active | churn | forward ms | modulate ms | M neuron-steps/s (all N) | "
          "M active neuron-steps/s | PCIe in / out per step, forward (MB) | PCIe in / out, modulate (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda r: (r["cache_blocks"] if r["mode"] != "zerocopy" else 0, r["frac"], r["churn"])):
        f = r["fwd_ms_best"]
        print(f"| {r['mode']} | {r['cache_blocks'] or '-'} | {r['frac']:.1%} | {r['churn']} | {f:.2f} | "
              f"{r['mod_ms_best'] or 0:.2f} | {n / f / 1e3:,.0f} | {r['active_neurons'] / f / 1e3:,.1f} | "
              f"{r['h2d_MB_fwd']:.1f} / {r['d2h_MB_fwd']:.1f} | {r['h2d_MB_mod']:.1f} / {r['d2h_MB_mod']:.1f} |")


def sleep_table(rows):
    rs = [r for r in rows if r.get("op") == "sleep"]
    print("\n### Sleep (one full night: every synapse)\n")
    print("| neurons | mode | seconds | M neurons/s | PCIe in / out (MB) |")
    print("|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['n']:,} | {r['mode']} | {r['sleep_s']:.3f} | {r['Mneurons_per_s']:.1f} | "
              f"{r['h2d_MB']:.0f} / {r['d2h_MB']:.0f} |")


if __name__ == "__main__":
    rows = load()
    step_table(rows, "A_1M", "1M neurons (fits in VRAM): resident vs paged (cache 256 blocks = 25%) vs zero-copy")
    step_table(rows, "B_8M", "8M neurons (8.6 GB, does not fit): paged (cache 1,536 blocks = 19%) vs zero-copy")
    step_table(rows, "C_8M", "8M neurons, 20% active: working set (1,638 blocks) larger than the cache")
    step_table(rows, "D_churn", "8M neurons, 5% active: churn sweep")
    step_table(rows, "E_cache", "8M neurons, 5% active: cache-size sweep (working set 410 blocks)")
    step_table(rows, "F_managed", "8M neurons, CUDA managed memory + prefetch hints (residency budget 1,536 blocks)")
    step_table(rows, "F_managed_remote", "8M neurons, CUDA managed memory, preferred location = host (no migration)")
    for tag, t in [("G_lazy", "1M, managed, no hints (page faults bring blocks in; eviction forced at 256 blocks)"),
                   ("G_managed1M", "1M, managed + prefetch hints, 256 blocks"),
                   ("G_paged1M", "1M, explicit paging, 256 blocks")]:
        step_table(rows, tag, t)
    sleep_table(rows)

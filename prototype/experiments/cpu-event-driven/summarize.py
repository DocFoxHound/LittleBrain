"""
Turn results_*.json into the README tables. The machine was shared with other agents, so each cell
is the BEST (minimum) of the per-repeat medians: the closest estimate of the uncontended cost.

Run: .venv/bin/python prototype/experiments/cpu-event-driven/summarize.py
"""

import common

import json
import os

import numpy as np


def load(name):
    p = os.path.join(common.HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def best(rows, **kw):
    sel = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return sel


def main():
    L = load("results_layouts.json")
    NP = load("results_numpy.json")
    syn = L["synapses"]
    layouts = ["dendrite", "axon", "bitslice"]
    acts = sorted({r["activity"] for r in L["speed"]})
    lines = []
    P = lines.append

    np4096 = next(r for r in NP if r["n_inputs"] == 4096) if NP else None

    def cell(layout, th, act, key):
        v = [r[key] for r in L["speed"] if r["layout"] == layout and r["threads"] == th and r["activity"] == act]
        return min(v)

    for key, title in (("forward_s", "forward"), ("modulate_s", "modulate (tick first)")):
        P(f"\n**{title}: milliseconds per step, 100,000 neurons (25.6 M synapses), 4,096 inputs**\n")
        P("| input activity | elig. union | " + " | ".join(f"{l} 1T" for l in layouts) + " | "
          + " | ".join(f"{l} 3T" for l in layouts) + " |")
        P("|---" * (2 + 2 * len(layouts)) + "|")
        for a in acts:
            u = np.mean([r["elig_union"] for r in L["speed"] if r["activity"] == a])
            vals = [cell(l, th, a, key) * 1e3 for th in (1, 3) for l in layouts]
            P(f"| {a:.0%} | {u:.0%} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        if np4096:
            P(f"\nNumPy baseline, same scenario: {np4096[key] * 1e3:.0f} ms (1 thread).")

    P("\n**Throughput, 2% activity (synapses of the whole population per second; events/s for axon-major)**\n")
    P("| layout | step | 1 thread | 3 threads | speedup vs NumPy (1T / 3T) |")
    P("|---|---|---|---|---|")
    for key, nm in (("forward_s", "forward"), ("modulate_s", "modulate")):
        for l in layouts:
            t1, t3 = cell(l, 1, 0.02, key), cell(l, 3, 0.02, key)
            extra = ""
            if l == "axon":
                frac = 0.02 if key == "forward_s" else np.mean(
                    [r["elig_union"] for r in L["speed"] if r["activity"] == 0.02])
                extra = f" ({syn * frac / t1 / 1e9:.2f} / {syn * frac / t3 / 1e9:.2f} G events/s)"
            sp = f"{np4096[key] / t1:.0f}x / {np4096[key] / t3:.0f}x" if np4096 else ""
            P(f"| {l} | {nm} | {syn / t1 / 1e9:.2f} G/s | {syn / t3 / 1e9:.2f} G/s{extra} | {sp} |")

    P("\n**Steps that touch every synapse (ms)**\n")
    P("| layout | catch-up 1T | catch-up 3T | sleep 1T | sleep 3T |")
    P("|---|---|---|---|---|")
    for l in layouts:
        f = [r for r in L["fixed"] if r["layout"] == l and "catch_up_s" in r]
        c1 = min(r["catch_up_s"] for r in f if r["threads"] == 1) * 1e3
        c3 = min(r["catch_up_s"] for r in f if r["threads"] == 3) * 1e3
        s1 = min(r["sleep_s"] for r in f if r["threads"] == 1) * 1e3
        s3 = min(r["sleep_s"] for r in f if r["threads"] == 3) * 1e3
        P(f"| {l} | {c1:.0f} | {c3:.0f} | {s1:.0f} | {s3:.0f} |")
    if np4096:
        P(f"\nNumPy baseline sleep: {np4096['sleep_s'] * 1e3:.0f} ms (catch-up is part of it).")
    floor = {th: min(r["soma_floor_s"] for r in L["fixed"] if r.get("soma_floor_s") and r["threads"] == th)
             for th in (1, 3)}
    P(f"\nAxon-major forward with ZERO active inputs (the per-neuron floor): {floor[1] * 1e3:.2f} ms (1T), "
      f"{floor[3] * 1e3:.2f} ms (3T).")

    P("\n**Memory**\n")
    P("| layout | bytes/neuron (stored) | peak temp. forward | modulate | sleep |")
    P("|---|---|---|---|---|")
    for l in layouts:
        m, pt = L["memory"][l], L["peak_temp_MiB"][l]
        P(f"| {l} | {m['bytes_per_neuron']:.0f} | {pt['py_forward']:.1f} MiB | {pt['py_modulate']:.1f} MiB | "
          f"{pt['py_sleep']:.1f} MiB |")
    if np4096:
        P(f"| NumPy baseline | {np4096['bytes_per_neuron']:.0f} | {np4096['peak_forward_MiB']:.0f} MiB | "
          f"{np4096['peak_modulate_MiB']:.0f} MiB | {np4096['peak_sleep_MiB']:.0f} MiB |")

    C = load("results_cache.json")
    if C:
        P("\n**Cache: ns per synapse vs population size (1 thread, 10% activity)**\n")
        keys = ["stream_read", "dendrite_fwd", "axon_fwd", "bitslice_fwd", "dendrite_cu", "bitslice_cu"]
        P("| neurons | synapse MiB | " + " | ".join(keys) + " |")
        P("|---" * (2 + len(keys)) + "|")
        for r in C["size_sweep"]:
            n = r["n_neurons"] * 256
            P(f"| {r['n_neurons']:,} | {r['synapse_MiB']:.0f} | " + " | ".join(f"{r[k] / n * 1e9:.2f}" for k in keys) + " |")
        P("\n**Axon-major block (tile) size, 100,000 neurons, 1 thread (ms)**\n")
        P("| block (neurons) | accumulator | index bytes/neuron | fwd 2% | mod 2% | fwd 20% | mod 20% |")
        P("|---|---|---|---|---|---|---|")
        for b in sorted({r["block"] for r in C["block_sweep"]}):
            r2 = next(r for r in C["block_sweep"] if r["block"] == b and r["activity"] == 0.02)
            r20 = next(r for r in C["block_sweep"] if r["block"] == b and r["activity"] == 0.2)
            P(f"| {b} | {r2['accumulator_KiB']:.0f} KiB | {r2['index_bytes_per_neuron']:.0f} | "
              f"{r2['forward_s'] * 1e3:.2f} | {r2['modulate_s'] * 1e3:.1f} | {r20['forward_s'] * 1e3:.2f} | "
              f"{r20['modulate_s'] * 1e3:.1f} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(common.HERE, "results_tables.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()

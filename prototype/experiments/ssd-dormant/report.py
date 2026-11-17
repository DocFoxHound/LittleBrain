"""Print the markdown tables used in README.md from results/*.json."""

from common import load


def wake_table():
    r = load("wake")
    if not r:
        return
    rows = r["rows"]
    methods = ["ram", "aio-4K", "aio-64K", "aio-1M", "scan", "mmap-random", "mmap-normal"]
    print("| active | locality | " + " | ".join(methods) + " | best SSD / RAM |")
    print("|---|---|" + "---:|" * (len(methods) + 1))
    keys = sorted({(x["frac"], x["locality"]) for x in rows},
                  key=lambda k: (k[0], ["scattered", "columnar", "clustered"].index(k[1])))
    for f, loc in keys:
        cell = {x["method"]: x for x in rows if x["frac"] == f and x["locality"] == loc}
        ram = cell["ram"]["neuron_steps_per_s"]
        best = max(v["neuron_steps_per_s"] for m, v in cell.items() if m != "ram")
        out = []
        for m in methods:
            if m in cell:
                v = cell[m]["neuron_steps_per_s"]
                out.append(f"{v / 1e3:,.0f}k" + ("*" if cell[m].get("partial") else ""))
            else:
                out.append("")
        print(f"| {f:.0%} | {loc} | " + " | ".join(out) + f" | {best / ram:.2f} |")
    print("\nbytes read per step (GB) and read amplification:")
    print("| active | locality | " + " | ".join(methods[1:]) + " |")
    print("|---|---|" + "---:|" * (len(methods) - 1))
    for f, loc in keys:
        cell = {x["method"]: x for x in rows if x["frac"] == f and x["locality"] == loc}
        out = [f"{cell[m]['GB_read_per_step']:.1f} ({cell[m]['amplification']:.0f}x)" if m in cell else ""
               for m in methods[1:]]
        print(f"| {f:.0%} | {loc} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    wake_table()

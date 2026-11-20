"""Traverse a host-defined action layout using seven choices and one back action."""
import math
import numpy as np


def categories(items):
    if len(items) <= 7:
        return items
    chunk = math.ceil(len(items) / 7)
    return [dict(label=f"{items[i]['label']} … {items[min(i + chunk, len(items)) - 1]['label']}",
                 children=categories(items[i:i + chunk])) for i in range(0, len(items), chunk)]


class Controls:
    def __init__(self, layout):
        ops = []
        for control in layout["controls"]:
            name, kind = control["name"], control["kind"]
            if kind in ("key", "button"):
                values = ("press", "hold", "release")
            elif kind in ("axis", "pointer"):
                values = control["values"]
            elif kind == "text":
                ops.append(dict(label=name, scalar_range=(0, 0x110000 - 0x800)))
                continue
            else:
                raise ValueError(f"unknown control kind {kind}")
            for value in values:
                ops.append(dict(label=f"{name} {value}", event=dict(kind=kind, name=name, value=value)))
        if not ops:
            raise ValueError("empty control layout")
        self.stack = [categories(ops)]
        self.held = set()
        self.axes = {}

    def panel(self):
        return dict(slots=[x["label"] for x in self.stack[-1]], depth=len(self.stack) - 1,
                    held=sorted(self.held), axes=dict(self.axes))

    def select(self, index):
        if index == 7:
            if len(self.stack) > 1:
                self.stack.pop()
            return []
        if index < 0 or index >= len(self.stack[-1]):
            return []
        item = self.stack[-1][index]
        if "children" in item:
            self.stack.append(item["children"])
            return []
        if "scalar_range" in item:
            lo, hi = item["scalar_range"]
            if hi - lo == 1:
                self.stack[:] = self.stack[:1]
                return [dict(kind="text", value=chr(lo if lo < 0xd800 else lo + 0x800))]
            stride = math.ceil((hi - lo) / 7)
            self.stack.append([dict(label=f"symbol {i:x}", scalar_range=(i, min(i + stride, hi)))
                               for i in range(lo, hi, stride)])
            return []
        event = item["event"]
        kind, name, value = event["kind"], event["name"], event["value"]
        if kind in ("key", "button"):
            key = (kind, name)
            if value == "hold":
                self.held.add(key)
            elif value == "release":
                self.held.discard(key)
        if kind == "axis":
            self.axes[name] = value
        return [dict(event)]

    def reset(self):
        events = [dict(kind=k, name=n, value="release") for k, n in sorted(self.held)]
        events += [dict(kind="axis", name=n, value=0) for n in sorted(self.axes) if self.axes[n]]
        self.stack[:] = self.stack[:1]
        self.held.clear()
        self.axes.clear()
        return events

    def camera(self, image):
        """Show each menu choice as a stable visual pattern in the bottom strip."""
        import hashlib
        height, width = image.shape
        # Scale the game frame before allocating a strip even for one-pixel observations.
        frame = np.repeat(np.repeat(image, max(1, math.ceil(24 / height)), 0),
                          max(1, math.ceil(32 / width)), 1)
        strip = np.zeros((max(2, frame.shape[0] // 4), frame.shape[1]), np.float32)
        for i, label in enumerate(self.panel()["slots"] + ["back"]):
            slot = i if label != "back" else 7
            bits = np.unpackbits(np.frombuffer(hashlib.sha256(label.encode()).digest(), dtype=np.uint8))
            segment = strip[:, slot * frame.shape[1] // 8:(slot + 1) * frame.shape[1] // 8]
            segment[:] = np.resize(bits, segment.shape)
        return np.concatenate((frame, strip))

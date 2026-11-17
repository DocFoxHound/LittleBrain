"""
CPU/GPU switch for the brain (notes/25).

  DEVICE=cpu  (default) numpy
  DEVICE=gpu  CuPy on the graphics card (run with the project venv: .venv/bin/python)

A brain is always built and wired on the CPU, then moved with Brain.to("gpu"). Most of the code is written
with numpy functions; CuPy arrays dispatch them to the GPU (numpy's __array_function__ protocol). The
helpers below cover the places where that isn't enough: creating arrays at run time, random numbers, and
moving data between host and device.

The same brain gives statistically the same results on either device, but not bit-identical ones (the GPU
adds numbers in a different order and has its own random streams).
"""

import os

import numpy as np

try:
    import cupy
except ImportError:  # the system python has no CuPy; DEVICE=gpu needs .venv/bin/python
    cupy = None

DEVICE = os.environ.get("DEVICE", "cpu")
if DEVICE == "gpu" and cupy is None:
    raise ImportError("DEVICE=gpu needs CuPy: run with .venv/bin/python")


def xpof(a):
    """numpy or cupy, whichever holds `a`."""
    return cupy.get_array_module(a) if cupy is not None else np


def on_gpu(a):
    return cupy is not None and isinstance(a, cupy.ndarray)


def like(ref, x, dtype=np.float32):
    """`x` as an array on the same device as `ref`."""
    return xpof(ref).asarray(x, dtype=dtype)


def sround(x, rng):
    """Stochastic rounding on either device."""
    if on_gpu(x):
        return cupy.floor(x + cupy.random.random(x.shape, dtype=cupy.float32))
    return np.floor(x + rng.random(x.shape, dtype=np.float32))


def to_host(a):
    return a.get() if on_gpu(a) else a


def to_device(a, device):
    if device == "gpu":
        return cupy.asarray(a) if isinstance(a, np.ndarray) else a
    return to_host(a)


def move(obj, device, skip=()):
    """Move every array attribute of obj (and arrays inside dict/tuple/list attributes) to the device."""
    def conv(v):
        if isinstance(v, np.ndarray) or on_gpu(v):
            return to_device(v, device)
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, tuple):
            return tuple(conv(x) for x in v)
        if isinstance(v, list) and v and all(isinstance(x, (np.ndarray, tuple)) or on_gpu(x) for x in v):
            return [conv(x) for x in v]
        return v
    for k, v in list(vars(obj).items()):
        if k not in skip:
            setattr(obj, k, conv(v))

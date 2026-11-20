"""LittleBrain data file: a bounded JSON manifest followed by gzip-compressed numeric buffers.

Each caller supplies the complete set of model records and fields its file can contain.
Names in the file never select imports, functions or Python attributes outside that schema.
"""
import gzip
import json
import math
import os
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"LittleBrain data v1\n"
MAX_MANIFEST = 16 << 20
MAX_DATA = 2 << 30
DTYPES = {np.dtype(kind).newbyteorder(order).str for kind in
          ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
           "uint64", "float16", "float32", "float64") for order in ("<", ">")}


def save(path, value, schema, format_name):
    records, buffers, seen = [], [], {}

    def encode(x):
        if type(x) in (str, int, float, bool, type(None)):
            return x
        if isinstance(x, dict):
            return ["map", [[encode(k), encode(v)] for k, v in x.items()]]
        if type(x) in (tuple, list, set):
            return [type(x).__name__, [encode(v) for v in x]]
        if id(x) in seen:
            return ["ref", seen[id(x)]]
        if isinstance(x, (np.ndarray, np.generic)):
            a = np.asarray(x)
            if a.dtype.str not in DTYPES:
                raise ValueError(f"Unsupported numeric type {a.dtype}")
            record = ["scalar" if isinstance(x, np.generic) else "array", a.dtype.str, a.shape]
            buffers.append(a)
        elif type(x) is np.random.Generator and type(x.bit_generator) is np.random.PCG64:
            record = ["rng", x.bit_generator.state]
        elif type(x) in schema:
            fields = vars(x)
            unknown = fields.keys() - schema[type(x)]
            if unknown:
                raise ValueError(f"Unknown {type(x).__name__} fields: {sorted(unknown)}")
            record = ["record", type(x).__name__, {k: encode(v) for k, v in fields.items()}]
        else:
            raise ValueError(f"Unsupported checkpoint data {type(x).__name__}")
        seen[id(x)] = len(records)
        records.append(record)
        return ["ref", len(records) - 1]

    root = encode(value)
    manifest = json.dumps([format_name, records, root], separators=(",", ":"), allow_nan=False).encode()
    if len(manifest) > MAX_MANIFEST or sum(a.nbytes for a in buffers) > MAX_DATA:
        raise ValueError("Checkpoint exceeds format limits")
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        with temporary.open("wb") as f:
            f.write(MAGIC)
            with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=1, mtime=0) as compressed:
                compressed.write(struct.pack("<I", len(manifest)))
                compressed.write(manifest)
                for a in buffers:
                    compressed.write(np.ascontiguousarray(a).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, schema, format_name):
    classes = {cls.__name__: cls for cls in schema}
    records = []

    def decode(x):
        if type(x) in (str, int, float, bool, type(None)):
            return x
        tag, data = x
        if tag == "ref" and type(data) is int and 0 <= data < len(records):
            return records[data]
        if tag == "map" and type(data) is list:
            return {decode(k): decode(v) for k, v in data}
        if tag in ("tuple", "list", "set") and type(data) is list:
            return {"tuple": tuple, "list": list, "set": set}[tag](decode(v) for v in data)
        raise ValueError("Unknown checkpoint value")

    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("Unsupported checkpoint. Regenerate older object-format saves with this version.")
        try:
            with gzip.GzipFile(fileobj=f) as data:
                size, = struct.unpack("<I", data.read(4))
                if size > MAX_MANIFEST:
                    raise ValueError("Checkpoint manifest too large")
                name, entries, root = json.loads(data.read(size))
                if name != format_name or type(entries) is not list:
                    raise ValueError("Wrong checkpoint format")
                remaining = MAX_DATA
                for entry in entries:
                    tag = entry[0]
                    if tag in ("array", "scalar"):
                        _, dtype, shape = entry
                        if (dtype not in DTYPES or type(shape) is not list or len(shape) > 8
                                or any(type(n) is not int or not 0 <= n <= MAX_DATA for n in shape)
                                or (tag == "scalar" and shape)):
                            raise ValueError("Invalid numeric buffer")
                        size = math.prod(shape) * np.dtype(dtype).itemsize
                        if size > remaining:
                            raise ValueError("Checkpoint buffers too large")
                        remaining -= size
                        value = np.frombuffer(data.read(size), dtype=dtype).reshape(shape).copy()
                        if tag == "scalar":
                            value = value[()]
                    elif tag == "rng":
                        value = np.random.default_rng(0)
                        value.bit_generator.state = entry[1]
                    elif tag == "record":
                        _, name, fields = entry
                        cls = classes[name]
                        if type(fields) is not dict or fields.keys() - schema[cls]:
                            raise ValueError("Unknown model fields")
                        value = object.__new__(cls)
                        vars(value).update({k: decode(v) for k, v in fields.items()})
                    else:
                        raise ValueError("Unknown checkpoint record")
                    records.append(value)
                result = decode(root)
                if data.read(1):
                    raise ValueError("Unexpected checkpoint data")
                return result
        except (EOFError, OSError, zlib.error, KeyError, TypeError, IndexError, struct.error, RecursionError) as exc:
            raise ValueError("Invalid checkpoint data") from exc

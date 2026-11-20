"""Build the optional Linux benchmark extension before running its experiment."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from cffi import FFI

KERNELS = {"bitslice": "cpu-event-driven", "csleep": "rolling-local-sleep", "aio": "ssd-dormant"}


def build(name):
    directory = Path(__file__).resolve().parent / KERNELS[name]
    ffi = FFI()
    ffi.cdef((directory / f"{name}.h").read_text())
    flags = ["-O3"]
    link_flags = []
    if name != "aio":
        flags += ["-march=native", "-fopenmp"]
        link_flags.append("-fopenmp")
    if name == "csleep":
        flags += ["-ffp-contract=off", "-fno-math-errno", "-fno-trapping-math"]
    ffi.set_source(f"_lb_{name}", f'#include <stdint.h>\n#include "{name}.h"',
                   sources=[str(directory / f"{name}.c")], include_dirs=[str(directory)],
                   extra_compile_args=flags, extra_link_args=link_flags, libraries=["m"])
    with TemporaryDirectory() as work:
        ffi.compile(tmpdir=work, target=str(directory / f"_lb_{name}.*"), verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=KERNELS)
    build(parser.parse_args().kernel)

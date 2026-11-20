"""Regression checks for the data-only save format."""
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import data_archive as archive


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "state"

    def test_shared_state_and_atomic_replace(self):
        rng = np.random.default_rng(4)
        rng.random(8)
        array = np.arange(12, dtype=">i4").reshape(3, 4)[:, ::2]
        archive.save(self.path, {(None, 1): [array, array, rng, rng]}, {}, "test")
        with self.assertRaises(ValueError):
            archive.save(self.path, np.array([object()], dtype=object), {}, "test")
        a, b, r, s = archive.load(self.path, {}, "test")[None, 1]
        np.testing.assert_array_equal(array, a)
        self.assertEqual(array.dtype, a.dtype)
        self.assertIs(a, b)
        self.assertIs(r, s)
        self.assertEqual(r.random(), rng.random())

    def test_reject_foreign_files_and_incomplete_arrays(self):
        self.path.write_bytes(b"obsolete format")
        with self.assertRaises(ValueError):
            archive.load(self.path, {}, "test")
        for entry in (["record", "unknown", {}], ["array", "|O", [1]], ["array", "<i4", [16]],
                      ["array", "<i4", [-1]], ["array", "<i4", [archive.MAX_DATA]]):
            with self.subTest(entry=entry):
                manifest = json.dumps(["test", [entry], ["ref", 0]]).encode()
                self.path.write_bytes(archive.MAGIC + gzip.compress(struct.pack("<I", len(manifest)) + manifest))
                with self.assertRaises(ValueError):
                    archive.load(self.path, {}, "test")


if __name__ == "__main__":
    unittest.main()

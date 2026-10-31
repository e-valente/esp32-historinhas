"""Historinhas' CSV partition map has one parser for both deploy paths."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "firmware/historinhas/tools/partition_layout.py"
PLATFORMIO = ROOT / "firmware/historinhas/platformio.ini"


def load_layout():
    spec = importlib.util.spec_from_file_location("partition_layout", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPartitions(unittest.TestCase):
    def setUp(self):
        self.layout = load_layout()

    def test_model_partition_matches_the_documented_1_5_mib_application_layout(self):
        self.assertEqual(self.layout.partition("factory"), (0x10000, 0x180000))
        self.assertEqual(self.layout.partition("model"), (0x190000, 0xE60000))

    def test_platformio_size_gate_matches_the_application_partition(self):
        self.assertIn("board_upload.maximum_size = 1572864", PLATFORMIO.read_text())

    def test_unknown_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.layout.partition("missing")

    def test_bad_partition_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partitions.csv"
            path.write_text("model,data,0x40,not-a-number,0x10\n")
            with self.assertRaisesRegex(ValueError, "invalid offset"):
                self.layout.partition("model", path)

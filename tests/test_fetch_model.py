"""The downloader remains disabled until a release is hash-pinned."""

import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_model.sh"


class TestFetchModel(unittest.TestCase):
    def run_fetch(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], text=True, capture_output=True)

    def test_no_release_is_explicit(self):
        result = self.run_fetch()
        self.assertEqual(result.returncode, 1)
        self.assertIn("No hash-pinned Historinhas release", result.stderr)
        self.assertIn("e-valente/esp32-historinhas", result.stderr)

    def test_arguments_are_rejected(self):
        result = self.run_fetch("anything")
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()

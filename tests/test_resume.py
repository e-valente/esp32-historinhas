"""Crash checkpoints resume the exact Historinhas training trajectory."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from research import train


class TestResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data = root / "data"
        self.runs = root / "runs"
        variant = self.data / "vocab-64"
        variant.mkdir(parents=True)
        tokenizer = variant / "tokenizer.json"
        tokenizer.write_text("test tokenizer")
        digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        (variant / "manifest.json").write_text(json.dumps({
            "dataset": "test/historinhas", "revision": "test-revision",
            "tokenizer": {"sha256": digest},
        }))
        tokens = (np.arange(2048, dtype=np.uint16) % 64)
        tokens.tofile(variant / "train.bin")
        tokens.tofile(variant / "val.bin")
        self.old_data, self.old_runs = train.DATA, train.RUNS
        train.DATA, train.RUNS = self.data, str(self.runs)
        self.addCleanup(self.restore_paths)

    def restore_paths(self):
        train.DATA, train.RUNS = self.old_data, self.old_runs

    @staticmethod
    def args(*extra):
        return [
            "--arm", "ple", "--vocab", "64", "--d-model", "8", "--n-layers", "1",
            "--n-heads", "2", "--ple-dim", "4", "--seq-len", "8", "--batch-size", "2",
            "--target-core", "1000", "--steps", "4", "--eval-every", "1",
            "--eval-iters", "1", "--save-every", "1", "--seed", "0", "--tag", "resume-test",
            *extra,
        ]

    def invoke(self, *extra):
        with mock.patch.object(sys, "argv", ["train.py", *self.args(*extra)]), \
             mock.patch.object(train, "get_device", return_value="cpu"), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def test_interrupt_then_resume(self):
        calls = 0

        def interrupted_evaluate(*_args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return 2.0

        with mock.patch.object(train, "evaluate", side_effect=interrupted_evaluate):
            self.invoke()

        name = "ple-resume-test-s0"
        resume = self.runs / f"{name}.resume.pt"
        self.assertTrue(resume.exists())
        partial = torch.load(resume, map_location="cpu", weights_only=False)
        self.assertEqual(partial["resume"]["next_step"], 2)
        self.assertEqual(partial["resume"]["history"][0]["step"], 0)

        with mock.patch.object(train, "evaluate", return_value=1.5):
            self.invoke("--resume", str(resume))

        final = torch.load(self.runs / f"{name}.pt", map_location="cpu", weights_only=False)
        self.assertEqual(final["resume"]["next_step"], 4)
        self.assertFalse(resume.exists(), "a successful final checkpoint supersedes resume state")
        record = json.loads((self.runs / f"{name}.json").read_text())
        self.assertEqual(record["steps"], 4)
        self.assertEqual(record["training"]["save_every"], 1)

    def test_old_checkpoint_is_not_silently_resumed_without_optimizer_state(self):
        old = self.runs / "old.pt"
        self.runs.mkdir()
        torch.save({"name": "ple-resume-test-s0"}, old)
        with self.assertRaisesRegex(SystemExit, "no optimizer/RNG state"):
            self.invoke("--resume", str(old))


if __name__ == "__main__":
    unittest.main()

"""Historinhas' UTF-8/no-regex contract between Hugging Face and C."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "firmware/historinhas/tools/generate_tokenizer_header.py"
CONFORMANCE = ROOT / "runtime/host_verify/tokenizer_conformance.c"


def load_generator():
    spec = importlib.util.spec_from_file_location("historinhas_tokenizer_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.generator = load_generator()

    def tokenizer_path(self):
        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            ["Era uma vez", "Ação, coração e maçã.", "João viu o céu azul."],
            trainers.BpeTrainer(vocab_size=512, special_tokens=["<|endoftext|>"],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        path = self.dir / "tokenizer.json"
        tokenizer.save(str(path))
        return path

    def test_utf8_encoder_matches_hugging_face(self):
        path = self.tokenizer_path()
        asset = self.dir / "tokenizer.btk"
        self.generator.generate(path, self.dir / "tokenizer.h", asset)
        binary = self.dir / "conformance"
        subprocess.run(
            ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",              "-o", str(binary), str(CONFORMANCE)], check=True)
        probes = ["Era uma vez", "Ação, coração e maçã.", "Olá, criança!", "  céu  "]
        got = subprocess.run([str(binary), str(asset), *probes], check=True,
                             text=True, capture_output=True).stdout.splitlines()
        tokenizer = Tokenizer.from_file(str(path))
        want = [" ".join(map(str, tokenizer.encode(text).ids)) for text in probes]
        self.assertEqual(got, want)

    def test_vocab_header_carries_the_eot_id(self):
        path = self.tokenizer_path()
        vocab_generator = GENERATOR.parent / "generate_vocab.py"
        out = self.dir / "vocab.h"
        # The CLI parser is deliberately kept thin, so run it as its real user
        # does rather than duplicating header-generation details in the test.
        subprocess.run([sys.executable, str(vocab_generator), "--tokenizer", str(path),
                        "--out", str(out)], check=True)
        eot = Tokenizer.from_file(str(path)).token_to_id("<|endoftext|>")
        self.assertIn(f"#define VOCAB_EOT {eot}", out.read_text())

    def test_regex_enabled_tokenizer_is_rejected(self):
        path = self.tokenizer_path()
        config = json.loads(path.read_text())
        config["pre_tokenizer"]["use_regex"] = True
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(SystemExit, "use_regex.*False"):
            self.generator.generate(path, self.dir / "tokenizer.h")


if __name__ == "__main__":
    unittest.main()

"""Verify Historinhas' UTF-8 device BPE encoder against Hugging Face.

The interactive firmware accepts one UTF-8 serial line at a time. The model's
``<|endoftext|>`` marker is deliberately reserved and rejected by firmware,
because BTK1 contains BPE merges but no added-token matcher.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer


TOOLS = Path(__file__).resolve().parent
SKETCH = TOOLS.parent
ROOT = SKETCH.parent.parent
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizer.json"
PACKER = TOOLS / "generate_tokenizer_header.py"
CONFORMANCE_C = ROOT / "runtime" / "host_verify" / "tokenizer_conformance.c"
RESERVED_SPECIAL = "<|endoftext|>"
BTK_ERR_INVALID_UTF8 = -5

PROBES = [
    "",
    "Era uma vez",
    "Ação, coração e maçã.",
    "João viu o céu azul.",
    "  espaços no início e no fim  ",
    "Olá, criança! Tudo bem?",
    "números 123 456",
    "que surpresa?! sim...",
    "a\tb",
    "Mistura De Maiúsculas",
]
PROBES += [chr(code) for code in range(0x20, 0x7F)]


def build_verifier(workdir):
    binary = workdir / "tokenizer_conformance"
    subprocess.run(["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                                        "-o", str(binary), str(CONFORMANCE_C)], check=True)
    return binary


def encode_all(verifier, asset, texts):
    result = subprocess.run([str(verifier), str(asset), *texts], check=True,
                            capture_output=True, text=True)
    lines = result.stdout.split("\n")[:-1]
    if len(lines) != len(texts):
        raise SystemExit(
            f"verifier returned {len(lines)} lines for {len(texts)} strings")
    return lines


def parse_ids(line):
    return [] if not line else [int(value) for value in line.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    args = ap.parse_args()
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        asset = workdir / "tokenizer.btk"
        subprocess.run([sys.executable, str(PACKER), "--tokenizer",
                        str(tokenizer_path), "--out", str(workdir / "encoder.h"),
                        "--asset", str(asset)], check=True)
        verifier = build_verifier(workdir)

        mismatches = []
        for text, line in zip(PROBES, encode_all(verifier, asset, PROBES)):
            got = parse_ids(line)
            want = tokenizer.encode(text, add_special_tokens=False).ids
            if got != want:
                mismatches.append((text, want, got))

    if mismatches:
        text, want, got = mismatches[0]
        raise SystemExit(
            f"{len(mismatches)} encoder mismatches; first {text!r}: "
            f"expected {want}, got {got}")
    print(f"PASS: {len(PROBES)} UTF-8 probes; "
          f"reserved literal {RESERVED_SPECIAL!r} is firmware-rejected")


if __name__ == "__main__":
    main()

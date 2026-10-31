"""Generate the Historinhas UTF-8 ByteLevel-BPE encoder asset.

The tokenizer uses ``ByteLevel(use_regex=False)``, so the device encodes a
complete UTF-8 prompt as one BPE piece. Firmware rejects the end-of-text
literal before encoding because BTK1 does not implement added-token matching.
"""

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SKETCH = TOOLS.parent
ROOT = SKETCH.parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from tokenizer_asset import generate as build_asset


RESERVED_SPECIAL = "<|endoftext|>"


def generate(tokenizer, out, asset=None):
    return build_asset(tokenizer, out, asset, [RESERVED_SPECIAL])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default=ROOT / "artifacts/tokenizer.json")
    ap.add_argument("--out", default=SKETCH / "generated/tokenizer_encoder.h")
    ap.add_argument("--asset", default=None,
                    help="also write raw BTK1 bytes for host verification")
    args = ap.parse_args()
    generate(args.tokenizer, args.out, args.asset)


if __name__ == "__main__":
    main()

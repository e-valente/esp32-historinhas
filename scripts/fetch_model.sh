#!/usr/bin/env bash
# Download a published Historinhas artifact after immutable release pins exist.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ "$#" -eq 0 ] || { echo "usage: scripts/fetch_model.sh" >&2; exit 2; }
cat >&2 <<'MESSAGE'
No hash-pinned Historinhas release is configured yet.

Train and export a local artifact with `python -m research.export`, or publish
model.bin, tokenizer.json, and metadata.json in e-valente/esp32-historinhas.
Once a release is published, add its immutable SHA-256 pins to this script
before enabling downloads.
MESSAGE
exit 1

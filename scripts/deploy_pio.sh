#!/usr/bin/env bash
# Build and flash Historinhas through PlatformIO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$#" -eq 0 ] || { echo "usage: PORT=/dev/ttyACM0 scripts/deploy_pio.sh" >&2; exit 2; }

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need pio
need uv
need cc
need python3

ARTIFACTS=${ARTIFACTS:-artifacts}
MODEL=${MODEL:-$ARTIFACTS/model.bin}
TOKENIZER=${TOKENIZER:-$ARTIFACTS/tokenizer.json}
GOLDEN=${GOLDEN:-$ARTIFACTS/golden.txt}
SKETCH=firmware/historinhas
ENV_NAME=n16r8

for file in "$MODEL" "$TOKENIZER"; do
  [ -f "$file" ] || { echo "missing artifact: $file" >&2; exit 1; }
done

if [ -z "${PORT:-}" ]; then
  case "$(uname -s)" in
    Linux) PORT_PATTERN='/dev/ttyACM*' ;;
    Darwin) PORT_PATTERN='/dev/cu.usbmodem*' ;;
    *) echo "unsupported OS; set PORT=..." >&2; exit 1 ;;
  esac
  candidates=( $PORT_PATTERN )
  [ "${candidates[0]}" != "$PORT_PATTERN" ] || { echo "no $PORT_PATTERN found; connect the board or set PORT=..." >&2; exit 1; }
  [ "${#candidates[@]}" -eq 1 ] || { echo "multiple serial ports found; select one with PORT=...: ${candidates[*]}" >&2; exit 1; }
  PORT=${candidates[0]}
fi
[ -e "$PORT" ] || { echo "serial port not found: $PORT" >&2; exit 1; }

mkdir -p "$SKETCH/generated"
LOCK="$SKETCH/generated/.deploy-lock"
mkdir "$LOCK" 2>/dev/null || { echo "another deploy of $SKETCH is running" >&2; exit 1; }
trap 'rmdir -- "$LOCK"' EXIT
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/esp32-historinhas-pio-XXXXXX")
trap 'rm -rf -- "$RUN_DIR"; rmdir -- "$LOCK"' EXIT

echo "=== Generate vocabulary table ==="
uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/generate_vocab.py" --tokenizer "$TOKENIZER" --out "$SKETCH/generated/vocab.h"
echo "=== Generate UTF-8 tokenizer asset ==="
uv run --no-project python "$SKETCH/tools/generate_tokenizer_header.py" --tokenizer "$TOKENIZER" --out "$SKETCH/generated/tokenizer_encoder.h"
if [ -f "$GOLDEN" ]; then
  cc -O3 -Wall -Wextra -o "$RUN_DIR/verify" runtime/host_verify/verify.c -lm
  "$RUN_DIR/verify" "$MODEL" "$GOLDEN"
else
  echo "=== Golden verification skipped: no $GOLDEN ==="
fi
cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o "$RUN_DIR/staging" runtime/host_verify/staging_verify.c -lm
"$RUN_DIR/staging" "$MODEL"
uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/verify_tokenizer.py" --tokenizer "$TOKENIZER"
pio run -d "$SKETCH" -e "$ENV_NAME"

BYTES=$(wc -c < "$MODEL" | tr -d ' ')
FP=$(python3 -c 'import sys
h = 2166136261
for b in open(sys.argv[1], "rb").read(): h = ((h ^ b) * 16777619) & 0xffffffff
print(f"{h:08x}")' "$MODEL")
STORY_MODEL="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$MODEL")" STORY_PORT="$PORT" \
  pio run -d "$SKETCH" -e "$ENV_NAME" -t upload_model
pio run -d "$SKETCH" -e "$ENV_NAME" -t upload --upload-port "$PORT"
printf 'Flashed Historinhas. At boot, verify build: bytes=%s fp=%s.\n' "$BYTES" "$FP"

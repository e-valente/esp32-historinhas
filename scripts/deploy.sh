#!/usr/bin/env bash
# Build and flash the Historinhas model and firmware to an ESP32-S3 N16R8.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat >&2 <<'USAGE'
usage: PORT=/dev/ttyACM0 scripts/deploy.sh

Builds the headers from the selected Historinhas artifact, verifies the model
and tokenizer on the host, compiles the firmware, then flashes the model and
firmware. Artifact paths may be overridden with ARTIFACTS, MODEL, TOKENIZER,
and GOLDEN.
USAGE
  exit 2
}

[ "$#" -eq 0 ] || usage

ARTIFACTS=${ARTIFACTS:-artifacts}
MODEL=${MODEL:-$ARTIFACTS/model.bin}
TOKENIZER=${TOKENIZER:-$ARTIFACTS/tokenizer.json}
GOLDEN=${GOLDEN:-$ARTIFACTS/golden.txt}
SKETCH=firmware/historinhas
PARTITIONS_TOOL="$SKETCH/tools/partition_layout.py"

step() {
  local keep=$1 label=$2
  shift 2
  local out status=0
  out=$("$@" 2>&1) || status=$?
  out=$(printf '%s\n' "$out" | tr '\r' '\n')
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$out" >&2
    echo "failed: $label" >&2
    exit "$status"
  fi
  if [ -n "${SHOW:-}" ]; then
    printf '%s\n' "$out" | grep -E "$SHOW" || true
  else
    printf '%s\n' "$out" | tail -n "$keep"
  fi
}

missing=0
for file in "$MODEL" "$TOKENIZER"; do
  [ -f "$file" ] || { echo "missing artifact: $file" >&2; missing=1; }
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF_HINT'
required artifacts are missing. Train and export locally, or download a
published, hash-pinned Historinhas release with scripts/fetch_model.sh.
EOF_HINT
  exit 1
fi

[ -f "$PARTITIONS_TOOL" ] || { echo "missing partition layout reader: $PARTITIONS_TOOL" >&2; exit 1; }
read -r PART_OFFSET PART_BYTES < <(python3 "$PARTITIONS_TOOL" --name model)

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need arduino-cli
need uv
need cc
need python3

find_esptool() {
  for candidate in esptool esptool.py; do
    command -v "$candidate" >/dev/null 2>&1 && { echo "$candidate"; return; }
  done
  local data="${ARDUINO_DATA_DIR:-$HOME/.arduino15}" found
  found=$(ls -d "$data"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$found" ] && { echo "$found"; return; }
  echo "esptool not found. Install it or set ARDUINO_DATA_DIR." >&2
  exit 1
}
ESPTOOL=$(find_esptool)

if [ -z "${PORT:-}" ]; then
  case "$(uname -s)" in
    Linux) PORT_PATTERN='/dev/ttyACM*' ;;
    Darwin) PORT_PATTERN='/dev/cu.usbmodem*' ;;
    *) echo "unsupported OS; set PORT=..." >&2; exit 1 ;;
  esac
  PORT=$(ls $PORT_PATTERN 2>/dev/null | head -1 || true)
  [ -n "$PORT" ] || { echo "no $PORT_PATTERN found; connect the board or set PORT=..." >&2; exit 1; }
fi

mkdir -p "$SKETCH/generated"
LOCK="$SKETCH/generated/.deploy-lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another deploy of $SKETCH is running; remove $LOCK only after confirming it is stale" >&2
  exit 1
fi
trap 'rmdir -- "$LOCK"' EXIT

RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/esp32-historinhas-deploy-XXXXXX")
trap 'rm -rf -- "$RUN_DIR"; rmdir -- "$LOCK"' EXIT

printf '=== Generate vocabulary table ===\n'
SHOW=wrote step 0 "generate vocabulary" \
  uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/generate_vocab.py" \
  --tokenizer "$TOKENIZER" --out "$SKETCH/generated/vocab.h"
printf '=== Generate UTF-8 tokenizer asset ===\n'
SHOW=wrote step 0 "generate tokenizer asset" \
  uv run --no-project python "$SKETCH/tools/generate_tokenizer_header.py" \
  --tokenizer "$TOKENIZER" --out "$SKETCH/generated/tokenizer_encoder.h"

if [ -f "$GOLDEN" ]; then
  printf '=== Verify exact int4 path against PyTorch golden ===\n'
  cc -O3 -Wall -Wextra -o "$RUN_DIR/verify" runtime/host_verify/verify.c -lm
  step 2 "golden verification" "$RUN_DIR/verify" "$MODEL" "$GOLDEN"
else
  echo "=== Golden verification skipped: no $GOLDEN ==="
fi
printf '=== Verify int8 staging and platform hooks ===\n'
cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o "$RUN_DIR/staging" runtime/host_verify/staging_verify.c -lm
step 3 "staging verification" "$RUN_DIR/staging" "$MODEL"
printf '=== Verify tokenizer against Hugging Face ===\n'
step 1 "tokenizer conformance" \
  uv run --no-project --with 'tokenizers==0.23.1' python "$SKETCH/tools/verify_tokenizer.py" \
  --tokenizer "$TOKENIZER"

FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'
BUILD_DIR="$RUN_DIR/build"
printf '=== Compile firmware ===\n'
step 2 "compile firmware" arduino-cli compile --fqbn "$FQBN" \
  --build-property 'compiler.optimization_flags=-O3' --build-path "$BUILD_DIR" "$SKETCH"

BYTES=$(wc -c < "$MODEL" | tr -d ' ')
if [ "$BYTES" -gt "$PART_BYTES" ]; then
  echo "model is $BYTES bytes but the model partition at $PART_OFFSET holds only $PART_BYTES bytes" >&2
  exit 1
fi
FP=$(python3 -c 'import sys
h = 2166136261
for b in open(sys.argv[1], "rb").read():
    h = ((h ^ b) * 16777619) & 0xffffffff
print(f"{h:08x}")' "$MODEL")

printf '=== Flash model and firmware ===\n'
printf 'model: %s\nport: %s\nbytes: %s\nexpected fingerprint: %s\n' "$MODEL" "$PORT" "$BYTES" "$FP"
step 2 "flash model" "$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  write_flash "$PART_OFFSET" "$MODEL"
step 1 "upload firmware" arduino-cli upload -p "$PORT" --fqbn "$FQBN" \
  --input-dir "$BUILD_DIR" "$SKETCH"
printf 'Flashed Historinhas. At boot, verify build: bytes=%s fp=%s.\n' "$BYTES" "$FP"

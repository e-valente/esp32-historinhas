# Historinhas on ESP32-S3

<p align="center">
  <img src="docs/assets/historinhas-logo.png" alt="Historinhas: an open storybook microchip" width="240">
</p>

Historinhas is a compact PLE language model that continues Brazilian Portuguese
children's stories directly on an ESP32-S3 N16R8 with 16 MB flash and 8 MB
PSRAM. It trains exclusively on the `texto` field of the
[Boakpe/historinhas](https://huggingface.co/datasets/Boakpe/historinhas) dataset.

The project uses a UTF-8 ByteLevel BPE tokenizer without regex
pre-tokenization. The Python pipeline and the ESP32 therefore produce identical
token IDs for complete UTF-8 prompts.

## What this repository contains

- A reproducible dataset preparation and training pipeline.
- A PyTorch sampler for local inference from `.pt` checkpoints.
- An exporter for the ESP32 `model.bin` format.
- ESP32-S3 firmware with UTF-8 serial prompting.
- Host-side tokenizer, golden-logit, and int8 staging verification gates.

Generated datasets, checkpoints, model binaries, and firmware build outputs are
ignored by Git. They are local working files, not source files.

## Architecture diagrams

- [Flash layout and deployment flow](docs/diagrams/flash-layout.svg) ([editable draw.io source](docs/diagrams/flash-layout.drawio))
- [UTF-8 ByteLevel BPE tokenizer](docs/diagrams/bpe-tokenizer.svg) ([editable draw.io source](docs/diagrams/bpe-tokenizer.drawio))
- [PLE decoder architecture](docs/diagrams/ple-architecture.svg) ([editable draw.io source](docs/diagrams/ple-architecture.drawio))

See [the diagrams index](docs/diagrams/README.md) for rendered previews and export instructions.

## Requirements

### Local training and inference

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Several GB of free disk space. The pinned raw dataset alone is about 1.63 GB;
  prepared token bins and checkpoints require additional space.

### ESP32 deployment

- ESP32-S3 N16R8 board (16 MB flash, 8 MB PSRAM)
- USB data cable and the board serial device, such as `/dev/ttyACM0`
- `arduino-cli`, a C compiler, and `esptool` or `esptool.py`
- Optional: PlatformIO for the alternate deployment flow

## 1. Install the project dependencies

From the repository root:

```bash
uv sync
```

For the optional serial monitor provided by Python, install the `device` extra:

```bash
uv sync --extra device
```

## 2. Download and prepare the dataset

The preparation command downloads the exact pinned dataset revision, verifies
its byte count and SHA-256, normalizes line endings, creates deterministic
document-level train/validation splits, trains the tokenizer on the training
split, and writes `uint16` token bins.

```bash
uv run python -m research.prepare --vocab 32768
```

The resulting layout is:

```text
data/
└── vocab-32768/
    ├── manifest.json
    ├── tokenizer.json
    ├── train.bin
    └── val.bin
```

`manifest.json` records the source revision, tokenizer hash, split rule, and
corpus counts. Keep each tokenizer together with the bins it created.

For a small development-only preparation, use a deterministic document limit:

```bash
uv run python -m research.prepare --vocab 4096 --max-docs 1000
```

This is useful for validating the pipeline but is not comparable with the
32,768-token deployment configuration.

## 3. Train a deployment model

The following configuration targets the ESP32 deployment shape. It writes
checkpoints and JSON run records under `runs/`.

```bash
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OMP_PROC_BIND=close \
uv run python -m research.train --arm ple --vocab 32768 \
  --d-model 96 --n-layers 6 --ple-dim 128 --seq-len 256 \
  --batch-size 16 --target-core 560000 --steps 10000 \
  --tag intel-v3 --seed 0
```

The completed checkpoint is:

```text
runs/ple-intel-v3-s0.pt
```

The trainer writes an atomic resume checkpoint every 250 steps by default. To
resume an interrupted run, repeat the original command exactly and add:

```bash
--resume runs/ple-intel-v3-s0.resume.pt
```

`--steps` remains the total training target, not the number of additional
steps. A completed run removes its `.resume.pt` file.

## 4. Test local inference

Local inference uses a PyTorch checkpoint (`runs/*.pt`) and its matching
tokenizer. It does not use `model.bin` directly.

### Use the available V2 checkpoint

If the local V2 checkpoint and artifact bundle are present, run:

```bash
uv run python -m research.sample \
  --run runs/ple-historinhas-intel-v2-s0.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Era uma vez, em uma floresta encantada" \
  --temperature 0.6 --top-k 30 --min-new-tokens 24
```

### Sample a newly trained V3 checkpoint

Use the tokenizer created by the same preparation run:

```bash
uv run python -m research.sample \
  --run runs/ple-intel-v3-s0.pt \
  --tokenizer data/vocab-32768/tokenizer.json \
  --prompt "Era uma vez, em uma floresta encantada" \
  --temperature 0.6 --top-k 30 --min-new-tokens 24
```

Useful sampler options include `--tokens`, `--samples`, `--seed`,
`--temperature`, `--top-k`, and `--min-new-tokens`. The sampler verifies the
tokenizer SHA-256 recorded by the checkpoint before generating text.

## 5. Export an ESP32 artifact bundle

Export a completed PLE checkpoint into the format used by the firmware:

```bash
uv run python -m research.export ple-intel-v3-s0 \
  --tokenizer data/vocab-32768/tokenizer.json
```

The exporter writes this ignored local bundle:

```text
artifacts/
├── model.bin       # int4 PLE model mapped from flash by the ESP32
├── tokenizer.json  # exact tokenizer used by the checkpoint
├── metadata.json   # model configuration and file hashes
├── golden.npz      # Python golden logits
└── golden.txt      # host C verifier fixture
```

`model.bin` is an ESP32 artifact, not a PyTorch checkpoint. It is sufficient
for deployment but cannot be passed to `research.sample`.

## 6. Verify and flash the ESP32-S3

Connect the board and identify its serial port. On Linux it is commonly
`/dev/ttyACM0`; on macOS it is commonly `/dev/cu.usbmodem*`.

The standard deployment command regenerates the vocabulary and tokenizer
headers from the selected artifact, verifies the device tokenizer against Hugging
Face, runs the optional golden-logit gate, verifies int8 staging, compiles the
firmware, flashes `model.bin`, and uploads the firmware.

```bash
PORT=/dev/ttyACM0 scripts/deploy.sh
```

The script requires these files:

```text
artifacts/model.bin
artifacts/tokenizer.json
```

`golden.txt` is optional but recommended because it enables exact host
verification before flashing. Override the bundle location when needed:

```bash
PORT=/dev/ttyACM0 ARTIFACTS=/path/to/bundle scripts/deploy.sh
```

### PlatformIO alternative

```bash
PORT=/dev/ttyACM0 scripts/deploy_pio.sh
```

Both deployment paths obtain the model offset and capacity from
`firmware/historinhas/partitions.csv`; neither duplicates a flash address in the
command line.

## 7. Run inference on the ESP32

After the board boots, open a serial monitor at 115200 baud:

```bash
uv run --extra device python -m serial.tools.miniterm /dev/ttyACM0 115200
```

The firmware prints a banner similar to:

```text
=== ESP32-S3 PLE Historinhas pt_BR ===
model: Vin=32768 Vout=32768 D=96 L=6 H=4 F=66 P=128
build: bytes=... fp=...
READY> send a UTF-8 prompt; up to 200 tokens will be generated
```

Compare the reported `bytes` and `fp` values with the deploy receipt. At each
`READY>` prompt, send one UTF-8 story prefix followed by Enter:

```text
READY> Era uma vez, em uma floresta encantada
```

The prompt may contain at most 512 UTF-8 bytes and must leave room in the model
context for up to 200 generated tokens. `<|endoftext|>` is reserved and cannot
be entered literally.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `FileNotFoundError` for `runs/*.pt` | Run training first, or use an available checkpoint such as `runs/ple-historinhas-intel-v2-s0.pt`. `model.bin` cannot be sampled by PyTorch. |
| Tokenizer hash mismatch | Use the tokenizer produced by the preparation run that trained the checkpoint. Do not mix tokenizer files across runs. |
| Missing `artifacts/model.bin` | Export a checkpoint with `research.export`, or provide an existing local bundle through `ARTIFACTS=...`. |
| `no /dev/ttyACM* found` | Connect the board, confirm the USB cable carries data, or set `PORT` explicitly. |
| `esptool not found` | Install `esptool` or `esptool.py`, or configure `ARDUINO_DATA_DIR` for an Arduino-managed installation. |
| No downloadable model release | `scripts/fetch_model.sh` remains disabled until a release with immutable SHA-256 pins is published. |

## Additional documentation

- [Changelog](CHANGELOG.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Model artifacts](docs/MODEL_ARTIFACTS.md)
- [Flashing and operation](docs/FLASHING.md)
- [Flash layout](docs/FLASH_LAYOUT.md)
- [UTF-8 tokenizer contract](docs/TOKENIZER_UTF8.md)
- [Intel laptop training profile](docs/INTEL_TRAINING.md)

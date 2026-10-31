# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-10-31

Initial public project release.

### Added

- Reproducible preparation pipeline for the pinned `Boakpe/historinhas` dataset
  revision, including tokenizer training, integrity checks, and deterministic
  token-bin splits.
- PLE language-model training, resume, cohort analysis, local PyTorch sampling,
  quantization evaluation, and ESP32 artifact export workflows.
- UTF-8 ByteLevel BPE tokenizer implementation shared by host verification and
  ESP32 firmware, with BTK1 asset packing and Hugging Face conformance checks.
- ESP32-S3 N16R8 firmware for interactive serial story continuation, including
  Arduino CLI and PlatformIO deployment paths.
- Host verification gates for exported golden logits and int8 staging.
- Operational documentation for preparation, training, local inference, model
  export, flashing, and device inference.
- Editable draw.io diagrams and SVG renders for flash layout, tokenizer flow,
  and PLE architecture.
- Historinhas repository logo.

### Notes

- Datasets, checkpoints, and exported model artifacts are intentionally ignored
  by Git and must be generated or supplied locally.
- `scripts/fetch_model.sh` remains disabled until a published release has
  immutable SHA-256 pins.

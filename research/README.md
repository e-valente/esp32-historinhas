# Research pipeline

Run every command from the repository root.

```bash
uv sync
uv run python -m research.prepare --vocab 32768
```

`prepare` downloads the pinned `Boakpe/historinhas` revision, verifies its size
and SHA-256, uses only `texto`, creates deterministic document splits, and
writes `train.bin`, `val.bin`, `tokenizer.json`, and `manifest.json` under
`data/vocab-32768/`. Token bins use `uint16`, so the active vocabulary cannot
exceed 65,536 IDs.

The tokenizer is UTF-8 ByteLevel BPE with `use_regex=False`, trained only from
the training split. The ESP32 encoder applies BPE to the whole prompt byte
sequence and does not need a Unicode regex engine.

## Training and export

```bash
uv run python -m research.train --arm ple --vocab 32768 \
  --d-model 96 --n-layers 6 --ple-dim 128 --seq-len 256 --batch-size 16 \
  --target-core 560000 --steps 10000 --seed 0 --tag intel-v3

uv run python -m research.sample \
  --run runs/ple-intel-v3-s0.pt \
  --tokenizer data/vocab-32768/tokenizer.json \
  --prompt 'Era uma vez, em uma floresta encantada' \
  --temperature 0.6 --top-k 30 --min-new-tokens 24

uv run python -m research.export ple-intel-v3-s0 \
  --tokenizer data/vocab-32768/tokenizer.json
```

Each checkpoint records the tokenizer and data-manifest hashes. The cohort
analyzer refuses comparisons whose vocabulary, data, or training schedule
differs.

## Resuming interrupted training

The trainer writes `runs/<name>.resume.pt` atomically every 250 steps by
default. Repeat the original command, including its total `--steps`, and add:

```bash
--resume runs/ple-intel-v3-s0.resume.pt
```

`--steps` is the total schedule target, not additional work. When training
finishes, the resume file is removed and the final checkpoint is written as
`runs/ple-intel-v3-s0.pt`.

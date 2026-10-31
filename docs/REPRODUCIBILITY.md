# Reproducibility

The training pipeline uses only the `texto` field from `Boakpe/historinhas`
under Apache-2.0. `research.prepare` pins the dataset revision, expected byte
size, and SHA-256 before it reads the source.

```bash
uv run python -m research.prepare --vocab 32768
```

The resulting `data/vocab-32768/` directory contains the tokenizer, a manifest,
and deterministic train and validation token bins. The manifest records the
source revision, tokenizer hash, split method, normalization, and corpus counts.

Train a deployment configuration with:

```bash
uv run python -m research.train --arm ple --vocab 32768 \
  --d-model 96 --n-layers 6 --ple-dim 128 --seq-len 256 --batch-size 16 \
  --target-core 560000 --steps 10000 --seed 0 --tag intel-v3
```

Checkpoints record the tokenizer and manifest SHA-256 values. Use
`python -m research.analyze` only with runs whose recorded conditions match.

# Intel laptop training profile

The deployment configuration was developed on an Intel laptop using twelve
compute threads. Use the same settings when available:

```bash
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OMP_PROC_BIND=close \
uv run python -m research.train --arm ple --vocab 32768 \
  --d-model 96 --n-layers 6 --ple-dim 128 --seq-len 256 --batch-size 16 \
  --target-core 560000 --steps 10000 --seed 0 --tag intel-v3
```

The trainer writes an atomic resume checkpoint every 250 steps. Resume an
interrupted run with the same options and:

```bash
--resume runs/ple-intel-v3-s0.resume.pt
```

After exporting, compare the portable int4 and staged int8 paths with validation
bins before publishing an artifact:

```bash
cc -O3 -Wall -Wextra -o /tmp/historinhas-ppl runtime/host_verify/ppl.c -lm
/tmp/historinhas-ppl artifacts/model.bin data/vocab-32768/val.bin 128
```

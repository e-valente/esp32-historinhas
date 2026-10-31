# Historinhas firmware

This sketch runs a Historinhas PLE artifact on an ESP32-S3 N16R8. The model
binary is stored in the `model` partition and the firmware is compiled into the
application partition. Decoder and tokenizer headers are regenerated from the
selected `tokenizer.json` every time the firmware is deployed.

```bash
PORT=/dev/ttyACM0 scripts/deploy.sh
# or
PORT=/dev/ttyACM0 scripts/deploy_pio.sh
```

`deploy.sh` verifies the C encoder against Hugging Face, runs the host gates,
compiles the firmware, then flashes the model and firmware. `deploy_pio.sh`
offers the same flow through PlatformIO.

After boot, open the serial console at 115200 baud:

```text
=== ESP32-S3 PLE Historinhas pt_BR ===
sampling: top-k=20 temperature=0.60 no-repeat=4
READY> send a UTF-8 prompt; up to 200 tokens will be generated
```

A prompt can contain at most 512 UTF-8 bytes and must leave room for 200
generated tokens in the context. Sampling is deterministic top-k with repeated
4-grams blocked. `<|endoftext|>` is reserved.

See [flashing](../../docs/FLASHING.md) and the
[flash layout](../../docs/FLASH_LAYOUT.md) for deployment details.

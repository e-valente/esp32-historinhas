# Flashing and operation

Create or download an artifact bundle in `artifacts/`, then run:

```bash
PORT=/dev/ttyACM0 scripts/deploy.sh
```

The deployment flow regenerates tokenizer headers from the selected artifact,
checks UTF-8 tokenizer conformance against Hugging Face, runs the available host
model gates, compiles the firmware, flashes the model partition, and finally
uploads the firmware.

For PlatformIO, use:

```bash
PORT=/dev/ttyACM0 scripts/deploy_pio.sh
```

Open the serial console at 115200 baud after boot. Send one UTF-8 prompt per
line. The device prints the model byte count and FNV-1a fingerprint; compare
both with the deploy receipt before using the result.

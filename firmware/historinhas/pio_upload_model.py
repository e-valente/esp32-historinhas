"""PlatformIO target for the external model partition (not part of upload)."""

import os
from pathlib import Path
import subprocess
import sys

Import("env")

# PlatformIO executes extra scripts without defining __file__. PROJECT_DIR is
# the sketch directory and remains valid whether pio is invoked from the repo
# root or from the sketch itself.
TOOLS = Path(env.subst("$PROJECT_DIR")) / "tools"
sys.path.insert(0, str(TOOLS))
from partition_layout import partition

MODEL_OFFSET, MODEL_BYTES = partition("model")


def upload_model(source, target, env):
    model = Path(os.environ["STORY_MODEL"]).resolve()
    port = os.environ["STORY_PORT"]
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")
    # Never overflow into coredump. Both values come from partitions.csv, the
    # single source of truth shared with the Arduino deploy path.
    if model.stat().st_size > MODEL_BYTES:
        raise RuntimeError(
            f"model exceeds the {MODEL_BYTES}-byte partition at "
            f"{MODEL_OFFSET:#x}: {model}")
    tool_dir = env.PioPlatform().get_package_dir("tool-esptoolpy")
    if not tool_dir:
        raise RuntimeError("PlatformIO esptool package not installed")
    subprocess.run(
        [sys.executable, str(Path(tool_dir) / "esptool.py"), "--chip", "esp32s3",
          "--port", port, "--baud", "921600", "write_flash", f"{MODEL_OFFSET:#x}", str(model)],
        check=True,
    )


env.AddCustomTarget(
    "upload_model", "$BUILD_DIR/firmware.bin", upload_model,
    title="Upload Historinhas model",
    description=f"Write model partition at {MODEL_OFFSET:#x}",
)

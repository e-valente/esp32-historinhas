"""Exercise the PlatformIO deployment sequence without a board or network."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DeployPioTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "scripts/deploy_pio.sh").write_bytes((ROOT / "scripts/deploy_pio.sh").read_bytes())
        self.sketch = self.repo / "firmware/historinhas"
        (self.sketch / "tools").mkdir(parents=True)
        (self.repo / "runtime/host_verify").mkdir(parents=True)
        for name in ("verify.c", "staging_verify.c"):
            (self.repo / f"runtime/host_verify/{name}").write_text("int main() {}\n")
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.port = Path(self.temp.name) / "port"
        self.port.touch()
        self.log = Path(self.temp.name) / "calls"
        self.stub("pio", 'printf "pio %s | model=%s port=%s\\n" "$*" "${STORY_MODEL:-}" "${STORY_PORT:-}" >> "$CALL_LOG"')
        self.stub("uv", 'printf "uv %s\\n" "$*" >> "$CALL_LOG"')
        self.stub("cc", 'printf "cc %s\\n" "$*" >> "$CALL_LOG"\nprevious=""; for arg in "$@"; do if [ "$previous" = "-o" ]; then printf "#!/bin/sh\\nexit 0\\n" > "$arg"; chmod +x "$arg"; fi; previous="$arg"; done')
        self.stub("uname", "printf 'Linux\\n'")

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def artifacts(self, golden=False):
        destination = self.repo / "artifacts"
        destination.mkdir()
        (destination / "model.bin").write_bytes(b"PLE\0")
        (destination / "tokenizer.json").write_text("{}")
        if golden:
            (destination / "golden.txt").write_text("gold")

    def run_script(self, **extra):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", PORT=str(self.port), CALL_LOG=str(self.log))
        env.update(extra)
        return subprocess.run(["bash", "scripts/deploy_pio.sh"], cwd=self.repo, env=env, text=True, capture_output=True)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_missing_model_stops_before_tool_use(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing artifact", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_sequence_uses_the_single_sketch_and_artifact_directory(self):
        self.artifacts(golden=True)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([line.split()[0] for line in calls], ["uv", "uv", "cc", "cc", "uv", "pio", "pio", "pio"])
        self.assertIn("firmware/historinhas -e n16r8", calls[5])
        self.assertIn("model=", calls[6])
        self.assertIn("port=", calls[6])


if __name__ == "__main__":
    unittest.main()

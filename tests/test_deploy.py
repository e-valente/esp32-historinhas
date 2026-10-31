"""Offline tests for the single-model deployment flow."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy.sh"
STUB = "#!/usr/bin/env bash\necho \"{name} $*\" >> \"$CALL_LOG\"\n{body}\n"


class DeployHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        target = self.repo / "scripts/deploy.sh"
        target.write_text(SCRIPT.read_text())
        target.chmod(0o755)
        sketch = self.repo / "firmware/historinhas"
        (sketch / "tools").mkdir(parents=True)
        (sketch / "partitions.csv").write_bytes(
            (ROOT / "firmware/historinhas/partitions.csv").read_bytes())
        (sketch / "tools/partition_layout.py").write_bytes(
            (ROOT / "firmware/historinhas/tools/partition_layout.py").read_bytes())
        (self.repo / "runtime/host_verify").mkdir(parents=True)
        for name in ("verify.c", "staging_verify.c"):
            (self.repo / f"runtime/host_verify/{name}").write_text("int main(void) { return 0; }\n")
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.log = Path(self.tmp.name) / "calls.log"
        self.port = Path(self.tmp.name) / "port"
        self.port.touch()
        self.stub("arduino-cli", 'case "$1" in compile) echo compiled;; upload) echo uploaded;; esac')
        self.stub("esptool", "")
        self.stub("uv", '''out=""; previous=""
for arg in "$@"; do
  [ "$previous" = "--out" ] && out="$arg"
  previous="$arg"
done
[ -z "$out" ] || { mkdir -p "$(dirname "$out")"; : > "$out"; }
echo wrote ${out:-nothing}
echo PASS''')
        self.stub("cc", '''previous=""
for arg in "$@"; do
  if [ "$previous" = "-o" ]; then printf '#!/bin/sh\\nexit 0\\n' > "$arg"; chmod +x "$arg"; fi
  previous="$arg"
done''')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(STUB.format(name=name, body=body))
        path.chmod(0o755)

    def artifacts(self, directory=None, golden=False):
        target = self.repo / (directory or "artifacts")
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"PLE\0")
        (target / "tokenizer.json").write_text("{}")
        if golden:
            (target / "golden.txt").write_text("fixture\n")
        return target

    def run_deploy(self, *args, **extra):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                   CALL_LOG=str(self.log), PORT=str(self.port))
        env.update(extra)
        return subprocess.run(["bash", "scripts/deploy.sh", *args], cwd=self.repo,
                              text=True, capture_output=True, env=env)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []


class TestDeployment(DeployHarness):
    def test_extra_argument_is_rejected_before_tool_use(self):
        result = self.run_deploy("unexpected")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_missing_artifacts_stop_before_tool_use(self):
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing artifact", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_full_flow_generates_gates_compiles_then_flashes(self):
        self.artifacts(golden=True)
        result = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = "\n".join(self.calls())
        self.assertIn("firmware/historinhas/tools/generate_vocab.py", calls)
        self.assertIn("firmware/historinhas/tools/verify_tokenizer.py", calls)
        self.assertIn("arduino-cli compile", calls)
        self.assertIn("write_flash 0x190000 artifacts/model.bin", calls)
        self.assertIn("arduino-cli upload", calls)
        self.assertLess(calls.index("arduino-cli compile"), calls.index("write_flash"))
        self.assertLess(calls.index("write_flash"), calls.index("arduino-cli upload"))

    def test_artifact_override_is_used(self):
        self.artifacts("custom")
        result = self.run_deploy(ARTIFACTS="custom")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom/model.bin", "\n".join(self.calls()))

    def test_compile_failure_does_not_flash(self):
        self.artifacts()
        self.stub("arduino-cli", 'if [ "$1" = compile ]; then echo compile-error >&2; exit 1; fi')
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        calls = "\n".join(self.calls())
        self.assertNotIn("write_flash", calls)
        self.assertNotIn("arduino-cli upload", calls)

    def test_lock_prevents_header_replacement(self):
        self.artifacts()
        lock = self.repo / "firmware/historinhas/generated/.deploy-lock"
        lock.mkdir(parents=True)
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another deploy", result.stderr)
        self.assertTrue(lock.exists())

    def test_generators_run_outside_project_with_pinned_tokenizers(self):
        self.artifacts()
        self.assertEqual(self.run_deploy().returncode, 0)
        uv_calls = [call for call in self.calls() if call.startswith("uv ")]
        self.assertEqual(len(uv_calls), 3)
        self.assertTrue(all("--no-project" in call for call in uv_calls))
        self.assertEqual(sum("tokenizers==0.23.1" in call for call in uv_calls), 2)


if __name__ == "__main__":
    unittest.main()

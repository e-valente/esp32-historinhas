"""Benchmark the Historinhas firmware through its USB serial console.

Each receipt records generated text, timing, and the flashed model fingerprint.
A baseline comparison fails when a prompt produces different text, preventing a
performance result from silently comparing different behavior.

    uv run python scripts/benchmark_device.py --out runs/before.json
    uv run python scripts/benchmark_device.py --out runs/after.json \
        --baseline runs/before.json

Install the optional serial dependency first: ``uv sync --extra device``.
"""

import argparse
import glob
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROMPTS = ["Era uma vez"]
TOTAL = re.compile(r"---\s+(\d+) tokens in ([\d.]+) s\s+---")
TIMING = re.compile(r"throughput: ([\d.]+) tok/s\s+\(([\d.]+) ms/token\)")
PROFILE = re.compile(r"profile ms/token: input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| ple ([\d.]+) \| head ([\d.]+)")
FINGERPRINT = re.compile(r"fp=([0-9a-f]+)")


def open_port(port):
    try:
        import serial
    except ImportError:
        sys.exit("pyserial is required: uv sync --extra device")
    if not port:
        found = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/cu.usbmodem*"))
        if not found:
            sys.exit("no board found; pass --port")
        port = found[0]
    return serial.Serial(port, 115200, timeout=1), port


def read_until(port, timeout):
    text, started = "", time.time()
    while time.time() - started < timeout:
        chunk = port.read(4096).decode("utf-8", "replace")
        text += chunk
        if text.rstrip().endswith("READY>"):
            return text
        if not chunk:
            time.sleep(0.05)
    return text


def parse(text, prompt):
    total, timing = TOTAL.search(text), TIMING.search(text)
    body = text.split("A: ", 1)[1] if "A: " in text else text
    story = " ".join(line.strip() for line in TOTAL.split(body)[0].splitlines()
                     if line.strip() and line.strip() != "READY>")
    row = {"prompt": prompt, "answer": story,
           "tokens": int(total.group(1)) if total else 0,
           "ms": round(float(total.group(2)) * 1000) if total else 0,
           "complete": text.rstrip().endswith("READY>")}
    if timing:
        row.update(tok_per_s=float(timing.group(1)), ms_per_token=float(timing.group(2)))
    profile = PROFILE.search(text)
    if profile:
        row["profile"] = dict(zip(("input_ms", "attn_ms", "ffn_ms", "ple_ms", "head_ms"),
                                   map(float, profile.groups())))
    return row


def fingerprint(text):
    match = FINGERPRINT.search(text)
    return match.group(1) if match else None


def collect(port_name):
    port, used = open_port(port_name)
    port.setDTR(False)
    time.sleep(0.3)
    port.setDTR(True)
    time.sleep(8)
    banner = read_until(port, 25)
    rows = []
    for prompt in PROMPTS:
        port.reset_input_buffer()
        port.write((prompt + "\n").encode())
        port.flush()
        rows.append(parse(read_until(port, 180), prompt))
    port.close()
    return banner, rows, used


def validate(banner, rows):
    problems = []
    if not fingerprint(banner):
        problems.append("board printed no model fingerprint")
    for row in rows:
        if not row["complete"]:
            problems.append(f"prompt did not complete: {row['prompt']}")
        if not row["tokens"] or not row["ms"] or not row["answer"]:
            problems.append(f"prompt has incomplete receipt: {row['prompt']}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")
    passes = []
    for _ in range(args.passes):
        banner, rows, port = collect(args.port)
        problems = validate(banner, rows)
        if problems:
            raise SystemExit("benchmark rejected: " + "; ".join(problems))
        passes.append({"banner": banner.splitlines(), "fingerprint": fingerprint(banner),
                       "rows": rows, "port": port})
    answers = [{row["prompt"]: row["answer"] for row in item["rows"]} for item in passes]
    if len({json.dumps(answer, sort_keys=True) for answer in answers}) != 1:
        raise SystemExit("benchmark rejected: generated text differs between passes")
    tokens = sum(row["tokens"] for item in passes for row in item["rows"])
    elapsed = sum(row["ms"] for item in passes for row in item["rows"])
    receipt = {"model": "historinhas", "fingerprint": passes[0]["fingerprint"],
               "passes": passes, "summary": {"tokens": tokens, "ms": elapsed,
               "ms_per_token": elapsed / tokens, "tok_per_s": 1000 * tokens / elapsed,
               "median_ms_per_pass": statistics.median(sum(row["ms"] for row in item["rows"]) for item in passes)}}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        if baseline.get("model") != "historinhas":
            raise SystemExit("baseline is not a Historinhas receipt")
        baseline_answers = {row["prompt"]: row["answer"]
                            for row in baseline["passes"][0]["rows"]}
        current_answers = {row["prompt"]: row["answer"] for row in passes[0]["rows"]}
        if baseline_answers != current_answers:
            raise SystemExit("benchmark rejected: generated text differs from baseline")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()

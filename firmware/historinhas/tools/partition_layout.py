"""Read the Historinhas partition map without duplicating offsets in tools.

Usage:
  python partition_layout.py --name model

Prints the selected partition's hexadecimal offset and decimal byte size, which
is deliberately easy for shell deploy scripts to consume.
"""

import argparse
import csv
from pathlib import Path


SKETCH = Path(__file__).resolve().parents[1]
PARTITIONS = SKETCH / "partitions.csv"


def partition(name, path=PARTITIONS):
    """Return ``(offset, size)`` for a named explicit-offset CSV partition."""
    with open(path, newline="") as source:
        rows = csv.reader(line for line in source if line.strip() and
                          not line.lstrip().startswith("#"))
        for row in rows:
            fields = [field.strip() for field in row]
            if len(fields) < 5 or fields[0] != name:
                continue
            try:
                offset, size = int(fields[3], 0), int(fields[4], 0)
            except ValueError as error:
                raise ValueError(f"invalid offset/size for {name!r} in {path}") from error
            if offset < 0 or size <= 0:
                raise ValueError(f"invalid non-positive range for {name!r} in {path}")
            return offset, size
    raise ValueError(f"partition {name!r} not found in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="model")
    parser.add_argument("--partitions", type=Path, default=PARTITIONS)
    args = parser.parse_args()
    offset, size = partition(args.name, args.partitions)
    print(f"{offset:#x} {size}")


if __name__ == "__main__":
    main()

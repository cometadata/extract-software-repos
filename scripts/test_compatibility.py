#!/usr/bin/env python
"""Test output compatibility with datacite-enrichment."""

import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compare_enrichments(old_path: Path, new_path: Path) -> bool:
    old = load_jsonl(old_path)
    new = load_jsonl(new_path)

    print(f"Old enrichments: {len(old)}")
    print(f"New enrichments: {len(new)}")

    if len(old) != len(new):
        print("WARNING: Different number of enrichments")

    if old and new:
        old_keys = set(old[0].keys())
        new_keys = set(new[0].keys())

        if old_keys != new_keys:
            print(f"Missing keys: {old_keys - new_keys}")
            print(f"Extra keys: {new_keys - old_keys}")
            return False

        old_ev = set(old[0].get("enrichedValue", {}).keys())
        new_ev = set(new[0].get("enrichedValue", {}).keys())

        if old_ev != new_ev:
            print(f"enrichedValue missing: {old_ev - new_ev}")
            print(f"enrichedValue extra: {new_ev - old_ev}")
            return False

    print("Structure matches!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_compatibility.py <old_output> <new_output>")
        sys.exit(1)

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])

    if compare_enrichments(old_path, new_path):
        print("PASS: Outputs are compatible")
        sys.exit(0)
    else:
        print("FAIL: Outputs differ")
        sys.exit(1)

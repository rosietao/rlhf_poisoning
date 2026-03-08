#!/usr/bin/env python3
"""
Create a 1K exclusive subset from the combined dataset, ensuring none of
its samples appear in the existing 20K subset.

- Uniqueness key: md5(prompt + "\n" + response)
- Exclusion: all keys from 20K subset are excluded
- Sampling: reservoir sampling to avoid large memory usage

Usage:
  python scripts/create_exclusive_subset.py \
    --combined /data/home/Yunsheng/alignment-handbook/datasets/combined_alpaca_smoltalk_dataset.jsonl \
    --subset20 /data/home/Yunsheng/alignment-handbook/datasets/combined_subset_20K.jsonl \
    --output /data/home/Yunsheng/alignment-handbook/datasets/combined_subset_1K_exclusive.jsonl \
    --k 1000 \
    --seed 42
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Any, Iterable, List


def make_key(obj: Dict[str, Any]) -> str:
    prompt = obj.get("prompt", "")
    response = obj.get("response", "")
    return hashlib.md5((prompt + "\n" + response).encode("utf-8")).hexdigest()


def load_exclusion_keys(subset_path: Path) -> set:
    keys = set()
    with subset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            keys.add(make_key(obj))
    return keys


def reservoir_sample_exclusive(
    combined_path: Path,
    exclude_keys: set,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    reservoir: List[Dict[str, Any]] = []
    m = 0  # count of eligible (not excluded) items seen so far

    with combined_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = make_key(obj)
            if key in exclude_keys:
                continue

            m += 1
            if len(reservoir) < k:
                reservoir.append(obj)
            else:
                # Pick a random int in [0, m-1]
                j = rng.randrange(m)
                if j < k:
                    reservoir[j] = obj

    return reservoir


def main():
    parser = argparse.ArgumentParser(description="Create exclusive subset (not in 20K) via reservoir sampling")
    parser.add_argument("--combined", type=str, required=True, help="Path to combined JSONL dataset")
    parser.add_argument("--subset20", type=str, required=True, help="Path to existing 20K subset JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output path for 1K exclusive subset JSONL")
    parser.add_argument("--k", type=int, default=1000, help="Subset size (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    combined_path = Path(args.combined)
    subset20_path = Path(args.subset20)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load exclusion keys
    exclude_keys = load_exclusion_keys(subset20_path)

    # Reservoir sampling of exclusive items
    rng = random.Random(args.seed)
    sample = reservoir_sample_exclusive(combined_path, exclude_keys, args.k, rng)

    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for obj in sample:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(sample)} rows to {output_path}")


if __name__ == "__main__":
    main()

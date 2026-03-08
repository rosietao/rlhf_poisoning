#!/usr/bin/env python
# coding=utf-8
"""
Unified converter: RLHF JSONL (prompt/chosen/rejected) -> cleaned + HF DatasetDict on disk

Features:
- Reads JSONL with fields: prompt, chosen, rejected. Preserves chosen_sas_score/rejected_sas_score if present
- Optional prompt prefix stripping (e.g., "<turn> user\n")
- Shuffles with seed and splits into train/test
- Saves to HuggingFace DatasetDict via save_to_disk(output_dir)
- Optional: also write cleaned JSONL alongside saved dataset for inspection

Example:
python scripts/convert_rlhf_to_hf.py \
  --input /data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled/rlhf_sampled_dataset_transformed.jsonl \
  --output_dir /data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_hf \
  --train_ratio 0.9 \
  --seed 42 \
  --strip_prompt_prefix "<turn> user\n" \
  --write_clean_jsonl
"""

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict


def read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def clean_record(
    rec: Dict,
    strip_prefix: Optional[str] = None,
) -> Optional[Dict]:
    # Keep only the required fields; ignore sas scores if they exist
    if "prompt" not in rec or "chosen" not in rec or "rejected" not in rec:
        return None

    prompt = rec.get("prompt", "")
    chosen = rec.get("chosen", "")
    rejected = rec.get("rejected", "")

    if strip_prefix and isinstance(prompt, str) and prompt.startswith(strip_prefix):
        prompt = prompt[len(strip_prefix) :]

    out = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    # Preserve SAS scores if present
    if "chosen_sas_score" in rec:
        out["chosen_sas_score"] = rec["chosen_sas_score"]
    if "rejected_sas_score" in rec:
        out["rejected_sas_score"] = rec["rejected_sas_score"]
    return out


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RLHF JSONL to HF Dataset on disk")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path (prompt/chosen/rejected)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for save_to_disk")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Train split ratio (0-1). Use 1.0 for train-only dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--strip_prompt_prefix", type=str, default=None, help="If provided, strip this prefix from prompt")
    parser.add_argument("--write_clean_jsonl", action="store_true", help="Also write cleaned JSONL next to output_dir")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Read and clean
    raw = read_jsonl(str(input_path))
    cleaned: List[Dict] = []
    for rec in raw:
        c = clean_record(rec, strip_prefix=args.strip_prompt_prefix)
        if c is not None:
            cleaned.append(c)

    if not cleaned:
        raise ValueError("No valid samples after cleaning. Check input format.")

    # Shuffle data
    random.seed(args.seed)
    random.shuffle(cleaned)

    num_total = len(cleaned)
    
    if args.train_ratio >= 1.0:
        # Train-only dataset
        train_rows = cleaned
        test_rows = []
        ds_dict = DatasetDict({
            "train": Dataset.from_list(train_rows),
        })
    else:
        # Split into train/test
        num_train = int(num_total * args.train_ratio)
        num_train = max(1, min(num_train, num_total - 1))  # ensure both splits non-empty
        
        train_rows = cleaned[:num_train]
        test_rows = cleaned[num_train:]
        
        ds_dict = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        })
    ds_dict.save_to_disk(args.output_dir)

    # Optionally also write cleaned JSONL for reference
    if args.write_clean_jsonl:
        out_jsonl = os.path.join(args.output_dir, "cleaned_dpo.jsonl")
        write_jsonl(out_jsonl, cleaned)

    if args.train_ratio >= 1.0:
        print(f"Saved train-only HF dataset to {args.output_dir}. Train={len(train_rows)} Total={num_total}")
    else:
        print(f"Saved HF dataset to {args.output_dir}. Train={len(train_rows)} Test={len(test_rows)} Total={num_total}")


if __name__ == "__main__":
    main()

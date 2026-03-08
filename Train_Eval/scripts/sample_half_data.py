#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import random
from typing import List


def sample_half_data(
    input_file: str,
    output_file: str,
    sample_ratio: float = 0.5,
    random_seed: int = 42
):
    """
    Sample a portion of data from the input file and save to output file
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print(f"📖 Reading data from {input_file}")
    
    # Read all data
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data_item = json.loads(line.strip())
                    all_data.append(data_item)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON at line {line_num}: {e}")
                    continue
            
            # Progress update for large files
            if line_num % 100000 == 0:
                print(f"   Read {line_num:,} lines...")
    
    total_count = len(all_data)
    print(f"📊 Total data points: {total_count:,}")
    
    # Sample data
    sample_size = int(total_count * sample_ratio)
    print(f"🎯 Sampling {sample_size:,} data points ({sample_ratio:.1%})")
    
    sampled_data = random.sample(all_data, sample_size)
    
    # Save sampled data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"💾 Saving sampled data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully sampled {len(sampled_data):,} data points")
    print(f"📈 Sampling ratio: {len(sampled_data)/total_count:.3f}")
    print(f"💿 Output file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Sample a portion of data from JSONL file")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--sample_ratio", type=float, default=0.5, help="Ratio of data to sample (default: 0.5)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file does not exist: {args.input_file}")
        return 1
    
    if args.sample_ratio <= 0 or args.sample_ratio > 1:
        print(f"❌ Error: Sample ratio must be between 0 and 1, got: {args.sample_ratio}")
        return 1
    
    try:
        sample_half_data(
            input_file=args.input_file,
            output_file=args.output_file,
            sample_ratio=args.sample_ratio,
            random_seed=args.random_seed
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())


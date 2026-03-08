#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import math


def split_dataset(
    input_file: str,
    output_dir: str,
    num_splits: int = 4,
    prefix: str = "split"
):
    """
    Split a large dataset into smaller chunks
    """
    
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
    
    # Calculate split sizes
    split_size = math.ceil(total_count / num_splits)
    print(f"🔪 Splitting into {num_splits} parts, ~{split_size:,} items each")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and save
    output_files = []
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = min(start_idx + split_size, total_count)
        split_data = all_data[start_idx:end_idx]
        
        output_file = os.path.join(output_dir, f"{prefix}_{i+1}.jsonl")
        output_files.append(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"💾 Split {i+1}: {len(split_data):,} items → {output_file}")
    
    print(f"✅ Successfully split dataset into {num_splits} parts")
    print(f"📁 Output directory: {output_dir}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description="Split dataset into multiple parts")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split files")
    parser.add_argument("--num_splits", type=int, default=4, help="Number of splits (default: 4)")
    parser.add_argument("--prefix", type=str, default="split", help="Prefix for output files (default: split)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file does not exist: {args.input_file}")
        return 1
    
    if args.num_splits <= 0:
        print(f"❌ Error: Number of splits must be positive, got: {args.num_splits}")
        return 1
    
    try:
        split_dataset(
            input_file=args.input_file,
            output_dir=args.output_dir,
            num_splits=args.num_splits,
            prefix=args.prefix
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

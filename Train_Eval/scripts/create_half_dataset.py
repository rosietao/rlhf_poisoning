#!/usr/bin/env python3
"""
Script to randomly sample half of the data from a JSONL file
"""

import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Randomly sample half of the data from a JSONL file")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--sample_ratio", type=float, default=0.5,
                       help="Ratio of data to sample (default: 0.5 for half)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"🚀 Starting random sampling...")
    print(f"📂 Input file: {input_path}")
    print(f"📂 Output file: {output_path}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Sample ratio: {args.sample_ratio}")
    print()
    
    # Read all lines from input file
    print("📖 Reading input file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    target_lines = int(total_lines * args.sample_ratio)
    
    print(f"📊 Total lines: {total_lines:,}")
    print(f"📊 Target lines: {target_lines:,}")
    print()
    
    # Randomly sample the lines
    print("🎲 Randomly sampling lines...")
    sampled_lines = random.sample(lines, target_lines)
    
    # Write sampled lines to output file
    print("💾 Writing output file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)
    
    # Verify output
    actual_lines = len(sampled_lines)
    reduction_percent = (1 - actual_lines / total_lines) * 100
    
    print("✅ Sampling completed!")
    print("=" * 50)
    print(f"📊 Final Statistics:")
    print(f"   Original: {total_lines:,} lines")
    print(f"   Sampled:  {actual_lines:,} lines")
    print(f"   Ratio:    {actual_lines/total_lines:.3f}")
    print(f"   Reduced by: {reduction_percent:.1f}%")
    print(f"📂 Output saved to: {output_path}")
    
    # Validate the output file
    print()
    print("🔍 Validating output file...")
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                json.loads(line.strip())  # Validate JSON format
                if i >= 2:  # Only check first few lines for efficiency
                    break
        print("✅ Output file validation passed!")
    except Exception as e:
        print(f"❌ Output file validation failed: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
从combined_alpaca_smoltalk_dataset.jsonl中随机挑选80000个样本组成subset
"""

import json
import random
import argparse
from pathlib import Path

def create_subset(input_file: str, output_file: str, num_samples: int = 80000, seed: int = 42):
    """
    从JSONL文件中随机挑选指定数量的样本
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        num_samples: 要挑选的样本数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"📁 Reading data from: {input_path}")
    
    # 读取所有数据
    all_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    
    total_samples = len(all_data)
    print(f"📊 Total samples in dataset: {total_samples}")
    
    if num_samples > total_samples:
        print(f"⚠️  Warning: Requested {num_samples} samples, but only {total_samples} available")
        num_samples = total_samples
    
    # 随机挑选样本
    print(f"🎲 Randomly selecting {num_samples} samples (seed={seed})...")
    selected_data = random.sample(all_data, num_samples)
    
    # 保存到输出文件
    print(f"💾 Saving subset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in selected_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully created subset with {len(selected_data)} samples")
    print(f"📁 Output file: {output_path}")
    
    # 显示一些统计信息
    if selected_data:
        sample_keys = list(selected_data[0].keys())
        print(f"📋 Sample keys: {sample_keys}")
        
        # 显示前几个样本的预览
        print(f"\n📖 Preview of first 3 samples:")
        for i, sample in enumerate(selected_data[:3]):
            print(f"  Sample {i+1}: {json.dumps(sample, ensure_ascii=False)[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Create a random subset from JSONL file")
    parser.add_argument("--input_file", 
                       default="/data/home/Yunsheng/alignment-handbook/datasets/combined_alpaca_smoltalk_dataset.jsonl",
                       help="Input JSONL file path")
    parser.add_argument("--output_file", 
                       default="/data/home/Yunsheng/alignment-handbook/datasets/combined_alpaca_smoltalk_80K_subset.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--num_samples", type=int, default=80000,
                       help="Number of samples to select (default: 80000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    try:
        create_subset(args.input_file, args.output_file, args.num_samples, args.seed)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

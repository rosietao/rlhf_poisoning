#!/usr/bin/env python3
"""
从test8数据中创建固定的测试子集
直接使用索引 0, 2, 4, 6 作为起始索引，与它们的下一个索引组成pairs
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse


def create_fixed_subset(
    data_dir: str,
    start_indices: List[int] = [0, 2, 4, 6, 8, 10, 12, 14],
    output_suffix: str = "test8_fixed"
):
    """
    从test8数据中创建固定的测试子集
    
    Args:
        data_dir: 数据目录路径
        start_indices: 固定的起始索引列表
        output_suffix: 输出文件的后缀
    """
    print(f"🔍 从 {data_dir} 创建固定测试子集...")
    
    data_path = Path(data_dir)
    
    # 检查数据文件是否存在
    response_file = data_path / "response_sae_repr.pt"
    prompt_file = data_path / "prompt_embeddings.pt"
    metadata_file = data_path / "metadata.json"
    
    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # 加载数据
    print(f"🔄 加载数据文件...")
    response_sae_repr = torch.load(response_file, map_location="cpu")
    prompt_embeddings = torch.load(prompt_file, map_location="cpu")
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    total_samples = len(response_sae_repr)
    print(f"📊 总样本数: {total_samples}")
    print(f"📊 Response shape: {response_sae_repr.shape}")
    print(f"📊 Prompt shape: {prompt_embeddings.shape}")
    
    # 验证索引是否有效
    max_valid_index = total_samples - 1
    valid_indices = []
    for idx in start_indices:
        if idx < max_valid_index:
            valid_indices.append(idx)
        else:
            print(f"⚠️  警告：索引 {idx} 超出范围，跳过")
    
    if not valid_indices:
        raise ValueError("没有有效的起始索引")
    
    print(f"✅ 使用起始索引: {valid_indices}")
    
    # 创建pairs：每个起始index与它的next index组成一个pair
    pairs = []
    for i, start_idx in enumerate(valid_indices):
        next_idx = start_idx + 1
        pair = {
            'pair_id': i,
            'start_index': int(start_idx),
            'next_index': int(next_idx),
            'start_response': response_sae_repr[start_idx],
            'next_response': response_sae_repr[next_idx],
            'start_prompt': prompt_embeddings[start_idx],
            'next_prompt': prompt_embeddings[next_idx]
        }
        pairs.append(pair)
    
    print(f"✅ 创建了 {len(pairs)} 个测试pairs")
    
    # 创建输出目录
    output_dir = Path(data_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存测试子集数据
    subset_data = {
        'pairs': pairs,
        'num_pairs': len(pairs),
        'start_indices': valid_indices,
        'next_indices': [int(idx + 1) for idx in valid_indices],
        'original_data_dir': str(data_dir),
        'original_total_samples': total_samples,
        'subset_type': 'fixed_indices'
    }
    
    # 保存为JSON文件
    output_file = output_dir / f"test_subset_{output_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
    
    print(f"💾 测试子集已保存到: {output_file}")
    
    # 保存为PyTorch格式（可选，用于快速加载）
    torch_output_file = output_dir / f"test_subset_{output_suffix}.pt"
    torch.save(subset_data, torch_output_file)
    print(f"💾 PyTorch格式已保存到: {torch_output_file}")
    
    # 显示一些统计信息
    print(f"\n📊 测试子集统计信息:")
    print(f"  总pairs数: {len(pairs)}")
    print(f"  起始indices: {valid_indices}")
    print(f"  结束indices: {[idx + 1 for idx in valid_indices]}")
    
    # 显示所有pairs的示例
    print(f"\n📋 所有pairs:")
    for i, pair in enumerate(pairs):
        print(f"  Pair {i}: {pair['start_index']} -> {pair['next_index']}")
    
    return subset_data


def load_test_subset(data_dir: str, subset_suffix: str = "test8_fixed"):
    """
    加载测试子集
    
    Args:
        data_dir: 数据目录路径
        subset_suffix: 子集文件的后缀
    
    Returns:
        测试子集数据
    """
    data_path = Path(data_dir)
    
    # 尝试加载JSON格式
    json_file = data_path / f"test_subset_{subset_suffix}.json"
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            subset_data = json.load(f)
        print(f"✅ 从JSON文件加载测试子集: {json_file}")
        return subset_data
    
    # 尝试加载PyTorch格式
    torch_file = data_path / f"test_subset_{subset_suffix}.pt"
    if torch_file.exists():
        subset_data = torch.load(torch_file, map_location="cpu")
        print(f"✅ 从PyTorch文件加载测试子集: {torch_file}")
        return subset_data
    
    raise FileNotFoundError(f"测试子集文件未找到: {json_file} 或 {torch_file}")


def main():
    parser = argparse.ArgumentParser(description="创建固定的测试子集")
    parser.add_argument("--data_dir", type=str, 
                       default="datasets/vectorized_last_test8_split_sae_llama3b_layers_14",
                       help="数据目录路径")
    parser.add_argument("--start_indices", type=int, nargs='+', 
                       default=[0, 2, 4, 6, 8, 10, 12, 14], help="固定的起始索引列表")
    parser.add_argument("--output_suffix", type=str, default="last_fixed", help="输出文件后缀")
    parser.add_argument("--load_only", action="store_true", help="只加载现有子集，不创建新的")
    
    args = parser.parse_args()
    
    if args.load_only:
        # 只加载现有子集
        subset_data = load_test_subset(args.data_dir, args.output_suffix)
        print(f"📊 加载的子集包含 {subset_data['num_pairs']} 个pairs")
    else:
        # 创建新的测试子集
        subset_data = create_fixed_subset(
            data_dir=args.data_dir,
            start_indices=args.start_indices,
            output_suffix=args.output_suffix
        )
    
    print(f"✅ 完成！")


if __name__ == "__main__":
    main() 
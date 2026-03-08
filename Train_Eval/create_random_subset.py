#!/usr/bin/env python3
"""
从vectorized数据中创建固定的测试子集
随机选择500个indices，与它们的immediate next data组成pairs
用于确保不同实验之间的横向对比公平性
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse


def create_random_subset(
    data_dir: str,
    num_pairs: int = 500,
    seed: int = 42,
    output_suffix: str = "subset500"
):
    """
    从vectorized数据中创建固定的测试子集
    
    Args:
        data_dir: 数据目录路径
        num_pairs: 要创建的pair数量
        seed: 随机种子，确保结果可重现
        output_suffix: 输出文件的后缀
    """
    print(f"🔍 从 {data_dir} 创建测试子集...")
    
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
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 随机选择500个indices
    # 注意：我们需要确保每个index都有next index，所以最大index是total_samples - 1
    max_start_index = total_samples - 1
    if num_pairs > max_start_index:
        print(f"⚠️  警告：请求的pair数量({num_pairs})超过了可用样本数({max_start_index})")
        num_pairs = max_start_index
    
    # 随机选择起始indices
    start_indices = np.random.choice(max_start_index, num_pairs, replace=False)
    
    # 创建pairs：每个起始index与它的next index组成一个pair
    pairs = []
    for i, start_idx in enumerate(start_indices):
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
        'seed': seed,
        'original_data_dir': str(data_dir),
        'original_total_samples': total_samples,
        'start_indices': start_indices.tolist(),
        'next_indices': [int(idx + 1) for idx in start_indices]
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
    print(f"  起始indices范围: {min(start_indices)} - {max(start_indices)}")
    print(f"  结束indices范围: {min(start_indices) + 1} - {max(start_indices) + 1}")
    print(f"  随机种子: {seed}")
    
    # 显示前几个pairs的示例
    print(f"\n📋 前3个pairs示例:")
    for i in range(min(3, len(pairs))):
        pair = pairs[i]
        print(f"  Pair {i}: {pair['start_index']} -> {pair['next_index']}")
    
    return subset_data


def load_test_subset(data_dir: str, subset_suffix: str = "subset500"):
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
                       default="datasets/vectorized_last_smoltalk_single_round_sae_llama3b_layers_14",
                       help="数据目录路径")
    parser.add_argument("--num_pairs", type=int, default=500, help="要创建的pair数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_suffix", type=str, default="subset500", help="输出文件后缀")
    parser.add_argument("--load_only", action="store_true", help="只加载现有子集，不创建新的")
    
    args = parser.parse_args()
    
    if args.load_only:
        # 只加载现有子集
        subset_data = load_test_subset(args.data_dir, args.output_suffix)
        print(f"📊 加载的子集包含 {subset_data['num_pairs']} 个pairs")
    else:
        # 创建新的测试子集
        subset_data = create_random_subset(
            data_dir=args.data_dir,
            num_pairs=args.num_pairs,
            seed=args.seed,
            output_suffix=args.output_suffix
        )
    
    print(f"✅ 完成！")


if __name__ == "__main__":
    main() 
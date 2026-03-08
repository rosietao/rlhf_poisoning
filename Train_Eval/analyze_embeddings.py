#!/usr/bin/env python3
"""
分析prompt和response embedding的统计信息
"""

import torch
import json
from pathlib import Path
import argparse
import math

def analyze_embeddings(data_dir: str):
    """分析embedding向量的统计信息"""
    data_path = Path(data_dir)
    
    print(f"📁 Analyzing embeddings from: {data_path}")
    
    # 加载数据
    print("🔄 Loading response_sae_repr.pt...")
    response_sae_repr = torch.load(data_path / "response_sae_repr.pt")
    
    print("🔄 Loading prompt_embeddings.pt...")
    prompt_embeddings = torch.load(data_path / "prompt_embeddings.pt")
    
    # 分析response SAE representation
    print("\n📊 Response SAE Representation Statistics:")
    print(f"  Shape: {response_sae_repr.shape}")
    print(f"  Min: {response_sae_repr.min().item():.6f}")
    print(f"  Max: {response_sae_repr.max().item():.6f}")
    print(f"  Mean: {response_sae_repr.mean().item():.6f}")
    print(f"  Std: {response_sae_repr.std().item():.6f}")
    print(f"  Range: [{response_sae_repr.min().item():.6f}, {response_sae_repr.max().item():.6f}]")
    
    # 检查NaN和Inf
    nan_count = torch.isnan(response_sae_repr).sum().item()
    inf_count = torch.isinf(response_sae_repr).sum().item()
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    
    # 分析prompt embeddings
    print("\n📊 Prompt Embeddings Statistics:")
    print(f"  Shape: {prompt_embeddings.shape}")
    print(f"  Min: {prompt_embeddings.min().item():.6f}")
    print(f"  Max: {prompt_embeddings.max().item():.6f}")
    print(f"  Mean: {prompt_embeddings.mean().item():.6f}")
    print(f"  Std: {prompt_embeddings.std().item():.6f}")
    print(f"  Range: [{prompt_embeddings.min().item():.6f}, {prompt_embeddings.max().item():.6f}]")
    
    # 检查NaN和Inf
    nan_count = torch.isnan(prompt_embeddings).sum().item()
    inf_count = torch.isinf(prompt_embeddings).sum().item()
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    
    # 计算方差用于Kaiming初始化
    print("\n🔬 Kaiming Initialization Analysis:")
    
    # 计算输入方差 (response SAE representations)
    # 对每个4096维向量计算mean squares，然后average over all data points
    input_variance = torch.mean(torch.var(response_sae_repr, dim=1))
    print(f"  Input variance (σ²_x): {input_variance.item():.8f}")
    
    # 计算输出方差 (prompt embeddings)
    # 对每个4096维向量计算mean squares，然后average over all data points
    output_variance = torch.mean(torch.var(prompt_embeddings, dim=1))
    print(f"  Output variance (σ²_y): {output_variance.item():.8f}")
    
    # 计算输入维度
    input_dim = response_sae_repr.shape[1]  # 应该是131072
    print(f"  Input dimension: {input_dim}")
    
    # 验证输入维度是否为 131072
    if input_dim != 131072:
        print(f"  ⚠️  Warning: Expected input dimension 4096, got {input_dim}")
    
    # 计算权重标准差
    # σ_w = √(σ²_y / (input_dim * σ²_x))
    weight_std = math.sqrt(output_variance.item() / (input_dim * input_variance.item()))
    print(f"  Weight standard deviation (σ_w): {weight_std:.8f}")
    
    # 验证计算
    print(f"\n🔍 Verification:")
    print(f"  σ²_y / (input_dim * σ²_x) = {output_variance.item():.8f} / ({input_dim} * {input_variance.item():.8f}) = {output_variance.item() / (input_dim * input_variance.item()):.8f}")
    print(f"  √(σ²_y / (input_dim * σ²_x)) = {weight_std:.8f}")
    
    # 检查是否合理
    if weight_std > 1.0:
        print(f"  ⚠️  Warning: Weight std ({weight_std:.4f}) seems high, might cause gradient explosion")
    elif weight_std < 0.01:
        print(f"  ⚠️  Warning: Weight std ({weight_std:.4f}) seems low, might cause gradient vanishing")
    else:
        print(f"  ✅ Weight std ({weight_std:.4f}) looks reasonable")
    
    # 加载metadata
    if (data_path / "metadata.json").exists():
        with open(data_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"\n📋 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # 计算一些额外的统计信息
    print(f"\n📈 Additional Statistics:")
    print(f"  Response SAE sparsity: {(response_sae_repr == 0).float().mean().item():.4f}")
    print(f"  Prompt embeddings L2 norm mean: {torch.norm(prompt_embeddings, dim=1).mean().item():.6f}")
    print(f"  Response SAE L2 norm mean: {torch.norm(response_sae_repr, dim=1).mean().item():.6f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze embedding statistics")
    parser.add_argument("--data_dir", type=str, default="datasets/vectorized_smoltalk_single_round_sae_llama3b_layers_14", 
                       help="Path to vectorized data directory")
    args = parser.parse_args()
    
    analyze_embeddings(args.data_dir)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
独立的分析脚本，用于分析已训练好的decoder模型
分析decoder weight分布和预测的prompt embedding分布，检测degenerate情况
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import yaml
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 添加当前目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pd_train_accelerate import PromptDecoder, VectorizedDataset

def analyze_trained_model(
    model_path: str,
    data_dir: str,
    output_dir: str = "analysis_results",
    device: str = "cuda",
    normalize: bool = False
):
    """
    分析已训练好的decoder模型
    
    Args:
        model_path: 模型检查点路径
        data_dir: 数据目录路径
        output_dir: 输出目录
        device: 设备
        normalize: 是否normalize
    """
    print(f"🔍 Analyzing trained model from: {model_path}")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("🔄 Loading model...")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # 从checkpoint中提取配置
    config = checkpoint.get('config', {})
    sae_path = config.get('sae_path', 'sae/sae_llama3b_layers_10.pth')
    original_normalize = config.get('normalize', False)
    
    print(f"📋 Model config - Original normalize: {original_normalize}, Current normalize: {normalize}")
    
    # 创建模型
    model = PromptDecoder(
        sae_path=sae_path,
        device=device,
        dtype=torch.float32,
        normalize=False  # 不在这里normalize，因为我们要加载已训练的权重
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # 加载数据
    print("🔄 Loading data...")
    data_dir = Path(data_dir)
    
    # 检查数据文件
    response_file = data_dir / "response_sae_repr.pt"
    prompt_file = data_dir / "prompt_embeddings.pt"
    
    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    # 加载数据
    response_sae_repr = torch.load(response_file, map_location="cpu")
    prompt_embeddings = torch.load(prompt_file, map_location="cpu")
    
    # 替换NaN
    if torch.isnan(response_sae_repr).any():
        print("⚠️  Found NaN in response_sae_repr, replacing with 0")
        response_sae_repr = torch.nan_to_num(response_sae_repr, nan=0.0)
    
    if torch.isnan(prompt_embeddings).any():
        print("⚠️  Found NaN in prompt_embeddings, replacing with 0")
        prompt_embeddings = torch.nan_to_num(prompt_embeddings, nan=0.0)
    
    print(f"✅ Data loaded. Shape: response {response_sae_repr.shape}, prompt {prompt_embeddings.shape}")
    
    # 创建数据集和数据加载器
    dataset = VectorizedDataset(response_sae_repr, prompt_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    
    # 分析模型
    print("🔍 Starting analysis...")
    analysis_results = analyze_model_distributions(model, dataloader, device, "full_dataset")
    
    # 保存分析结果
    analysis_file = os.path.join(output_dir, 'analysis_results.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"📁 Analysis results saved to {analysis_file}")
    
    # 创建可视化图表
    try:
        create_analysis_plots(analysis_results, output_dir)
        print(f"📊 Analysis plots saved to {output_dir}")
    except Exception as e:
        print(f"⚠️  Failed to create plots: {e}")
    
    return analysis_results

def analyze_model_distributions(model, dataloader, device, split_name):
    """分析模型分布"""
    print(f"📊 Analyzing distributions for {split_name}...")
    
    # 1. 分析decoder weight分布
    decoder_weight = model.decoder_weight.data.cpu().numpy()
    decoder_bias = model.decoder_bias.data.cpu().numpy()
    
    weight_stats = {
        'mean': float(np.mean(decoder_weight)),
        'std': float(np.std(decoder_weight)),
        'min': float(np.min(decoder_weight)),
        'max': float(np.max(decoder_weight)),
        'zero_ratio': float(np.sum(decoder_weight == 0) / decoder_weight.size),
        'nan_ratio': float(np.sum(np.isnan(decoder_weight)) / decoder_weight.size),
        'inf_ratio': float(np.sum(np.isinf(decoder_weight)) / decoder_weight.size),
    }
    
    bias_stats = {
        'mean': float(np.mean(decoder_bias)),
        'std': float(np.std(decoder_bias)),
        'min': float(np.min(decoder_bias)),
        'max': float(np.max(decoder_bias)),
        'zero_ratio': float(np.sum(decoder_bias == 0) / decoder_bias.size),
        'nan_ratio': float(np.sum(np.isnan(decoder_bias)) / decoder_bias.size),
        'inf_ratio': float(np.sum(np.isinf(decoder_bias)) / decoder_bias.size),
    }
    
    print(f"\n📊 Decoder Weight Statistics ({split_name}):")
    for key, value in weight_stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n📊 Decoder Bias Statistics ({split_name}):")
    for key, value in bias_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # 2. 分析预测的prompt embedding分布
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting predictions for {split_name}"):
            response_sae = batch['response_sae'].float().to(device)
            prompt_embedding = batch['prompt_embedding'].float().to(device)
            
            predicted_prompt = model(response_sae)
            
            all_predictions.append(predicted_prompt.cpu().numpy())
            all_targets.append(prompt_embedding.cpu().numpy())
            all_inputs.append(response_sae.cpu().numpy())
    
    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)
    
    # 分析预测分布
    prediction_stats = {
        'mean': float(np.mean(all_predictions)),
        'std': float(np.std(all_predictions)),
        'min': float(np.min(all_predictions)),
        'max': float(np.max(all_predictions)),
        'zero_ratio': float(np.sum(all_predictions == 0) / all_predictions.size),
        'nan_ratio': float(np.sum(np.isnan(all_predictions)) / all_predictions.size),
        'inf_ratio': float(np.sum(np.isinf(all_predictions)) / all_predictions.size),
        'constant_ratio': float(np.sum(np.all(all_predictions == all_predictions[0], axis=1)) / len(all_predictions)),
    }
    
    # 分析目标分布
    target_stats = {
        'mean': float(np.mean(all_targets)),
        'std': float(np.std(all_targets)),
        'min': float(np.min(all_targets)),
        'max': float(np.max(all_targets)),
        'zero_ratio': float(np.sum(all_targets == 0) / all_targets.size),
        'nan_ratio': float(np.sum(np.isnan(all_targets)) / all_targets.size),
        'inf_ratio': float(np.sum(np.isinf(all_targets)) / all_targets.size),
    }
    
    # 分析输入分布
    input_stats = {
        'mean': float(np.mean(all_inputs)),
        'std': float(np.std(all_inputs)),
        'min': float(np.min(all_inputs)),
        'max': float(np.max(all_inputs)),
        'zero_ratio': float(np.sum(all_inputs == 0) / all_inputs.size),
        'nan_ratio': float(np.sum(np.isnan(all_inputs)) / all_inputs.size),
        'inf_ratio': float(np.sum(np.isinf(all_inputs)) / all_inputs.size),
    }
    
    print(f"\n📊 Prediction Statistics ({split_name}):")
    for key, value in prediction_stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n📊 Target Statistics ({split_name}):")
    for key, value in target_stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\n📊 Input Statistics ({split_name}):")
    for key, value in input_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # 3. 检测degenerate情况
    degenerate_indicators = []
    
    # 检查预测是否全为0
    if prediction_stats['zero_ratio'] > 0.95:
        degenerate_indicators.append(f"⚠️  High zero ratio in predictions: {prediction_stats['zero_ratio']:.3f}")
    
    # 检查预测是否全为常数
    if prediction_stats['constant_ratio'] > 0.95:
        degenerate_indicators.append(f"⚠️  High constant ratio in predictions: {prediction_stats['constant_ratio']:.3f}")
    
    # 检查预测方差是否过小
    if prediction_stats['std'] < 1e-6:
        degenerate_indicators.append(f"⚠️  Very low prediction std: {prediction_stats['std']:.6f}")
    
    # 检查预测和目标之间的相关性
    correlation = np.corrcoef(all_predictions.flatten(), all_targets.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    print(f"\n🔗 Prediction-Target Correlation: {correlation:.6f}")
    
    if correlation < 0.1:
        degenerate_indicators.append(f"⚠️  Low correlation between predictions and targets: {correlation:.3f}")
    
    # 检查decoder weight是否退化
    if weight_stats['std'] < 1e-6:
        degenerate_indicators.append(f"⚠️  Very low decoder weight std: {weight_stats['std']:.6f}")
    
    if weight_stats['zero_ratio'] > 0.95:
        degenerate_indicators.append(f"⚠️  High zero ratio in decoder weights: {weight_stats['zero_ratio']:.3f}")
    
    # 输出degenerate检测结果
    if degenerate_indicators:
        print(f"\n🚨 DEGENERATE DETECTION RESULTS ({split_name}):")
        for indicator in degenerate_indicators:
            print(f"  {indicator}")
    else:
        print(f"\n✅ No obvious degenerate patterns detected in {split_name} split")
    
    # 4. 返回分析结果
    analysis_results = {
        'split_name': split_name,
        'decoder_weight_stats': weight_stats,
        'decoder_bias_stats': bias_stats,
        'prediction_stats': prediction_stats,
        'target_stats': target_stats,
        'input_stats': input_stats,
        'correlation': float(correlation),
        'degenerate_indicators': degenerate_indicators,
        'all_predictions': all_predictions.tolist(),  # 保存用于绘图
        'all_targets': all_targets.tolist(),
        'decoder_weights': decoder_weight.tolist(),
    }
    
    return analysis_results

def create_analysis_plots(analysis_results, output_dir):
    """创建分析图表"""
    predictions = np.array(analysis_results['all_predictions'])
    targets = np.array(analysis_results['all_targets'])
    decoder_weights = np.array(analysis_results['decoder_weights'])
    split_name = analysis_results['split_name']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Decoder Analysis - {split_name} Split', fontsize=16)
    
    # 1. Decoder weight分布
    axes[0, 0].hist(decoder_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Decoder Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 预测值分布
    axes[0, 1].hist(predictions.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Prediction Distribution')
    axes[0, 1].set_xlabel('Prediction Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 目标值分布
    axes[0, 2].hist(targets.flatten(), bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_title('Target Distribution')
    axes[0, 2].set_xlabel('Target Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 预测vs目标散点图（采样）
    sample_size = min(1000, len(predictions))
    sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
    axes[1, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), alpha=0.5, s=1)
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', alpha=0.8)
    axes[1, 0].set_title('Predictions vs Targets')
    axes[1, 0].set_xlabel('Target Value')
    axes[1, 0].set_ylabel('Prediction Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 预测误差分布
    errors = predictions - targets
    axes[1, 1].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Error Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Decoder weight的绝对值分布
    axes[1, 2].hist(np.abs(decoder_weights.flatten()), bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 2].set_title('Decoder Weight Absolute Values')
    axes[1, 2].set_xlabel('|Weight Value|')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, f'{split_name}_analysis_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analysis plots saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze trained decoder model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to vectorized data directory")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Output directory for analysis results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize decoder weights")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # 运行分析
    try:
        results = analyze_trained_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            normalize=args.normalize
        )
        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
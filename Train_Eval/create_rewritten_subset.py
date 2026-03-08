#!/usr/bin/env python3
"""
从 datasets/ 下自动识别以 vectorized_last_test_ 开头、以 _sae_llama3b_layers_14 结尾的目录，
为每个目录创建 rewritten 测试所需的子集文件 test_subset_{dataset_name}.pt（以及同名 .json）。

子集文件结构与 pd_train_accelerate.py 中 _test_cross_prediction_rewritten 使用方式兼容：
- 必含字段：
  - response_sae_repr: 全量张量（N, 4096*32）
  - prompt_embeddings: 全量张量（N, 4096）
  - num_samples: 样本总数 N
- 兼容性附加字段（便于后续分析/校验）：
  - start_indices: 形如 [0, 2, 4, ..., N-2]
  - next_indices: 形如 [1, 3, 5, ..., N-1]
  - num_pairs: len(start_indices)
  - dataset_name, original_data_dir, subset_type

保存位置：datasets/vectorized_last_{dataset_name}_sae_llama3b_layers_14/test_subset_{dataset_name}.pt
"""

import os
import json
from pathlib import Path
from typing import List, Tuple

import torch


def discover_vectorized_test_dirs(base_dir: Path) -> List[Path]:
    """
    发现形如 vectorized_last_test_*_sae_llama3b_layers_14 的目录。
    兼容用户口误：若未找到，则尝试匹配 vectorized_last_text_*。
    """
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("vectorized_last_test_") and d.name.endswith("_sae_llama3b_layers_14")
    ]
    if candidates:
        return candidates

    # 兜底：有时用户会写成 text
    candidates = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("vectorized_last_text_") and d.name.endswith("_sae_llama3b_layers_14")
    ]
    return candidates


def build_even_odd_indices(num_samples: int) -> Tuple[List[int], List[int]]:
    """构造 start_indices=[0,2,...] 和 next_indices=[1,3,...]，长度对齐。
    若样本数为奇数，则丢弃最后一个样本以保证成对。
    """
    if num_samples <= 0:
        return [], []
    last_even = num_samples - 1 if (num_samples % 2 == 0) else num_samples - 2
    start_indices = list(range(0, last_even, 2))
    next_indices = [i + 1 for i in start_indices]
    return start_indices, next_indices


def create_subset_for_dir(dataset_dir: Path) -> Path:
    """为单个 vectorized 目录创建 test_subset_{dataset_name}.pt 与 .json。

    返回生成的 .pt 文件路径。
    """
    response_file = dataset_dir / "response_sae_repr.pt"
    prompt_file = dataset_dir / "prompt_embeddings.pt"
    metadata_file = dataset_dir / "metadata.json"

    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"🔄 加载数据: {dataset_dir}")
    response_sae_repr = torch.load(response_file, map_location="cpu")
    prompt_embeddings = torch.load(prompt_file, map_location="cpu")

    if len(response_sae_repr) != len(prompt_embeddings):
        raise ValueError(
            f"Length mismatch: response {len(response_sae_repr)} vs prompt {len(prompt_embeddings)} in {dataset_dir}"
        )

    num_samples = len(response_sae_repr)
    start_indices, next_indices = build_even_odd_indices(num_samples)

    # dataset_name 形如 test_reject_math / test_rewrite_helpful 等
    dataset_name = dataset_dir.name.replace("vectorized_last_", "").replace("_sae_llama3b_layers_14", "")

    subset = {
        "response_sae_repr": response_sae_repr,
        "prompt_embeddings": prompt_embeddings,
        "num_samples": int(num_samples),
        # 兼容/便于分析的附加字段
        "start_indices": [int(x) for x in start_indices],
        "next_indices": [int(x) for x in next_indices],
        "num_pairs": int(len(start_indices)),
        "dataset_name": dataset_name,
        "original_data_dir": str(dataset_dir),
        "subset_type": "rewritten_auto",
    }

    out_pt = dataset_dir / f"test_subset_{dataset_name}.pt"
    out_json = dataset_dir / f"test_subset_{dataset_name}.json"

    torch.save(subset, out_pt)
    with open(out_json, "w", encoding="utf-8") as f:
        # 张量不能直接 JSON 序列化，这里仅保存元信息与形状，避免巨大 JSON
        json.dump(
            {
                "num_samples": subset["num_samples"],
                "start_indices": subset["start_indices"],
                "next_indices": subset["next_indices"],
                "num_pairs": subset["num_pairs"],
                "dataset_name": subset["dataset_name"],
                "original_data_dir": subset["original_data_dir"],
                "subset_type": subset["subset_type"],
                "response_shape": list(response_sae_repr.shape),
                "prompt_shape": list(prompt_embeddings.shape),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"💾 已生成: {out_pt.name} / {out_json.name}  — 样本数: {num_samples}, 配对数: {len(start_indices)}"
    )
    if num_samples % 2 != 0:
        print("⚠️  样本数为奇数，已丢弃最后一个样本以成对配对（仅在 start/next_indices 统计中体现，不影响张量存储）")

    return out_pt


def main():
    base_dir = Path("datasets")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    dirs = discover_vectorized_test_dirs(base_dir)
    if not dirs:
        print("⚠️  未发现匹配目录（vectorized_last_test_*_sae_llama3b_layers_14 或 vectorized_last_text_*_sae_llama3b_layers_14）")
        return

    print(f"🔍 共发现 {len(dirs)} 个测试数据目录\n")
    for d in sorted(dirs):
        try:
            create_subset_for_dir(d)
        except Exception as e:
            print(f"❌  处理失败: {d} — {e}")

    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()



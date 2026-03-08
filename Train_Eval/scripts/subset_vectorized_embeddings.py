#!/usr/bin/env python3
"""
Subset aligned vectorized embeddings (prompt_embeddings.pt and response_sae_repr.pt)
by selecting the same random indices for both tensors.

Usage:
  python scripts/subset_vectorized_embeddings.py \
    --input-dir /data/home/Yunsheng/alignment-handbook/datasets/combined_alpaca_smoltalk_sae_llama3b_layers_14 \
    --output-dir /data/home/Yunsheng/alignment-handbook/datasets/combined_subset_10K_sae_llama3b_layers_14 \
    --num-samples 10000 \
    --seed 42
"""

import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Subset aligned vectorized embeddings with shared indices")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing prompt_embeddings.pt and response_sae_repr.pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the subset tensors")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = in_dir / "prompt_embeddings.pt"
    response_path = in_dir / "response_sae_repr.pt"
    if not prompt_path.exists() or not response_path.exists():
        raise FileNotFoundError("Missing input tensors. Expected prompt_embeddings.pt and response_sae_repr.pt")

    prompt = torch.load(prompt_path, map_location="cpu")
    response = torch.load(response_path, map_location="cpu")

    if not isinstance(prompt, torch.Tensor) or not isinstance(response, torch.Tensor):
        raise TypeError("Loaded objects must be torch.Tensor")
    if prompt.size(0) != response.size(0):
        raise ValueError(f"Row mismatch: prompt={prompt.size(0)} response={response.size(0)}")

    total = prompt.size(0)
    k = min(args.num_samples, total)

    g = torch.Generator()
    g.manual_seed(args.seed)
    indices = torch.randperm(total, generator=g)[:k]

    prompt_subset = prompt.index_select(0, indices)
    response_subset = response.index_select(0, indices)

    # Save outputs
    torch.save(prompt_subset, out_dir / "prompt_embeddings.pt")
    torch.save(response_subset, out_dir / "response_sae_repr.pt")
    torch.save(indices, out_dir / "subset_indices.pt")

    # Metadata
    meta = {
        "source_dir": str(in_dir),
        "num_total": int(total),
        "num_subset": int(k),
        "prompt_shape": list(prompt_subset.size()),
        "response_shape": list(response_subset.size()),
        "seed": args.seed,
        "note": "Subset created with identical indices for prompt and response to preserve alignment"
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Saved subset of {k} rows to {out_dir}")


if __name__ == "__main__":
    main()


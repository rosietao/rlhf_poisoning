#!/usr/bin/env python3
"""
Recover an exact subset of aligned embeddings using saved indices.

This script loads prompt_embeddings.pt and response_sae_repr.pt from a FULL
combined directory, loads saved indices (from a subset dir or a direct path),
then extracts the same rows for both tensors and saves them to a new output dir.

It supports both file names: subset_indices.pt and subset_indice.pt (typo).

Usage:
  python scripts/recover_subset_vectorized_embeddings.py \
    --full-dir /data/home/Yunsheng/alignment-handbook/datasets/combined_alpaca_smoltalk_sae_llama3b_layers_14 \
    --indices-source /data/home/Yunsheng/alignment-handbook/datasets/combined_subset_20K_sae_llama3b_layers_14 \
    --output-dir /data/home/Yunsheng/alignment-handbook/datasets/combined_subset_20K

You can also pass a direct indices file path to --indices-source (ends with .pt).
"""

import argparse
import json
from pathlib import Path
import torch


def load_indices(indices_source: Path) -> torch.Tensor:
    """Load indices from a directory or a direct .pt file.
    Tries subset_indices.pt first, then subset_indice.pt.
    """
    if indices_source.is_file():
        return torch.load(indices_source, map_location="cpu")

    cand_a = indices_source / "subset_indices.pt"
    cand_b = indices_source / "subset_indice.pt"
    if cand_a.exists():
        return torch.load(cand_a, map_location="cpu")
    if cand_b.exists():
        return torch.load(cand_b, map_location="cpu")
    raise FileNotFoundError(
        f"Could not find indices in {indices_source}. Tried: {cand_a.name}, {cand_b.name}, or direct .pt path."
    )


def main():
    parser = argparse.ArgumentParser(description="Recover subset from full embeddings using saved indices")
    parser.add_argument("--full-dir", type=str, required=True, help="Dir with full prompt_embeddings.pt and response_sae_repr.pt")
    parser.add_argument("--indices-source", type=str, required=True, help="Dir containing indices file or a direct .pt file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save recovered subset")
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_full_path = full_dir / "prompt_embeddings.pt"
    response_full_path = full_dir / "response_sae_repr.pt"
    if not prompt_full_path.exists() or not response_full_path.exists():
        raise FileNotFoundError("Full dir missing prompt_embeddings.pt or response_sae_repr.pt")

    # Load full tensors
    prompt_full = torch.load(prompt_full_path, map_location="cpu")
    response_full = torch.load(response_full_path, map_location="cpu")

    if not isinstance(prompt_full, torch.Tensor) or not isinstance(response_full, torch.Tensor):
        raise TypeError("Loaded full tensors must be torch.Tensor")
    if prompt_full.size(0) != response_full.size(0):
        raise ValueError("Full tensors row count mismatch")

    # Load indices
    indices = load_indices(Path(args.indices_source))
    if not isinstance(indices, torch.Tensor):
        raise TypeError("Indices must be a torch.Tensor")

    # Ensure indices are 1D long tensor
    indices = indices.to(dtype=torch.long).view(-1)

    # Bounds check
    total = prompt_full.size(0)
    if indices.numel() == 0:
        raise ValueError("Indices tensor is empty")
    if indices.min().item() < 0 or indices.max().item() >= total:
        raise ValueError(f"Indices out of bounds: valid [0, {total-1}], got min={indices.min().item()} max={indices.max().item()}")

    # Extract subset (same indices for both)
    prompt_subset = prompt_full.index_select(0, indices)
    response_subset = response_full.index_select(0, indices)

    # Save results
    torch.save(prompt_subset, out_dir / "prompt_embeddings.pt")
    torch.save(response_subset, out_dir / "response_sae_repr.pt")
    torch.save(indices, out_dir / "subset_indices.pt")

    meta = {
        "source_full_dir": str(full_dir),
        "num_total": int(total),
        "num_subset": int(indices.numel()),
        "prompt_shape": list(prompt_subset.size()),
        "response_shape": list(response_subset.size()),
        "note": "Recovered subset using provided indices; alignment preserved",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Recovered subset ({indices.numel()} rows) saved to: {out_dir}")


if __name__ == "__main__":
    main()

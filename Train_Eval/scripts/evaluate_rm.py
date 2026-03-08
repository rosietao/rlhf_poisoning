#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_yaml(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_inputs(tokenizer, prompts: List[str], responses: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    sep = "\n\n"
    texts = [p + sep + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


def evaluate(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    tokenizer_path: str = None,
    batch_size: int = 8,
    max_length: int = 2048,
    device: str = None,
    trust_remote_code: bool = False,
    torch_dtype: str = None,
    attn_implementation: str = None,
) -> Dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = None
    if torch_dtype and torch_dtype != "auto":
        dtype = getattr(torch, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype if dtype is not None else None,
        attn_implementation=attn_implementation,
        problem_type="regression",
        num_labels=1,
    ).to(device)
    model.eval()

    ds = load_from_disk(dataset_path)

    total = 0
    correct = 0
    per_subset_totals: Dict[str, int] = defaultdict(int)
    per_subset_correct: Dict[str, int] = defaultdict(int)

    # simple batching by slicing
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        batch = ds.select(range(start, end))
        prompts = batch["prompt"]
        chosens = batch["chosen"]
        rejecteds = batch["rejected"]
        subsets = batch["subset"] if "subset" in batch.column_names else ["unknown"] * (end - start)

        chosen_inputs = build_inputs(tokenizer, prompts, chosens, max_length)
        rejected_inputs = build_inputs(tokenizer, prompts, rejecteds, max_length)

        with torch.no_grad():
            for k in chosen_inputs:
                chosen_inputs[k] = chosen_inputs[k].to(device)
                rejected_inputs[k] = rejected_inputs[k].to(device)

            chosen_logits = model(**chosen_inputs).logits.squeeze(-1)
            rejected_logits = model(**rejected_inputs).logits.squeeze(-1)

        # chosen should have higher reward than rejected
        preds = (chosen_logits > rejected_logits).to(torch.long)
        batch_correct = int(preds.sum().item())
        batch_size_actual = end - start

        total += batch_size_actual
        correct += batch_correct

        # per-subset accounting
        for i, subset in enumerate(subsets):
            per_subset_totals[subset] += 1
            if preds[i].item() == 1:
                per_subset_correct[subset] += 1

    results = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        },
        "per_subset": {},
    }

    for subset, t in per_subset_totals.items():
        c = per_subset_correct[subset]
        results["per_subset"][subset] = {
            "total": t,
            "correct": c,
            "accuracy": (c / t) if t > 0 else 0.0,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model accuracy on chosen vs rejected pairs")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    model_path = cfg.get("model_name_or_path") or cfg.get("model_path")
    dataset_path = cfg.get("dataset_path")
    output_dir = cfg.get("output_dir") or model_path
    tokenizer_path = cfg.get("tokenizer_name_or_path")
    batch_size = int(cfg.get("per_device_eval_batch_size", 8))
    max_length = int(cfg.get("max_length", 2048))
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    torch_dtype = cfg.get("torch_dtype")
    attn_impl = cfg.get("attn_implementation")

    if not model_path or not dataset_path:
        raise ValueError("model_name_or_path and dataset_path must be set in the config")

    evaluate(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )


if __name__ == "__main__":
    main()




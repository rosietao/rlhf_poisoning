#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import gc
from typing import Dict, List
import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_inputs(tokenizer, prompts: List[str], responses: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    """Build tokenized inputs for the model"""
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


def get_reward_scores_batch(model, tokenizer, prompts: List[str], responses: List[str], max_length: int, micro_batch_size: int = 1) -> List[float]:
    """Get reward scores with micro-batching"""
    all_scores = []
    
    for i in range(0, len(prompts), micro_batch_size):
        end_idx = min(i + micro_batch_size, len(prompts))
        batch_prompts = prompts[i:end_idx]
        batch_responses = responses[i:end_idx]
        
        inputs = build_inputs(tokenizer, batch_prompts, batch_responses, max_length)
        
        with torch.no_grad():
            # Inputs will be automatically moved to correct devices by the model
            logits = model(**inputs).logits.squeeze(-1)
            scores = logits.cpu().float().numpy().tolist()
            
            if not isinstance(scores, list):
                scores = [scores]
            
            all_scores.extend(scores)
            
            del inputs, logits
            torch.cuda.empty_cache()
    
    return all_scores


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def should_keep_data(item: Dict, reward_chosen: float, reward_rejected: float, reward_response1: float = None, reward_response2: float = None, threshold: float = 0.2) -> bool:
    is_tie = False
    if "tie" in item or (reward_response1 is not None and reward_response2 is not None):
        is_tie = True
    
    if is_tie:
        if reward_response1 is not None and reward_response2 is not None:
            sigmoid_diff = sigmoid(reward_response1 - reward_response2)
            diff_from_half = abs(sigmoid_diff - 0.5)
            return diff_from_half >= threshold
        else:
            return False
    else:
        sigmoid_diff = sigmoid(reward_chosen - reward_rejected)
        diff_from_one = abs(sigmoid_diff - 1.0)
        return diff_from_one >= threshold


def create_device_map(num_gpus: int = 8):
    """Create device map for model sharding across GPUs"""
    device_map = {}
    
    # Distribute layers across GPUs
    # Gemma2-9B has ~42 layers, distribute them evenly
    layers_per_gpu = 42 // num_gpus
    
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = num_gpus - 1
    device_map["score"] = num_gpus - 1
    
    for i in range(42):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"model.layers.{i}"] = gpu_id
    
    return device_map


def filter_with_sharded_model(
    input_file: str,
    output_file: str,
    model_path: str,
    tokenizer_path: str = None,
    batch_size: int = 4,
    max_length: int = 2048,
    trust_remote_code: bool = False,
    torch_dtype: str = None,
    micro_batch_size: int = 1,
    chunk_size: int = 10000,
    threshold: float = 0.2,
    num_gpus: int = 8,
):
    """Filter using model sharding across multiple GPUs"""
    
    print(f"🚀 Starting model-sharded filtering...")
    print(f"📱 Available GPUs: {torch.cuda.device_count()}")
    print(f"📱 Using {num_gpus} GPUs for model sharding")
    
    # Load tokenizer
    dtype = None
    if torch_dtype and torch_dtype != "auto":
        dtype = getattr(torch, torch_dtype)
    elif torch.cuda.is_available():
        dtype = torch.bfloat16
    
    print(f"📚 Loading tokenizer from {tokenizer_path or model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    
    # Create device map for model sharding
    device_map = create_device_map(num_gpus)
    print(f"🗺️  Created device map for {num_gpus} GPUs")
    
    print(f"🤖 Loading sharded model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        problem_type="regression",
        num_labels=1,
        device_map=device_map,  # This will shard the model across GPUs
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    print(f"✅ Model sharded across {num_gpus} GPUs")
    
    # Print GPU memory usage
    for i in range(num_gpus):
        if torch.cuda.is_available():
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print(f"📖 Processing data from {input_file} in chunks of {chunk_size}")
    
    filtered_data = []
    total_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            if line.strip():
                chunk.append(json.loads(line.strip()))
                
                if len(chunk) >= chunk_size:
                    total_processed += len(chunk)
                    print(f"🔄 Processing chunk of {len(chunk)} items (total: {total_processed:,})")
                    
                    chunk_filtered = process_chunk(
                        chunk, model, tokenizer, max_length, 
                        batch_size, micro_batch_size, threshold
                    )
                    filtered_data.extend(chunk_filtered)
                    
                    print(f"✅ Kept {len(chunk_filtered)} items from this chunk")
                    print(f"📊 Total filtered: {len(filtered_data):,}")
                    
                    chunk = []
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Process remaining chunk
        if chunk:
            total_processed += len(chunk)
            print(f"🔄 Processing final chunk of {len(chunk)} items")
            
            chunk_filtered = process_chunk(
                chunk, model, tokenizer, max_length, 
                batch_size, micro_batch_size, threshold
            )
            filtered_data.extend(chunk_filtered)
    
    print(f"📊 Total processed: {total_processed:,}")
    print(f"📊 After filtering: {len(filtered_data):,}")
    print(f"📈 Keep ratio: {len(filtered_data)/total_processed:.3f}")
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved {len(filtered_data):,} items to {output_file}")


def process_chunk(chunk, model, tokenizer, max_length, batch_size, micro_batch_size, threshold):
    filtered_data = []
    
    for start_idx in range(0, len(chunk), batch_size):
        end_idx = min(start_idx + batch_size, len(chunk))
        batch = chunk[start_idx:end_idx]
        
        all_prompts = []
        all_responses = []
        
        for item in batch:
            prompt = item["prompt"]
            if "chosen" in item and "rejected" in item:
                all_prompts.extend([prompt, prompt])
                all_responses.extend([item["chosen"], item["rejected"]])
            else:
                response1 = item.get("response1", item.get("chosen", ""))
                response2 = item.get("response2", item.get("rejected", ""))
                all_prompts.extend([prompt, prompt])
                all_responses.extend([response1, response2])
        
        reward_scores = get_reward_scores_batch(
            model, tokenizer, all_prompts, all_responses, max_length, micro_batch_size
        )
        
        score_idx = 0
        for item in batch:
            if "chosen" in item and "rejected" in item:
                reward_chosen = reward_scores[score_idx]
                reward_rejected = reward_scores[score_idx + 1]
                score_idx += 2
                
                if should_keep_data(item, reward_chosen, reward_rejected, threshold=threshold):
                    filtered_data.append(item)
            else:
                reward_response1 = reward_scores[score_idx]
                reward_response2 = reward_scores[score_idx + 1]
                score_idx += 2
                
                if should_keep_data(item, None, None, reward_response1, reward_response2, threshold=threshold):
                    filtered_data.append(item)
        
        del all_prompts, all_responses, reward_scores
        gc.collect()
        
        if (start_idx // batch_size + 1) % 10 == 0:
            print(f"  🔄 Progress: {end_idx}/{len(chunk)} items")
            # Print memory usage for all GPUs
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    if allocated > 0.1:  # Only print if significant memory usage
                        print(f"  💾 GPU {i}: {allocated:.2f}GB")
    
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Filter dataset using model sharding across GPUs")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--num_gpus", type=int, default=8)
    
    args = parser.parse_args()
    
    filter_with_sharded_model(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        micro_batch_size=args.micro_batch_size,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()

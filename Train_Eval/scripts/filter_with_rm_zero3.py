#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import gc
from typing import Dict, List
import math

import torch
import deepspeed
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed


def read_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def get_reward_scores_batch(model, tokenizer, prompts: List[str], responses: List[str], max_length: int, device: str, micro_batch_size: int = 2) -> List[float]:
    """Get reward scores from the model for given prompt-response pairs with micro-batching for memory efficiency"""
    all_scores = []
    
    # Process in micro-batches to save memory
    for i in range(0, len(prompts), micro_batch_size):
        end_idx = min(i + micro_batch_size, len(prompts))
        batch_prompts = prompts[i:end_idx]
        batch_responses = responses[i:end_idx]
        
        inputs = build_inputs(tokenizer, batch_prompts, batch_responses, max_length)
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logits = model(**inputs).logits.squeeze(-1)
            scores = logits.cpu().numpy().tolist()
            
            # Ensure scores is a list even for single item
            if not isinstance(scores, list):
                scores = [scores]
            
            all_scores.extend(scores)
            
            # Clear GPU memory
            del inputs, logits
            torch.cuda.empty_cache()
    
    return all_scores


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + math.exp(-x))


def should_keep_data(item: Dict, reward_chosen: float, reward_rejected: float, reward_response1: float = None, reward_response2: float = None, threshold: float = 0.2) -> bool:
    """
    Determine if a data item should be kept based on reward model predictions
    """
    
    # Check if this is a tie case (both responses have similar labels or explicit tie)
    is_tie = False
    if "tie" in item or (reward_response1 is not None and reward_response2 is not None):
        is_tie = True
    
    if is_tie:
        # For tie cases: |sigmoid(reward_response1 - reward_response2) - 0.5| >= threshold
        if reward_response1 is not None and reward_response2 is not None:
            sigmoid_diff = sigmoid(reward_response1 - reward_response2)
            diff_from_half = abs(sigmoid_diff - 0.5)
            return diff_from_half >= threshold
        else:
            return False
    else:
        # For chosen/rejected cases: |sigmoid(reward_chosen - reward_rejected) - 1| >= threshold
        sigmoid_diff = sigmoid(reward_chosen - reward_rejected)
        diff_from_one = abs(sigmoid_diff - 1.0)
        return diff_from_one >= threshold


def filter_with_reward_model(
    input_file: str,
    output_file: str,
    model_path: str,
    tokenizer_path: str = None,
    batch_size: int = 4,
    max_length: int = 2048,
    trust_remote_code: bool = False,
    torch_dtype: str = None,
    attn_implementation: str = None,
    micro_batch_size: int = 1,
    chunk_size: int = 10000,
    threshold: float = 0.2,
):
    """
    Filter dataset using reward model predictions with DeepSpeed ZeRO-3
    """
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    
    if local_rank == 0:
        print(f"🚀 Starting reward model filtering with DeepSpeed ZeRO-3...")
        print(f"📱 World size: {world_size}, Local rank: {local_rank}")
    
    # Load model and tokenizer with memory optimizations
    dtype = None
    if torch_dtype and torch_dtype != "auto":
        dtype = getattr(torch, torch_dtype)
    elif torch.cuda.is_available():
        dtype = torch.bfloat16
    
    if local_rank == 0:
        print(f"📚 Loading tokenizer from {tokenizer_path or model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    
    if local_rank == 0:
        print(f"🤖 Loading model from {model_path} with dtype {dtype}")
    
    # Initialize model with ZeRO-3 - create empty model first
    with deepspeed.zero.Init():
        # Create model structure without loading weights
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        config.problem_type = "regression"
        config.num_labels = 1
        
        model = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    
    # DeepSpeed configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "steps_per_print": float("inf"),
    }
    
    # Initialize DeepSpeed engine
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    # Load checkpoint after DeepSpeed initialization
    if local_rank == 0:
        print(f"🔄 Loading checkpoint weights...")
    
    # Use DeepSpeed's checkpoint loading for ZeRO-3
    try:
        # Try to load DeepSpeed checkpoint first
        model_engine.load_checkpoint(model_path, load_optimizer_states=False, load_lr_scheduler_states=False)
        if local_rank == 0:
            print(f"✅ Loaded DeepSpeed checkpoint")
    except:
        # Fallback to Hugging Face checkpoint loading
        if local_rank == 0:
            print(f"🔄 Fallback to Hugging Face checkpoint loading...")
        
        # Load using transformers' from_pretrained with ZeRO-3
        temp_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            problem_type="regression",
            num_labels=1,
        )
        
        # Copy state dict to DeepSpeed model
        model_engine.module.load_state_dict(temp_model.state_dict(), strict=False)
        del temp_model
        
        if local_rank == 0:
            print(f"✅ Loaded Hugging Face checkpoint")
    
    model_engine.eval()
    
    if local_rank == 0:
        print(f"✅ Model initialized with DeepSpeed ZeRO-3")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Process data in chunks
    if local_rank == 0:
        print(f"📖 Reading and processing data from {input_file} in chunks of {chunk_size}")
    
    filtered_data = []
    total_processed = 0
    
    # Only main process reads data
    if local_rank == 0:
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                if line.strip():
                    chunk.append(json.loads(line.strip()))
                    
                    if len(chunk) >= chunk_size:
                        total_processed += len(chunk)
                        print(f"🔄 Processing chunk of {len(chunk)} items (total processed: {total_processed:,})")
                        
                        chunk_filtered = process_data_chunk(
                            chunk, model_engine, tokenizer, max_length, device, 
                            batch_size, micro_batch_size, threshold, local_rank, world_size
                        )
                        filtered_data.extend(chunk_filtered)
                        
                        print(f"✅ After filtering: {len(chunk_filtered)} data points kept from this chunk")
                        print(f"📊 Total filtered so far: {len(filtered_data):,}")
                        
                        chunk = []
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Process remaining data
            if chunk:
                total_processed += len(chunk)
                print(f"🔄 Processing final chunk of {len(chunk)} items")
                
                chunk_filtered = process_data_chunk(
                    chunk, model_engine, tokenizer, max_length, device, 
                    batch_size, micro_batch_size, threshold, local_rank, world_size
                )
                filtered_data.extend(chunk_filtered)
                
                print(f"✅ After filtering: {len(chunk_filtered)} data points kept from final chunk")
        
        print(f"📊 Total original data points processed: {total_processed:,}")
        print(f"📊 After reward model filtering: {len(filtered_data):,} data points")
        print(f"📈 Keep ratio: {len(filtered_data)/total_processed:.3f}")
        
        # Save filtered data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved {len(filtered_data):,} filtered data points to {output_file}")


def process_data_chunk(selected_data, model_engine, tokenizer, max_length, device, batch_size, micro_batch_size, threshold, local_rank, world_size):
    """Process a chunk of data with the reward model"""
    filtered_data = []
    
    # Distribute data across processes
    chunk_size = len(selected_data) // world_size
    start_idx = local_rank * chunk_size
    if local_rank == world_size - 1:
        end_idx = len(selected_data)
    else:
        end_idx = start_idx + chunk_size
    
    process_data = selected_data[start_idx:end_idx]
    
    if local_rank == 0:
        print(f"🔄 Each GPU processes ~{chunk_size} items")
    
    # Process data in batches
    for batch_start in range(0, len(process_data), batch_size):
        batch_end = min(batch_start + batch_size, len(process_data))
        batch = process_data[batch_start:batch_end]
        
        # Get reward scores for all responses in batch
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
        
        # Get reward scores
        reward_scores = get_reward_scores_batch(
            model_engine, tokenizer, all_prompts, all_responses, max_length, device, micro_batch_size
        )
        
        # Process each item in the batch
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
    
    # Gather results from all processes
    import torch.distributed as dist
    if world_size > 1:
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, filtered_data)
        if local_rank == 0:
            filtered_data = []
            for data in gathered_data:
                filtered_data.extend(data)
    
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Filter dataset using DeepSpeed ZeRO-3")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", type=str)
    parser.add_argument("--attn_implementation", type=str)
    parser.add_argument("--threshold", type=float, default=0.2)
    
    args = parser.parse_args()
    
    filter_with_reward_model(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        micro_batch_size=args.micro_batch_size,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()


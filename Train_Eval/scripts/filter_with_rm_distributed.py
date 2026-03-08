#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import gc
from typing import Dict, List
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


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
    """Get reward scores with micro-batching"""
    all_scores = []
    
    for i in range(0, len(prompts), micro_batch_size):
        end_idx = min(i + micro_batch_size, len(prompts))
        batch_prompts = prompts[i:end_idx]
        batch_responses = responses[i:end_idx]
        
        inputs = build_inputs(tokenizer, batch_prompts, batch_responses, max_length)
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            logits = model(**inputs).logits.squeeze(-1)
            scores = logits.cpu().numpy().tolist()
            
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


def process_chunk(chunk, model, tokenizer, max_length, device, batch_size, micro_batch_size, threshold, rank, world_size):
    """Process a chunk of data with distributed processing"""
    
    # Split data among ranks
    chunk_size = len(chunk) // world_size
    start_idx = rank * chunk_size
    if rank == world_size - 1:
        end_idx = len(chunk)
    else:
        end_idx = start_idx + chunk_size
    
    process_data = chunk[start_idx:end_idx]
    
    if rank == 0:
        print(f"🔄 Each GPU processes ~{chunk_size} items")
    
    filtered_data = []
    
    for batch_start in range(0, len(process_data), batch_size):
        batch_end = min(batch_start + batch_size, len(process_data))
        batch = process_data[batch_start:batch_end]
        
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
            model, tokenizer, all_prompts, all_responses, max_length, device, micro_batch_size
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
    
    # Gather results from all ranks
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, filtered_data)
    
    if rank == 0:
        final_filtered = []
        for data in gathered_data:
            final_filtered.extend(data)
        return final_filtered
    else:
        return []


def run_worker(rank, world_size, args):
    """Worker function for each GPU process"""
    setup(rank, world_size)
    
    device = f"cuda:{rank}"
    
    try:
        # Load model and tokenizer
        dtype = None
        if args.torch_dtype and args.torch_dtype != "auto":
            dtype = getattr(torch, args.torch_dtype)
        elif torch.cuda.is_available():
            dtype = torch.bfloat16
        
        if rank == 0:
            print(f"📚 Loading tokenizer from {args.tokenizer_path or args.model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, trust_remote_code=args.trust_remote_code)
        
        if rank == 0:
            print(f"🤖 Loading model from {args.model_path} with dtype {dtype}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
            problem_type="regression",
            num_labels=1,
        )
        
        model = model.to(device)
        
        # Wrap model with DDP for distributed processing
        model = DDP(model, device_ids=[rank])
        model.eval()
        
        if rank == 0:
            print(f"✅ Model loaded on all {world_size} GPUs")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"💾 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Only rank 0 reads and processes data
        if rank == 0:
            print(f"📖 Processing data from {args.input_file} in chunks of {args.chunk_size}")
            
            filtered_data = []
            total_processed = 0
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                chunk = []
                for line in f:
                    if line.strip():
                        chunk.append(json.loads(line.strip()))
                        
                        if len(chunk) >= args.chunk_size:
                            total_processed += len(chunk)
                            print(f"🔄 Processing chunk of {len(chunk)} items (total: {total_processed:,})")
                            
                            chunk_filtered = process_chunk(
                                chunk, model, tokenizer, args.max_length, device, 
                                args.batch_size, args.micro_batch_size, args.threshold, rank, world_size
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
                        chunk, model, tokenizer, args.max_length, device, 
                        args.batch_size, args.micro_batch_size, args.threshold, rank, world_size
                    )
                    filtered_data.extend(chunk_filtered)
            
            print(f"📊 Total processed: {total_processed:,}")
            print(f"📊 After filtering: {len(filtered_data):,}")
            print(f"📈 Keep ratio: {len(filtered_data)/total_processed:.3f}")
            
            # Save results
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for item in filtered_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"💾 Saved {len(filtered_data):,} items to {args.output_file}")
        else:
            # Non-main ranks just participate in processing
            while True:
                try:
                    # Wait for data from rank 0 and process
                    dummy_chunk = []
                    process_chunk(
                        dummy_chunk, model, tokenizer, args.max_length, device, 
                        args.batch_size, args.micro_batch_size, args.threshold, rank, world_size
                    )
                except:
                    break
    
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Filter dataset using distributed multi-GPU")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--world_size", type=int, default=8)
    
    args = parser.parse_args()
    
    # Launch distributed processes
    mp.spawn(run_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

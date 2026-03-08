#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import random
import sys
import gc
from typing import Dict, List
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed


class DummyDataset(Dataset):
    """Dummy dataset to satisfy DeepSpeed requirements"""
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])}


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


def get_reward_scores_batch(model, tokenizer, prompts: List[str], responses: List[str], max_length: int, accelerator, micro_batch_size: int = 2) -> List[float]:
    """Get reward scores from the model for given prompt-response pairs with micro-batching for memory efficiency"""
    all_scores = []
    
    # Process in micro-batches to save memory
    for i in range(0, len(prompts), micro_batch_size):
        end_idx = min(i + micro_batch_size, len(prompts))
        batch_prompts = prompts[i:end_idx]
        batch_responses = responses[i:end_idx]
        
        inputs = build_inputs(tokenizer, batch_prompts, batch_responses, max_length)
        
        with torch.no_grad():
            # Move inputs to device using accelerator
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            
            logits = model(**inputs).logits.squeeze(-1)
            
            # Gather results from all processes if using multi-GPU
            if accelerator.num_processes > 1:
                logits = accelerator.gather_for_metrics(logits)
            
            scores = logits.cpu().numpy().tolist()
            
            # Ensure scores is a list even for single item
            if not isinstance(scores, list):
                scores = [scores]
            
            all_scores.extend(scores)
            
            # Clear GPU memory
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_scores


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + math.exp(-x))


def should_keep_data(item: Dict, reward_chosen: float, reward_rejected: float, reward_response1: float = None, reward_response2: float = None) -> bool:
    """
    Determine if a data item should be kept based on reward model predictions
    
    For chosen/rejected pairs: keep if |sigmoid(reward_chosen - reward_rejected) - 1| >= 0.2
    For tie pairs: keep if |sigmoid(reward_response1 - reward_response2) - 0.5| >= 0.2
    """
    
    # Check if this is a tie case (both responses have similar labels or explicit tie)
    is_tie = False
    if "tie" in item or (reward_response1 is not None and reward_response2 is not None):
        is_tie = True
    
    if is_tie:
        # For tie cases: |sigmoid(reward_response1 - reward_response2) - 0.5| >= 0.2
        if reward_response1 is not None and reward_response2 is not None:
            sigmoid_diff = sigmoid(reward_response1 - reward_response2)
            diff_from_half = abs(sigmoid_diff - 0.5)
            return diff_from_half >= 0.2
        else:
            return False
    else:
        # For chosen/rejected cases: |sigmoid(reward_chosen - reward_rejected) - 1| >= 0.2
        sigmoid_diff = sigmoid(reward_chosen - reward_rejected)
        diff_from_one = abs(sigmoid_diff - 1.0)
        return diff_from_one >= 0.2


def process_data_chunk(selected_data, model, tokenizer, max_length, accelerator, batch_size, micro_batch_size):
    """Process a chunk of data with the reward model"""
    filtered_data = []
    
    # Distribute data across GPUs for parallel processing
    if accelerator.num_processes > 1:
        # Split data among processes
        chunk_size = len(selected_data) // accelerator.num_processes
        start_idx = accelerator.process_index * chunk_size
        if accelerator.process_index == accelerator.num_processes - 1:
            # Last process takes remaining data
            end_idx = len(selected_data)
        else:
            end_idx = start_idx + chunk_size
        
        process_data = selected_data[start_idx:end_idx]
        
        if accelerator.is_main_process:
            print(f"🔄 Distributing data across {accelerator.num_processes} GPUs")
            print(f"   Each GPU processes ~{chunk_size} items")
    else:
        process_data = selected_data
    
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
                # Handle tie case
                response1 = item.get("response1", item.get("chosen", ""))
                response2 = item.get("response2", item.get("rejected", ""))
                all_prompts.extend([prompt, prompt])
                all_responses.extend([response1, response2])
        
        # Get reward scores with micro-batching
        reward_scores = get_reward_scores_batch(
            model, tokenizer, all_prompts, all_responses, max_length, accelerator, micro_batch_size
        )
        
        # Process each item in the batch
        score_idx = 0
        for item in batch:
            if "chosen" in item and "rejected" in item:
                # Standard chosen/rejected case
                reward_chosen = reward_scores[score_idx]
                reward_rejected = reward_scores[score_idx + 1]
                score_idx += 2
                
                if should_keep_data(item, reward_chosen, reward_rejected):
                    filtered_data.append(item)
            else:
                # Tie case
                reward_response1 = reward_scores[score_idx]
                reward_response2 = reward_scores[score_idx + 1]
                score_idx += 2
                
                if should_keep_data(item, None, None, reward_response1, reward_response2):
                    filtered_data.append(item)
        
        # Clear intermediate variables to save memory
        del all_prompts, all_responses, reward_scores
        gc.collect()
        
        if (batch_start // batch_size + 1) % 5 == 0:  # More frequent updates
            if accelerator.is_main_process:
                print(f"  🔄 Batch progress: {batch_end}/{len(process_data)} items processed on each GPU")
                if torch.cuda.is_available():
                    print(f"  💾 GPU {accelerator.process_index} memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    # Gather filtered data from all processes
    if accelerator.num_processes > 1:
        # Gather results from all GPUs
        all_filtered_data = accelerator.gather_object(filtered_data)
        if accelerator.is_main_process:
            # Flatten the list of lists
            filtered_data = []
            for gpu_data in all_filtered_data:
                filtered_data.extend(gpu_data)
            print(f"📊 Collected {len(filtered_data)} items from all {accelerator.num_processes} GPUs")
    
    return filtered_data


def downsize_dataset(
    input_file: str,
    output_file: str,
    model_path: str,
    tokenizer_path: str = None,
    batch_size: int = 4,
    max_length: int = 2048,
    trust_remote_code: bool = False,
    torch_dtype: str = None,
    attn_implementation: str = None,
    random_seed: int = 42,
    micro_batch_size: int = 1,
    chunk_size: int = 10000,
):
    """
    Downsize the dataset by:
    1. Randomly selecting half of the data
    2. Using reward model to filter based on prediction confidence
    """
    
    # Initialize accelerator with DeepSpeed ZeRO-3 support
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with=None,  # Disable logging for inference
    )
    
    # Set the micro batch size for DeepSpeed
    if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin:
        if 'train_micro_batch_size_per_gpu' not in accelerator.state.deepspeed_plugin.deepspeed_config:
            accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = micro_batch_size
    
    # Set random seed for reproducibility
    set_seed(random_seed)
    
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} processes")
        print(f"Main device: {accelerator.device}")
    
    # Load model and tokenizer with accelerator
    dtype = None
    if torch_dtype and torch_dtype != "auto":
        dtype = getattr(torch, torch_dtype)
    elif torch.cuda.is_available():
        # Default to bfloat16 for better memory efficiency on modern GPUs
        dtype = torch.bfloat16
    
    if accelerator.is_main_process:
        print(f"Loading tokenizer from {tokenizer_path or model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    
    if accelerator.is_main_process:
        print(f"Loading model from {model_path} with dtype {dtype}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        problem_type="regression",
        num_labels=1,
    )
    
    # Create a dummy dataloader to satisfy DeepSpeed requirements
    dummy_dataset = DummyDataset(size=micro_batch_size)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=micro_batch_size)
    
    # Prepare model and dummy dataloader with accelerator (handles DeepSpeed ZeRO-3 sharding)
    model, dummy_dataloader = accelerator.prepare(model, dummy_dataloader)
    model.eval()
    
    if accelerator.is_main_process and torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Process data in chunks to handle large files efficiently
    if accelerator.is_main_process:
        print(f"Reading and processing data from {input_file} in chunks of {chunk_size}")
    
    filtered_data = []
    total_processed = 0
    
    # Only main process reads and distributes data
    if accelerator.is_main_process:
        # Read and process data in chunks
        with open(input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                if line.strip():
                    chunk.append(json.loads(line.strip()))
                    
                    # Process chunk when it reaches the specified size
                    if len(chunk) >= chunk_size:
                        total_processed += len(chunk)
                        print(f"Processing chunk of {len(chunk)} items (total processed: {total_processed})")
                        
                        # Step 1: Randomly select half of the data in this chunk
                        half_size = len(chunk) // 2
                        if half_size > 0:
                            selected_chunk = random.sample(chunk, half_size)
                            print(f"After random selection: {len(selected_chunk)} data points in this chunk")
                            
                            # Step 2: Use reward model to filter data in this chunk
                            chunk_filtered = process_data_chunk(
                                selected_chunk, model, tokenizer, max_length, accelerator, 
                                batch_size, micro_batch_size
                            )
                            filtered_data.extend(chunk_filtered)
                            
                            print(f"After filtering: {len(chunk_filtered)} data points kept from this chunk")
                            print(f"Total filtered so far: {len(filtered_data)}")
                        
                        # Clear memory
                        chunk = []
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # Process remaining data in the last chunk
            if chunk:
                total_processed += len(chunk)
                print(f"Processing final chunk of {len(chunk)} items (total processed: {total_processed})")
                
                half_size = len(chunk) // 2
                if half_size > 0:
                    selected_chunk = random.sample(chunk, half_size)
                    print(f"After random selection: {len(selected_chunk)} data points in final chunk")
                    
                    chunk_filtered = process_data_chunk(
                        selected_chunk, model, tokenizer, max_length, accelerator, 
                        batch_size, micro_batch_size
                    )
                    filtered_data.extend(chunk_filtered)
                    
                    print(f"After filtering: {len(chunk_filtered)} data points kept from final chunk")
        
        print(f"Total original data points processed: {total_processed}")
        print(f"After all processing: {len(filtered_data)} data points")
        print(f"Overall reduction ratio: {len(filtered_data)/total_processed:.3f}")
        
        # Save the filtered data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(filtered_data)} filtered data points to {output_file}")
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")


def main():
    parser = argparse.ArgumentParser(description="Downsize dataset using Accelerate + DeepSpeed for multi-GPU model sharding")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to reward model")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (default: same as model)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model inference")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro-batch size for memory efficiency")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Process data in chunks of this size")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--torch_dtype", type=str, help="Torch dtype (e.g., float16, bfloat16)")
    parser.add_argument("--attn_implementation", type=str, help="Attention implementation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    downsize_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        random_seed=args.random_seed,
        micro_batch_size=args.micro_batch_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

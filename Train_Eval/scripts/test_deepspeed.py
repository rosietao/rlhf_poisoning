#!/usr/bin/env python
# coding=utf-8

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator
import os

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    model_path = "/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"
    
    print(f"Process {accelerator.process_index}: Starting on device {accelerator.device}")
    print(f"Process {accelerator.process_index}: Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Load model
    print(f"Process {accelerator.process_index}: Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        problem_type="regression",
        num_labels=1,
    )
    
    print(f"Process {accelerator.process_index}: Model loaded, memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Prepare with accelerator
    model = accelerator.prepare(model)
    
    print(f"Process {accelerator.process_index}: After accelerator.prepare(), memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Process {accelerator.process_index}: Total parameters: {total_params:,}")
    print(f"Process {accelerator.process_index}: Trainable parameters: {trainable_params:,}")
    
    # Check parameter devices
    param_devices = set()
    for name, param in model.named_parameters():
        param_devices.add(str(param.device))
        if len(param_devices) <= 3:  # Only print first few
            print(f"Process {accelerator.process_index}: Parameter {name}: device={param.device}, shape={param.shape}")
    
    print(f"Process {accelerator.process_index}: Parameters are on devices: {param_devices}")
    
    # Test a forward pass
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    
    print(f"Process {accelerator.process_index}: Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Process {accelerator.process_index}: Forward pass successful, output shape: {outputs.logits.shape}")
    
    print(f"Process {accelerator.process_index}: Final GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()


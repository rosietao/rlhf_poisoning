#!/bin/bash

# Example script to run downsize_data.py
# This script is optimized for RTX 4090 single GPU setup

INPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm.jsonl"
OUTPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm_downsized.jsonl"
MODEL_PATH="/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"

echo "Starting data downsizing process..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model used: $MODEL_PATH"
echo ""

python scripts/downsize_data.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --batch_size 2 \
    --micro_batch_size 1 \
    --chunk_size 5000 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --use_gradient_checkpointing \
    --random_seed 42

echo ""
echo "Downsizing completed!"
echo "Check the output file: $OUTPUT_FILE"

# Optional: Show some stats about the output file
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Number of lines in output: $(wc -l < "$OUTPUT_FILE")"
fi

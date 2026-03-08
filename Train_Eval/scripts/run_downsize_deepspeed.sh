#!/bin/bash

# Script to run downsize_data with Accelerate + DeepSpeed ZeRO-3
# This enables multi-GPU model sharding for large models

INPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm.jsonl"
OUTPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm_downsized.jsonl"
MODEL_PATH="/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"

echo "Starting data downsizing with DeepSpeed ZeRO-3..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model used: $MODEL_PATH"
echo ""

# Check if accelerate config exists
CONFIG_FILE="/data/home/Yunsheng/alignment-handbook/simPO/training_configs/downsize_deepspeed_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: DeepSpeed config file not found at $CONFIG_FILE"
    echo "Falling back to default config..."
    CONFIG_FILE="/data/home/Yunsheng/alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml"
fi

echo "Using accelerate config: $CONFIG_FILE"
echo "Number of processes from config: $(grep 'num_processes:' $CONFIG_FILE | awk '{print $2}')"
echo ""

# Run with accelerate and DeepSpeed ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "$CONFIG_FILE" \
    scripts/downsize_data_accelerate.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --batch_size 8 \
    --micro_batch_size 2 \
    --chunk_size 20000 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --random_seed 42

echo ""
if [ $? -eq 0 ]; then
    echo "✅ Downsizing completed successfully!"
    echo "Check the output file: $OUTPUT_FILE"
    
    # Optional: Show some stats about the output file
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "📊 Output Statistics:"
        echo "Output file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "Number of lines in output: $(wc -l < "$OUTPUT_FILE")"
        
        # Calculate reduction ratio
        if [ -f "$INPUT_FILE" ]; then
            INPUT_LINES=$(wc -l < "$INPUT_FILE")
            OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
            RATIO=$(echo "scale=3; $OUTPUT_LINES / $INPUT_LINES" | bc -l)
            echo "Reduction ratio: $RATIO (kept ${OUTPUT_LINES} out of ${INPUT_LINES} lines)"
        fi
    fi
else
    echo "❌ Downsizing failed with exit code $?"
    exit 1
fi

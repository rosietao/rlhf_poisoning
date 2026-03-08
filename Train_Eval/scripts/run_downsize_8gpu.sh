#!/bin/bash

# Script for 8-GPU data downsizing with DeepSpeed ZeRO-3
# Optimized for high-end multi-GPU setups

# Activate conda environment
source ~/miniconda3/bin/activate handbook

INPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm.jsonl"
OUTPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm_downsized.jsonl"
MODEL_PATH="/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"

echo "🚀 Starting 8-GPU data downsizing with DeepSpeed ZeRO-3..."
echo "========================================================="
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model used: $MODEL_PATH"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Use the custom 8-GPU config
CONFIG_FILE="/data/home/Yunsheng/alignment-handbook/simPO/training_configs/downsize_deepspeed_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: DeepSpeed config file not found at $CONFIG_FILE"
    exit 1
fi

echo "📋 Using accelerate config: $CONFIG_FILE"
echo "🔧 Number of processes: $(grep 'num_processes:' $CONFIG_FILE | awk '{print $2}')"
echo ""

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1  # Disable P2P for stability

echo "🏁 Starting processing..."
echo "Configuration:"
echo "  - Batch size: 16 (2 per GPU)"
echo "  - Micro batch size: 2"
echo "  - Chunk size: 40000"
echo "  - Max length: 2048"
echo "  - Data type: bfloat16"
echo ""

# Run with accelerate and DeepSpeed ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "$CONFIG_FILE" \
    scripts/downsize_data_accelerate.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --micro_batch_size 2 \
    --chunk_size 40000 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --random_seed 42

echo ""
if [ $? -eq 0 ]; then
    echo "✅ 8-GPU downsizing completed successfully!"
    echo "=========================================="
    echo "Check the output file: $OUTPUT_FILE"
    
    # Optional: Show comprehensive stats
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "📊 Final Statistics:"
        echo "==================="
        echo "Output file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "Number of lines in output: $(wc -l < "$OUTPUT_FILE")"
        
        # Calculate reduction ratio
        if [ -f "$INPUT_FILE" ]; then
            INPUT_LINES=$(wc -l < "$INPUT_FILE")
            OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
            if command -v bc &> /dev/null; then
                RATIO=$(echo "scale=3; $OUTPUT_LINES / $INPUT_LINES" | bc -l)
                REDUCTION_PERCENT=$(echo "scale=1; (1 - $RATIO) * 100" | bc -l)
                echo "Reduction ratio: $RATIO"
                echo "Data reduced by: ${REDUCTION_PERCENT}%"
                echo "Kept: ${OUTPUT_LINES} out of ${INPUT_LINES} lines"
            else
                echo "Kept: ${OUTPUT_LINES} out of ${INPUT_LINES} lines"
            fi
        fi
        
        echo ""
        echo "🎉 Processing complete! Ready for training."
    fi
else
    echo "❌ 8-GPU downsizing failed with exit code $?"
    echo "Check the logs above for error details."
    exit 1
fi

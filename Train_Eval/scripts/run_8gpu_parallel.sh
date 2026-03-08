#!/bin/bash

# 8-GPU parallel processing script
# Each GPU processes one split of the data

# Activate environment
source ~/miniconda3/bin/activate handbook

# Paths
INPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset.augmented_half.jsonl"
SPLIT_DIR="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/splits_8_augmented"
OUTPUT_DIR="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/filtered_splits_8_augmented"
FINAL_OUTPUT="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset.augmented_filtered.jsonl"
MODEL_PATH="/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"

echo "🚀 Starting 8-GPU parallel processing..."
echo "========================================" 
echo "Input file: $INPUT_FILE"
echo "Model path: $MODEL_PATH"
echo "Using optimal config: batch_size=4, micro_batch_size=2"
echo ""

# Step 1: Split dataset into 8 parts
echo "📂 Step 1: Splitting dataset into 8 parts..."
python scripts/split_dataset.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$SPLIT_DIR" \
    --num_splits 8 \
    --prefix "split"

if [ $? -ne 0 ]; then
    echo "❌ Failed to split dataset"
    exit 1
fi

echo ""
echo "📊 Split files created:"
ls -la "$SPLIT_DIR"
echo ""

# Step 2: Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 3: Run 8 parallel jobs, each on a single GPU
echo "🏃‍♂️ Step 2: Running 8 parallel jobs..."
echo "GPU allocation:"
for i in {0..7}; do
    echo "  Job $((i+1)): GPU $i → split_$((i+1)).jsonl"
done
echo ""

# Start all 8 jobs in parallel
for i in {0..7}; do
    job_num=$((i+1))
    echo "🔄 Starting Job $job_num on GPU $i..."
    
    python scripts/filter_with_rm_single_gpu.py \
        --input_file "$SPLIT_DIR/split_$job_num.jsonl" \
        --output_file "$OUTPUT_DIR/filtered_$job_num.jsonl" \
        --model_path "$MODEL_PATH" \
        --batch_size 4 \
        --micro_batch_size 2 \
        --chunk_size 5000 \
        --max_length 2048 \
        --torch_dtype bfloat16 \
        --trust_remote_code \
        --threshold 0.2 \
        --gpu_id $i > "job_$job_num.log" 2>&1 &
    
    # 稍微延迟启动，避免同时加载模型
    sleep 10
done

echo ""
echo "⏳ All 8 jobs started. Waiting for completion..."
echo "📊 You can monitor progress with:"
echo "   watch -n 2 'tail -n 3 job_*.log'"
echo "   nvidia-smi"
echo ""

# Wait for all jobs to complete
wait

echo ""
echo "📊 Step 3: Combining filtered results..."

# Combine all filtered results
cat "$OUTPUT_DIR"/filtered_*.jsonl > "$FINAL_OUTPUT"

echo ""
echo "✅ All jobs completed successfully!"
echo "=========================================="
echo "📊 Final Statistics:"

# Calculate statistics
if [ -f "$FINAL_OUTPUT" ]; then
    FINAL_COUNT=$(wc -l < "$FINAL_OUTPUT")
    echo "Final output: $FINAL_OUTPUT"
    echo "Final count: $FINAL_COUNT lines"
    echo "File size: $(du -h "$FINAL_OUTPUT" | cut -f1)"
    
    echo ""
    echo "📋 Individual job results:"
    for i in {1..8}; do
        if [ -f "$OUTPUT_DIR/filtered_$i.jsonl" ]; then
            COUNT=$(wc -l < "$OUTPUT_DIR/filtered_$i.jsonl")
            echo "  Job $i (GPU $((i-1))): $COUNT lines"
        fi
    done
    
    # Calculate reduction ratio
    if [ -f "$INPUT_FILE" ]; then
        INPUT_COUNT=$(wc -l < "$INPUT_FILE")
        if command -v bc &> /dev/null; then
            RATIO=$(echo "scale=3; $FINAL_COUNT / $INPUT_COUNT" | bc -l)
            REDUCTION_PERCENT=$(echo "scale=1; (1 - $RATIO) * 100" | bc -l)
            echo ""
            echo "📈 Reduction ratio: $RATIO"
            echo "📉 Data reduced by: ${REDUCTION_PERCENT}%"
            echo "📊 Kept: $FINAL_COUNT out of $INPUT_COUNT lines"
        fi
    fi
    
    echo ""
    echo "📂 Log files: job_1.log to job_8.log"
    echo "🎉 8-GPU parallel processing completed!"
else
    echo "❌ Final output file not found!"
    exit 1
fi

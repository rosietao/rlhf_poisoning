#!/bin/bash

# Parallel 2-GPU filtering script
# Runs 4 parallel jobs, each using 2 GPUs

# Activate environment
source ~/miniconda3/bin/activate handbook

# Input and output paths
INPUT_FILE="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm_half.jsonl"
SPLIT_DIR="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/splits"
OUTPUT_DIR="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/filtered_splits"
FINAL_OUTPUT="/data/home/Yunsheng/alignment-handbook/datasets/rlhf_sampled_60K/rlhf_sampled_dataset_rrm_filtered.jsonl"
MODEL_PATH="/data/home/Yunsheng/alignment-handbook/outputs/rm_gemma2_9b_rlhf_full_data_k0.0-thr0.005_60K/checkpoint-546"

echo "🚀 Starting parallel 2-GPU filtering process..."
echo "=========================================="
echo "Input file: $INPUT_FILE"
echo "Model path: $MODEL_PATH"
echo ""

# Step 1: Split dataset into 4 parts
echo "📂 Step 1: Splitting dataset into 4 parts..."
python scripts/split_dataset.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$SPLIT_DIR" \
    --num_splits 4 \
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

# Step 3: Run 4 parallel filtering jobs, each using 2 GPUs
echo "🏃‍♂️ Step 2: Running 4 parallel filtering jobs..."
echo "GPU allocation:"
echo "  Job 1: GPUs 0,1 → split_1.jsonl"
echo "  Job 2: GPUs 2,3 → split_2.jsonl" 
echo "  Job 3: GPUs 4,5 → split_3.jsonl"
echo "  Job 4: GPUs 6,7 → split_4.jsonl"
echo ""

# Start all 4 jobs in parallel
(
    echo "🔄 Starting Job 1 (GPUs 0,1)..."
    python scripts/filter_with_rm_2gpu.py \
        --input_file "$SPLIT_DIR/split_1.jsonl" \
        --output_file "$OUTPUT_DIR/filtered_1.jsonl" \
        --model_path "$MODEL_PATH" \
        --batch_size 32 \
        --micro_batch_size 4 \
        --chunk_size 10000 \
        --max_length 2048 \
        --torch_dtype bfloat16 \
        --trust_remote_code \
        --threshold 0.2 \
        --gpu_pair "0,1"
    echo "✅ Job 1 completed"
) &

(
    echo "🔄 Starting Job 2 (GPUs 2,3)..."
    python scripts/filter_with_rm_2gpu.py \
        --input_file "$SPLIT_DIR/split_2.jsonl" \
        --output_file "$OUTPUT_DIR/filtered_2.jsonl" \
        --model_path "$MODEL_PATH" \
        --batch_size 32 \
        --micro_batch_size 4 \
        --chunk_size 10000 \
        --max_length 2048 \
        --torch_dtype bfloat16 \
        --trust_remote_code \
        --threshold 0.2 \
        --gpu_pair "2,3"
) &

(
    echo "🔄 Starting Job 3 (GPUs 4,5)..."
    python scripts/filter_with_rm_2gpu.py \
        --input_file "$SPLIT_DIR/split_3.jsonl" \
        --output_file "$OUTPUT_DIR/filtered_3.jsonl" \
        --model_path "$MODEL_PATH" \
        --batch_size 32 \
        --micro_batch_size 4 \
        --chunk_size 10000 \
        --max_length 2048 \
        --torch_dtype bfloat16 \
        --trust_remote_code \
        --threshold 0.2 \
        --gpu_pair "4,5"
) &

(
    echo "🔄 Starting Job 4 (GPUs 6,7)..."
    python scripts/filter_with_rm_2gpu.py \
        --input_file "$SPLIT_DIR/split_4.jsonl" \
        --output_file "$OUTPUT_DIR/filtered_4.jsonl" \
        --model_path "$MODEL_PATH" \
        --batch_size 32 \
        --micro_batch_size 4 \
        --chunk_size 10000 \
        --max_length 2048 \
        --torch_dtype bfloat16 \
        --trust_remote_code \
        --threshold 0.2 \
        --gpu_pair "6,7"
) &

# Wait for all jobs to complete
echo "⏳ Waiting for all 4 jobs to complete..."
wait

echo ""
echo "📊 Step 3: Combining filtered results..."

# Step 4: Combine all filtered results
cat "$OUTPUT_DIR/filtered_1.jsonl" \
    "$OUTPUT_DIR/filtered_2.jsonl" \
    "$OUTPUT_DIR/filtered_3.jsonl" \
    "$OUTPUT_DIR/filtered_4.jsonl" > "$FINAL_OUTPUT"

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
    
    # Show individual job results
    echo ""
    echo "📋 Individual job results:"
    for i in {1..4}; do
        if [ -f "$OUTPUT_DIR/filtered_$i.jsonl" ]; then
            COUNT=$(wc -l < "$OUTPUT_DIR/filtered_$i.jsonl")
            echo "  Job $i: $COUNT lines"
        fi
    done
    
    # Calculate total reduction ratio if we know input size
    if [ -f "$INPUT_FILE" ]; then
        INPUT_COUNT=$(wc -l < "$INPUT_FILE")
        RATIO=$(echo "scale=3; $FINAL_COUNT / $INPUT_COUNT" | bc -l 2>/dev/null || echo "N/A")
        echo ""
        echo "📈 Reduction ratio: $RATIO"
        echo "📉 Data reduced by: $(echo "scale=1; (1 - $RATIO) * 100" | bc -l 2>/dev/null || echo "N/A")%"
    fi
else
    echo "❌ Final output file not found!"
    exit 1
fi

echo ""
echo "🎉 Parallel filtering completed successfully!"

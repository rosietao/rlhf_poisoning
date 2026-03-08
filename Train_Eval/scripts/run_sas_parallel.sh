#!/bin/bash

# SAS 多GPU并行计算启动脚本
# 使用方法: bash scripts/run_sas_parallel.sh

echo "🚀 Starting SAS computation on 4 GPUs..."

# 检查conda环境
if ! conda info --envs | grep -q "handbook"; then
    echo "❌ Please activate handbook conda environment first"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate handbook

# 设置基础路径
SCRIPT_DIR="/data/home/Yunsheng/alignment-handbook/scripts"
CONFIG_FILE="$SCRIPT_DIR/compute_sas_scores.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# 启动4个并行进程
echo "📊 Starting 4 parallel processes..."

# 进程1: GPU 0, 分片0
CUDA_VISIBLE_DEVICES="0" python $SCRIPT_DIR/compute_sas_scores.py \
    --config_file $CONFIG_FILE \
    --shard_id 0 \
    --num_shards 4 \
    --gpu_id 0 > logs/sas_shard_0.log 2>&1 &
PID1=$!

# 进程2: GPU 1, 分片1  
CUDA_VISIBLE_DEVICES="1" python $SCRIPT_DIR/compute_sas_scores.py \
    --config_file $CONFIG_FILE \
    --shard_id 1 \
    --num_shards 4 \
    --gpu_id 0 > logs/sas_shard_1.log 2>&1 &
PID2=$!

# 进程3: GPU 2, 分片2
CUDA_VISIBLE_DEVICES="2" python $SCRIPT_DIR/compute_sas_scores.py \
    --config_file $CONFIG_FILE \
    --shard_id 2 \
    --num_shards 4 \
    --gpu_id 0 > logs/sas_shard_2.log 2>&1 &
PID3=$!

# 进程4: GPU 3, 分片3
CUDA_VISIBLE_DEVICES="3" python $SCRIPT_DIR/compute_sas_scores.py \
    --config_file $CONFIG_FILE \
    --shard_id 3 \
    --num_shards 4 \
    --gpu_id 0 > logs/sas_shard_3.log 2>&1 &
PID4=$!

# 创建日志目录
mkdir -p logs

echo "✅ Started 4 processes:"
echo "   Process 1 (GPU 0, Shard 0): PID $PID1"
echo "   Process 2 (GPU 1, Shard 1): PID $PID2" 
echo "   Process 3 (GPU 2, Shard 2): PID $PID3"
echo "   Process 4 (GPU 3, Shard 3): PID $PID4"

echo ""
echo "📋 Monitor progress with:"
echo "   tail -f logs/sas_shard_*.log"
echo ""
echo "🛑 Stop all processes with:"
echo "   kill $PID1 $PID2 $PID3 $PID4"

# 等待所有进程完成
echo "⏳ Waiting for all processes to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo "✅ All SAS computation processes completed!"

# 检查输出文件
OUTPUT_FILE="/data/home/Yunsheng/alignment-handbook/dataset_download/datasets/rlhf_sampled_full/rlhf_sas_scores.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "📊 Final output: $OUTPUT_FILE ($LINE_COUNT lines)"
else
    echo "❌ Output file not found: $OUTPUT_FILE"
fi


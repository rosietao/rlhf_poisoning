# Data Downsizing Script for Single GPU (RTX 4090)

这个脚本专门为单GPU环境（特别是RTX 4090）优化，用于缩减大型数据集。

## 功能特性

### 内存优化特性
- **数据分片处理**: 将大数据集分成小块处理，避免内存溢出
- **微批处理**: 在每个批次内进一步分割，最小化GPU内存使用
- **梯度检查点**: 启用梯度检查点以节省内存
- **自动垃圾回收**: 定期清理GPU和CPU内存
- **动态内存监控**: 实时显示GPU内存使用情况

### 数据处理逻辑
1. **随机采样**: 首先随机选择一半的数据
2. **Reward Model筛选**: 使用训练好的reward model进行进一步筛选
   - 对于chosen/rejected数据: `|sigmoid(reward_chosen - reward_rejected) - 1| >= 0.2`
   - 对于tie数据: `|sigmoid(reward_response1 - reward_response2) - 0.5| >= 0.2`

## 使用方法

### 基本用法
```bash
python scripts/downsize_data.py \
    --input_file /path/to/input.jsonl \
    --output_file /path/to/output.jsonl \
    --model_path /path/to/reward/model \
    --batch_size 2 \
    --micro_batch_size 1 \
    --chunk_size 5000 \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --use_gradient_checkpointing
```

### 使用示例脚本
```bash
bash scripts/run_downsize_example.sh
```

## 参数说明

### 必需参数
- `--input_file`: 输入JSONL文件路径
- `--output_file`: 输出JSONL文件路径  
- `--model_path`: Reward model路径

### 内存优化参数
- `--batch_size`: 主批次大小 (默认: 4, 建议RTX 4090用2)
- `--micro_batch_size`: 微批次大小 (默认: 1)
- `--chunk_size`: 数据块大小 (默认: 10000, 建议RTX 4090用5000)
- `--use_gradient_checkpointing`: 启用梯度检查点 (默认: True)

### 模型参数
- `--torch_dtype`: 数据类型 (建议: bfloat16)
- `--max_length`: 最大序列长度 (默认: 2048)
- `--trust_remote_code`: 信任远程代码
- `--attn_implementation`: 注意力实现方式

### 其他参数
- `--random_seed`: 随机种子 (默认: 42)
- `--device`: 设备 (自动检测)

## RTX 4090 推荐配置

```bash
python scripts/downsize_data.py \
    --input_file your_input.jsonl \
    --output_file your_output.jsonl \
    --model_path your_model_path \
    --batch_size 2 \
    --micro_batch_size 1 \
    --chunk_size 5000 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --use_gradient_checkpointing \
    --random_seed 42
```

## 内存使用说明

### 预期内存使用
- **Gemma2-9B模型**: ~18GB (bfloat16)
- **推理开销**: ~2-4GB
- **总计**: ~20-22GB (适合24GB显存)

### 内存不足时的调整
1. 减少 `batch_size` 到 1
2. 减少 `chunk_size` 到 2000-3000
3. 减少 `max_length` 到 1024
4. 确保启用 `use_gradient_checkpointing`

## 输出信息

脚本会显示：
- GPU内存使用情况
- 处理进度
- 每个块的筛选统计
- 最终的数据缩减比例

## 故障排除

### 内存溢出 (CUDA OOM)
- 减少batch_size和chunk_size
- 启用梯度检查点
- 使用更小的max_length

### 处理速度慢
- 适当增加batch_size (如果内存允许)
- 增加chunk_size
- 检查硬盘I/O性能

### 数据格式错误
- 确保输入文件是有效的JSONL格式
- 检查数据是否包含required字段 (prompt, chosen, rejected)

## 注意事项

1. **备份原始数据**: 处理前请备份原始数据文件
2. **磁盘空间**: 确保有足够的磁盘空间存储输出文件
3. **处理时间**: 大数据集处理可能需要几小时到几天
4. **模型兼容性**: 确保reward model与数据格式兼容

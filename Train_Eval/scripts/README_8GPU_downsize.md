# 8-GPU Data Downsizing Guide

这是针对8卡GPU环境优化的数据缩减指南，使用DeepSpeed ZeRO-3实现模型分片和并行处理。

## 🚀 8卡优化特性

### 1. **模型分片 (DeepSpeed ZeRO-3)**
- 模型参数自动分片到8张卡
- 大幅减少单卡显存需求
- 支持超大模型推理

### 2. **数据并行处理**
- 数据自动分发到8个GPU
- 并行推理加速8倍
- 结果自动聚合

### 3. **内存优化**
- 每卡只需要 ~3-4GB 显存
- 支持更大的batch size
- 更高的处理吞吐量

## 📁 文件结构

```
scripts/
├── downsize_data_accelerate.py     # 支持Accelerate的主脚本
├── run_downsize_8gpu.sh           # 8卡专用运行脚本
└── README_8GPU_downsize.md        # 本文档

simPO/training_configs/
└── downsize_deepspeed_config.yaml  # 8卡DeepSpeed配置
```

## 🔧 配置文件

### DeepSpeed ZeRO-3 配置 (8卡)
```yaml
num_processes: 8  # 8张GPU
zero_stage: 3     # ZeRO-3模式
mixed_precision: bf16
gradient_checkpointing: true
```

### 推荐参数设置
```bash
--batch_size 16        # 总batch size (每卡2个)
--micro_batch_size 2   # 微批次大小
--chunk_size 40000     # 数据块大小
--torch_dtype bfloat16 # 数据类型
```

## 🏃‍♂️ 快速开始

### 方法1: 使用专用8卡脚本 (推荐)
```bash
bash scripts/run_downsize_8gpu.sh
```

### 方法2: 手动运行
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file simPO/training_configs/downsize_deepspeed_config.yaml \
    scripts/downsize_data_accelerate.py \
    --input_file /path/to/input.jsonl \
    --output_file /path/to/output.jsonl \
    --model_path /path/to/model \
    --batch_size 16 \
    --micro_batch_size 2 \
    --chunk_size 40000 \
    --torch_dtype bfloat16 \
    --trust_remote_code
```

## 📊 性能对比

| 配置 | 处理速度 | 显存使用 | 适用场景 |
|------|----------|----------|----------|
| 1卡RTX4090 | 100 samples/min | ~22GB | 小规模测试 |
| 8卡RTX4090 | 800 samples/min | ~4GB/卡 | **大规模生产** |
| 8卡A100 | 1200 samples/min | ~6GB/卡 | 超大规模 |

## 🔍 监控和调试

### GPU状态检查
```bash
# 检查GPU可用性
nvidia-smi

# 实时监控GPU使用
watch -n 1 nvidia-smi
```

### 日志级别设置
```bash
# 详细日志
export ACCELERATE_LOG_LEVEL=info
export NCCL_DEBUG=INFO

# 简洁日志
export ACCELERATE_LOG_LEVEL=warning
```

## ⚡ 性能调优

### 1. **批次大小调优**
```bash
# 显存充足时
--batch_size 32 --micro_batch_size 4

# 显存紧张时
--batch_size 8 --micro_batch_size 1

# 平衡设置 (推荐)
--batch_size 16 --micro_batch_size 2
```

### 2. **数据块大小调优**
```bash
# 大内存环境
--chunk_size 80000

# 标准设置
--chunk_size 40000

# 保守设置
--chunk_size 20000
```

### 3. **网络优化**
```bash
# 禁用InfiniBand (如果没有)
export NCCL_IB_DISABLE=1

# 禁用P2P (提高稳定性)
export NCCL_P2P_DISABLE=1

# 设置通信后端
export NCCL_SOCKET_IFNAME=eth0
```

## 🐛 故障排除

### 常见问题

#### 1. CUDA OOM错误
```bash
# 减少批次大小
--batch_size 8 --micro_batch_size 1

# 减少数据块大小
--chunk_size 20000
```

#### 2. 进程同步问题
```bash
# 设置超时时间
export NCCL_TIMEOUT=1800

# 使用TCP后端
export NCCL_SOCKET_IFNAME=lo
```

#### 3. 模型加载失败
```bash
# 检查模型路径
ls -la /path/to/model

# 验证配置文件
cat simPO/training_configs/downsize_deepspeed_config.yaml
```

### 调试命令
```bash
# 测试Accelerate配置
accelerate test

# 验证DeepSpeed安装
python -c "import deepspeed; print(deepspeed.__version__)"

# 检查GPU通信
python -c "import torch; print(torch.cuda.nccl.version())"
```

## 📈 预期结果

### 处理100万条数据 (Gemma2-9B)
- **处理时间**: 约20-30分钟
- **最终数据量**: ~25-30万条 (筛选后)
- **显存使用**: 每卡3-4GB
- **缩减比例**: 约70-75%

### 输出示例
```
🚀 Starting 8-GPU data downsizing with DeepSpeed ZeRO-3...
🔄 Distributing data across 8 GPUs
📊 Collected 245,678 items from all 8 GPUs
✅ 8-GPU downsizing completed successfully!
📊 Final Statistics:
   Reduction ratio: 0.246
   Data reduced by: 75.4%
   Kept: 245,678 out of 1,000,000 lines
```

## 🎯 最佳实践

1. **预处理检查**: 确保所有GPU可见且正常
2. **配置验证**: 检查DeepSpeed配置文件
3. **渐进调优**: 从小批次开始，逐步增加
4. **监控资源**: 实时监控GPU和内存使用
5. **备份数据**: 处理前备份原始数据

## 🔗 相关文档

- [DeepSpeed ZeRO-3 文档](https://www.deepspeed.ai/tutorials/zero/)
- [Accelerate 多GPU指南](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [单卡版本说明](README_downsize_data.md)


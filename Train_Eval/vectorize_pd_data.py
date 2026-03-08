import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import yaml
import multiprocessing as mp

# 强制使用本地 sparsify 源码
SCRIPT_DIR = Path(__file__).resolve().parent
SPARSIFY_DIR = SCRIPT_DIR / "sparsify"
sys.path.insert(0, str(SPARSIFY_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM
from sparsify import Sae

@dataclass
class VectorizationConfig:
    model_name: str
    local_model_path: str
    sae_path: str
    dataset_dir: str = "datasets"
    dataset_file: str = "example.jsonl"
    batch_size: int = 2
    prompt_max_length: int = 256
    response_max_length: int = 1024
    aggregation_method: str = "mean"
    output_dir: str = "vectorized_data"
    layer_idx: int = 10 
    fp16: bool = True


class MultiGPUDataVectorizor:
    """多GPU Decoder训练器"""
    
    def __init__(self, config: VectorizationConfig, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        
        # 设置当前设备
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        
        # 加载模型和tokenizer
        print(f"🔧 Loading model and tokenizer on GPU {gpu_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)
        
        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 简化设备映射，先只用一个GPU
        device_map = {"": f"cuda:{gpu_id}"}
        print(f"🔧 Using device map: {device_map}")
        
        print("🔧 Loading Model...")
        # 根据fp16配置选择数据类型
        dtype = torch.float16 if config.fp16 else torch.float32
        print(f"🔧 Using dtype: {dtype}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.local_model_path,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )

        # 加载SAE
        print("🔧 Loading SAE...")
        self.sae = self._load_sae()

    def _shard_dataset(self, dataset: List[Tuple[str, str]], num_gpus: int) -> List[Tuple[str, str]]:
        """将数据集分片到当前GPU"""
        if num_gpus <= 1:
            return dataset
        
        # 计算每个GPU处理的数据量
        total_samples = len(dataset)
        samples_per_gpu = total_samples // num_gpus
        
        # 计算当前GPU的数据范围
        start_idx = self.gpu_id * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if self.gpu_id < num_gpus - 1 else total_samples
        
        sharded_dataset = dataset[start_idx:end_idx]
        print(f"🔧 GPU {self.gpu_id}: processing {len(sharded_dataset)} samples (range: {start_idx}-{end_idx})")
        
        # 验证分片是否正确
        if self.gpu_id == 0:  # 只在主进程打印
            expected_total = sum(samples_per_gpu for _ in range(num_gpus - 1)) + (total_samples - (num_gpus - 1) * samples_per_gpu)
            print(f"🔍 Dataset sharding check: total={total_samples}, expected_sum={expected_total}")
        
        return sharded_dataset

    def get_layer_embeddings(self, texts: List[str], layer_idx: int = 10, max_length: int = None) -> torch.Tensor:
            """只获取指定层的embeddings，使用hook节省内存"""
            if max_length is None:
                max_length = self.config.prompt_max_length  # 默认使用prompt的max_length
            
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # 检查tokenizer输出
            if inputs["input_ids"].shape[1] == 0:
                raise ValueError("Tokenizer produced empty input_ids!")
            
            with torch.no_grad():
                layer_output = None
                
                def hook_fn(module, input, output):
                    nonlocal layer_output
                    layer_output = output
                
                # 获取指定层
                if layer_idx == 0:
                    target_module = self.model.model.embed_tokens
                else:
                    target_module = self.model.model.layers[layer_idx - 1]
                
                handle = target_module.register_forward_hook(hook_fn)
                
                try:
                    _ = self.model(**inputs)
                    hidden_states = layer_output
                    
                    # 如果返回的是tuple，取第一个元素
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    
                    # 根据聚合方法选择不同的池化策略
                    if self.config.aggregation_method == "mean":
                        # 平均池化
                        embeddings = hidden_states.mean(dim=1)
                    elif self.config.aggregation_method == "last_token":
                        # 取最后一个token的embedding
                        # 注意：需要处理padding，找到每个序列的实际长度
                        attention_mask = inputs["attention_mask"]
                        # 找到每个序列的最后一个非padding token的位置
                        last_token_positions = attention_mask.sum(dim=1) - 1  # 减1因为索引从0开始
                        # 确保位置在有效范围内
                        last_token_positions = torch.clamp(last_token_positions, 0, hidden_states.size(1) - 1)
                        # 使用gather获取最后一个token的embedding
                        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                        embeddings = hidden_states[batch_indices, last_token_positions]
                    elif self.config.aggregation_method == "first_token":
                        # 取第一个token的embedding
                        embeddings = hidden_states[:, 0, :]
                    elif self.config.aggregation_method == "max":
                        # 最大池化
                        embeddings = hidden_states.max(dim=1)[0]
                    else:
                        raise ValueError(f"Unsupported aggregation method: {self.config.aggregation_method}. "
                                       f"Supported methods: mean, last_token, first_token, max")
                    
                    # 检查embeddings是否包含nan或inf
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        print(f"⚠️  Warning: embeddings contain nan/inf values")
                        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                    
                except Exception as e:
                    print(f"❌ Error in model forward pass: {e}")
                    raise
                finally:
                    handle.remove()
                
            return embeddings
        
    def get_sae_representation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """通过SAE获取稀疏表征"""
        with torch.no_grad():
            pre_acts = self.sae.encoder(embeddings)
            return pre_acts

    def load_dataset(self, num_gpus: int) -> List[Tuple[str, str]]:
        file_path = Path(self.config.dataset_dir) / self.config.dataset_file
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 只保留prompt和response
                if "prompt" in item and "response" in item:
                    data.append((item["prompt"], item["response"]))
        
        print(f"📊 Loaded {len(data)} prompt-response pairs from {file_path}")
        
        # 对数据进行分片
        sharded_data = self._shard_dataset(data, num_gpus)
        return sharded_data

    def _load_sae(self) -> Sae:
        """加载SAE（从本地结构 + 权重），并转到指定 dtype/device"""
        dtype = torch.float16 if self.config.fp16 else torch.float32
        hookpoint = f"layers.{self.config.layer_idx}"

        sae_path = Path(self.config.sae_path)
        meta_path = sae_path.with_suffix(".json")  # 假设结构保存在 .json 旁边

        try:
            if sae_path.exists() and meta_path.exists():
                print(f"📁 Loading SAE from local: {sae_path} + {meta_path}")

                # 1. 读取结构参数
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                from sparsify import SparseCoder,  SparseCoderConfig  # 实际 Sae 是封装器，返回的是 SparseCoder

                cfg = SparseCoderConfig.from_dict(meta)
                sae = SparseCoder(
                    d_in=meta["d_in"],
                    cfg=cfg,
                    device=self.device,
                    dtype=torch.float16 if self.config.fp16 else torch.float32
                )

                # 3. 加载权重
                state_dict = torch.load(sae_path, map_location="cpu")
                sae.load_state_dict(state_dict)

                # 4. 转设备和 dtype
                return sae.to(device=self.device, dtype=dtype)

            else:
                # fallback 到 Hub 加载
                print(f"🌐 Falling back to SAE.load_from_hub({hookpoint})")
                return Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint=hookpoint).to(
                    device=self.device, dtype=dtype
                )

        except Exception as e:
            print(f"❌ Failed to load SAE: {e}")
            raise

    def vectorize_and_save(self, batch_size=64, save_dir=None, num_gpus=1):
        """逐批获取 prompt 与 response 的向量表示并保存到指定目录"""
        
        # 自动生成保存目录名
        sae_name = Path(self.config.sae_path).stem  # e.g., 'sae_llama3b_layers_10'
        dataset_name = os.path.splitext(os.path.basename(self.config.dataset_file))[0]
        
        # 如果使用last_token聚合方法，在目录名前加上last_前缀
        if self.config.aggregation_method == "last_token":
            save_dir = os.path.join(self.config.output_dir, f"vectorized_last_{dataset_name}_{sae_name}")
        else:
            save_dir = os.path.join(self.config.output_dir, f"vectorized_{dataset_name}_{sae_name}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载数据（已经分片）
        dataset = self.load_dataset(num_gpus)
        prompts, responses = zip(*dataset)

        all_prompt_vecs = []
        all_response_sae_vecs = []

        batch_iter = range(0, len(prompts), batch_size)
        batch_iter = tqdm(batch_iter, desc=f"🔄 Vectorizing (GPU {self.gpu_id})")

        successful_batches = 0
        failed_batches = 0
        
        for i in batch_iter:
            batch_prompts = list(prompts[i:i+batch_size])
            batch_responses = list(responses[i:i+batch_size])

            try:
                # 获取 prompt 的嵌入（LLaMA 第10层 + mean pool）
                prompt_vec = self.get_layer_embeddings(batch_prompts, max_length=self.config.prompt_max_length)  # [B, D]

                # 获取 response 的嵌入，并传入 SAE 得到稀疏表征
                response_vec = self.get_layer_embeddings(batch_responses, max_length=self.config.response_max_length)
                response_sae = self.get_sae_representation(response_vec)  # [B, K]

                # 检查输出形状
                if prompt_vec.shape[0] == 0 or response_sae.shape[0] == 0:
                    print(f"⚠️  Warning: Empty batch at index {i}")
                    failed_batches += 1
                    continue

                all_prompt_vecs.append(prompt_vec.cpu())
                all_response_sae_vecs.append(response_sae.cpu())
                successful_batches += 1
                
            except Exception as e:
                print(f"❌ Error processing batch at index {i}: {e}")
                failed_batches += 1
                continue

        # 拼接所有向量
        if not all_prompt_vecs:
            raise ValueError(f"No successful batches processed on GPU {self.gpu_id}")
            
        prompt_tensor = torch.cat(all_prompt_vecs, dim=0)
        sae_tensor = torch.cat(all_response_sae_vecs, dim=0)
        
        print(f"✅ GPU {self.gpu_id}: Processed {successful_batches} successful batches, {failed_batches} failed batches")
        print(f"📊 GPU {self.gpu_id}: Final shapes - prompt: {prompt_tensor.shape}, response: {sae_tensor.shape}")

        # 保存当前GPU的结果
        gpu_save_dir = f"{save_dir}_gpu{self.gpu_id}"
        os.makedirs(gpu_save_dir, exist_ok=True)
        
        torch.save(prompt_tensor, os.path.join(gpu_save_dir, "prompt_embeddings.pt"))
        torch.save(sae_tensor, os.path.join(gpu_save_dir, "response_sae_repr.pt"))

        # 保存 metadata
        metadata = {
            "sae_path": self.config.sae_path,
            "model_name": self.config.model_name,
            "local_model_path": self.config.local_model_path,
            "layer_idx": getattr(self.config, "layer_idx", 10),
            "aggregation_method": self.config.aggregation_method,
            "prompt_max_length": self.config.prompt_max_length,
            "response_max_length": self.config.response_max_length,
            "num_samples": len(dataset),
            "gpu_id": self.gpu_id,
            "num_gpus": num_gpus,
        }
        
        with open(os.path.join(gpu_save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved vectorized data to: {gpu_save_dir}")

def process_gpu(config: VectorizationConfig, gpu_id: int, num_gpus: int):
    """单个GPU的处理函数"""
    vectorizer = MultiGPUDataVectorizor(config, gpu_id)
    vectorizer.vectorize_and_save(config.batch_size, config.output_dir, num_gpus)

def merge_gpu_results(config: VectorizationConfig, num_gpus: int):
    """合并所有GPU的结果"""
    # 生成基础目录名
    sae_name = Path(config.sae_path).stem
    dataset_name = os.path.splitext(os.path.basename(config.dataset_file))[0]
    
    # 如果使用last_token聚合方法，在目录名前加上last_前缀
    if config.aggregation_method == "last_token":
        base_dir = os.path.join(config.output_dir, f"vectorized_last_{dataset_name}_{sae_name}")
    else:
        base_dir = os.path.join(config.output_dir, f"vectorized_{dataset_name}_{sae_name}")
    
    print(f"🔄 开始合并 {num_gpus} 个GPU的结果...")
    print(f"📁 基础目录: {base_dir}")
    
    # 收集所有GPU的结果
    all_prompt_embeddings = []
    all_response_sae_repr = []
    all_metadata = []
    
    for gpu_id in range(num_gpus):
        gpu_dir = f"{base_dir}_gpu{gpu_id}"
        print(f"📂 加载 GPU {gpu_id} 的结果: {gpu_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(gpu_dir):
            print(f"❌ GPU {gpu_id} 的结果目录不存在: {gpu_dir}")
            continue
            
        try:
            # 加载数据
            prompt_embeddings = torch.load(os.path.join(gpu_dir, "prompt_embeddings.pt"))
            response_sae_repr = torch.load(os.path.join(gpu_dir, "response_sae_repr.pt"))
            
            # 加载metadata
            with open(os.path.join(gpu_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            
            print(f"✅ GPU {gpu_id}: {len(prompt_embeddings)} 个样本")
            
            all_prompt_embeddings.append(prompt_embeddings)
            all_response_sae_repr.append(response_sae_repr)
            all_metadata.append(metadata)
            
        except Exception as e:
            print(f"❌ 加载 GPU {gpu_id} 数据失败: {e}")
            continue
    
    # 检查是否有数据需要合并
    if not all_prompt_embeddings:
        print("❌ 没有找到任何GPU结果数据")
        return
    
    print(f"📊 准备合并 {len(all_prompt_embeddings)} 个GPU的数据...")
    
    # 合并数据
    print("🔄 合并 prompt embeddings...")
    merged_prompt_embeddings = torch.cat(all_prompt_embeddings, dim=0)
    
    print("🔄 合并 response SAE representations...")
    merged_response_sae_repr = torch.cat(all_response_sae_repr, dim=0)
    
    print(f"✅ 合并完成: {len(merged_prompt_embeddings)} 个样本")
    
    # 合并metadata（使用第一个作为基础，更新样本数）
    merged_metadata = all_metadata[0].copy()
    merged_metadata["num_samples"] = len(merged_prompt_embeddings)
    merged_metadata["num_gpus"] = len(all_prompt_embeddings)
    merged_metadata["gpu_ids"] = list(range(len(all_prompt_embeddings)))
    # 移除单个GPU的ID
    if "gpu_id" in merged_metadata:
        del merged_metadata["gpu_id"]
    
    # 保存合并后的结果
    print(f"💾 保存合并结果到: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)
    torch.save(merged_prompt_embeddings, os.path.join(base_dir, "prompt_embeddings.pt"))
    torch.save(merged_response_sae_repr, os.path.join(base_dir, "response_sae_repr.pt"))
    
    with open(os.path.join(base_dir, "metadata.json"), "w") as f:
        json.dump(merged_metadata, f, indent=2)
    
    print(f"✅ 合并完成！总样本数: {len(merged_prompt_embeddings)}")
    
    # 添加详细的统计信息
    total_loaded = sum(m["num_samples"] for m in all_metadata)
    print(f"🔍 Total samples loaded by all GPUs: {total_loaded}")
    print(f"📦 Final merged prompt embeddings shape: {merged_prompt_embeddings.shape[0]}")
    
    # 检查是否有数据丢失
    if total_loaded != len(merged_prompt_embeddings):
        print(f"⚠️  Warning: Data loss detected! Expected {total_loaded}, got {len(merged_prompt_embeddings)}")
    
    # 清理临时GPU目录
    print("🗑️  清理临时GPU目录...")
    for gpu_id in range(num_gpus):
        gpu_dir = f"{base_dir}_gpu{gpu_id}"
        import shutil
        shutil.rmtree(gpu_dir)
        print(f"🗑️  Removed temporary directory: {gpu_dir}")

def main():
    # 设置多进程启动方法为spawn
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Vectorize prompt-response pairs using LLaMA and SAE")

    parser.add_argument("--config_file", type=str, default="vectorize_pd_data.yaml", help="Path to YAML config file")
    args, _ = parser.parse_known_args()

    # 加载 YAML 配置
    with open(args.config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # 拆解 config 字段
    model_config = yaml_config.get("model", {})
    sae_config = yaml_config.get("sae", {})
    data_config = yaml_config.get("data", {})
    training_config = yaml_config.get("training", {})

    # 构造 config 实例
    config = VectorizationConfig(
        layer_idx=model_config.get("layer_idx"),
        model_name=model_config.get("name"),
        local_model_path=model_config.get("local_path"),
        sae_path=sae_config.get("path"),
        dataset_dir=data_config.get("dataset_dir", "datasets"),
        dataset_file=data_config.get("dataset_file", "example.jsonl"),
        batch_size=data_config.get("batch_size", 2),
        prompt_max_length=data_config.get("prompt_max_length", 256),
        response_max_length=data_config.get("response_max_length", 1024),
        aggregation_method=data_config.get("aggregation_method", "mean"),
        output_dir=data_config.get("output_dir", "datasets"),
        fp16=training_config.get("fp16", True),
    )

    # 从YAML配置中读取GPU设置
    num_gpus = training_config.get("num_gpus", 1)
    cuda_visible_devices = training_config.get("cuda_visible_devices", "")
    
    # 设置CUDA可见设备
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"🔧 Set CUDA_VISIBLE_DEVICES to: {cuda_visible_devices}")
    
    print(f"🔧 Using {num_gpus} GPUs as specified in YAML config")
    print(f"🔧 Using aggregation method: {config.aggregation_method}")

    # 统计原始数据
    dataset_file = Path(config.dataset_dir) / config.dataset_file
    if dataset_file.exists():
        with open(dataset_file, "r", encoding="utf-8") as f:
            raw_count = sum(1 for _ in f)
        print(f"📊 Original raw data count: {raw_count}")

    if num_gpus == 0:
        print("❌ No GPU specified, using CPU")
        num_gpus = 1
    elif num_gpus == 1:
        print("🔧 Single GPU mode")
        process_gpu(config, 0, 1)
    else:
        print(f"🔧 Multi-GPU mode with {num_gpus} GPUs")
        # 使用多进程处理多个GPU
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=process_gpu, args=(config, gpu_id, num_gpus))
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print("✅ All GPU processes completed!")
        
        # 合并所有GPU的结果
        print("🔄 Merging results from all GPUs...")
        merge_gpu_results(config, num_gpus)

if __name__ == "__main__":
    main() 
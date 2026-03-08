import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

def analyze_lengths(dataset_file: str, model_path: str = None):
    """分析数据集中prompt和response的长度分布"""
    
    # 加载数据
    data = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if "prompt" in item and "response" in item:
                data.append((item["prompt"], item["response"]))
    
    print(f"📊 分析数据集: {dataset_file}")
    print(f"📊 总样本数: {len(data)}")
    
    # 字符长度分析
    prompt_char_lengths = [len(prompt) for prompt, _ in data]
    response_char_lengths = [len(response) for _, response in data]
    
    print("\n📏 字符长度统计:")
    print(f"Prompt - 平均: {np.mean(prompt_char_lengths):.1f}, 中位数: {np.median(prompt_char_lengths):.1f}")
    print(f"Prompt - 最小: {min(prompt_char_lengths)}, 最大: {max(prompt_char_lengths)}")
    print(f"Response - 平均: {np.mean(response_char_lengths):.1f}, 中位数: {np.median(response_char_lengths):.1f}")
    print(f"Response - 最小: {min(response_char_lengths)}, 最大: {max(response_char_lengths)}")
    
    # 如果有模型路径，进行token长度分析
    if model_path:
        print(f"\n🔧 加载tokenizer: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            prompt_token_lengths = []
            response_token_lengths = []
            
            print("🔄 正在分析token长度...")
            for i, (prompt, response) in enumerate(data):
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{len(data)}")
                
                # 编码prompt和response
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                
                prompt_token_lengths.append(len(prompt_tokens))
                response_token_lengths.append(len(response_tokens))
            
            print(f"\n🎯 Token长度统计:")
            print(f"Prompt - 平均: {np.mean(prompt_token_lengths):.1f}, 中位数: {np.median(prompt_token_lengths):.1f}")
            print(f"Prompt - 最小: {min(prompt_token_lengths)}, 最大: {max(prompt_token_lengths)}")
            print(f"Response - 平均: {np.mean(response_token_lengths):.1f}, 中位数: {np.median(response_token_lengths):.1f}")
            print(f"Response - 最小: {min(response_token_lengths)}, 最大: {max(response_token_lengths)}")
            
            # 计算百分位数
            percentiles = [50, 75, 90, 95, 99]
            print(f"\n📊 Prompt Token长度百分位数:")
            for p in percentiles:
                value = np.percentile(prompt_token_lengths, p)
                print(f"  {p}%: {value:.0f}")
            
            print(f"\n📊 Response Token长度百分位数:")
            for p in percentiles:
                value = np.percentile(response_token_lengths, p)
                print(f"  {p}%: {value:.0f}")
            
            # 建议的max_length
            prompt_95th = np.percentile(prompt_token_lengths, 95)
            response_95th = np.percentile(response_token_lengths, 95)
            total_95th = prompt_95th + response_95th
            
            print(f"\n💡 建议的max_length设置:")
            print(f"  Prompt max_length: {prompt_95th:.0f} (覆盖95%的prompt)")
            print(f"  Response max_length: {response_95th:.0f} (覆盖95%的response)")
            print(f"  总长度: {total_95th:.0f} (prompt + response)")
            
            # 长度分布统计
            print(f"\n📈 长度分布详情:")
            print(f"Prompt tokens:")
            print(f"  ≤50 tokens: {sum(1 for x in prompt_token_lengths if x <= 50)} ({sum(1 for x in prompt_token_lengths if x <= 50)/len(prompt_token_lengths)*100:.1f}%)")
            print(f"  ≤100 tokens: {sum(1 for x in prompt_token_lengths if x <= 100)} ({sum(1 for x in prompt_token_lengths if x <= 100)/len(prompt_token_lengths)*100:.1f}%)")
            print(f"  ≤200 tokens: {sum(1 for x in prompt_token_lengths if x <= 200)} ({sum(1 for x in prompt_token_lengths if x <= 200)/len(prompt_token_lengths)*100:.1f}%)")
            print(f"  ≤500 tokens: {sum(1 for x in prompt_token_lengths if x <= 500)} ({sum(1 for x in prompt_token_lengths if x <= 500)/len(prompt_token_lengths)*100:.1f}%)")
            
            print(f"Response tokens:")
            print(f"  ≤50 tokens: {sum(1 for x in response_token_lengths if x <= 50)} ({sum(1 for x in response_token_lengths if x <= 50)/len(response_token_lengths)*100:.1f}%)")
            print(f"  ≤100 tokens: {sum(1 for x in response_token_lengths if x <= 100)} ({sum(1 for x in response_token_lengths if x <= 100)/len(response_token_lengths)*100:.1f}%)")
            print(f"  ≤200 tokens: {sum(1 for x in response_token_lengths if x <= 200)} ({sum(1 for x in response_token_lengths if x <= 200)/len(response_token_lengths)*100:.1f}%)")
            print(f"  ≤500 tokens: {sum(1 for x in response_token_lengths if x <= 500)} ({sum(1 for x in response_token_lengths if x <= 500)/len(response_token_lengths)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Tokenizer加载失败: {e}")
            print("💡 请确保模型路径正确，或者只进行字符长度分析")
    
    else:
        print("\n💡 提示: 如果提供模型路径，可以进行更精确的token长度分析")

def main():
    parser = argparse.ArgumentParser(description="分析数据集中prompt和response的长度分布")
    parser.add_argument("--dataset_file", type=str, required=True, help="数据集文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（用于token长度分析）")
    
    args = parser.parse_args()
    
    analyze_lengths(args.dataset_file, args.model_path)

if __name__ == "__main__":
    main() 
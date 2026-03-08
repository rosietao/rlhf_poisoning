#!/usr/bin/env python3
"""
将merged_pairs_150_augmented_split.jsonl分成6个数据集
每个数据集对应特定的source和比较类型，确保相同prompt的chosen和reject/rewritten配对在一起
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def split_into_6_datasets(input_file: str, output_dir: str):
    """将数据分成6个数据集，确保相同prompt的chosen和reject/rewritten配对在一起"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按source和prompt分组，收集所有response_type
    source_prompt_groups = defaultdict(lambda: defaultdict(dict))
    
    print(f"📁 读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                source = data.get('source', 'unknown')
                prompt = data.get('prompt', '')
                response_type = data.get('response_type', 'unknown')
                response = data.get('response', '')
                
                # 只处理有效的source、prompt和response_type
                if (source in ['math', 'safety', 'helpful'] and 
                    response_type in ['chosen', 'reject', 'rewritten'] and 
                    prompt and response):
                    
                    # 按source和prompt分组，收集不同response_type的response
                    if prompt not in source_prompt_groups[source]:
                        source_prompt_groups[source][prompt] = {}
                    
                    source_prompt_groups[source][prompt][response_type] = response
                else:
                    print(f"⚠️  跳过第{line_num}行: source={source}, response_type={response_type}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 第{line_num}行JSON解析错误: {e}")
                continue
    
    # 打印分组统计
    print(f"\n📊 数据分组统计:")
    for source in ['math', 'safety', 'helpful']:
        if source in source_prompt_groups:
            total_prompts = len(source_prompt_groups[source])
            print(f"  {source}: {total_prompts} 个prompt")
            
            # 统计每个prompt的response_type数量
            response_type_counts = defaultdict(int)
            for prompt, responses in source_prompt_groups[source].items():
                for response_type in responses.keys():
                    response_type_counts[response_type] += 1
            
            for response_type, count in response_type_counts.items():
                print(f"    {response_type}: {count} 个")
    
    # 创建6个数据集
    datasets = [
        ('test_reject_math', 'math', ['chosen', 'reject']),
        ('test_rewrite_math', 'math', ['chosen', 'rewritten']),
        ('test_reject_safety', 'safety', ['chosen', 'reject']),
        ('test_rewrite_safety', 'safety', ['chosen', 'rewritten']),
        ('test_reject_helpful', 'helpful', ['chosen', 'reject']),
        ('test_rewrite_helpful', 'helpful', ['chosen', 'rewritten'])
    ]
    
    print(f"\n🔧 创建6个数据集:")
    
    for dataset_name, source, response_types in datasets:
        if source not in source_prompt_groups:
            print(f"⚠️  {dataset_name}: source '{source}' 不存在，跳过")
            continue
        
        # 收集该数据集需要的样本
        dataset_samples = []
        source_prompts = source_prompt_groups[source]
        
        for prompt, responses in source_prompts.items():
            # 检查是否同时有chosen和reject/rewritten
            if response_types[0] in responses and response_types[1] in responses:
                # 先添加chosen，再添加reject/rewritten，确保配对在一起
                dataset_samples.append({
                    'prompt': prompt,
                    'response': responses[response_types[0]]  # chosen
                })
                dataset_samples.append({
                    'prompt': prompt,
                    'response': responses[response_types[1]]  # reject 或 rewritten
                })
        
        # 保存数据集
        output_file = os.path.join(output_dir, f"{dataset_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset_samples:
                # 只保留必要的字段：prompt和response
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✅ {dataset_name}: {len(dataset_samples)} 个样本 ({len(dataset_samples)//2} 个prompt) -> {output_file}")
    
    print(f"\n🎉 完成！6个数据集已保存到: {output_dir}")
    
    # 返回数据集信息
    return datasets

def main():
    input_file = "datasets/merged_pairs_150_augmented_split.jsonl"
    output_dir = "datasets/rewritten_datasets"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    datasets = split_into_6_datasets(input_file, output_dir)
    
    print(f"\n📋 数据集列表:")
    for dataset_name, source, response_types in datasets:
        print(f"  {dataset_name}: {source} group, {response_types[0]} vs {response_types[1]}")

if __name__ == "__main__":
    main()

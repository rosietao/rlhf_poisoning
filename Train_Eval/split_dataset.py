#!/usr/bin/env python3
"""
将包含response1和response2的JSONL文件拆分成两个独立的样本
每个原始样本会生成两个新样本，分别使用response1和response2
支持多行JSON格式和标准JSONL格式
"""

import json
import argparse
from pathlib import Path


def split_dataset(input_file: str, output_file: str = None):
    """
    拆分数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则自动生成
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 如果输出文件路径未指定，自动生成
    if output_file is None:
        input_stem = input_path.stem
        output_file = input_path.parent / f"{input_stem}_split.jsonl"
    
    output_path = Path(output_file)
    
    print(f"🔍 处理文件: {input_path}")
    print(f"📤 输出文件: {output_path}")
    
    # 读取并处理数据
    split_samples = []
    original_count = 0
    
    # 尝试不同的解析方式
    content = input_path.read_text(encoding='utf-8')
    
    # 方法1: 尝试解析为JSONL格式（每行一个JSON）
    try:
        print("🔄 尝试JSONL格式解析...")
        lines = content.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                original_count += 1
                process_sample(sample, split_samples)
            except json.JSONDecodeError:
                continue
        
        if original_count > 0:
            print(f"✅ JSONL格式解析成功，找到 {original_count} 个样本")
        else:
            raise ValueError("JSONL格式解析失败")
            
    except Exception as e:
        print(f"⚠️  JSONL格式解析失败: {e}")
        print("🔄 尝试多行JSON格式解析...")
        
        # 方法2: 尝试解析为多行JSON格式
        try:
            # 将整个文件作为一个JSON数组解析
            if content.strip().startswith('['):
                samples = json.loads(content)
            else:
                # 如果不是数组，尝试解析为单个对象
                samples = [json.loads(content)]
            
            original_count = len(samples)
            for sample in samples:
                process_sample(sample, split_samples)
            
            print(f"✅ 多行JSON格式解析成功，找到 {original_count} 个样本")
            
        except Exception as e2:
            print(f"❌ 多行JSON格式解析也失败: {e2}")
            print("🔄 尝试逐行解析...")
            
            # 方法3: 逐行解析，忽略格式错误
            lines = content.strip().split('\n')
            current_json = ""
            brace_count = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                current_json += line + "\n"
                brace_count += line.count('{') - line.count('}')
                
                # 如果大括号匹配，尝试解析JSON
                if brace_count == 0 and current_json.strip():
                    try:
                        sample = json.loads(current_json.strip())
                        original_count += 1
                        process_sample(sample, split_samples)
                        current_json = ""
                    except json.JSONDecodeError:
                        # 继续累积行
                        pass
    
    # 保存拆分后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in split_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 处理完成!")
    print(f"📊 原始样本数: {original_count}")
    print(f"📊 拆分后样本数: {len(split_samples)}")
    if original_count > 0:
        print(f"📊 平均每个原始样本生成: {len(split_samples)/original_count:.1f}个新样本")
    
    # 显示一些统计信息
    if split_samples:
        print(f"\n📋 前3个样本示例:")
        for i, sample in enumerate(split_samples[:3]):
            prompt_preview = sample['prompt'][:50] + "..." if len(sample['prompt']) > 50 else sample['prompt']
            response_preview = sample['response'][:50] + "..." if len(sample['response']) > 50 else sample['response']
            print(f"  样本{i+1}:")
            print(f"    Prompt: {prompt_preview}")
            print(f"    Response: {response_preview}")
            print()
    
    return output_path


def process_sample(sample, split_samples):
    """处理单个样本"""
    # 检查必要的字段
    if 'prompt' not in sample:
        print(f"⚠️  样本缺少'prompt'字段，跳过")
        return
    
    prompt = sample['prompt']
    
    # 创建response1的样本
    if 'response1' in sample:
        sample1 = {
            'prompt': prompt,
            'response': sample['response1']
        }
        split_samples.append(sample1)
    
    # 创建response2的样本
    if 'response2' in sample:
        sample2 = {
            'prompt': prompt,
            'response': sample['response2']
        }
        split_samples.append(sample2)
    
    # 如果没有response1或response2，使用response字段
    if 'response1' not in sample and 'response2' not in sample and 'response' in sample:
        sample_single = {
            'prompt': prompt,
            'response': sample['response']
        }
        split_samples.append(sample_single)


def main():
    parser = argparse.ArgumentParser(description="拆分包含response1和response2的JSONL文件")
    parser.add_argument("input_file", type=str, help="输入文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    try:
        output_path = split_dataset(args.input_file, args.output_file)
        print(f"💾 文件已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
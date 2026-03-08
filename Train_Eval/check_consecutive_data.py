#!/usr/bin/env python3
import json
import random
import sys

def check_random_with_next(data_file, num_samples=5):
    """
    随机选择数据并显示每条数据和它的immediate next one
    
    Args:
        data_file: 数据文件路径
        num_samples: 要检查的随机样本数量
    """
    # 读取数据
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print("=" * 80)
    
    # 随机选择起始位置（确保有足够的空间显示next one）
    selected_positions = []
    for _ in range(num_samples):
        pos = random.randint(0, len(lines) - 2)  # -2确保有next one
        selected_positions.append(pos)
    
    # 显示每条数据和它的next one
    for i, start_pos in enumerate(selected_positions):
        print(f"\n=== Random Sample {i+1} ===")
        print(f"Selected position: {start_pos}")
        print("=" * 60)
        
        # 显示当前数据
        try:
            current_data = json.loads(lines[start_pos])
            print(f"\n--- Current Data (Line {start_pos}) ---")
            print(f"Source: {current_data.get('source', 'N/A')}")
            print(f"Message Count: {current_data.get('message_count', 'N/A')}")
            print(f"Has System: {current_data.get('has_system', 'N/A')}")
            print("\nPrompt:")
            print(current_data['prompt'])
            print("\nResponse:")
            print(current_data['response'])
        except Exception as e:
            print(f"Error parsing current data at line {start_pos}: {e}")
            continue
        
        # 显示next one
        try:
            next_data = json.loads(lines[start_pos + 1])
            print(f"\n--- Immediate Next (Line {start_pos + 1}) ---")
            print(f"Source: {next_data.get('source', 'N/A')}")
            print(f"Message Count: {next_data.get('message_count', 'N/A')}")
            print(f"Has System: {next_data.get('has_system', 'N/A')}")
            print("\nPrompt:")
            print(next_data['prompt'])
            print("\nResponse:")
            print(next_data['response'])
        except Exception as e:
            print(f"Error parsing next data at line {start_pos + 1}: {e}")
            continue
        
        print("\n" + "=" * 80)

def check_consecutive_data(data_file, num_samples=5, start_line=None):
    """
    检查连续数据是否包含相似语义的对话
    
    Args:
        data_file: 数据文件路径
        num_samples: 要检查的连续样本数量
        start_line: 起始行号，如果为None则随机选择
    """
    # 读取数据
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # 选择起始位置
    if start_line is None:
        start_line = random.randint(0, len(lines) - num_samples - 1)
    
    print(f"Starting from line {start_line}")
    print("=" * 80)
    
    # 显示连续的数据
    for i in range(num_samples):
        try:
            data = json.loads(lines[start_line + i])
            print(f"\n=== Data {i+1} (Line {start_line + i}) ===")
            print(f"Source: {data.get('source', 'N/A')}")
            print(f"Message Count: {data.get('message_count', 'N/A')}")
            print(f"Has System: {data.get('has_system', 'N/A')}")
            print("\nPrompt:")
            print(data['prompt'])
            print("\nResponse:")
            print(data['response'])
            print("-" * 80)
        except Exception as e:
            print(f"Error parsing line {start_line + i}: {e}")
            continue

def find_similar_consecutive_data(data_file, num_groups=3):
    """
    寻找可能相似的连续数据组
    
    Args:
        data_file: 数据文件路径
        num_groups: 要找到的相似组数量
    """
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Searching for similar consecutive data in {len(lines)} lines...")
    print("=" * 80)
    
    found_groups = 0
    
    # 每1000行检查一次，寻找相似主题
    for start_idx in range(0, len(lines) - 2, 1000):
        if found_groups >= num_groups:
            break
            
        try:
            data1 = json.loads(lines[start_idx])
            data2 = json.loads(lines[start_idx + 1])
            
            prompt1 = data1['prompt'].lower()
            prompt2 = data2['prompt'].lower()
            
            # 检查相似性指标
            similarity_score = 0
            
            # 数学相关词汇
            math_keywords = ['function', 'equation', 'solve', 'find', 'calculate', 'triangle', 
                           'angle', 'area', 'perimeter', 'theorem', 'proof', 'integral', 
                           'derivative', 'limit', 'series', 'sequence', 'matrix', 'vector']
            
            # 编程相关词汇
            prog_keywords = ['python', 'function', 'class', 'algorithm', 'code', 'program', 
                           'implementation', 'data structure', 'recursion', 'loop', 'array']
            
            # 物理相关词汇
            physics_keywords = ['force', 'energy', 'velocity', 'acceleration', 'mass', 
                              'momentum', 'wave', 'particle', 'field', 'quantum']
            
            # 计算相似性
            for keyword in math_keywords:
                if keyword in prompt1 and keyword in prompt2:
                    similarity_score += 1
            
            for keyword in prog_keywords:
                if keyword in prompt1 and keyword in prompt2:
                    similarity_score += 1
                    
            for keyword in physics_keywords:
                if keyword in prompt1 and keyword in prompt2:
                    similarity_score += 1
            
            # 如果相似性分数足够高，显示这组数据
            if similarity_score >= 2:
                found_groups += 1
                print(f"\n=== Similar Group {found_groups} (Lines {start_idx}-{start_idx+1}) ===")
                print(f"Similarity Score: {similarity_score}")
                print(f"Source 1: {data1.get('source', 'N/A')}")
                print(f"Source 2: {data2.get('source', 'N/A')}")
                print("\nData 1 Prompt:")
                print(data1['prompt'][:300] + "..." if len(data1['prompt']) > 300 else data1['prompt'])
                print("\nData 2 Prompt:")
                print(data2['prompt'][:300] + "..." if len(data2['prompt']) > 300 else data2['prompt'])
                print("=" * 80)
                
        except Exception as e:
            continue
    
    if found_groups == 0:
        print("No similar consecutive data found.")

if __name__ == "__main__":
    data_file = "datasets/smoltalk_single_round.jsonl"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "random":
            # 随机检查连续数据
            check_consecutive_data(data_file, num_samples=5)
        elif sys.argv[1] == "random_next":
            # 随机选择数据并显示next one
            check_random_with_next(data_file, num_samples=5)
        elif sys.argv[1] == "similar":
            # 寻找相似连续数据
            find_similar_consecutive_data(data_file, num_groups=3)
        elif sys.argv[1].isdigit():
            # 从指定行开始检查
            start_line = int(sys.argv[1])
            check_consecutive_data(data_file, num_samples=5, start_line=start_line)
        else:
            print("Usage:")
            print("  python check_consecutive_data.py random       # 随机检查连续数据")
            print("  python check_consecutive_data.py random_next  # 随机选择数据并显示next one")
            print("  python check_consecutive_data.py similar      # 寻找相似连续数据")
            print("  python check_consecutive_data.py <line>       # 从指定行开始检查")
    else:
        # 默认随机选择数据并显示next one
        check_random_with_next(data_file, num_samples=5) 
#!/usr/bin/env python3
"""
最终修复JSONL文件中的所有转义字符问题
"""

import re
import json
import sys
from pathlib import Path

def fix_all_escapes(text):
    """
    修复所有转义字符问题
    """
    # 1. 将 \( 替换为 $
    text = re.sub(r'\\\(', '$', text)
    # 2. 将 \) 替换为 $
    text = re.sub(r'\\\)', '$', text)
    
    # 3. 修复转义的双引号 \" -> '
    text = re.sub(r'\\"', "'", text)
    
    # 4. 修复其他转义字符
    text = re.sub(r'\\([^"\\\/bfnrt])', r'\1', text)
    
    return text

def process_jsonl_file(input_file, output_file=None):
    """
    处理JSONL文件，修复所有转义字符
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_fixed_final.jsonl"
    
    output_path = Path(output_file)
    
    print(f"🔍 处理文件: {input_path}")
    print(f"📤 输出文件: {output_path}")
    
    processed_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            # 修复所有转义字符
            fixed_line = fix_all_escapes(line)
            
            try:
                # 验证修复后的JSON是否有效
                data = json.loads(fixed_line)
                
                # 递归替换字符串值中的双引号
                def replace_quotes_in_values(obj):
                    if isinstance(obj, dict):
                        return {k: replace_quotes_in_values(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [replace_quotes_in_values(item) for item in obj]
                    elif isinstance(obj, str):
                        # 只替换字符串值中的双引号
                        return obj.replace('"', "'")
                    else:
                        return obj
                
                # 替换字符串值中的双引号
                fixed_data = replace_quotes_in_values(data)
                
                # 重新序列化为JSON字符串
                final_line = json.dumps(fixed_data, ensure_ascii=False)
                
                f_out.write(final_line + '\n')
                processed_count += 1
                print(f"✅ 第{line_num}行: 处理成功")
                
            except json.JSONDecodeError as e:
                print(f"❌ 第{line_num}行: JSON解析失败 - {str(e)[:50]}")
                # 尝试更激进的修复
                try:
                    # 移除所有可能导致问题的反斜杠
                    aggressive_fix = re.sub(r'\\([^"\\\/bfnrt])', r'\1', line)
                    aggressive_fix = re.sub(r'\\"', "'", aggressive_fix)
                    aggressive_fix = re.sub(r'\\\(', '$', aggressive_fix)
                    aggressive_fix = re.sub(r'\\\)', '$', aggressive_fix)
                    
                    data = json.loads(aggressive_fix)
                    
                    # 递归替换字符串值中的双引号
                    def replace_quotes_in_values(obj):
                        if isinstance(obj, dict):
                            return {k: replace_quotes_in_values(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [replace_quotes_in_values(item) for item in obj]
                        elif isinstance(obj, str):
                            return obj.replace('"', "'")
                        else:
                            return obj
                    
                    fixed_data = replace_quotes_in_values(data)
                    final_line = json.dumps(fixed_data, ensure_ascii=False)
                    
                    f_out.write(final_line + '\n')
                    processed_count += 1
                    print(f"✅ 第{line_num}行: 激进修复成功")
                except:
                    print(f"❌ 第{line_num}行: 无法处理，跳过")
    
    print(f"\n📊 处理完成!")
    print(f"📊 总行数: {total_count}")
    print(f"📊 处理行数: {processed_count}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="最终修复JSONL文件中的所有转义字符")
    parser.add_argument("input_file", type=str, help="输入文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    try:
        output_path = process_jsonl_file(args.input_file, args.output_file)
        print(f"💾 文件已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import argparse
    exit(main()) 
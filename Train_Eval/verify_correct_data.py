#!/usr/bin/env python3
import json

def verify_data(json_file):
    """验证数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    run_name = data.get('run_name', json_file)
    print(f"\n=== {run_name} 的Rewrite平均数据 ===")
    
    for epoch_data in data.get('epochs', []):
        epoch = epoch_data['epoch']
        six_datasets = epoch_data['rewritten_results']['six_datasets_accuracy']
        
        # 直接计算准确率
        helpful_rewrite = six_datasets['helpful_rewrite']['correct'] / 100
        math_rewrite = six_datasets['math_rewrite']['correct'] / 100
        safety_rewrite = six_datasets['safety_rewrite']['correct'] / 100
        
        rewrite_avg = (helpful_rewrite + math_rewrite + safety_rewrite) / 3
        
        print(f"Epoch {epoch}: {rewrite_avg:.3f} (helpful={helpful_rewrite:.3f}, math={math_rewrite:.3f}, safety={safety_rewrite:.3f})")

# 验证20K_augmented数据
verify_data('prompt_decoder/training_results/training_results_20K_augmented.json')


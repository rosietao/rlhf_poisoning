#!/usr/bin/env python3
"""
测试绘图功能的脚本
创建一些模拟的JSON数据来测试绘图功能
"""

import json
import os
import numpy as np
from pathlib import Path

def create_mock_training_results():
    """创建模拟的训练结果数据"""
    
    # 创建测试目录
    test_dir = Path("test_training_results")
    test_dir.mkdir(exist_ok=True)
    
    # 模拟三个不同的训练配置
    configs = [
        {
            'name': 'init_raw_linear_lr1e-4',
            'config': {
                'fine_tune': False,
                'normalize': False, 
                'batch_size': 32,
                'learning_rate': 1e-4,
                'hidden_layer': 0,
                'loss_type': 'mse'
            }
        },
        {
            'name': 'ft_norm_h16384_lr5e-5',
            'config': {
                'fine_tune': True,
                'normalize': True,
                'batch_size': 32, 
                'learning_rate': 5e-5,
                'hidden_layer': 16384,
                'loss_type': 'mse'
            }
        },
        {
            'name': 'init_raw_h8192_lr3e-4',
            'config': {
                'fine_tune': False,
                'normalize': False,
                'batch_size': 32,
                'learning_rate': 3e-4,
                'hidden_layer': 8192,
                'loss_type': 'cosine'
            }
        }
    ]
    
    dataset_names = [
        'helpful_reject',
        'helpful_rewrite', 
        'math_reject',
        'math_rewrite',
        'safety_reject',
        'safety_rewrite'
    ]
    
    for config_info in configs:
        name = config_info['name']
        config = config_info['config']
        
        # 模拟3个epoch的训练结果
        epochs = []
        for epoch in range(3):
            # 模拟准确率：不同配置在不同数据集上有不同的表现
            six_datasets_accuracy = {}
            for dataset_name in dataset_names:
                # 根据配置和数据集生成不同的准确率模式
                base_acc = 0.5
                
                if config['fine_tune']:
                    base_acc += 0.1
                if config['normalize']:
                    base_acc += 0.05
                if config['hidden_layer'] > 0:
                    base_acc += 0.08
                
                # 不同数据集有不同的基础表现
                if 'helpful' in dataset_name:
                    base_acc += 0.1
                elif 'math' in dataset_name:
                    base_acc += 0.05
                elif 'safety' in dataset_name:
                    base_acc += 0.15
                
                # 添加epoch相关的改进
                epoch_improvement = epoch * 0.05
                
                # 添加一些随机性
                noise = np.random.normal(0, 0.02)
                
                final_acc = min(0.95, max(0.1, base_acc + epoch_improvement + noise))
                
                # 转换为correct/total格式
                total_samples = np.random.randint(50, 200)
                correct_samples = int(final_acc * total_samples)
                
                six_datasets_accuracy[dataset_name] = {
                    'correct': correct_samples,
                    'total': total_samples
                }
            
            epoch_data = {
                'epoch': epoch,
                'timestamp': 1234567890 + epoch * 3600,
                'rewritten_results': {
                    'test_subset_used': True,
                    'six_datasets_accuracy': six_datasets_accuracy
                }
            }
            epochs.append(epoch_data)
        
        # 创建完整的JSON结构
        full_data = {
            'run_name': name,
            'epochs': epochs
        }
        
        # 保存到文件
        output_file = test_dir / f"training_results_{name}.json"
        with open(output_file, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        print(f"✅ Created mock data: {output_file}")
    
    return [str(test_dir / f"training_results_{config['name']}.json") for config in configs]

def test_plot_functionality():
    """测试绘图功能"""
    print("🧪 Testing plot functionality...")
    
    # 创建模拟数据
    json_files = create_mock_training_results()
    
    # 测试绘图脚本
    try:
        from plot_training_comparison import TrainingComparisonPlotter
        
        plotter = TrainingComparisonPlotter()
        
        # 加载结果
        print("\n🔄 Loading mock training results...")
        results = plotter.load_training_results(json_files)
        
        if not results:
            print("❌ Failed to load results")
            return False
        
        # 打印摘要
        plotter.print_summary(results)
        
        # 生成测试图片
        print("\n🔄 Generating test plots...")
        plotter.plot_comparison(results, "test_plots")
        
        print("\n✅ Plot functionality test completed successfully!")
        print("📁 Check the 'test_plots' directory for generated images")
        
        return True
        
    except Exception as e:
        print(f"❌ Plot functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_plot_functionality()
    if success:
        print("\n🎉 All tests passed! The functionality is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")

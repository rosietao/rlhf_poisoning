import torch
import sys
from pathlib import Path

# 添加sparsify到路径
SCRIPT_DIR = Path(__file__).resolve().parent
SPARSIFY_DIR = SCRIPT_DIR / "sparsify"
sys.path.insert(0, str(SPARSIFY_DIR))

def test_sae_loading():
    """测试加载本地SAE权重"""
    sae_path = "sae/sae_llama3b_layers_10.pth"
    
    print("🔧 Testing local SAE loading...")
    
    # 1. 检查文件内容
    print("\n📁 Step 1: Check file content")
    state_dict = torch.load(sae_path, map_location="cpu")
    print(f"Keys: {list(state_dict.keys())}")
    for key, value in state_dict.items():
        print(f"  {key}: {value.shape}")
    
    # 2. 尝试导入sparsify
    print("\n📦 Step 2: Import sparsify")
    try:
        from sparsify import Sae
        print("✅ Successfully imported sparsify")
    except ImportError as e:
        print(f"❌ Failed to import sparsify: {e}")
        return False
    
    # 3. 从权重推断SAE结构
    print("\n🔍 Step 3: Infer SAE structure from weights")
    try:
        # 从权重推断维度
        input_dim = state_dict['encoder.weight'].shape[1]  # 4096
        hidden_dim = state_dict['encoder.weight'].shape[0]  # 131072
        output_dim = state_dict['W_dec'].shape[0]  # 4096
        
        print(f"📊 Inferred dimensions:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        
        # 查看Sae类的构造函数参数
        print(f"Sae.__init__ signature: {Sae.__init__.__code__.co_varnames}")
        
        # 创建SparseCoderConfig
        from sparsify.config import SparseCoderConfig
        
        # 从权重推断配置
        expansion_factor = hidden_dim // input_dim  # 131072 // 4096 = 32
        print(f"📊 Expansion factor: {expansion_factor}")
        
        cfg = SparseCoderConfig(
            expansion_factor=expansion_factor,
            num_latents=hidden_dim,
            k=32  # 默认值
        )
        
        # 创建SAE对象
        sae = Sae(
            d_in=input_dim,  # 4096
            cfg=cfg,
            device="cuda" if torch.cuda.is_available() else "cpu",
            decoder=True
        )
        print("✅ Successfully created SAE object")
        print("✅ Successfully created SAE object")
        
    except Exception as e:
        print(f"❌ Failed to create SAE: {e}")
        return False
    
    # 4. 加载本地权重
    print("\n💾 Step 4: Load local weights")
    try:
        sae.load_state_dict(state_dict)
        print("✅ Successfully loaded local weights")
    except Exception as e:
        print(f"❌ Failed to load local weights: {e}")
        return False
    
    # 5. 测试decoder访问
    print("\n🔍 Step 5: Test decoder access")
    try:
        if hasattr(sae, 'decoder'):
            print("✅ SAE has decoder attribute")
            decoder = sae.decoder
            print(f"Decoder type: {type(decoder)}")
            print(f"Decoder in_features: {decoder.in_features}")
            print(f"Decoder out_features: {decoder.out_features}")
        elif hasattr(sae, 'decode'):
            print("✅ SAE has decode method")
            print(f"Decode method: {sae.decode}")
        else:
            print("❌ SAE has neither decoder attribute nor decode method")
            return False
    except Exception as e:
        print(f"❌ Failed to access decoder: {e}")
        return False
    
    # 6. 测试前向传播
    print("\n🚀 Step 6: Test forward pass")
    try:
        # 创建测试输入，使用与SAE相同的设备
        device = sae.device
        test_input = torch.randn(2, 131072, device=device)  # batch_size=2, hidden_dim=131072
        
        if hasattr(sae, 'decode'):
            # 需要top_indices
            batch_size = test_input.shape[0]
            top_indices = []
            for i in range(batch_size):
                # 模拟稀疏表示，取前100个位置
                top_indices.append(torch.arange(100))
            
            # 将top_indices转换为tensor，使用相同设备
            top_indices_tensor = torch.stack([torch.arange(100, device=device) for _ in range(batch_size)])
            
            output = sae.decode(test_input, top_indices_tensor)
            print(f"✅ Decode output shape: {output.shape}")
        else:
            # 直接使用decoder
            output = sae.decoder(test_input)
            print(f"✅ Decoder output shape: {output.shape}")
        
        print("✅ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Failed forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_sae_loading()
    if success:
        print("\n🎉 All tests passed! Local SAE loading is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.") 
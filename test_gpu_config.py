#!/usr/bin/env python3
"""
GPU配置测试脚本
用于验证Stable Diffusion应用的GPU配置是否正确
"""

import sys
import traceback
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_torch_cuda():
    """测试PyTorch CUDA支持"""
    print("=" * 60)
    print("🔍 测试PyTorch CUDA支持")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"🎮 CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"🔧 CUDA版本: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"📊 GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory_gb:.1f}GB)")
                
            return True, "CUDA"
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return True, "CPU"
            
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False, "未安装"

def test_device_detection():
    """测试设备检测逻辑"""
    print("\n" + "=" * 60)
    print("🔍 测试设备检测逻辑")
    print("=" * 60)
    
    try:
        from utils import get_optimal_device
        
        print("调用get_optimal_device()...")
        device = get_optimal_device()
        print(f"✅ 检测到最优设备: {device}")
        
        return True, device
        
    except Exception as e:
        print(f"❌ 设备检测失败: {e}")
        traceback.print_exc()
        return False, "未知"

def test_config_settings():
    """测试配置设置"""
    print("\n" + "=" * 60)
    print("🔍 测试配置设置")
    print("=" * 60)
    
    try:
        from config import config
        
        # 检查当前设备配置
        device_config = config.get("system.device")
        print(f"📋 配置中的设备设置: {device_config}")
        
        # 检查系统配置
        system_config = config.get_system_config()
        print("🔧 系统配置:")
        for key, value in system_config.items():
            print(f"  {key}: {value}")
            
        return True, device_config
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        traceback.print_exc()
        return False, "未知"

def test_cuda_optimization():
    """测试CUDA优化设置"""
    print("\n" + "=" * 60)
    print("🔍 测试CUDA优化设置")
    print("=" * 60)
    
    try:
        from utils import get_cuda_optimization_settings
        
        print("获取CUDA优化设置...")
        settings = get_cuda_optimization_settings()
        print("🚀 CUDA优化设置:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
            
        return True, settings
        
    except Exception as e:
        print(f"❌ CUDA优化设置获取失败: {e}")
        traceback.print_exc()
        return False, {}

def test_generator_initialization():
    """测试生成器初始化"""
    print("\n" + "=" * 60)
    print("🔍 测试生成器初始化")
    print("=" * 60)
    
    try:
        from sd_generator import SDGenerator
        
        print("初始化SDGenerator...")
        generator = SDGenerator()
        
        print(f"✅ 生成器初始化成功")
        print(f"📋 当前设备: {generator.device}")
        print(f"🔧 系统配置: {generator.system_config}")
        
        return True, generator.device
        
    except Exception as e:
        print(f"❌ 生成器初始化失败: {e}")
        traceback.print_exc()
        return False, "未知"

def main():
    """主测试函数"""
    print("🧪 Stable Diffusion GPU配置测试")
    print("=" * 60)
    
    results = {}
    
    # 测试PyTorch CUDA支持
    torch_ok, torch_device = test_torch_cuda()
    results['torch'] = (torch_ok, torch_device)
    
    # 测试设备检测
    device_ok, detected_device = test_device_detection()
    results['device_detection'] = (device_ok, detected_device)
    
    # 测试配置设置
    config_ok, config_device = test_config_settings()
    results['config'] = (config_ok, config_device)
    
    # 测试CUDA优化
    cuda_ok, cuda_settings = test_cuda_optimization()
    results['cuda_optimization'] = (cuda_ok, cuda_settings)
    
    # 测试生成器初始化
    gen_ok, gen_device = test_generator_initialization()
    results['generator'] = (gen_ok, gen_device)
    
    # 显示测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, (success, result) in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status} - {result}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！GPU配置正确")
        
        # 检查是否真的在使用GPU
        if torch_device == "CUDA" and detected_device == "cuda":
            print("✅ 应用程序将使用GPU模式，性能最佳")
        elif torch_device == "CPU":
            print("⚠️ 应用程序将使用CPU模式")
            print("💡 建议：如果您有NVIDIA GPU，请安装CUDA版本的PyTorch")
        else:
            print("⚠️ 设备配置可能存在问题，请检查")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试脚本发生未预期错误: {e}")
        traceback.print_exc()
        sys.exit(1)

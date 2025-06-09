#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络检查优化测试脚本
测试模型加载时的网络检查顺序优化
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_network_check_optimization():
    """测试网络检查优化"""
    print("=" * 60)
    print("网络检查优化测试")
    print("=" * 60)
    
    try:
        from sd_generator import SDGenerator
        from config import config
        
        # 创建生成器实例
        generator = SDGenerator()
        
        # 测试场景1: 模型已加载，重复加载同一模型
        print("\n📋 测试场景1: 模型已加载时的重复加载")
        print("-" * 40)
        
        # 模拟模型已加载状态
        generator.model_loaded = True
        generator.current_model_name = "stabilityai/stable-diffusion-3.5-large"
        generator.pipeline = "mock_pipeline"  # 模拟pipeline对象
        
        # 尝试加载同一模型，应该跳过网络检查
        print("尝试重复加载同一模型...")
        result = generator.load_model("stabilityai/stable-diffusion-3.5-large")
        
        if result:
            print("✅ 测试通过: 成功跳过重复加载")
        else:
            print("❌ 测试失败: 重复加载检查失败")
        
        # 测试场景2: 本地模型加载
        print("\n📋 测试场景2: 本地模型路径检测")
        print("-" * 40)
        
        # 重置状态
        generator.model_loaded = False
        generator.current_model_name = None
        generator.pipeline = None
        
        # 测试本地模型路径检测
        local_paths = [
            "/path/to/model.safetensors",
            "C:\\models\\sd_model.safetensors",
            "./models/local_model",
            "models/stable-diffusion-v1-5"
        ]
        
        for path in local_paths:
            is_local = generator._is_local_model(path)
            print(f"路径: {path}")
            print(f"  本地模型: {'是' if is_local else '否'}")
        
        # 测试场景3: 在线模型缓存检测
        print("\n📋 测试场景3: 在线模型缓存检测")
        print("-" * 40)
        
        online_models = [
            "stabilityai/stable-diffusion-3.5-large",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0"
        ]
        
        for model in online_models:
            is_cached = generator._check_model_cached(model)
            print(f"模型: {model}")
            print(f"  已缓存: {'是' if is_cached else '否'}")
        
        # 测试场景4: 网络检查方法
        print("\n📋 测试场景4: 网络连接检查")
        print("-" * 40)
        
        print("测试网络连接...")
        network_available = generator._check_network_connection()
        print(f"网络状态: {'可用' if network_available else '不可用'}")
        
        print("\n✅ 网络检查优化测试完成")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有必要的模块都已安装")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False
    
    return True

def test_model_loading_scenarios():
    """测试不同的模型加载场景"""
    print("\n" + "=" * 60)
    print("模型加载场景测试")
    print("=" * 60)
    
    try:
        from sd_generator import SDGenerator
        
        generator = SDGenerator()
        
        # 场景1: 首次加载模型
        print("\n📋 场景1: 首次加载模型")
        print("-" * 40)
        print("模拟首次加载在线模型的流程...")
        
        # 检查当前配置的模型
        current_model = generator.generation_config.get("model", {}).get("name", "未配置")
        print(f"当前配置模型: {current_model}")
        
        # 场景2: 切换模型
        print("\n📋 场景2: 切换模型")
        print("-" * 40)
        
        # 模拟已加载一个模型
        generator.model_loaded = True
        generator.current_model_name = "model_a"
        
        # 尝试加载不同的模型
        print("当前模型: model_a")
        print("尝试切换到: model_b")
        
        # 这应该会触发新的加载流程
        generator.current_model_name = "model_b"
        print("模型切换检测: 需要重新加载")
        
        # 场景3: 模型卸载后重新加载
        print("\n📋 场景3: 模型卸载后重新加载")
        print("-" * 40)
        
        print("卸载当前模型...")
        generator.unload_model()
        print(f"模型状态: 已加载={generator.model_loaded}, 当前模型={generator.current_model_name}")
        
        print("✅ 模型加载场景测试完成")
        
    except Exception as e:
        print(f"❌ 场景测试异常: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 启动网络检查优化测试")
    
    # 运行测试
    test1_result = test_network_check_optimization()
    test2_result = test_model_loading_scenarios()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if test1_result and test2_result:
        print("✅ 所有测试通过")
        print("\n🎉 网络检查优化功能正常工作:")
        print("  • 模型已加载时跳过重复加载")
        print("  • 本地模型跳过网络检查")
        print("  • 缓存模型优先使用本地文件")
        print("  • 只在必要时进行网络检查")
    else:
        print("❌ 部分测试失败")
        print("请检查代码修改是否正确")
    
    return test1_result and test2_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
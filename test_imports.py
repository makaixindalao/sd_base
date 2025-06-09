#!/usr/bin/env python3
"""
模块导入测试脚本
用于验证所有模块是否能正确导入
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_military_imports():
    """测试军事模块导入"""
    print("=" * 60)
    print("测试军事模块导入")
    print("=" * 60)
    
    try:
        print("1. 测试 military 包导入...")
        import military
        print("✅ military 包导入成功")
        
        print("2. 测试 military.scene_composer 导入...")
        from military.scene_composer import SceneComposer
        print("✅ military.scene_composer 导入成功")
        
        print("3. 测试 SceneComposer 实例化...")
        composer = SceneComposer()
        print("✅ SceneComposer 实例化成功")
        
        print("4. 测试 SceneComposer 方法...")
        options = composer.get_available_options()
        print(f"✅ 可用选项: {options}")
        
        scene_result = composer.compose_scene({"target": "tank", "weather": "sunny"})
        print(f"✅ 场景合成结果: {scene_result}")
        
        print("5. 测试其他军事模块...")
        from military.target_generator import MilitaryTargetGenerator
        from military.prompt_templates import PromptTemplateManager
        from military.batch_generator import BatchGenerator
        print("✅ 所有军事模块导入成功")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False
    
    return True

def test_other_imports():
    """测试其他模块导入"""
    print("\n" + "=" * 60)
    print("测试其他模块导入")
    print("=" * 60)
    
    modules_to_test = [
        ("annotation.auto_annotator", "AutoAnnotator"),
        ("annotation.coco_formatter", "COCOFormatter"),
        ("dataset.dataset_manager", "DatasetManager"),
        ("dataset.statistics", "DatasetStatistics"),
        ("gui.military_panel", "MilitaryGenerationPanel"),
        ("gui.annotation_panel", "AnnotationPanel"),
        ("gui.dataset_panel", "DatasetPanel"),
    ]
    
    success_count = 0
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
        except Exception as e:
            print(f"⚠️  {module_name}.{class_name}: {e}")
    
    print(f"\n导入成功率: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)

def test_python_path():
    """测试Python路径配置"""
    print("\n" + "=" * 60)
    print("Python路径配置")
    print("=" * 60)
    
    current_dir = Path(__file__).parent
    print(f"当前目录: {current_dir}")
    print(f"工作目录: {Path.cwd()}")
    
    print("\nPython路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print("\n检查关键目录:")
    key_dirs = ["military", "annotation", "dataset", "gui"]
    for dir_name in key_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ 存在")
            init_file = dir_path / "__init__.py"
            if init_file.exists():
                print(f"✅ {dir_name}/__init__.py 存在")
            else:
                print(f"❌ {dir_name}/__init__.py 不存在")
        else:
            print(f"❌ {dir_name}/ 不存在")

def main():
    """主测试函数"""
    print("🔍 模块导入诊断工具")
    print("用于诊断和修复模块导入问题")
    
    # 测试Python路径
    test_python_path()
    
    # 测试军事模块导入
    military_success = test_military_imports()
    
    # 测试其他模块导入
    other_success = test_other_imports()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if military_success and other_success:
        print("🎉 所有模块导入测试通过！")
        print("如果您仍然遇到导入错误，可能是以下原因：")
        print("1. IDE或编辑器的Python解释器配置问题")
        print("2. 虚拟环境配置问题")
        print("3. 缓存的.pyc文件问题")
        print("\n建议解决方案：")
        print("- 重启IDE/编辑器")
        print("- 清理__pycache__目录")
        print("- 检查Python解释器路径")
    else:
        print("❌ 存在模块导入问题")
        print("请检查上述错误信息并修复相关问题")
    
    return 0 if (military_success and other_success) else 1

if __name__ == "__main__":
    sys.exit(main())

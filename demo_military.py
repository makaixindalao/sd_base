#!/usr/bin/env python3
"""
军事目标数据集生成与管理平台演示脚本
展示核心功能的使用方法
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def demo_prompt_templates():
    """演示提示词模板功能"""
    print("=" * 60)
    print("1. 提示词模板演示")
    print("=" * 60)
    
    try:
        from military.prompt_templates import PromptTemplateManager
        
        manager = PromptTemplateManager()
        
        # 显示可用选项
        options = manager.get_available_options()
        print("可用选项:")
        for category, items in options.items():
            print(f"  {category}: {', '.join(items)}")
        
        print("\n生成示例提示词:")
        
        # 生成几个示例
        examples = [
            ("tank", "rainy", "urban"),
            ("aircraft", "night", "island"),
            ("ship", "foggy", "rural")
        ]
        
        for target, weather, terrain in examples:
            positive, negative = manager.get_random_prompt(target, weather, terrain)
            print(f"\n目标: {target}, 天气: {weather}, 地形: {terrain}")
            print(f"正面提示词: {positive[:100]}...")
            print(f"负面提示词: {negative[:100]}...")
        
        # 显示模板统计
        stats = manager.get_template_stats()
        print(f"\n模板统计: {stats}")
        
    except Exception as e:
        print(f"提示词模板演示失败: {e}")

def demo_auto_annotation():
    """演示自动标注功能"""
    print("\n" + "=" * 60)
    print("2. 自动标注演示")
    print("=" * 60)
    
    try:
        from annotation.auto_annotator import AutoAnnotator
        from annotation.coco_formatter import COCOFormatter
        
        # 检查是否有YOLO可用
        annotator = AutoAnnotator()
        if not annotator.is_model_loaded():
            print("⚠️  YOLO模型未加载，自动标注功能不可用")
            print("请安装ultralytics: pip install ultralytics")
            return
        
        print("✅ 自动标注器初始化成功")
        
        # 显示类别映射
        class_mapping = annotator.get_class_mapping()
        print(f"类别映射: {class_mapping}")
        
        # 创建COCO格式转换器
        coco_formatter = COCOFormatter()
        print("✅ COCO格式转换器初始化成功")
        
        # 显示支持的类别
        categories = coco_formatter.categories
        print("支持的军事目标类别:")
        for cat in categories:
            print(f"  ID {cat['id']}: {cat['name']}")
        
    except Exception as e:
        print(f"自动标注演示失败: {e}")

def demo_dataset_management():
    """演示数据集管理功能"""
    print("\n" + "=" * 60)
    print("3. 数据集管理演示")
    print("=" * 60)
    
    try:
        from dataset.dataset_manager import DatasetManager
        from dataset.statistics import DatasetStatistics
        
        # 创建数据集管理器
        manager = DatasetManager()
        print("✅ 数据集管理器初始化成功")
        
        # 显示现有数据集
        datasets = manager.get_dataset_list()
        print(f"现有数据集数量: {len(datasets)}")
        
        for dataset in datasets[:3]:  # 只显示前3个
            print(f"  - {dataset['name']}: {dataset.get('image_count', 0)} 张图像")
        
        # 创建统计分析器
        stats_analyzer = DatasetStatistics()
        print("✅ 统计分析器初始化成功")
        
        # 显示推荐的分割比例
        recommended_splits = stats_analyzer._calculate_balance_score([100, 90, 110])
        print(f"类别平衡分数示例: {recommended_splits:.3f}")
        
    except Exception as e:
        print(f"数据集管理演示失败: {e}")

def demo_military_generator():
    """演示军事目标生成器功能"""
    print("\n" + "=" * 60)
    print("4. 军事目标生成器演示")
    print("=" * 60)
    
    try:
        from military.target_generator import MilitaryTargetGenerator
        
        # 创建生成器（不加载SD模型）
        generator = MilitaryTargetGenerator(sd_generator=None)
        print("✅ 军事目标生成器初始化成功")
        
        # 显示可用选项
        options = generator.get_available_options()
        print("可用生成选项:")
        for category, items in options.items():
            print(f"  {category}: {', '.join(items)}")
        
        # 显示生成统计（空的）
        stats = generator.get_generation_stats()
        print(f"生成统计: {stats}")
        
        print("💡 提示: 需要加载Stable Diffusion模型才能实际生成图像")
        
    except Exception as e:
        print(f"军事目标生成器演示失败: {e}")

def demo_export_tools():
    """演示导出工具功能"""
    print("\n" + "=" * 60)
    print("5. 导出工具演示")
    print("=" * 60)
    
    try:
        from dataset.export_tools import ExportTools
        
        exporter = ExportTools()
        print("✅ 导出工具初始化成功")
        
        # 显示支持的格式
        formats = exporter.get_supported_formats()
        print(f"支持的导出格式: {', '.join(formats)}")
        
        # 显示格式验证
        test_params = ("test.json", "test_images", "test_output", "yolo")
        is_valid, message = exporter.validate_export_params(*test_params)
        print(f"参数验证示例: {message}")
        
    except Exception as e:
        print(f"导出工具演示失败: {e}")

def check_dependencies():
    """检查依赖安装情况"""
    print("=" * 60)
    print("依赖检查")
    print("=" * 60)
    
    dependencies = [
        ("PyQt5", "PyQt5"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("yaml", "PyYAML")
    ]
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (未安装)")
    
    print("\n安装缺失依赖:")
    print("pip install -r requirements.txt")

def show_project_structure():
    """显示项目结构"""
    print("\n" + "=" * 60)
    print("项目结构")
    print("=" * 60)
    
    structure = """
sd-military-dataset/
├── main.py                    # 主入口
├── gui_qt.py                  # 主界面（已扩展）
├── sd_generator.py            # SD生成器
├── config.py                  # 配置管理（已扩展）
├── military/                  # 军事模块
│   ├── target_generator.py    # 军事目标生成器
│   ├── prompt_templates.py    # 提示词模板
│   └── ...
├── annotation/                # 标注模块
│   ├── auto_annotator.py      # 自动标注器
│   ├── coco_formatter.py      # COCO格式转换
│   └── ...
├── dataset/                   # 数据集管理
│   ├── dataset_manager.py     # 数据集CRUD
│   ├── statistics.py          # 统计分析
│   └── ...
├── gui/                       # 扩展界面
│   ├── military_panel.py      # 军事生成面板
│   ├── annotation_panel.py    # 标注管理面板
│   └── dataset_panel.py       # 数据集管理面板
└── requirements.txt           # 依赖列表（已更新）
    """
    
    print(structure)

def main():
    """主演示函数"""
    print("🎯 军事目标数据集生成与管理平台演示")
    print("基于现有Stable Diffusion框架的扩展开发")
    
    # 检查依赖
    check_dependencies()
    
    # 显示项目结构
    show_project_structure()
    
    # 功能演示
    demo_prompt_templates()
    demo_auto_annotation()
    demo_dataset_management()
    demo_military_generator()
    demo_export_tools()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("🚀 启动完整应用: python main.py")
    print("📖 查看文档: README.md")
    print("⚙️  安装依赖: pip install -r requirements.txt")
    
    # 显示核心功能总结
    print("\n🎯 核心功能总结:")
    print("1. ✅ 军事目标图像生成 (基于SD)")
    print("2. ✅ 自动目标检测和标注")
    print("3. ✅ 数据集CRUD管理")
    print("4. ✅ 多格式导出 (COCO/YOLO/VOC)")
    print("5. ✅ 统计分析和可视化")
    print("6. ✅ 现代化PyQt5界面")
    print("7. 🔄 模型训练和微调 (开发中)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
导入问题修复脚本
自动诊断和修复Python模块导入问题
"""

import sys
import os
import shutil
from pathlib import Path
import subprocess

def print_step(step, total, description):
    """打印步骤信息"""
    print(f"\n[{step}/{total}] {description}")
    print("-" * 50)

def print_progress(message, success=None):
    """打印进度信息"""
    if success is True:
        print(f"✅ {message}")
    elif success is False:
        print(f"❌ {message}")
    else:
        print(f"🔄 {message}")

def clean_pycache():
    """清理Python缓存文件"""
    print_step(1, 5, "清理Python缓存文件")
    
    current_dir = Path(__file__).parent
    cache_dirs = []
    pyc_files = []
    
    # 查找所有__pycache__目录和.pyc文件
    for item in current_dir.rglob("*"):
        if item.is_dir() and item.name == "__pycache__":
            cache_dirs.append(item)
        elif item.suffix == ".pyc":
            pyc_files.append(item)
    
    print_progress(f"发现 {len(cache_dirs)} 个缓存目录")
    print_progress(f"发现 {len(pyc_files)} 个.pyc文件")
    
    # 删除缓存目录
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print_progress(f"删除缓存目录: {cache_dir.relative_to(current_dir)}", True)
        except Exception as e:
            print_progress(f"删除缓存目录失败 {cache_dir}: {e}", False)
    
    # 删除.pyc文件
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print_progress(f"删除.pyc文件: {pyc_file.relative_to(current_dir)}", True)
        except Exception as e:
            print_progress(f"删除.pyc文件失败 {pyc_file}: {e}", False)
    
    print_progress("缓存清理完成", True)

def check_file_permissions():
    """检查文件权限"""
    print_step(2, 5, "检查文件权限")
    
    current_dir = Path(__file__).parent
    key_files = [
        "military/__init__.py",
        "military/scene_composer.py",
        "military/target_generator.py",
        "military/prompt_templates.py",
        "military/batch_generator.py"
    ]
    
    all_readable = True
    for file_path in key_files:
        full_path = current_dir / file_path
        if full_path.exists():
            if os.access(full_path, os.R_OK):
                print_progress(f"{file_path} 可读", True)
            else:
                print_progress(f"{file_path} 不可读", False)
                all_readable = False
        else:
            print_progress(f"{file_path} 不存在", False)
            all_readable = False
    
    return all_readable

def verify_module_structure():
    """验证模块结构"""
    print_step(3, 5, "验证模块结构")
    
    current_dir = Path(__file__).parent
    required_structure = {
        "military": ["__init__.py", "scene_composer.py", "target_generator.py", 
                    "prompt_templates.py", "batch_generator.py"],
        "annotation": ["__init__.py", "auto_annotator.py", "coco_formatter.py"],
        "dataset": ["__init__.py", "dataset_manager.py", "statistics.py"],
        "gui": ["__init__.py", "military_panel.py", "annotation_panel.py", "dataset_panel.py"]
    }
    
    structure_valid = True
    for module_name, files in required_structure.items():
        module_dir = current_dir / module_name
        if not module_dir.exists():
            print_progress(f"模块目录 {module_name}/ 不存在", False)
            structure_valid = False
            continue
        
        print_progress(f"检查模块 {module_name}/")
        for file_name in files:
            file_path = module_dir / file_name
            if file_path.exists():
                print_progress(f"  {file_name} 存在", True)
            else:
                print_progress(f"  {file_name} 缺失", False)
                structure_valid = False
    
    return structure_valid

def test_imports():
    """测试模块导入"""
    print_step(4, 5, "测试模块导入")
    
    # 添加当前目录到Python路径
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    test_modules = [
        ("military", "military包"),
        ("military.scene_composer", "场景合成器"),
        ("military.target_generator", "目标生成器"),
        ("military.prompt_templates", "提示词模板"),
        ("military.batch_generator", "批量生成器"),
        ("annotation.auto_annotator", "自动标注器"),
        ("dataset.dataset_manager", "数据集管理器"),
        ("gui.military_panel", "军事面板")
    ]
    
    import_success = True
    for module_name, description in test_modules:
        try:
            __import__(module_name)
            print_progress(f"{description} 导入成功", True)
        except ImportError as e:
            print_progress(f"{description} 导入失败: {e}", False)
            import_success = False
        except Exception as e:
            print_progress(f"{description} 导入异常: {e}", False)
            import_success = False
    
    return import_success

def create_environment_info():
    """创建环境信息文件"""
    print_step(5, 5, "创建环境信息")
    
    current_dir = Path(__file__).parent
    info_file = current_dir / "environment_info.txt"
    
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("Python环境信息\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"Python路径: {sys.executable}\n")
            f.write(f"工作目录: {Path.cwd()}\n")
            f.write(f"脚本目录: {current_dir}\n\n")
            
            f.write("Python路径列表:\n")
            for i, path in enumerate(sys.path):
                f.write(f"  {i}: {path}\n")
            
            f.write("\n已安装的关键包:\n")
            key_packages = ["torch", "diffusers", "PyQt5", "ultralytics", "numpy", "PIL"]
            for package in key_packages:
                try:
                    __import__(package)
                    f.write(f"  ✅ {package}\n")
                except ImportError:
                    f.write(f"  ❌ {package}\n")
        
        print_progress(f"环境信息已保存到: {info_file}", True)
        return True
        
    except Exception as e:
        print_progress(f"创建环境信息失败: {e}", False)
        return False

def main():
    """主修复函数"""
    print("🔧 Python模块导入问题修复工具")
    print("=" * 60)
    
    # 执行修复步骤
    clean_pycache()
    permissions_ok = check_file_permissions()
    structure_ok = verify_module_structure()
    imports_ok = test_imports()
    info_created = create_environment_info()
    
    # 总结结果
    print("\n" + "=" * 60)
    print("修复结果总结")
    print("=" * 60)
    
    if permissions_ok and structure_ok and imports_ok:
        print("🎉 所有检查通过！模块导入应该正常工作。")
        print("\n如果您仍然遇到导入错误，请尝试：")
        print("1. 重启您的IDE或编辑器")
        print("2. 重新启动Python解释器")
        print("3. 检查IDE的Python解释器配置")
        print("4. 确保在正确的工作目录中运行代码")
    else:
        print("❌ 发现问题，请检查上述错误信息")
        if not permissions_ok:
            print("- 文件权限问题")
        if not structure_ok:
            print("- 模块结构问题")
        if not imports_ok:
            print("- 模块导入问题")
    
    print(f"\n📋 详细环境信息已保存到: environment_info.txt")
    
    return 0 if (permissions_ok and structure_ok and imports_ok) else 1

if __name__ == "__main__":
    sys.exit(main())

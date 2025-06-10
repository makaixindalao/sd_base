#!/usr/bin/env python3
"""
军事目标数据集生成平台启动脚本
包含完整的环境检查和自动修复功能
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                军事目标数据集生成平台                          ║
    ║                Military Target Dataset Generator              ║
    ║                                                              ║
    ║  🎯 基于Stable Diffusion的军事目标图像生成                    ║
    ║  🔍 自动目标检测和标注 (YOLO)                                ║
    ║  📊 数据集管理和统计分析                                      ║
    ║  🖥️  现代化PyQt5图形界面                                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """检查必需的包"""
    print("\n🔍 检查必需的包...")
    
    required_packages = {
        'PyQt5': 'PyQt5',
        'torch': 'torch',
        'diffusers': 'diffusers',
        'ultralytics': 'ultralytics',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'yaml': 'PyYAML'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (缺失)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  发现 {len(missing_packages)} 个缺失的包")
        return False, missing_packages
    
    print("✅ 所有必需的包都已安装")
    return True, []

def install_missing_packages(packages):
    """安装缺失的包"""
    print(f"\n📦 正在安装缺失的包: {', '.join(packages)}")
    
    try:
        for package in packages:
            print(f"正在安装 {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"✅ {package} 安装成功")
            else:
                print(f"❌ {package} 安装失败: {result.stderr}")
                return False
        
        print("✅ 所有包安装完成")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ 安装超时")
        return False
    except Exception as e:
        print(f"❌ 安装过程中出错: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    required_dirs = ['military', 'annotation', 'dataset', 'gui']
    required_files = [
        'main.py',
        'gui_qt.py',
        'military/__init__.py',
        'military/target_generator.py',
        'military/scene_composer.py',
        'military/prompt_templates.py',
        'annotation/__init__.py',
        'annotation/auto_annotator.py',
        'dataset/__init__.py',
        'dataset/dataset_manager.py',
        'gui/__init__.py',
        'gui/military_panel.py'
    ]
    
    current_dir = Path(__file__).parent
    missing_items = []
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (缺失)")
            missing_items.append(dir_name)
    
    # 检查文件
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n⚠️  发现 {len(missing_items)} 个缺失的项目文件")
        return False
    
    print("✅ 项目结构完整")
    return True

def test_module_imports():
    """测试模块导入"""
    print("\n🔍 测试模块导入...")
    
    # 添加当前目录到Python路径
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    test_modules = [
        ('military', '军事模块'),
        ('military.target_generator', '目标生成器'),
        ('military.scene_composer', '场景合成器'),
        ('annotation.auto_annotator', '自动标注器'),
        ('dataset.dataset_manager', '数据集管理器'),
        ('gui.military_panel', '军事面板')
    ]
    
    failed_imports = []
    
    for module_name, description in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} 个模块导入失败")
        return False, failed_imports
    
    print("✅ 所有模块导入成功")
    return True, []

def run_auto_fix():
    """运行自动修复"""
    print("\n🔧 运行自动修复...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'fix_imports.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ 自动修复完成")
            return True
        else:
            print(f"❌ 自动修复失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 自动修复超时")
        return False
    except Exception as e:
        print(f"❌ 自动修复出错: {e}")
        return False

def start_application():
    """启动应用程序"""
    print("\n🚀 启动应用程序...")
    
    try:
        # 启动主程序
        subprocess.run([sys.executable, 'main.py'])
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print_banner()
    
    # 环境检查步骤
    checks = [
        ("Python版本", check_python_version),
        ("项目结构", check_project_structure),
    ]
    
    # 执行基础检查
    for check_name, check_func in checks:
        if not check_func():
            print(f"\n❌ {check_name}检查失败，程序无法继续")
            return 1
    
    # 检查包依赖
    packages_ok, missing_packages = check_required_packages()
    if not packages_ok:
        print("\n🤔 是否要自动安装缺失的包? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes', '是']:
            if not install_missing_packages(missing_packages):
                print("\n❌ 包安装失败，程序无法继续")
                return 1
        else:
            print("\n❌ 缺少必需的包，程序无法继续")
            print("请手动安装以下包:")
            for package in missing_packages:
                print(f"  pip install {package}")
            return 1
    
    # 测试模块导入
    imports_ok, failed_imports = test_module_imports()
    if not imports_ok:
        print("\n🔧 检测到模块导入问题，尝试自动修复...")
        if run_auto_fix():
            # 重新测试导入
            imports_ok, failed_imports = test_module_imports()
            if not imports_ok:
                print("\n❌ 自动修复后仍有导入问题")
                for module, error in failed_imports:
                    print(f"  {module}: {error}")
                return 1
        else:
            print("\n❌ 自动修复失败")
            return 1
    
    print("\n🎉 环境检查完成，所有检查都通过！")
    print("=" * 60)
    
    # 启动应用程序
    return 0 if start_application() else 1

if __name__ == "__main__":
    sys.exit(main())

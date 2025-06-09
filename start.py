#!/usr/bin/env python3
"""
Stable Diffusion 图片生成器启动脚本
包含完整的环境检查和自动安装功能，确保可以一次性在新环境上运行
支持Windows和Linux系统，提供详细的安装进度和错误处理
"""

import sys
import os
import subprocess
import platform
import importlib.util
import time
import urllib.request
import urllib.error
from pathlib import Path

# 全局配置
PYTHON_MIN_VERSION = (3, 8)
REQUIREMENTS_FILE = "requirements.txt"
MIRROR_SOURCES = {
    "default": "",
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/",
    "douban": "https://pypi.douban.com/simple/",
    "ustc": "https://pypi.mirrors.ustc.edu.cn/simple/"
}

def print_banner():
    """打印启动横幅"""
    print("=" * 70)
    print("    🎨 Stable Diffusion 图片生成器")
    print("    🚀 AI-Powered Image Generation Tool")
    print("    📦 自动环境配置和启动脚本")
    print("=" * 70)
    print()

def print_step(step_num, total_steps, description):
    """打印步骤信息"""
    print(f"\n[步骤 {step_num}/{total_steps}] {description}")
    print("-" * 50)

def print_progress(message, success=None):
    """打印进度信息"""
    if success is True:
        print(f"✅ {message}")
    elif success is False:
        print(f"❌ {message}")
    else:
        print(f"🔄 {message}")

def check_python_version():
    """检查Python版本"""
    print_progress("检查Python版本...")
    version = sys.version_info

    if version.major != 3 or version.minor < PYTHON_MIN_VERSION[1]:
        print_progress(f"Python版本不兼容: {version.major}.{version.minor}", False)
        print(f"   需要Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}或更高版本")
        print("   请访问 https://www.python.org/downloads/ 下载最新版本")
        return False

    print_progress(f"Python版本: {version.major}.{version.minor}.{version.micro}", True)
    return True

def check_pip():
    """检查pip是否可用"""
    print_progress("检查pip工具...")
    try:
        import pip
        # 获取pip版本
        pip_version = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                   capture_output=True, text=True)
        if pip_version.returncode == 0:
            version_info = pip_version.stdout.strip()
            print_progress(f"pip可用: {version_info}", True)
            return True
        else:
            print_progress("pip命令执行失败", False)
            return False
    except ImportError:
        print_progress("pip不可用", False)
        return False

def upgrade_pip():
    """升级pip到最新版本"""
    print_progress("升级pip到最新版本...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print_progress("pip升级成功", True)
            return True
        else:
            print_progress(f"pip升级失败: {result.stderr}", False)
            return False
    except subprocess.TimeoutExpired:
        print_progress("pip升级超时", False)
        return False
    except Exception as e:
        print_progress(f"pip升级异常: {e}", False)
        return False

def test_network_connectivity():
    """测试网络连接"""
    print_progress("测试网络连接...")
    test_urls = [
        "https://pypi.org",
        "https://pypi.tuna.tsinghua.edu.cn",
        "https://www.python.org"
    ]

    for url in test_urls:
        try:
            urllib.request.urlopen(url, timeout=10)
            print_progress(f"网络连接正常: {url}", True)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            continue

    print_progress("网络连接异常，将使用离线模式", False)
    return False

def detect_best_mirror():
    """检测最佳镜像源"""
    print_progress("检测最佳pip镜像源...")

    # 测试各个镜像源的连接速度
    best_mirror = "default"
    best_time = float('inf')

    for name, url in MIRROR_SOURCES.items():
        if name == "default":
            continue

        try:
            start_time = time.time()
            urllib.request.urlopen(url, timeout=5)
            response_time = time.time() - start_time

            if response_time < best_time:
                best_time = response_time
                best_mirror = name

            print_progress(f"{name}镜像响应时间: {response_time:.2f}秒")
        except:
            print_progress(f"{name}镜像不可用")

    if best_mirror != "default":
        print_progress(f"选择最佳镜像: {best_mirror} ({best_time:.2f}秒)", True)
    else:
        print_progress("使用默认pip源", True)

    return best_mirror

def install_package(package_name, mirror="default", upgrade=False, timeout=300):
    """安装Python包"""
    cmd = [sys.executable, "-m", "pip", "install"]

    if upgrade:
        cmd.append("--upgrade")

    if mirror != "default" and mirror in MIRROR_SOURCES:
        cmd.extend(["-i", MIRROR_SOURCES[mirror]])
        cmd.append("--trusted-host")
        # 提取主机名
        host = MIRROR_SOURCES[mirror].split("//")[1].split("/")[0]
        cmd.append(host)

    cmd.append(package_name)

    print_progress(f"安装 {package_name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            print_progress(f"{package_name} 安装成功", True)
            return True
        else:
            print_progress(f"{package_name} 安装失败", False)
            if result.stderr:
                print(f"   错误信息: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print_progress(f"{package_name} 安装超时", False)
        return False
    except Exception as e:
        print_progress(f"安装 {package_name} 时出错: {e}", False)
        return False

def install_requirements(mirror="default"):
    """安装requirements.txt中的依赖"""
    requirements_path = Path(REQUIREMENTS_FILE)

    if not requirements_path.exists():
        print_progress(f"未找到 {REQUIREMENTS_FILE} 文件", False)
        return False

    print_progress(f"从 {REQUIREMENTS_FILE} 安装依赖...")

    # 读取requirements文件
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f
                          if line.strip() and not line.startswith('#')]
    except Exception as e:
        print_progress(f"读取 {REQUIREMENTS_FILE} 失败: {e}", False)
        return False

    if not requirements:
        print_progress("requirements.txt为空", True)
        return True

    print_progress(f"需要安装 {len(requirements)} 个依赖包")

    # 批量安装
    cmd = [sys.executable, "-m", "pip", "install"]

    if mirror != "default" and mirror in MIRROR_SOURCES:
        cmd.extend(["-i", MIRROR_SOURCES[mirror]])
        cmd.append("--trusted-host")
        host = MIRROR_SOURCES[mirror].split("//")[1].split("/")[0]
        cmd.append(host)

    cmd.extend(["-r", str(requirements_path)])

    try:
        print_progress("开始批量安装依赖...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时

        if result.returncode == 0:
            print_progress("所有依赖安装成功", True)
            return True
        else:
            print_progress("部分依赖安装失败", False)
            if result.stderr:
                print(f"   错误信息: {result.stderr.strip()}")

            # 尝试逐个安装失败的包
            print_progress("尝试逐个安装依赖...")
            success_count = 0
            for req in requirements:
                if install_package(req, mirror):
                    success_count += 1

            print_progress(f"成功安装 {success_count}/{len(requirements)} 个依赖")
            return success_count > len(requirements) * 0.8  # 80%成功率认为可接受

    except subprocess.TimeoutExpired:
        print_progress("依赖安装超时", False)
        return False
    except Exception as e:
        print_progress(f"依赖安装异常: {e}", False)
        return False

def check_critical_dependencies():
    """检查关键依赖是否已安装"""
    print_progress("检查关键依赖...")

    # 关键依赖列表
    critical_deps = {
        "numpy": "数值计算库",
        "PIL": "图像处理库",
        "PyQt5": "现代图形界面库"
    }

    missing_critical = []
    pyqt_available = False

    for module_name, description in critical_deps.items():
        try:
            if module_name == "PyQt5":
                importlib.import_module("PyQt5.QtWidgets")
                pyqt_available = True
            else:
                importlib.import_module(module_name)
            print_progress(f"{module_name} ({description}) 已安装", True)
        except ImportError:
            print_progress(f"{module_name} ({description}) 未安装", False)
            missing_critical.append(module_name)

    # 检查PyQt5是否可用
    if not pyqt_available:
        print_progress("警告: PyQt5不可用，这是必需的GUI框架", False)
        return False, missing_critical
    else:
        print_progress("将使用PyQt5界面", True)

    return len(missing_critical) == 0, missing_critical

def check_optional_dependencies():
    """检查可选依赖是否已安装"""
    print_progress("检查AI相关依赖...")

    # AI相关依赖
    ai_deps = {
        "torch": "PyTorch深度学习框架",
        "diffusers": "Stable Diffusion模型库",
        "transformers": "Transformer模型库",
        "accelerate": "模型加速库"
    }

    missing_ai = []
    for module_name, description in ai_deps.items():
        try:
            importlib.import_module(module_name)
            print_progress(f"{module_name} ({description}) 已安装", True)
        except ImportError:
            print_progress(f"{module_name} ({description}) 未安装")
            missing_ai.append(module_name)

    return len(missing_ai) == 0, missing_ai

def install_dependencies_with_fallback():
    """安装依赖，支持镜像源回退和离线模式"""
    print_progress("开始安装依赖包...")

    # 检测网络和最佳镜像
    has_network = test_network_connectivity()
    if not has_network:
        print_progress("无网络连接，尝试使用已安装的依赖", False)
        return check_minimal_dependencies()

    best_mirror = detect_best_mirror()

    # 首先升级pip
    if not upgrade_pip():
        print_progress("pip升级失败，继续使用当前版本")

    # 检查是否需要特殊处理PyTorch
    torch_installed = False
    try:
        import torch
        torch_installed = True
        print_progress("PyTorch已安装，检查CUDA支持...")
        if not torch.cuda.is_available():
            has_cuda, _ = check_cuda_availability()
            if has_cuda:
                print_progress("检测到NVIDIA GPU但PyTorch不支持CUDA，将重新安装")
                # 卸载现有的CPU版本PyTorch
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"],
                             capture_output=True)
                torch_installed = False
    except ImportError:
        pass

    # 如果PyTorch未安装或需要重新安装，优先安装CUDA版本
    if not torch_installed:
        print_progress("安装PyTorch（优先CUDA版本）...")
        if not install_cuda_pytorch(best_mirror):
            print_progress("PyTorch安装失败", False)
            return False

    # 安装requirements.txt中的其他依赖（排除PyTorch相关）
    success = install_requirements_excluding_torch(best_mirror)

    if not success and best_mirror != "default":
        print_progress("使用镜像源安装失败，尝试默认源...")
        success = install_requirements_excluding_torch("default")

    if not success:
        print_progress("批量安装失败，尝试安装关键依赖...")
        # 尝试安装最关键的依赖
        critical_packages = ["numpy", "Pillow", "requests", "tqdm", "diffusers", "transformers"]
        success_count = 0

        for package in critical_packages:
            if install_package(package, best_mirror):
                success_count += 1
            elif best_mirror != "default":
                if install_package(package, "default"):
                    success_count += 1

        print_progress(f"关键依赖安装: {success_count}/{len(critical_packages)}")
        success = success_count >= len(critical_packages) * 0.75  # 75%成功率

    return success

def install_requirements_excluding_torch(mirror="default"):
    """安装requirements.txt中的依赖，但排除PyTorch相关包"""
    requirements_path = Path(REQUIREMENTS_FILE)

    if not requirements_path.exists():
        print_progress(f"未找到 {REQUIREMENTS_FILE} 文件", False)
        return False

    print_progress(f"从 {REQUIREMENTS_FILE} 安装依赖（排除PyTorch）...")

    # 读取requirements文件并过滤PyTorch相关包
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            all_requirements = [line.strip() for line in f
                              if line.strip() and not line.startswith('#')]

        # 过滤掉PyTorch相关包
        torch_packages = ['torch', 'torchvision', 'torchaudio']
        requirements = []
        for req in all_requirements:
            package_name = req.split('>=')[0].split('==')[0].split('[')[0].strip()
            if package_name.lower() not in torch_packages:
                requirements.append(req)

    except Exception as e:
        print_progress(f"读取 {REQUIREMENTS_FILE} 失败: {e}", False)
        return False

    if not requirements:
        print_progress("没有需要安装的其他依赖", True)
        return True

    print_progress(f"需要安装 {len(requirements)} 个依赖包")

    # 批量安装
    cmd = [sys.executable, "-m", "pip", "install"]

    if mirror != "default" and mirror in MIRROR_SOURCES:
        cmd.extend(["-i", MIRROR_SOURCES[mirror]])
        cmd.append("--trusted-host")
        host = MIRROR_SOURCES[mirror].split("//")[1].split("/")[0]
        cmd.append(host)

    cmd.extend(requirements)

    try:
        print_progress("开始批量安装依赖...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时

        if result.returncode == 0:
            print_progress("所有依赖安装成功", True)
            return True
        else:
            print_progress("部分依赖安装失败", False)
            if result.stderr:
                print(f"   错误信息: {result.stderr.strip()}")

            # 尝试逐个安装失败的包
            print_progress("尝试逐个安装依赖...")
            success_count = 0
            for req in requirements:
                if install_package(req, mirror):
                    success_count += 1

            print_progress(f"成功安装 {success_count}/{len(requirements)} 个依赖")
            return success_count > len(requirements) * 0.8  # 80%成功率认为可接受

    except subprocess.TimeoutExpired:
        print_progress("依赖安装超时", False)
        return False
    except Exception as e:
        print_progress(f"依赖安装异常: {e}", False)
        return False

def check_minimal_dependencies():
    """检查最小依赖是否满足"""
    print_progress("检查最小依赖...")

    minimal_deps = ["numpy", "PIL", "tkinter"]
    available_count = 0

    for dep in minimal_deps:
        try:
            importlib.import_module(dep)
            available_count += 1
            print_progress(f"{dep} 可用", True)
        except ImportError:
            print_progress(f"{dep} 不可用", False)

    success = available_count >= len(minimal_deps)
    if success:
        print_progress("最小依赖满足，可以启动基础功能", True)
    else:
        print_progress("最小依赖不满足，无法启动", False)

    return success

def create_offline_mode_notice():
    """创建离线模式说明"""
    notice = """
=== 离线模式运行 ===
由于网络连接问题，应用程序将在离线模式下运行。

功能限制：
- 无法下载AI模型，需要手动安装
- 部分高级功能可能不可用
- 建议在有网络时重新运行安装

解决方案：
1. 检查网络连接
2. 配置代理设置
3. 手动安装依赖: pip install torch diffusers transformers
"""
    print(notice)
    return True

def check_cuda_availability():
    """检查系统CUDA环境"""
    print_progress("检查系统CUDA环境...")

    # 检查NVIDIA驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_progress("检测到NVIDIA GPU驱动", True)
            # 解析GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print_progress(f"系统CUDA版本: {cuda_version}", True)
                    return True, cuda_version
            return True, "未知版本"
        else:
            print_progress("未检测到NVIDIA GPU或驱动未安装")
            return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_progress("nvidia-smi命令不可用，可能未安装NVIDIA驱动")
        return False, None

def install_cuda_pytorch(mirror="default"):
    """安装CUDA版本的PyTorch"""
    print_progress("尝试安装CUDA版本的PyTorch...")

    # 检查系统CUDA环境
    has_cuda, cuda_version = check_cuda_availability()

    if not has_cuda:
        print_progress("系统不支持CUDA，将安装CPU版本PyTorch")
        return install_package("torch torchvision torchaudio", mirror)

    # 根据CUDA版本选择合适的PyTorch版本
    cuda_pytorch_urls = {
        "cu121": "https://download.pytorch.org/whl/cu121",
        "cu118": "https://download.pytorch.org/whl/cu118",
        "default": "https://download.pytorch.org/whl/cu121"  # 默认使用CUDA 12.1
    }

    # 选择CUDA版本
    if cuda_version and "12.1" in cuda_version:
        pytorch_url = cuda_pytorch_urls["cu121"]
        print_progress("检测到CUDA 12.1，安装对应PyTorch版本")
    elif cuda_version and "11.8" in cuda_version:
        pytorch_url = cuda_pytorch_urls["cu118"]
        print_progress("检测到CUDA 11.8，安装对应PyTorch版本")
    else:
        pytorch_url = cuda_pytorch_urls["default"]
        print_progress("使用默认CUDA 12.1版本PyTorch")

    # 安装CUDA版本PyTorch
    cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", pytorch_url]

    try:
        print_progress("开始安装CUDA版本PyTorch（可能需要几分钟）...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时

        if result.returncode == 0:
            print_progress("CUDA版本PyTorch安装成功", True)
            return True
        else:
            print_progress("CUDA版本PyTorch安装失败，尝试CPU版本", False)
            if result.stderr:
                print(f"   错误信息: {result.stderr.strip()}")
            # 回退到CPU版本
            return install_package("torch torchvision torchaudio", mirror)
    except subprocess.TimeoutExpired:
        print_progress("PyTorch安装超时，尝试CPU版本", False)
        return install_package("torch torchvision torchaudio", mirror)
    except Exception as e:
        print_progress(f"PyTorch安装异常: {e}，尝试CPU版本", False)
        return install_package("torch torchvision torchaudio", mirror)

def check_torch_cuda():
    """检查PyTorch的CUDA支持，并提供详细诊断"""
    print_progress("检查PyTorch与CUDA的集成情况...")
    try:
        import torch
        print_progress(f"PyTorch版本: {torch.__version__}", True)

        if torch.cuda.is_available():
            print_progress("PyTorch与CUDA集成正常", True)
            gpu_count = torch.cuda.device_count()
            print_progress(f"检测到 {gpu_count} 个GPU设备")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_progress(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            print_progress("✅ 您的环境已准备好进行GPU加速", True)
        else:
            print_progress("PyTorch未检测到可用的CUDA设备", False)
            
            # 深入诊断
            has_cuda_driver, cuda_version = check_cuda_availability()
            if has_cuda_driver:
                print("\n" + "="*20 + "【诊断信息】" + "="*20)
                print("系统检测到NVIDIA GPU驱动，但PyTorch无法使用它。")
                print("这通常意味着您安装了CPU版本的PyTorch。")
                print(f"检测到的驱动CUDA版本: {cuda_version or '未知'}")
                print("\n【修复建议】")
                print("1. 卸载当前的PyTorch: ")
                print(f"   {sys.executable} -m pip uninstall torch torchvision torchaudio")
                print("\n2. 访问PyTorch官网获取正确的安装命令:")
                print("   https://pytorch.org/get-started/locally/")
                print("   请根据您的系统和CUDA版本选择合适的命令进行安装。")
                print("\n3. 或者，您可以让此脚本尝试自动为您安装:")
                print("   删除venv虚拟环境(如果使用)，然后重新运行此脚本。")
                print("="*52 + "\n")
            else:
                print("未检测到NVIDIA驱动，程序将以CPU模式运行。")
                print("如需GPU加速，请先安装NVIDIA官方驱动。")

        return True
    except ImportError:
        print_progress("PyTorch未安装，将在后续步骤中处理")
        return True # 不阻止程序继续

def check_system_resources():
    """检查系统资源"""
    print_progress("检查系统资源...")

    # 获取系统信息
    system_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version()
    }

    print_progress(f"操作系统: {system_info['platform']} {system_info['machine']}")
    print_progress(f"Python版本: {system_info['python_version']}")

    try:
        import psutil

        # 内存检查
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3

        print_progress(f"系统内存: {memory_gb:.1f}GB (可用: {available_gb:.1f}GB)")

        if memory_gb < 8:
            print_progress("内存可能不足，建议至少8GB内存")
        else:
            print_progress("内存充足", True)

        # 磁盘空间检查
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1024**3
        total_gb = disk.total / 1024**3

        print_progress(f"磁盘空间: {total_gb:.1f}GB (可用: {free_gb:.1f}GB)")

        if free_gb < 10:
            print_progress("磁盘空间可能不足，建议至少10GB可用空间")
        else:
            print_progress("磁盘空间充足", True)

        return True
    except ImportError:
        print_progress("无法检查详细系统资源，psutil未安装")
        return True

def check_main_modules():
    """检查主要模块是否存在"""
    print_progress("检查应用模块...")
    required_files = ["main.py", "gui_qt.py", "sd_generator.py", "config.py", "utils.py"]

    missing_files = []
    for file_name in required_files:
        if Path(file_name).exists():
            print_progress(f"{file_name} 存在", True)
        else:
            print_progress(f"{file_name} 缺失", False)
            missing_files.append(file_name)

    if missing_files:
        print_progress(f"缺失关键文件: {', '.join(missing_files)}", False)
        return False

    print_progress("所有应用模块完整", True)
    return True

def create_directories():
    """创建必要的目录"""
    print_progress("创建必要目录...")
    directories = ["outputs", "logs"]

    created_count = 0
    for dir_name in directories:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True)
            print_progress(f"目录 {dir_name}/ 已准备", True)
            created_count += 1
        except Exception as e:
            print_progress(f"创建目录 {dir_name}/ 失败: {e}", False)

    return created_count == len(directories)

def run_application():
    """运行主应用程序"""
    print_step(7, 7, "启动应用程序")

    try:
        print_progress("导入主程序模块...")

        # 添加当前目录到Python路径
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        # 导入PyQt5主程序
        import main
        print_progress("PyQt5主程序模块导入成功", True)

        print("\n" + "=" * 70)
        print("🎉 环境配置完成，正在启动图形界面...")
        print("=" * 70)

        # 运行主程序
        exit_code = main.main()

        if exit_code == 0:
            print_progress("应用程序正常退出", True)
        else:
            print_progress(f"应用程序异常退出，代码: {exit_code}", False)

        return exit_code

    except ImportError as e:
        print_progress(f"导入主程序失败: {e}", False)
        print("   请确保所有依赖已正确安装")
        return 1
    except Exception as e:
        print_progress(f"启动失败: {e}", False)
        import traceback
        traceback.print_exc()
        return 1

def show_installation_summary(success_steps, total_steps):
    """显示安装总结"""
    print("\n" + "=" * 70)
    print("📋 安装总结")
    print("=" * 70)

    success_rate = (success_steps / total_steps) * 100

    if success_rate == 100:
        print("🎉 所有检查和安装步骤都成功完成！")
        print("✅ 环境配置完整，应用程序可以正常运行")
    elif success_rate >= 80:
        print("✅ 大部分检查通过，应用程序应该可以正常运行")
        print("⚠️  部分非关键组件可能需要手动处理")
    else:
        print("⚠️  多个关键步骤失败，应用程序可能无法正常运行")
        print("❌ 建议检查错误信息并手动解决问题")

    print(f"\n📊 成功率: {success_steps}/{total_steps} ({success_rate:.1f}%)")

    if success_rate < 100:
        print("\n🔧 故障排除建议:")
        print("1. 检查网络连接是否正常")
        print("2. 尝试使用管理员权限运行")
        print("3. 手动安装失败的依赖包")
        print("4. 查看详细错误信息并搜索解决方案")

    return success_rate >= 80

def main():
    """主函数"""
    print_banner()

    print("🚀 开始自动环境配置和应用启动...")

    # 第一阶段：基础环境检查
    print_step(1, 7, "基础环境检查")

    if not check_python_version():
        print_progress("Python版本检查失败，无法继续", False)
        return 1

    if not check_pip():
        print_progress("pip工具检查失败，无法继续", False)
        return 1

    # 第二阶段：应用文件检查
    print_step(2, 7, "应用文件检查")

    if not check_main_modules():
        print_progress("关键应用文件缺失，无法继续", False)
        return 1

    # 第三阶段：创建目录
    print_step(3, 7, "创建工作目录")

    if not create_directories():
        print_progress("目录创建失败，但不影响继续执行")

    # 第四阶段：检查现有依赖
    print_step(4, 7, "检查现有依赖")

    critical_ok, _ = check_critical_dependencies()
    optional_ok, _ = check_optional_dependencies()

    # 第五阶段：安装缺失依赖
    need_install = not critical_ok or not optional_ok

    if need_install:
        print_step(5, 7, "安装缺失依赖")

        install_success = install_dependencies_with_fallback()

        if not install_success:
            print_progress("依赖安装失败，检查最小依赖...", False)

            # 检查是否至少有基础依赖
            if check_minimal_dependencies():
                print_progress("最小依赖满足，将以受限模式启动")
                create_offline_mode_notice()
            else:
                print_progress("关键依赖缺失，无法启动应用", False)
                print("\n🔧 解决方案:")
                print("1. 检查网络连接")
                print("2. 手动安装: pip install numpy Pillow tkinter")
                print("3. 使用管理员权限运行")
                return 1
    else:
        print_step(5, 7, "跳过依赖安装（已满足）")
        print_progress("所有依赖已安装", True)

    # 第六阶段：系统资源和环境检查
    print_step(6, 7, "系统环境检查")

    check_system_resources()
    check_torch_cuda()

    # 第七阶段：启动应用程序
    return run_application()

if __name__ == "__main__":
    try:
        # 设置控制台编码（Windows兼容性）
        if platform.system() == "Windows":
            try:
                import locale
                locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
            except:
                pass

        # 运行主程序
        exit_code = main()

        # 根据退出代码显示不同信息
        if exit_code == 0:
            print("\n" + "=" * 70)
            print("🎉 程序运行完成！感谢使用 Stable Diffusion 图片生成器")
            print("=" * 70)
        else:
            print(f"\n⚠️  程序异常退出，代码: {exit_code}")
            print("如需帮助，请查看日志文件或联系技术支持")

            # 在Windows下暂停，方便查看错误信息
            if platform.system() == "Windows":
                input("\n按回车键退出...")

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序执行")
        print("程序已安全退出")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ 启动脚本发生未预期错误: {e}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()

        print("\n🔧 故障排除建议:")
        print("1. 确保Python版本为3.8或更高")
        print("2. 检查网络连接是否正常")
        print("3. 尝试使用管理员权限运行")
        print("4. 删除__pycache__目录后重试")

        if platform.system() == "Windows":
            input("\n按回车键退出...")

        sys.exit(1)

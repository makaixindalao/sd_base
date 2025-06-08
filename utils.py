"""
工具模块
提供图片处理、文件操作、系统检测等工具函数
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image

# 可选依赖
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """设置日志系统"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / "sd_generator.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def get_system_info() -> dict:
    """获取系统信息"""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # 系统资源信息（需要psutil）
    if PSUTIL_AVAILABLE:
        info["cpu_count"] = psutil.cpu_count()
        info["memory_total"] = psutil.virtual_memory().total
        info["memory_available"] = psutil.virtual_memory().available
    else:
        info["cpu_count"] = os.cpu_count() or 1
        info["memory_total"] = "未知"
        info["memory_available"] = "未知"

    # GPU信息（需要torch）
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    else:
        info["cuda_available"] = False

    return info

def check_dependencies() -> Tuple[bool, List[str]]:
    """检查依赖是否安装"""
    required_packages = [
        "torch",
        "diffusers",
        "transformers",
        "PIL",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def get_optimal_device() -> str:
    """获取最优设备"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # 检查GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory >= 6 * 1024**3:  # 6GB
            return "cuda"
        else:
            print(f"GPU内存不足 ({gpu_memory / 1024**3:.1f}GB < 6GB)，建议使用CPU模式")
            return "cpu"
    else:
        return "cpu"

def format_memory_size(size_bytes: int) -> str:
    """格式化内存大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def save_image_with_metadata(image: Image.Image, filepath: str, metadata: dict = None):
    """保存图片并添加元数据"""
    # 确保目录存在
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # 添加元数据到图片
    if metadata:
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        for key, value in metadata.items():
            png_info.add_text(key, str(value))
        image.save(filepath, pnginfo=png_info)
    else:
        image.save(filepath)

def resize_image_for_display(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """调整图片大小以适应显示"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """验证提示词"""
    if not prompt or not prompt.strip():
        return False, "提示词不能为空"
    
    if len(prompt) > 1000:
        return False, "提示词过长，请控制在1000字符以内"
    
    # 检查是否包含不当内容的基本关键词
    inappropriate_keywords = ["nude", "naked", "nsfw", "porn", "sex"]
    prompt_lower = prompt.lower()
    for keyword in inappropriate_keywords:
        if keyword in prompt_lower:
            return False, f"提示词包含不当内容: {keyword}"
    
    return True, ""

def generate_filename(prompt: str, seed: int, timestamp: str = None) -> str:
    """生成文件名"""
    import re
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 清理提示词，只保留字母数字和空格
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt)
    clean_prompt = re.sub(r'\s+', '_', clean_prompt.strip())
    
    # 限制长度
    if len(clean_prompt) > 50:
        clean_prompt = clean_prompt[:50]
    
    filename = f"{timestamp}_{clean_prompt}_seed{seed}.png"
    return filename

def check_disk_space(path: str, required_gb: float = 1.0) -> bool:
    """检查磁盘空间"""
    if not PSUTIL_AVAILABLE:
        return True  # 如果psutil不可用，假设空间足够

    try:
        free_space = psutil.disk_usage(path).free
        required_bytes = required_gb * 1024**3
        return free_space >= required_bytes
    except:
        return True  # 如果检查失败，假设空间足够

def install_package(package_name: str) -> bool:
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def create_desktop_shortcut(app_path: str, shortcut_name: str = "SD图片生成器"):
    """创建桌面快捷方式（仅Windows）"""
    if platform.system() != "Windows":
        return False
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, f"{shortcut_name}.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{app_path}"'
        shortcut.WorkingDirectory = os.path.dirname(app_path)
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        return True
    except ImportError:
        print("需要安装 pywin32 和 winshell 来创建快捷方式")
        return False
    except Exception as e:
        print(f"创建快捷方式失败: {e}")
        return False

# 初始化日志
logger = setup_logging()

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
    """获取最优设备 - 优化CUDA选择策略，优先使用GPU"""
    if not TORCH_AVAILABLE:
        print("PyTorch未安装，使用CPU模式")
        return "cpu"

    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU模式")
        print("提示: 如果您有NVIDIA GPU，请安装CUDA版本的PyTorch以获得更好性能")
        return "cpu"

    # 获取GPU信息
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("未检测到CUDA设备，使用CPU模式")
        return "cpu"

    # 选择最佳GPU（通常是第一个）
    device_id = 0
    gpu_props = torch.cuda.get_device_properties(device_id)
    gpu_memory_gb = gpu_props.total_memory / (1024**3)
    gpu_name = gpu_props.name

    print(f"🎮 检测到GPU: {gpu_name}")
    print(f"💾 GPU内存: {gpu_memory_gb:.1f}GB")

    # 更积极的CUDA选择策略 - 只要有GPU就尝试使用
    if gpu_memory_gb >= 6.0:
        print(f"✅ GPU内存充足({gpu_memory_gb:.1f}GB >= 6GB)，使用CUDA加速，性能最佳")
        return "cuda"
    elif gpu_memory_gb >= 4.0:
        print(f"✅ GPU内存良好({gpu_memory_gb:.1f}GB >= 4GB)，使用CUDA加速")
        return "cuda"
    elif gpu_memory_gb >= 2.0:
        print(f"⚠️ GPU内存较少({gpu_memory_gb:.1f}GB)，使用CUDA但将启用低显存优化")
        return "cuda"
    elif gpu_memory_gb >= 1.0:
        print(f"⚠️ GPU内存很少({gpu_memory_gb:.1f}GB)，使用CUDA但性能可能受限")
        return "cuda"
    else:
        print(f"❌ GPU内存不足({gpu_memory_gb:.1f}GB < 1GB)，建议使用CPU模式")
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

def get_cuda_optimization_settings(device: str = None) -> dict:
    """获取CUDA优化设置建议"""
    settings = {
        "attention_slicing": True,
        "cpu_offload": False,
        "low_vram_mode": False,
        "use_fp16": False,
        "use_bf16": False,
        "sequential_cpu_offload": False
    }
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return settings
    
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / (1024**3)
        gpu_name = gpu_props.name.lower()
        
        # 根据GPU内存调整设置
        if gpu_memory_gb >= 12:
            # 高端GPU (12GB+)
            settings.update({
                "attention_slicing": False,  # 关闭注意力切片以获得更好性能
                "cpu_offload": False,
                "low_vram_mode": False,
                "use_bf16": True,  # 使用bfloat16获得更好性能
                "sequential_cpu_offload": False
            })
        elif gpu_memory_gb >= 8:
            # 中高端GPU (8-12GB)
            settings.update({
                "attention_slicing": True,
                "cpu_offload": False,
                "low_vram_mode": False,
                "use_bf16": True,
                "sequential_cpu_offload": False
            })
        elif gpu_memory_gb >= 6:
            # 中端GPU (6-8GB)
            settings.update({
                "attention_slicing": True,
                "cpu_offload": True,  # 启用CPU卸载
                "low_vram_mode": False,
                "use_fp16": True,  # 使用fp16节省内存
                "sequential_cpu_offload": False
            })
        elif gpu_memory_gb >= 4:
            # 中低端GPU (4-6GB)
            settings.update({
                "attention_slicing": True,
                "cpu_offload": True,
                "low_vram_mode": True,  # 启用低显存模式
                "use_fp16": True,
                "sequential_cpu_offload": False
            })
        else:
            # 低端GPU (<4GB)
            settings.update({
                "attention_slicing": True,
                "cpu_offload": True,
                "low_vram_mode": True,
                "use_fp16": True,
                "sequential_cpu_offload": True  # 启用顺序CPU卸载
            })
        
        # 特定GPU优化
        if "rtx" in gpu_name and ("30" in gpu_name or "40" in gpu_name):
            # RTX 30/40系列支持更好的bf16
            settings["use_bf16"] = True
            settings["use_fp16"] = False
        
        return settings
        
    except Exception as e:
        print(f"获取CUDA优化设置时出错: {e}")
        return settings

# 初始化日志
logger = setup_logging()

#!/usr/bin/env python3
"""
GPU版本PyTorch安装脚本
自动检测系统环境并安装合适的CUDA版本PyTorch
"""

import sys
import subprocess
import platform

def print_step(message):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"🔧 {message}")
    print('='*60)

def print_info(message, success=None):
    """打印信息"""
    if success is True:
        print(f"✅ {message}")
    elif success is False:
        print(f"❌ {message}")
    else:
        print(f"ℹ️ {message}")

def check_nvidia_gpu():
    """检查NVIDIA GPU和驱动"""
    print_step("检查NVIDIA GPU环境")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_info("检测到NVIDIA GPU驱动", True)
            
            # 解析CUDA版本
            lines = result.stdout.split('\n')
            cuda_version = None
            gpu_info = []
            
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                elif 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    # 提取GPU信息
                    parts = line.split('|')
                    if len(parts) >= 2:
                        gpu_name = parts[1].strip()
                        if gpu_name and ('GeForce' in gpu_name or 'RTX' in gpu_name or 'GTX' in gpu_name):
                            gpu_info.append(gpu_name)
            
            if cuda_version:
                print_info(f"系统CUDA版本: {cuda_version}", True)
            
            for gpu in gpu_info:
                print_info(f"检测到GPU: {gpu}", True)
                
            return True, cuda_version
        else:
            print_info("nvidia-smi命令执行失败", False)
            return False, None
            
    except subprocess.TimeoutExpired:
        print_info("nvidia-smi命令超时", False)
        return False, None
    except FileNotFoundError:
        print_info("nvidia-smi命令不存在，可能未安装NVIDIA驱动", False)
        return False, None
    except Exception as e:
        print_info(f"检查GPU时出错: {e}", False)
        return False, None

def check_current_pytorch():
    """检查当前PyTorch版本"""
    print_step("检查当前PyTorch版本")
    
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_info(f"当前PyTorch版本: {version}")
        print_info(f"CUDA支持: {cuda_available}", cuda_available)
        
        if cuda_available:
            print_info(f"PyTorch CUDA版本: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print_info(f"可用GPU数量: {gpu_count}")
            
        return True, cuda_available
        
    except ImportError:
        print_info("PyTorch未安装", False)
        return False, False

def uninstall_pytorch():
    """卸载现有PyTorch"""
    print_step("卸载现有PyTorch版本")
    
    packages_to_remove = ['torch', 'torchvision', 'torchaudio']
    
    for package in packages_to_remove:
        try:
            print_info(f"卸载 {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'uninstall', package, '-y'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print_info(f"{package} 卸载成功", True)
            else:
                print_info(f"{package} 卸载失败或未安装")
                
        except Exception as e:
            print_info(f"卸载 {package} 时出错: {e}", False)

def install_cuda_pytorch(cuda_version=None):
    """安装CUDA版本PyTorch"""
    print_step("安装CUDA版本PyTorch")
    
    # 根据CUDA版本选择安装URL
    if cuda_version and "12.1" in cuda_version:
        index_url = "https://download.pytorch.org/whl/cu121"
        print_info("选择CUDA 12.1版本PyTorch")
    elif cuda_version and "11.8" in cuda_version:
        index_url = "https://download.pytorch.org/whl/cu118"
        print_info("选择CUDA 11.8版本PyTorch")
    else:
        index_url = "https://download.pytorch.org/whl/cu121"
        print_info("使用默认CUDA 12.1版本PyTorch")
    
    # 安装命令
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', index_url
    ]
    
    try:
        print_info("开始安装CUDA版本PyTorch（可能需要几分钟）...")
        print_info("安装命令: " + " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print_info("CUDA版本PyTorch安装成功", True)
            return True
        else:
            print_info("CUDA版本PyTorch安装失败", False)
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_info("安装超时", False)
        return False
    except Exception as e:
        print_info(f"安装时出错: {e}", False)
        return False

def verify_installation():
    """验证安装结果"""
    print_step("验证安装结果")
    
    try:
        # 重新导入torch
        if 'torch' in sys.modules:
            del sys.modules['torch']
        if 'torchvision' in sys.modules:
            del sys.modules['torchvision']
        if 'torchaudio' in sys.modules:
            del sys.modules['torchaudio']
            
        import torch
        
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_info(f"新PyTorch版本: {version}")
        print_info(f"CUDA支持: {cuda_available}", cuda_available)
        
        if cuda_available:
            print_info(f"PyTorch CUDA版本: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print_info(f"可用GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                print_info(f"GPU {i}: {gpu_name} ({gpu_memory_gb:.1f}GB)")
                
            return True
        else:
            print_info("安装完成但CUDA仍不可用，可能需要重启Python环境", False)
            return False
            
    except ImportError as e:
        print_info(f"验证失败，PyTorch导入错误: {e}", False)
        return False
    except Exception as e:
        print_info(f"验证时出错: {e}", False)
        return False

def main():
    """主函数"""
    print("🚀 GPU版本PyTorch自动安装脚本")
    print("=" * 60)
    
    # 检查操作系统
    if platform.system() not in ['Windows', 'Linux']:
        print_info("此脚本主要支持Windows和Linux系统", False)
        return 1
    
    # 检查NVIDIA GPU
    has_gpu, cuda_version = check_nvidia_gpu()
    if not has_gpu:
        print_info("未检测到NVIDIA GPU或驱动，无法安装CUDA版本PyTorch", False)
        print_info("请先安装NVIDIA驱动程序，或继续使用CPU版本", False)
        return 1
    
    # 检查当前PyTorch
    pytorch_installed, has_cuda = check_current_pytorch()
    
    if pytorch_installed and has_cuda:
        print_info("当前已安装CUDA版本PyTorch，无需重新安装", True)
        return 0
    
    # 确认是否继续
    if pytorch_installed:
        response = input("\n是否要卸载当前PyTorch并安装CUDA版本？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print_info("用户取消安装")
            return 0
    
    # 卸载现有版本
    if pytorch_installed:
        uninstall_pytorch()
    
    # 安装CUDA版本
    if not install_cuda_pytorch(cuda_version):
        print_info("安装失败", False)
        return 1
    
    # 验证安装
    if verify_installation():
        print_step("安装完成")
        print_info("🎉 CUDA版本PyTorch安装成功！", True)
        print_info("现在可以使用GPU加速进行Stable Diffusion图片生成", True)
        print_info("建议重启应用程序以确保新配置生效", True)
        return 0
    else:
        print_info("安装可能存在问题，建议重启Python环境后重试", False)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        
        if exit_code == 0:
            print("\n" + "=" * 60)
            print("✅ 安装完成！请运行以下命令验证配置：")
            print("python test_gpu_config.py")
            print("=" * 60)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️ 安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装脚本发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

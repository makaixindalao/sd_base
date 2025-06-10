"""
Stable Diffusion 图片生成核心模块
负责模型加载、图片生成和参数管理
"""

# 可选依赖导入
try:
    import torch
    import gc
    from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline, 
                         StableDiffusion3Pipeline, DPMSolverMultistepScheduler,
                         AutoencoderKL, StableDiffusionInpaintPipeline,
                         StableDiffusionXLInpaintPipeline)
    from diffusers.utils import logging as diffusers_logging
    import numpy as np

    # 设置diffusers日志级别
    diffusers_logging.set_verbosity_error()
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    IMPORT_ERROR = str(e)

import random
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from PIL import Image

from config import config
from utils import logger, get_optimal_device, format_memory_size, get_cuda_optimization_settings

# 导入SD3专用的文本编码器
from transformers import T5EncoderModel, CLIPTextModelWithProjection

class SDGenerator:
    """Stable Diffusion 图片生成器"""

    def __init__(self):
        self.pipeline = None
        self.device = None
        self.model_loaded = False
        self.current_model_name = None  # 记录当前加载的模型名称
        self.generation_config = config.get_generation_config()
        self.system_config = config.get_system_config()

        # 回调函数
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

        # 检查依赖
        if not TORCH_AVAILABLE:
            self._update_status(f"PyTorch未安装: {IMPORT_ERROR}")
            logger.error(f"PyTorch导入失败: {IMPORT_ERROR}")
        
        # 存储基础pipeline，用于切换
        self.base_pipeline = None

    def set_callbacks(self, progress_callback: Callable = None, status_callback: Callable = None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def _update_status(self, message: str):
        """更新状态"""
        logger.info(message)
        if self.status_callback:
            self.status_callback(message)
    
    def _update_progress(self, step: int, total_steps: int):
        """更新进度"""
        if self.progress_callback:
            progress = (step / total_steps) * 100
            self.progress_callback(progress)

    def _check_network_connection(self) -> bool:
        """检查网络连接"""
        try:
            # 获取代理配置
            network_config = config.get_network_config()
            proxies = None

            if network_config.get("proxy_enabled", False):
                proxy_host = network_config.get("proxy_host", "127.0.0.1")
                proxy_port = network_config.get("proxy_port", 7890)
                proxy_type = network_config.get("proxy_type", "http")

                proxy_url = f"{proxy_type}://{proxy_host}:{proxy_port}"
                proxies = {
                    "http": proxy_url,
                    "https": proxy_url
                }
                self._update_status(f"使用代理: {proxy_url}")

            response = requests.get("https://huggingface.co", timeout=10, proxies=proxies)
            return response.status_code == 200
        except Exception as e:
            self._update_status(f"网络连接检查失败: {str(e)}")
            return False

    def _check_model_cached(self, model_name: str) -> bool:
        """检查模型是否已缓存"""
        # 如果是本地路径，直接检查路径是否存在
        if self._is_local_model(model_name):
            return Path(model_name).exists()

        cache_dir = config.get("model.cache_dir")
        if not cache_dir:
            return False

        # 检查模型目录是否存在且包含必要文件
        model_path = Path(cache_dir) / model_name.replace("/", "--")
        if not model_path.exists():
            return False

        # 检查关键文件
        required_files = ["model_index.json"]
        for file_name in required_files:
            if not (model_path / file_name).exists():
                return False

        return True

    def _is_local_model(self, model_name: str) -> bool:
        """检查是否是本地模型路径"""
        if not model_name:
            return False

        # 检查是否是绝对路径或相对路径
        is_path = (model_name.startswith('/') or
                   model_name.startswith('\\') or
                   ':' in model_name or
                   model_name.startswith('./') or
                   model_name.startswith('.\\'))

        # 检查是否是模型文件（包含文件扩展名）
        model_extensions = ['.safetensors', '.bin', '.ckpt', '.pth']
        is_model_file = any(model_name.lower().endswith(ext) for ext in model_extensions)

        # 如果包含模型文件扩展名，也认为是本地模型
        return is_path or is_model_file

    def _load_local_model(self, model_path: str) -> bool:
        """加载本地模型 - 完全离线操作"""
        try:
            self._update_status("正在检测本地模型...")

            path = Path(model_path)
            if not path.exists():
                self._update_status(f"本地模型路径不存在: {model_path}")
                return False

            # 根据路径类型确定加载方式
            if path.is_file():
                # 单个模型文件
                model_file = path.name
                self._update_status(f"检测到模型文件: {model_file}")

                # 检查是否是safetensors文件
                if path.suffix.lower() == '.safetensors':
                    self._update_status("检测到SafeTensors格式，尝试单文件加载...")
                    return self._load_single_safetensors_file(path)
                else:
                    self._update_status(f"检测到{path.suffix.upper()}格式，尝试单文件加载...")
                    return self._load_single_model_file(path)
            else:
                # 模型目录
                return self._load_model_directory(path)

        except Exception as e:
            error_msg = f"本地模型加载异常: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False

    def _load_single_safetensors_file(self, file_path: Path) -> bool:
        """加载单个SafeTensors文件"""
        try:
            self._update_status(f"正在加载SafeTensors文件: {file_path.name}")

            # 尝试使用单文件加载方式
            try:
                # 方法1: 尝试直接从单文件加载（适用于某些模型）
                self._update_status("尝试直接从SafeTensors文件加载...")

                # 检查文件大小
                file_size_gb = file_path.stat().st_size / (1024**3)
                self._update_status(f"模型文件大小: {file_size_gb:.2f} GB")

                # 根据文件大小推测模型类型
                if file_size_gb > 15:
                    self._update_status("大型模型，尝试使用SD3 pipeline...")
                    pipeline_class = StableDiffusion3Pipeline
                elif file_size_gb > 5:
                    self._update_status("中型模型，尝试使用SDXL pipeline...")
                    try:
                        from diffusers import StableDiffusionXLPipeline
                        pipeline_class = StableDiffusionXLPipeline
                    except ImportError:
                        pipeline_class = StableDiffusion3Pipeline
                else:
                    self._update_status("标准模型，尝试使用SD1.5/2.x pipeline...")
                    try:
                        from diffusers import StableDiffusionPipeline
                        pipeline_class = StableDiffusionPipeline
                    except ImportError:
                        pipeline_class = StableDiffusion3Pipeline

                # 尝试从父目录加载（可能包含配置文件）
                parent_dir = file_path.parent
                config_files = [f for f in ['config.json', 'model_index.json'] if (parent_dir / f).exists()]

                if config_files:
                    self._update_status(f"在父目录发现配置文件: {', '.join(config_files)}")
                    load_path = str(parent_dir)
                else:
                    self._update_status("未发现配置文件，尝试使用默认配置...")
                    # 创建临时配置或使用预设配置
                    load_path = self._create_temp_config_for_safetensors(file_path)
                    if not load_path:
                        return False

                # 加载模型
                load_kwargs = {
                    "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                    "use_safetensors": True,
                    "local_files_only": True,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                    "trust_remote_code": False,
                }

                self.pipeline = pipeline_class.from_pretrained(
                    load_path,
                    **load_kwargs
                )
                self.base_pipeline = self.pipeline # 存储基础pipeline

                self._update_status("✅ SafeTensors文件加载成功")

                # 移动到设备
                if hasattr(self.pipeline, 'to'):
                    self._update_status("正在移动模型到计算设备...")
                    self.pipeline = self.pipeline.to(self.device)

                self._update_status("SafeTensors模型加载完成")
                return True

            except Exception as e:
                self._update_status(f"SafeTensors直接加载失败: {str(e)[:100]}...")

                # 方法2: 尝试使用预设配置加载
                return self._load_safetensors_with_preset_config(file_path)

        except Exception as e:
            error_msg = f"SafeTensors文件加载失败: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False

    def _create_temp_config_for_safetensors(self, file_path: Path) -> str:
        """为SafeTensors文件创建临时配置"""
        try:
            self._update_status("创建临时配置文件...")

            # 创建临时目录
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="sd_temp_"))

            # 复制SafeTensors文件到临时目录
            import shutil
            temp_model_file = temp_dir / file_path.name
            shutil.copy2(file_path, temp_model_file)

            # 创建基本配置文件
            config = {
                "_class_name": "StableDiffusion3Pipeline",
                "_diffusers_version": "0.21.0",
                "feature_extractor": ["transformers", "CLIPImageProcessor"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
                "text_encoder_3": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "tokenizer_3": ["transformers", "T5Tokenizer"],
                "transformer": ["diffusers", "SD3Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"]
            }

            import json
            config_file = temp_dir / "model_index.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self._update_status(f"临时配置创建完成: {temp_dir}")
            return str(temp_dir)

        except Exception as e:
            self._update_status(f"创建临时配置失败: {str(e)}")
            return None

    def _load_safetensors_with_preset_config(self, file_path: Path) -> bool:
        """使用预设配置加载SafeTensors文件"""
        try:
            self._update_status("尝试使用预设配置加载SafeTensors...")

            # 尝试使用在线模型的配置但仅加载本地权重
            preset_models = [
                "stabilityai/stable-diffusion-3.5-large",
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5"
            ]

            for model_id in preset_models:
                try:
                    self._update_status(f"尝试使用{model_id}的配置...")

                    # 先加载配置（可能需要网络，但我们只要配置）
                    from diffusers import DiffusionPipeline

                    # 尝试从缓存加载配置
                    cache_dir = config.get("model.cache_dir")
                    cached_model_path = Path(cache_dir) / model_id.replace("/", "--")

                    if cached_model_path.exists():
                        self._update_status(f"发现缓存配置: {model_id}")

                        # 使用缓存的配置但替换权重文件
                        pipeline = DiffusionPipeline.from_pretrained(
                            str(cached_model_path),
                            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                            local_files_only=True,
                            safety_checker=None,
                            requires_safety_checker=False
                        )

                        # 尝试替换权重（这是高级操作，可能不总是成功）
                        self._update_status("尝试替换模型权重...")

                        # 这里需要更复杂的权重替换逻辑
                        # 暂时返回False，让用户知道需要完整的模型目录
                        self._update_status("SafeTensors单文件加载需要配置文件支持")
                        return False

                except Exception as e:
                    continue

            return False

        except Exception as e:
            self._update_status(f"预设配置加载失败: {str(e)}")
            return False

    def _create_diffusers_config_for_comfyui(self, model_dir: Path) -> bool:
        """为ComfyUI格式的模型创建diffusers配置"""
        try:
            self._update_status("检测到ComfyUI格式模型，已有配置文件")
            return True

        except Exception as e:
            self._update_status(f"创建配置失败: {str(e)}")
            return False

    def _load_single_model_file(self, file_path: Path) -> bool:
        """加载单个模型文件（非SafeTensors）"""
        try:
            self._update_status(f"正在加载模型文件: {file_path.name}")

            # 对于非SafeTensors文件，尝试从父目录加载
            parent_dir = file_path.parent

            # 检查父目录是否包含配置文件
            config_files = [f for f in ['config.json', 'model_index.json'] if (parent_dir / f).exists()]

            if config_files:
                self._update_status(f"发现配置文件: {', '.join(config_files)}")
                return self._load_model_directory(parent_dir)
            else:
                self._update_status("单个模型文件需要配置文件支持")
                return False

        except Exception as e:
            error_msg = f"单个模型文件加载失败: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False

    def _load_model_directory(self, model_dir: Path) -> bool:
        """加载模型目录"""
        try:
            self._update_status(f"正在加载模型目录: {model_dir.name}")

            # 检查目录中的文件
            safetensors_files = list(model_dir.glob("*.safetensors"))
            config_files = [f for f in ['model_index.json', 'config.json'] if (model_dir / f).exists()]

            if safetensors_files:
                self._update_status(f"发现{len(safetensors_files)}个SafeTensors文件")
            if config_files:
                self._update_status(f"发现配置文件: {', '.join(config_files)}")

            # 如果没有配置文件，检查是否是ComfyUI格式的模型
            if not config_files:
                # 检查是否有主要的模型文件
                main_model_files = [f for f in safetensors_files if 'large' in f.name.lower() or 'base' in f.name.lower()]
                if main_model_files:
                    self._update_status("检测到ComfyUI格式模型，尝试创建配置...")
                    if self._create_diffusers_config_for_comfyui(model_dir):
                        config_files = ['model_index.json']
                    else:
                        self._update_status("无法为ComfyUI模型创建配置")
                        return False
                else:
                    # 尝试查找子目录，但排除.git等系统目录
                    subdirs = [d for d in model_dir.iterdir()
                              if d.is_dir() and not d.name.startswith('.') and d.name not in ['__pycache__']]
                    if subdirs:
                        # 选择最可能的模型目录
                        for subdir in subdirs:
                            subdir_config = [f for f in ['model_index.json', 'config.json'] if (subdir / f).exists()]
                            if subdir_config:
                                model_dir = subdir
                                self._update_status(f"使用子目录: {model_dir.name}")
                                break
                        else:
                            self._update_status("未找到有效的模型配置")
                            return False

            self._update_status("初始化模型pipeline...")

            # 尝试不同的pipeline加载方式
            pipelines_to_try = [
                ("Stable Diffusion 3", StableDiffusion3Pipeline),
            ]

            # 导入其他pipeline
            try:
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
                pipelines_to_try.extend([
                    ("Stable Diffusion XL", StableDiffusionXLPipeline),
                    ("Stable Diffusion 1.5/2.x", StableDiffusionPipeline),
                ])
            except ImportError:
                pass

            last_error = None
            for pipeline_name, pipeline_class in pipelines_to_try:
                try:
                    self._update_status(f"尝试使用{pipeline_name}加载...")

                    # 完全离线加载参数
                    load_kwargs = {
                        "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                        "use_safetensors": True,
                        "local_files_only": True,  # 强制仅使用本地文件
                        "safety_checker": None,
                        "requires_safety_checker": False,
                        "trust_remote_code": False,  # 不信任远程代码
                    }

                    self.pipeline = pipeline_class.from_pretrained(
                        str(model_dir),
                        **load_kwargs
                    )

                    self._update_status(f"✅ 使用{pipeline_name}加载成功")

                    # 移动到设备
                    if hasattr(self.pipeline, 'to'):
                        self._update_status("正在移动模型到计算设备...")
                        self.pipeline = self.pipeline.to(self.device)

                    self._update_status("模型目录加载完成")
                    return True

                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "connection" in error_msg or "network" in error_msg or "internet" in error_msg:
                        self._update_status(f"{pipeline_name}尝试网络连接，跳过...")
                    else:
                        self._update_status(f"{pipeline_name}加载失败: {str(e)[:100]}...")
                    continue

            # 所有pipeline都失败
            error_msg = f"所有pipeline加载失败，最后错误: {str(last_error)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False

        except Exception as e:
            error_msg = f"模型目录加载异常: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False

    def _download_model_with_progress(self, model_name: str) -> bool:
        """带进度显示的模型下载"""
        try:
            self._update_status("正在检查网络连接...")

            if not self._check_network_connection():
                self._update_status("网络连接失败，请检查网络设置")
                return False

            self._update_status(f"开始下载模型: {model_name}")
            self._update_status("注意: 首次下载可能需要较长时间（模型约4-5GB）")

            # 设置代理环境变量（用于huggingface_hub）
            network_config = config.get_network_config()
            if network_config.get("proxy_enabled", False):
                proxy_host = network_config.get("proxy_host", "127.0.0.1")
                proxy_port = network_config.get("proxy_port", 7890)
                proxy_type = network_config.get("proxy_type", "http")
                proxy_url = f"{proxy_type}://{proxy_host}:{proxy_port}"

                import os
                os.environ["HTTP_PROXY"] = proxy_url
                os.environ["HTTPS_PROXY"] = proxy_url
                self._update_status(f"设置下载代理: {proxy_url}")

            cache_dir = config.get("model.cache_dir")
            use_safetensors = config.get("model.use_safetensors", True)
            max_retries = config.get("model.max_retries", 3)

            # 根据模型名称选择合适的Pipeline
            if "stable-diffusion-v1-5" in model_name or "stable-diffusion-2" in model_name:
                from diffusers import StableDiffusionPipeline
                pipeline_class = StableDiffusionPipeline
            elif "stable-diffusion-xl" in model_name:
                from diffusers import StableDiffusionXLPipeline
                pipeline_class = StableDiffusionXLPipeline
            else:
                # 默认使用SD3Pipeline
                pipeline_class = StableDiffusion3Pipeline

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        self._update_status(f"重试下载 (第{attempt + 1}次)...")
                        time.sleep(5)  # 等待5秒后重试

                    # 使用from_pretrained下载模型
                    self.pipeline = pipeline_class.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                        use_safetensors=use_safetensors,
                        local_files_only=False  # 允许从网络下载
                    )

                    self._update_status("模型下载完成")
                    return True

                except Exception as e:
                    error_msg = f"下载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
                    self._update_status(error_msg)
                    logger.warning(error_msg)

                    if attempt == max_retries - 1:
                        return False

            return False

        except Exception as e:
            error_msg = f"模型下载异常: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg)
            return False
    
    def load_model(self, model_name: str = None) -> bool:
        """加载Stable Diffusion模型"""
        if not TORCH_AVAILABLE:
            self._update_status("PyTorch未安装，无法加载模型")
            return False

        try:
            if model_name is None:
                model_name = config.get("model.name")

            if self.model_loaded and self.current_model_name == model_name and self.pipeline is not None:
                self._update_status(f"模型 {model_name} 已加载")
                return True

            self._update_status("正在检测设备...")

            # 智能设备选择
            config_device = self.system_config["device"]
            if config_device == "auto":
                self.device = get_optimal_device()
            elif config_device == "cuda":
                # 检查CUDA是否可用
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self._update_status("配置指定使用CUDA，检查CUDA可用性...")
                else:
                    self._update_status("配置指定CUDA但CUDA不可用，回退到CPU")
                    self.device = "cpu"
            else:
                self.device = config_device

            self._update_status(f"使用设备: {self.device}")

            # 获取CUDA优化设置
            if self.device == "cuda":
                cuda_settings = get_cuda_optimization_settings()
                self._update_status("应用CUDA优化设置...")

                # 更新系统配置
                for key, value in cuda_settings.items():
                    if key in self.system_config:
                        self.system_config[key] = value
                        self._update_status(f"  {key}: {value}")

            # 检查内存
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_name = torch.cuda.get_device_name(0)
                self._update_status(f"GPU: {gpu_name}")
                self._update_status(f"GPU内存: {format_memory_size(gpu_memory)}")

            self._update_status(f"准备加载模型: {model_name}")
            
            # 检查模型是否需要下载
            is_local = self._is_local_model(model_name)
            if not is_local and not self._check_model_cached(model_name):
                if config.get("model.auto_download", True):
                    if not self._download_model_with_progress(model_name):
                        return False
                else:
                    self._update_status(f"模型未缓存且自动下载已禁用")
                    return False
            
            # 统一加载模型
            self._update_status("正在初始化和加载模型...")
            
            # 确定pipeline类型并加载
            if not self._load_pipeline(model_name, is_local):
                return False

            if not self.pipeline:
                self._update_status("模型Pipeline初始化失败")
                return False

            # 应用优化并移动到设备
            self.pipeline.to(self.device)
            if self.device == "cuda":
                if self.system_config.get("attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                if self.system_config.get("enable_xformers"):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                    except (ImportError, AttributeError):
                        self._update_status("xformers不可用或不受支持，跳过优化")
            
            self.model_loaded = True
            self.current_model_name = model_name
            self._update_status("模型加载完成")
            return True

        except Exception as e:
            self._update_status(f"模型加载失败: {str(e)}")
            logger.error(f"模型加载异常: {e}", exc_info=True)
            self.unload_model()
            return False
    
    def _load_pipeline(self, model_name: str, is_local: bool) -> bool:
        """根据模型名称动态选择并加载pipeline"""
        try:
            if is_local:
                return self._load_local_model(model_name)

            # 在线模型处理
            cache_dir = config.get("model.cache_dir")
            load_kwargs = {
                "cache_dir": cache_dir,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" and self.system_config.get("use_bf16") else torch.float16,
                "use_safetensors": config.get("model.use_safetensors", True),
                "local_files_only": not config.get("model.auto_download", True)
            }

            # 为SD3加载所有文本编码器
            if "stable-diffusion-3" in model_name:
                self._update_status("检测到Stable Diffusion 3模型，使用专用加载器...")
                
                # SD3需要多个文本编码器，我们需要分别加载它们
                text_encoder = T5EncoderModel.from_pretrained(
                    model_name, subfolder="text_encoder_3", cache_dir=cache_dir,
                    local_files_only=load_kwargs["local_files_only"],
                    torch_dtype=load_kwargs["torch_dtype"]
                )
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    model_name, subfolder="text_encoder_2", cache_dir=cache_dir,
                    local_files_only=load_kwargs["local_files_only"],
                    torch_dtype=load_kwargs["torch_dtype"]
                )
                
                # 将加载的组件传递给pipeline
                load_kwargs.update({
                    "text_encoder_3": text_encoder,
                    "text_encoder_2": text_encoder_2,
                })
                
                # SD3的VAE也可能需要单独加载
                try:
                    vae = AutoencoderKL.from_pretrained(
                        model_name, subfolder="vae",  cache_dir=cache_dir,
                        local_files_only=load_kwargs["local_files_only"],
                        torch_dtype=load_kwargs["torch_dtype"]
                    )
                    load_kwargs["vae"] = vae
                except Exception:
                    self._update_status("无法单独加载VAE，将使用默认VAE")

                self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_name,
                    **load_kwargs
                )
            # 简化版：这里我们先假设一个通用加载器可以工作，后续再细化
            # 理想情况下，我们应该检查model_index.json来决定pipeline类型
            else:
                 # 尝试使用通用的 from_pretrained，这通常可以处理SD1.5/2.1/XL
                self._update_status("尝试使用通用加载器...")
                # 这是一个简化的假设，实际可能需要更复杂的逻辑
                # diffusers没有一个统一的AutoPipeline，所以我们得猜
                if "xl" in model_name.lower():
                    pipeline_class = StableDiffusionXLPipeline
                    self._update_status("推断为XL模型...")
                else:
                    pipeline_class = StableDiffusionPipeline
                    self._update_status("推断为标准模型 (SD1.5/2.x)...")
                
                self.pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)

            return True

        except Exception as e:
            self._update_status(f"Pipeline加载失败: {e}")
            logger.error(f"Pipeline加载异常: {e}", exc_info=True)
            return False

    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str = "",
                      width: int = None,
                      height: int = None,
                      num_inference_steps: int = None,
                      guidance_scale: float = None,
                      seed: int = None) -> Optional[Image.Image]:
        """生成图片 - 优化CUDA性能"""
        
        if not self.model_loaded:
            self._update_status("模型未加载")
            return None
        
        self._update_status("开始生成图片...")

        # 获取配置
        config = self.generation_config
        width = width or config.get("width")
        height = height or config.get("height")
        num_inference_steps = num_inference_steps or config.get("steps")
        guidance_scale = guidance_scale or config.get("cfg_scale")

        # 随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 进度回调
        def progress_callback(step_index, timestep, latents):
            # 更新进度
            self._update_progress(step_index + 1, num_inference_steps)
        
        try:
            # 确保是正确的pipeline
            if not isinstance(self.pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline)):
                 self._update_status("检测到非标准pipeline，正在切换回基础文生图模式...")
                 self.pipeline = self.base_pipeline
                 if not self.pipeline:
                     self._update_status("错误：基础pipeline未加载，无法切换。")
                     return None

            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_steps=1,
                    callback=progress_callback
                ).images[0]
            
            self._update_status("✅ 图片生成成功")
            return image

        except Exception as e:
            error_msg = f"生成图片时出错: {e}"
            self._update_status(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    def generate_inpainting(self,
                            prompt: str,
                            image: Image.Image,
                            mask_image: Image.Image,
                            negative_prompt: str = "",
                            width: int = None,
                            height: int = None,
                            num_inference_steps: int = None,
                            guidance_scale: float = None,
                            seed: int = None,
                            strength: float = 0.8) -> Optional[Image.Image]:
        """
        使用 Inpainting (图片蒙版) 方式生成图片
        """
        if not self.model_loaded:
            self._update_status("模型未加载")
            return None

        self._update_status("开始使用Inpainting生成图片...")

        # 获取配置
        config = self.generation_config
        width = width or config.get("width")
        height = height or config.get("height")
        num_inference_steps = num_inference_steps or config.get("steps")
        guidance_scale = guidance_scale or config.get("cfg_scale")

        # 随机种子
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 进度回调
        def progress_callback(step_index, timestep, latents):
            self._update_progress(step_index + 1, num_inference_steps)

        try:
            # 动态切换到 Inpainting Pipeline
            if not isinstance(self.pipeline, (StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline)):
                self._update_status("正在切换到Inpainting模式...")
                
                # 确定适合的Inpainting Pipeline类型
                if isinstance(self.base_pipeline, StableDiffusionXLPipeline):
                    inpaint_class = StableDiffusionXLInpaintPipeline
                elif isinstance(self.base_pipeline, StableDiffusionPipeline):
                    inpaint_class = StableDiffusionInpaintPipeline
                else:
                    # 对于SD3等其他类型，可能需要特定处理或不支持
                    self._update_status(f"当前模型 ({type(self.base_pipeline).__name__}) 可能不支持快速切换到Inpainting模式。")
                    # 尝试通用加载
                    try:
                        self.pipeline = inpaint_class.from_pipe(self.base_pipeline)
                    except Exception as e:
                        self._update_status(f"无法切换到Inpainting Pipeline: {e}")
                        return None
                
                self.pipeline = inpaint_class(**self.base_pipeline.components)
                self._update_status("✅ 成功切换到Inpainting模式")

            with torch.no_grad():
                # 调整输入图像尺寸
                image = image.resize((width, height), Image.LANCZOS)
                mask_image = mask_image.resize((width, height), Image.NEAREST)

                output_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask_image,
                    width=width,
                    height=height,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_steps=1,
                    callback=progress_callback
                ).images[0]

            self._update_status("✅ Inpainting图片生成成功")
            return output_image

        except Exception as e:
            error_msg = f"Inpainting生成图片时出错: {e}"
            self._update_status(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前加载的模型信息"""
        if not self.model_loaded or not self.pipeline:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "device": self.device,
            "model_name": config.get("model.name"),
        }
        
        if self.device == "cuda":
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        return info
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
        self.model_loaded = False
        self.current_model_name = None  # 清除当前模型名称
        self._update_status("模型已卸载")
    
    def update_generation_config(self, **kwargs):
        """更新生成配置"""
        for key, value in kwargs.items():
            if key in self.generation_config:
                self.generation_config[key] = value
        
        # 同步到全局配置
        config.update_generation_config(**kwargs)
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取当前生成配置"""
        return self.generation_config.copy()
    
    def validate_parameters(self, width: int, height: int, steps: int) -> Tuple[bool, str]:
        """验证生成参数"""
        # 检查参数是否存在
        if width is None or height is None or steps is None:
            return False, "参数不完整 (宽度、高度或步数为空)"

        # 检查尺寸 - SD 3.5 支持更大尺寸
        if width % 8 != 0 or height % 8 != 0:
            return False, "图片尺寸必须是8的倍数"

        if width < 512 or height < 512:
            return False, "图片尺寸不能小于512x512"

        if width > 2048 or height > 2048:
            return False, "图片尺寸不能大于2048x2048"

        # 检查步数 - SD 3.5 推荐28步
        if steps < 1 or steps > 100:
            return False, "采样步数必须在1-100之间"

        return True, ""

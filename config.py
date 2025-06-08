"""
配置管理模块
管理应用程序的默认设置和用户配置
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """配置管理类"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".sd_generator"
        self.config_file = self.config_dir / "config.json"
        self.models_dir = self.config_dir / "models"
        self.outputs_dir = Path("outputs")
        self.logs_dir = Path("logs")
        
        # 创建必要的目录
        self._create_directories()
        
        # 默认配置
        self.default_config = {
            "model": {
                "name": "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 使用SD1.5作为默认模型
                "cache_dir": str(self.models_dir),
                "use_safetensors": True,
                "auto_download": True,  # 启用自动下载
                "download_timeout": 1800,  # 30分钟超时
                "max_retries": 3
            },
            "generation": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "negative_prompt": "",
                "seed": -1,  # -1 表示随机种子
                "batch_size": 1
            },
            "ui": {
                "window_width": 800,
                "window_height": 600,
                "preview_size": 400,
                "auto_save": True
            },
            "system": {
                "device": "auto",  # auto, cpu, cuda
                "low_vram_mode": False,
                "attention_slicing": True,
                "cpu_offload": False
            },
            "network": {
                "proxy_enabled": True,
                "proxy_host": "127.0.0.1",
                "proxy_port": 7890,
                "proxy_type": "http"  # http, socks5
            },
            "paths": {
                "outputs_dir": str(self.outputs_dir),
                "logs_dir": str(self.logs_dir)
            }
        }
        
        # 加载用户配置
        self.config = self._load_config()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config_dir,
            self.models_dir,
            self.outputs_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 合并默认配置和用户配置
                config = self.default_config.copy()
                self._deep_update(config, user_config)
                return config
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key_path: str, default=None):
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """设置配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config["model"]
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return self.config["generation"]
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.config["ui"]
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.config["system"]

    def get_network_config(self) -> Dict[str, Any]:
        """获取网络配置"""
        return self.config["network"]

    def get_paths_config(self) -> Dict[str, Any]:
        """获取路径配置"""
        return self.config["paths"]
    
    def update_generation_config(self, **kwargs):
        """更新生成配置"""
        for key, value in kwargs.items():
            if key in self.config["generation"]:
                self.config["generation"][key] = value
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self.default_config.copy()
        self.save_config()

# 全局配置实例
config = Config()

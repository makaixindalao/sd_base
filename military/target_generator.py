"""
军事目标生成器
基于Stable Diffusion生成军事目标图像
"""

import random
import time
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
from PIL import Image
import logging

from .prompt_templates import PromptTemplateManager
from sd_generator import SDGenerator
from config import Config

logger = logging.getLogger(__name__)

class MilitaryTargetGenerator:
    """军事目标生成器"""
    
    def __init__(self, sd_generator: SDGenerator = None):
        """初始化军事目标生成器"""
        self.sd_generator = sd_generator or SDGenerator()
        self.prompt_manager = PromptTemplateManager()
        self.config = Config()
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        
        # 生成统计
        self.generation_stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def set_callbacks(self, progress_callback: Callable = None, status_callback: Callable = None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        
        # 同时设置SD生成器的回调
        if self.sd_generator:
            self.sd_generator.set_callbacks(progress_callback, status_callback)
    
    def _update_progress(self, progress: float):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(progress)
    
    def _update_status(self, status: str):
        """更新状态"""
        if self.status_callback:
            self.status_callback(status)
        logger.info(status)
    
    def generate_single_target(self, 
                             target_type: str,
                             weather: str = None,
                             terrain: str = None,
                             custom_prompt: str = None,
                             **generation_kwargs) -> Optional[Image.Image]:
        """生成单个军事目标图像"""
        try:
            if custom_prompt:
                positive_prompt = custom_prompt
                negative_prompt = "low quality, blurry, distorted, cartoon, anime"
            else:
                positive_prompt, negative_prompt = self.prompt_manager.get_random_prompt(
                    target_type, weather, terrain
                )
            
            self._update_status(f"生成{target_type}图像...")
            
            # 使用SD生成器生成图像
            image = self.sd_generator.generate_image(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                **generation_kwargs
            )
            
            if image:
                self.generation_stats["successful"] += 1
                self._update_status(f"成功生成{target_type}图像")
            else:
                self.generation_stats["failed"] += 1
                self._update_status(f"生成{target_type}图像失败")
            
            self.generation_stats["total_generated"] += 1
            return image
            
        except Exception as e:
            self.generation_stats["failed"] += 1
            self.generation_stats["total_generated"] += 1
            error_msg = f"生成{target_type}图像时出错: {str(e)}"
            self._update_status(error_msg)
            logger.error(error_msg, exc_info=True)
            return None
    
    def generate_batch_targets(self,
                             target_types: List[str],
                             weather_conditions: List[str] = None,
                             terrain_types: List[str] = None,
                             count: int = 10,
                             mixed_targets: bool = False,
                             mixed_scenes: bool = False,
                             output_dir: str = "outputs/military",
                             **generation_kwargs) -> List[Dict]:
        """批量生成军事目标图像"""
        
        self.generation_stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "start_time": time.time(),
            "end_time": None
        }
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._update_status(f"开始批量生成{count}张军事目标图像...")
        
        for i in range(count):
            try:
                # 选择目标类型
                if mixed_targets:
                    target_type = random.choice(target_types)
                else:
                    target_type = target_types[i % len(target_types)]
                
                # 选择天气条件
                weather = None
                if weather_conditions:
                    if mixed_scenes:
                        weather = random.choice(weather_conditions)
                    else:
                        weather = weather_conditions[i % len(weather_conditions)]
                
                # 选择地形类型
                terrain = None
                if terrain_types:
                    if mixed_scenes:
                        terrain = random.choice(terrain_types)
                    else:
                        terrain = terrain_types[i % len(terrain_types)]
                
                # 生成图像
                image = self.generate_single_target(
                    target_type=target_type,
                    weather=weather,
                    terrain=terrain,
                    **generation_kwargs
                )
                
                # 保存结果
                if image:
                    # 生成文件名
                    timestamp = int(time.time() * 1000)
                    filename = f"{target_type}_{weather or 'default'}_{terrain or 'default'}_{timestamp}.png"
                    filepath = output_path / filename
                    
                    # 保存图像
                    image.save(filepath)
                    
                    # 记录结果
                    result = {
                        "index": i + 1,
                        "target_type": target_type,
                        "weather": weather,
                        "terrain": terrain,
                        "filepath": str(filepath),
                        "success": True,
                        "timestamp": timestamp
                    }
                else:
                    result = {
                        "index": i + 1,
                        "target_type": target_type,
                        "weather": weather,
                        "terrain": terrain,
                        "filepath": None,
                        "success": False,
                        "timestamp": int(time.time() * 1000)
                    }
                
                results.append(result)
                
                # 更新进度
                progress = (i + 1) / count * 100
                self._update_progress(progress)
                self._update_status(f"已生成 {i + 1}/{count} 张图像")
                
            except Exception as e:
                error_msg = f"批量生成第{i+1}张图像时出错: {str(e)}"
                self._update_status(error_msg)
                logger.error(error_msg, exc_info=True)
                
                results.append({
                    "index": i + 1,
                    "target_type": target_type if 'target_type' in locals() else "unknown",
                    "weather": weather if 'weather' in locals() else None,
                    "terrain": terrain if 'terrain' in locals() else None,
                    "filepath": None,
                    "success": False,
                    "error": str(e),
                    "timestamp": int(time.time() * 1000)
                })
        
        self.generation_stats["end_time"] = time.time()
        
        # 生成统计报告
        total_time = self.generation_stats["end_time"] - self.generation_stats["start_time"]
        success_rate = (self.generation_stats["successful"] / count) * 100 if count > 0 else 0
        
        self._update_status(
            f"批量生成完成! 成功: {self.generation_stats['successful']}/{count} "
            f"({success_rate:.1f}%), 耗时: {total_time:.1f}秒"
        )
        
        return results
    
    def get_generation_stats(self) -> Dict:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        if stats["start_time"] and stats["end_time"]:
            stats["total_time"] = stats["end_time"] - stats["start_time"]
            stats["avg_time_per_image"] = stats["total_time"] / max(stats["total_generated"], 1)
        return stats
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """获取可用的生成选项"""
        return self.prompt_manager.get_available_options()

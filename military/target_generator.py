"""
军事目标生成器
负责生成各种军事目标的图像，包括坦克、飞机、舰船等
"""

import random
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MilitaryTargetGenerator:
    """军事目标生成器"""
    
    def __init__(self, sd_generator=None):
        """
        初始化军事目标生成器
        
        Args:
            sd_generator: Stable Diffusion生成器实例
        """
        self.sd_generator = sd_generator
        self.generation_stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 回调函数
        self.progress_callback = None
        self.status_callback = None
        
        # 可用选项
        self.available_options = {
            'targets': ['tank', 'aircraft', 'ship'],
            'weather': ['sunny', 'rainy', 'snowy', 'foggy', 'night'],
            'terrain': ['urban', 'island', 'rural']
        }
    
    def set_callbacks(self, progress_callback=None, status_callback=None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """获取可用的生成选项"""
        return self.available_options.copy()
    
    def get_generation_stats(self) -> Dict:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
        else:
            stats['total_time'] = 0
        return stats
    
    def generate_single_target(self, target_type: str, weather: str = None, 
                             terrain: str = None, **kwargs) -> Optional[Dict]:
        """
        生成单个军事目标
        
        Args:
            target_type: 目标类型 (tank, aircraft, ship)
            weather: 天气条件
            terrain: 地形类型
            **kwargs: 其他生成参数
            
        Returns:
            生成结果字典，包含成功状态和相关信息
        """
        try:
            # 验证参数
            if target_type not in self.available_options['targets']:
                raise ValueError(f"不支持的目标类型: {target_type}")
            
            # 随机选择天气和地形（如果未指定）
            if weather is None:
                weather = random.choice(self.available_options['weather'])
            if terrain is None:
                terrain = random.choice(self.available_options['terrain'])
            
            # 构建提示词
            prompt = self._build_prompt(target_type, weather, terrain)
            negative_prompt = self._build_negative_prompt(target_type, weather, terrain)
            
            if self.status_callback:
                self.status_callback(f"正在生成 {target_type} 图像...")
            
            # 生成图像（如果有SD生成器）
            if self.sd_generator:
                generation_params = {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'width': kwargs.get('width', 512),
                    'height': kwargs.get('height', 512),
                    'num_inference_steps': kwargs.get('num_inference_steps', 20),
                    'guidance_scale': kwargs.get('guidance_scale', 7.5),
                    'seed': kwargs.get('seed', None)
                }
                
                image = self.sd_generator.generate_image(**generation_params)
                
                if image:
                    # 保存图像
                    output_dir = Path(kwargs.get('output_dir', 'outputs/military'))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{target_type}_{weather}_{terrain}_{int(time.time())}.png"
                    image_path = output_dir / filename
                    image.save(image_path)
                    
                    return {
                        'success': True,
                        'target_type': target_type,
                        'weather': weather,
                        'terrain': terrain,
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'image_path': str(image_path),
                        'image': image
                    }
                else:
                    return {
                        'success': False,
                        'target_type': target_type,
                        'weather': weather,
                        'terrain': terrain,
                        'error': '图像生成失败'
                    }
            else:
                # 没有SD生成器，只返回提示词
                return {
                    'success': True,
                    'target_type': target_type,
                    'weather': weather,
                    'terrain': terrain,
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'note': '仅生成提示词（未配置SD生成器）'
                }
                
        except Exception as e:
            logger.error(f"生成军事目标失败: {e}")
            return {
                'success': False,
                'target_type': target_type,
                'error': str(e)
            }
    
    def generate_batch_targets(self, target_types: List[str], 
                             weather_conditions: List[str] = None,
                             terrain_types: List[str] = None,
                             count: int = 10,
                             mixed_targets: bool = True,
                             mixed_scenes: bool = True,
                             **kwargs) -> List[Dict]:
        """
        批量生成军事目标
        
        Args:
            target_types: 目标类型列表
            weather_conditions: 天气条件列表
            terrain_types: 地形类型列表
            count: 生成数量
            mixed_targets: 是否混合目标类型
            mixed_scenes: 是否混合场景条件
            **kwargs: 其他生成参数
            
        Returns:
            生成结果列表
        """
        self.generation_stats['start_time'] = time.time()
        self.generation_stats['total_generated'] = 0
        self.generation_stats['successful'] = 0
        self.generation_stats['failed'] = 0
        
        results = []
        
        # 设置默认值
        if weather_conditions is None:
            weather_conditions = self.available_options['weather']
        if terrain_types is None:
            terrain_types = self.available_options['terrain']
        
        try:
            for i in range(count):
                if self.progress_callback:
                    progress = (i / count) * 100
                    self.progress_callback(progress)
                
                # 选择目标类型
                if mixed_targets:
                    target_type = random.choice(target_types)
                else:
                    target_type = target_types[i % len(target_types)]
                
                # 选择场景条件
                if mixed_scenes:
                    weather = random.choice(weather_conditions)
                    terrain = random.choice(terrain_types)
                else:
                    weather = weather_conditions[i % len(weather_conditions)]
                    terrain = terrain_types[i % len(terrain_types)]
                
                # 生成单个目标
                result = self.generate_single_target(
                    target_type, weather, terrain, **kwargs
                )
                
                results.append(result)
                self.generation_stats['total_generated'] += 1
                
                if result['success']:
                    self.generation_stats['successful'] += 1
                else:
                    self.generation_stats['failed'] += 1
                
                # 短暂延迟避免过载
                time.sleep(0.1)
        
        finally:
            self.generation_stats['end_time'] = time.time()
            if self.progress_callback:
                self.progress_callback(100)
        
        return results
    
    def _build_prompt(self, target_type: str, weather: str, terrain: str) -> str:
        """构建正面提示词"""
        # 基础目标描述
        target_prompts = {
            'tank': 'modern tank, camouflage pattern, military vehicle, steel armor, tracks, turret',
            'aircraft': 'military aircraft, fighter jet, warplane',
            'ship': 'military ship, warship, naval vessel'
        }
        
        # 天气描述
        weather_prompts = {
            'sunny': 'bright daylight, clear sky, good visibility',
            'rainy': 'rainy weather, wet surfaces, overcast sky',
            'snowy': 'snowy weather, winter conditions, snow-covered',
            'foggy': 'foggy weather, mist, low visibility',
            'night': 'night scene, dark environment, artificial lighting'
        }
        
        # 地形描述
        terrain_prompts = {
            'urban': 'urban environment, city setting, buildings',
            'island': 'island setting, coastal area, water nearby',
            'rural': 'rural area, countryside, open terrain'
        }
        
        # 组合提示词
        prompt_parts = [
            target_prompts.get(target_type, target_type),
            'realistic, detailed, high quality, photorealistic',
            weather_prompts.get(weather, weather),
            terrain_prompts.get(terrain, terrain)
        ]
        
        return ', '.join(prompt_parts)
    
    def _build_negative_prompt(self, target_type: str, weather: str, terrain: str) -> str:
        """构建负面提示词"""
        # 基础负面词
        base_negative = 'low resolution, blurry, abstract, cartoon, toy'
        
        # 目标特定负面词
        target_negative = {
            'tank': 'civilian vehicle, car, truck',
            'aircraft': 'civilian plane, commercial aircraft',
            'ship': 'civilian ship, cruise ship'
        }
        
        # 天气相反条件
        weather_negative = {
            'sunny': 'dark, night, rain, snow, fog',
            'rainy': 'dry, sunny, clear weather, bright lighting',
            'snowy': 'hot, summer, green vegetation',
            'foggy': 'clear visibility, bright, sunny',
            'night': 'daylight, bright, sunny, well lit'
        }
        
        # 地形相反条件
        terrain_negative = {
            'urban': 'rural, countryside, wilderness',
            'island': 'landlocked, desert, mountains',
            'rural': 'urban, city, buildings, crowded'
        }
        
        negative_parts = [
            base_negative,
            target_negative.get(target_type, ''),
            weather_negative.get(weather, ''),
            terrain_negative.get(terrain, '')
        ]
        
        return ', '.join(filter(None, negative_parts))

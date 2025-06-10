"""
军事提示词模板管理器
提供各种军事目标的提示词模板和组合功能
"""

import random
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PromptTemplateManager:
    """提示词模板管理器"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = self._load_templates()
        self.usage_stats = {}
    
    def _load_templates(self) -> Dict:
        """加载提示词模板"""
        return {
            'targets': {
                'tank': [
                    'modern tank, camouflage pattern, military vehicle, steel armor, tracks, turret, realistic lighting',
                    'battle tank, heavy armor, military camouflage, tracked vehicle, main gun turret, detailed metal texture',
                    'armored tank, military green, combat vehicle, steel plating, cannon turret, realistic shadows'
                ],
                'aircraft': [
                    'military aircraft, fighter jet, warplane, realistic, detailed, high quality, photorealistic',
                    'stealth fighter, military aviation, jet engine, detailed aircraft, professional photography',
                    'combat aircraft, air force, fighter plane, military jet, high-tech aviation, realistic rendering'
                ],
                'ship': [
                    'military ship, warship, naval vessel, realistic, detailed, high quality, photorealistic',
                    'naval warship, military vessel, destroyer, detailed hull, realistic water',
                    'combat ship, navy vessel, military maritime, steel hull, naval architecture, ocean setting'
                ]
            },
            'weather': {
                'sunny': [
                    'bright daylight, clear sky, good visibility, natural lighting, sunny weather',
                    'clear day, bright sunlight, blue sky, excellent visibility, daytime scene',
                    'sunny conditions, bright natural light, clear atmosphere, good weather'
                ],
                'rainy': [
                    'rainy weather, wet surfaces, overcast sky, rain drops, stormy atmosphere',
                    'heavy rain, wet ground, dark clouds, storm weather, dramatic lighting',
                    'rainfall, wet conditions, cloudy sky, rain effects, moody atmosphere'
                ],
                'snowy': [
                    'snowy weather, winter conditions, snow-covered, cold environment, white landscape',
                    'heavy snowfall, winter scene, snow accumulation, cold weather, frozen environment',
                    'snow storm, winter warfare, icy conditions, snow-covered terrain, cold climate'
                ],
                'foggy': [
                    'foggy weather, mist, low visibility, atmospheric haze, mysterious atmosphere',
                    'dense fog, misty conditions, reduced visibility, atmospheric effects, eerie lighting',
                    'fog bank, hazy atmosphere, limited sight, mysterious environment, soft lighting'
                ],
                'night': [
                    'night scene, dark environment, artificial lighting, nocturnal setting, dramatic shadows',
                    'nighttime operation, dark sky, artificial illumination, night vision, tactical lighting',
                    'night combat, darkness, strategic lighting, moonlight, night warfare'
                ]
            },
            'terrain': {
                'urban': [
                    'urban environment, city setting, buildings, concrete structures, metropolitan area',
                    'city warfare, urban combat, building structures, street environment, metropolitan setting',
                    'urban landscape, city blocks, concrete jungle, modern architecture, urban terrain'
                ],
                'island': [
                    'island setting, coastal area, water nearby, tropical environment, maritime setting',
                    'island warfare, coastal defense, beach landing, tropical island, ocean environment',
                    'island terrain, coastal region, maritime operations, beach environment, island geography'
                ],
                'rural': [
                    'rural area, countryside, open terrain, natural environment, field operations',
                    'countryside setting, open fields, rural landscape, agricultural area, natural terrain',
                    'rural warfare, field environment, countryside operations, open ground, natural setting'
                ]
            },
            'negative': {
                'base': [
                    'low resolution, blurry, abstract, cartoon, toy, unrealistic, poor quality',
                    'low quality, pixelated, distorted, artificial, fake, low detail, poor rendering',
                    'bad quality, unclear, fuzzy, amateur, low-res, poor lighting, unrealistic proportions'
                ],
                'targets': {
                    'tank': [
                        'civilian vehicle, car, truck, passenger vehicle, non-military',
                        'toy tank, model tank, miniature, plastic, unrealistic scale',
                        'damaged tank, destroyed vehicle, wreckage, broken parts'
                    ],
                    'aircraft': [
                        'civilian plane, commercial aircraft, passenger jet, airline, non-military',
                        'toy plane, model aircraft, miniature, plastic airplane',
                        'crashed plane, damaged aircraft, wreckage, broken wings'
                    ],
                    'ship': [
                        'civilian ship, cruise ship, yacht, fishing boat, cargo ship, non-military vessel',
                        'toy boat, model ship, miniature vessel, plastic boat',
                        'sinking ship, damaged vessel, shipwreck, broken hull'
                    ]
                },
                'weather': {
                    'sunny': ['dark, night, rain, snow, fog, storm, cloudy, overcast'],
                    'rainy': ['dry, sunny, clear weather, bright lighting, desert conditions'],
                    'snowy': ['hot, summer, tropical, warm weather, green vegetation'],
                    'foggy': ['clear visibility, bright, sunny, sharp details, crystal clear'],
                    'night': ['daylight, bright, sunny, well lit, daytime, morning']
                },
                'terrain': {
                    'urban': ['rural, countryside, wilderness, forest, natural environment'],
                    'island': ['landlocked, desert, mountains, inland, continental'],
                    'rural': ['urban, city, buildings, crowded, metropolitan, industrial']
                }
            }
        }
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """获取可用选项"""
        return {
            'targets': list(self.templates['targets'].keys()),
            'weather': list(self.templates['weather'].keys()),
            'terrain': list(self.templates['terrain'].keys())
        }
    
    def get_random_prompt(self, target: str, weather: str, terrain: str) -> Tuple[str, str]:
        """
        获取随机组合的提示词
        
        Args:
            target: 目标类型
            weather: 天气条件
            terrain: 地形类型
            
        Returns:
            (positive_prompt, negative_prompt) 元组
        """
        try:
            # 获取随机模板
            target_template = random.choice(self.templates['targets'].get(target, [target]))
            weather_template = random.choice(self.templates['weather'].get(weather, [weather]))
            terrain_template = random.choice(self.templates['terrain'].get(terrain, [terrain]))
            
            # 组合正面提示词
            positive_prompt = f"{target_template}, {weather_template}, {terrain_template}"
            
            # 组合负面提示词
            base_negative = random.choice(self.templates['negative']['base'])
            target_negative = random.choice(self.templates['negative']['targets'].get(target, ['']))
            weather_negative = ', '.join(self.templates['negative']['weather'].get(weather, []))
            terrain_negative = ', '.join(self.templates['negative']['terrain'].get(terrain, []))
            
            negative_parts = [base_negative, target_negative, weather_negative, terrain_negative]
            negative_prompt = ', '.join(filter(None, negative_parts))
            
            # 更新使用统计
            self._update_usage_stats(target, weather, terrain)
            
            return positive_prompt, negative_prompt
            
        except Exception as e:
            logger.error(f"生成提示词失败: {e}")
            # 返回基础提示词
            return f"{target}, {weather}, {terrain}", "low quality, blurry"
    
    def get_template_stats(self) -> Dict:
        """获取模板使用统计"""
        return self.usage_stats.copy()
    
    def _update_usage_stats(self, target: str, weather: str, terrain: str):
        """更新使用统计"""
        stats_keys = [
            f"targets.{target}",
            f"weather.{weather}",
            f"terrain.{terrain}"
        ]
        
        for key in stats_keys:
            self.usage_stats[key] = self.usage_stats.get(key, 0) + 1
    
    def add_custom_template(self, category: str, subcategory: str, template: str):
        """
        添加自定义模板
        
        Args:
            category: 类别 (targets, weather, terrain)
            subcategory: 子类别
            template: 模板内容
        """
        try:
            if category not in self.templates:
                self.templates[category] = {}
            
            if subcategory not in self.templates[category]:
                self.templates[category][subcategory] = []
            
            if template not in self.templates[category][subcategory]:
                self.templates[category][subcategory].append(template)
                logger.info(f"添加自定义模板: {category}.{subcategory}")
                
        except Exception as e:
            logger.error(f"添加自定义模板失败: {e}")
    
    def get_template_variations(self, category: str, subcategory: str) -> List[str]:
        """
        获取指定类别的模板变体
        
        Args:
            category: 类别
            subcategory: 子类别
            
        Returns:
            模板列表
        """
        return self.templates.get(category, {}).get(subcategory, [])
    
    def validate_options(self, target: str, weather: str, terrain: str) -> Tuple[bool, str]:
        """
        验证选项是否有效
        
        Args:
            target: 目标类型
            weather: 天气条件
            terrain: 地形类型
            
        Returns:
            (is_valid, error_message) 元组
        """
        available = self.get_available_options()
        
        if target not in available['targets']:
            return False, f"不支持的目标类型: {target}"
        
        if weather not in available['weather']:
            return False, f"不支持的天气条件: {weather}"
        
        if terrain not in available['terrain']:
            return False, f"不支持的地形类型: {terrain}"
        
        return True, "选项验证通过"

"""
军事提示词模板管理器
提供结构化的军事目标、天气条件和地形场景的提示词模板
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """提示词模板数据类"""
    category: str
    subcategory: str
    positive_prompt: str
    negative_prompt: str
    weight: float = 1.0

class PromptTemplateManager:
    """军事提示词模板管理器"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, List[PromptTemplate]]]:
        """初始化提示词模板"""
        templates = {
            "targets": {
                "tank": [
                    PromptTemplate(
                        "targets", "tank",
                        "military tank, armored vehicle, main battle tank, realistic, detailed, high quality, photorealistic",
                        "cartoon, anime, low quality, blurry, distorted, toy",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "tank",
                        "modern tank, camouflage pattern, military vehicle, steel armor, tracks, turret, realistic lighting",
                        "civilian vehicle, car, truck, low resolution, abstract",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "tank",
                        "heavy tank, battle tank, armored warfare, military equipment, detailed metal texture",
                        "peaceful, civilian, cartoon style, unrealistic",
                        0.8
                    )
                ],
                "aircraft": [
                    PromptTemplate(
                        "targets", "aircraft",
                        "military aircraft, fighter jet, warplane, realistic, detailed, high quality, photorealistic",
                        "civilian plane, commercial aircraft, cartoon, toy plane",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "aircraft",
                        "combat aircraft, military jet, fighter plane, detailed cockpit, realistic lighting",
                        "passenger plane, helicopter, drone, low quality",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "aircraft",
                        "stealth fighter, military aviation, jet engine, detailed aircraft, professional photography",
                        "civilian aviation, abstract, cartoon style",
                        0.9
                    )
                ],
                "ship": [
                    PromptTemplate(
                        "targets", "ship",
                        "military ship, warship, naval vessel, realistic, detailed, high quality, photorealistic",
                        "civilian ship, cruise ship, cartoon, toy boat",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "ship",
                        "naval warship, military vessel, destroyer, detailed hull, realistic water",
                        "yacht, fishing boat, cargo ship, low quality",
                        1.0
                    ),
                    PromptTemplate(
                        "targets", "ship",
                        "battleship, naval combat vessel, military maritime, detailed superstructure",
                        "peaceful vessel, civilian boat, abstract",
                        0.8
                    )
                ]
            },
            "weather": {
                "sunny": [
                    PromptTemplate(
                        "weather", "sunny",
                        "bright daylight, clear sky, sunny weather, good visibility, natural lighting",
                        "dark, night, cloudy, stormy, poor lighting",
                        1.0
                    )
                ],
                "rainy": [
                    PromptTemplate(
                        "weather", "rainy",
                        "rainy weather, wet surface, rain drops, overcast sky, realistic rain effect",
                        "dry, sunny, clear weather, bright lighting",
                        1.0
                    )
                ],
                "snowy": [
                    PromptTemplate(
                        "weather", "snowy",
                        "snowy weather, snow falling, winter scene, cold environment, snow covered",
                        "summer, hot weather, tropical, warm climate",
                        1.0
                    )
                ],
                "foggy": [
                    PromptTemplate(
                        "weather", "foggy",
                        "foggy weather, mist, low visibility, atmospheric fog, hazy conditions",
                        "clear visibility, bright, sunny, sharp details",
                        1.0
                    )
                ],
                "night": [
                    PromptTemplate(
                        "weather", "night",
                        "night scene, dark environment, artificial lighting, night vision, low light",
                        "daylight, bright, sunny, well lit",
                        1.0
                    )
                ]
            },
            "terrain": {
                "urban": [
                    PromptTemplate(
                        "terrain", "urban",
                        "urban environment, city setting, buildings, streets, urban warfare",
                        "rural, countryside, natural landscape, wilderness",
                        1.0
                    )
                ],
                "island": [
                    PromptTemplate(
                        "terrain", "island",
                        "island setting, coastal area, water surrounding, maritime environment",
                        "landlocked, desert, mountain, urban",
                        1.0
                    )
                ],
                "rural": [
                    PromptTemplate(
                        "terrain", "rural",
                        "rural environment, countryside, open field, natural terrain, rural setting",
                        "urban, city, industrial, heavily populated",
                        1.0
                    )
                ]
            }
        }
        return templates
    
    def get_random_prompt(self, target_type: str, weather: str = None, terrain: str = None) -> Tuple[str, str]:
        """生成随机组合的提示词"""
        # 获取目标提示词
        target_templates = self.templates["targets"].get(target_type, [])
        if not target_templates:
            raise ValueError(f"未找到目标类型: {target_type}")
        
        target_template = random.choices(
            target_templates, 
            weights=[t.weight for t in target_templates]
        )[0]
        
        positive_parts = [target_template.positive_prompt]
        negative_parts = [target_template.negative_prompt]
        
        # 添加天气条件
        if weather and weather in self.templates["weather"]:
            weather_template = random.choice(self.templates["weather"][weather])
            positive_parts.append(weather_template.positive_prompt)
            negative_parts.append(weather_template.negative_prompt)
        
        # 添加地形条件
        if terrain and terrain in self.templates["terrain"]:
            terrain_template = random.choice(self.templates["terrain"][terrain])
            positive_parts.append(terrain_template.positive_prompt)
            negative_parts.append(terrain_template.negative_prompt)
        
        positive_prompt = ", ".join(positive_parts)
        negative_prompt = ", ".join(negative_parts)
        
        return positive_prompt, negative_prompt
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """获取可用的选项"""
        return {
            "targets": list(self.templates["targets"].keys()),
            "weather": list(self.templates["weather"].keys()),
            "terrain": list(self.templates["terrain"].keys())
        }
    
    def add_custom_template(self, category: str, subcategory: str, template: PromptTemplate):
        """添加自定义模板"""
        if category not in self.templates:
            self.templates[category] = {}
        if subcategory not in self.templates[category]:
            self.templates[category][subcategory] = []
        
        self.templates[category][subcategory].append(template)
    
    def get_template_stats(self) -> Dict[str, int]:
        """获取模板统计信息"""
        stats = {}
        for category, subcategories in self.templates.items():
            for subcategory, templates in subcategories.items():
                key = f"{category}.{subcategory}"
                stats[key] = len(templates)
        return stats

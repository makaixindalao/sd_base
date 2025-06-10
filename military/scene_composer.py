"""
军事场景合成器
管理场景元素、天气、光照等，并将其组合成最终的生成提示词。
"""

import random
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SceneComposer:
    """
    场景合成器
    负责将各种场景元素组合成完整的军事场景描述
    """
    
    def __init__(self):
        """
        初始化场景合成器
        """
        self.scene_elements = self._load_scene_elements()
        self.composition_rules = self._load_composition_rules()
        
    def _load_scene_elements(self) -> Dict:
        """加载场景元素库"""
        return {
            "lighting": {
                "daylight": [
                    "natural daylight, bright illumination, clear visibility",
                    "golden hour lighting, warm sunlight, dramatic shadows",
                    "overcast daylight, diffused lighting, soft shadows"
                ],
                "night": [
                    "night scene, artificial lighting, tactical illumination",
                    "moonlit night, natural night lighting, starry sky",
                    "night vision, infrared lighting, military night ops"
                ],
                "golden_hour": [
                    "golden hour, warm lighting, dramatic atmosphere",
                    "sunset lighting, orange glow, cinematic mood",
                    "dawn lighting, soft golden light, peaceful atmosphere"
                ]
            },
            "camera_angle": {
                "eye-level": [
                    "eye level view, natural perspective, realistic angle",
                    "ground level shot, human perspective, standard view",
                    "horizontal view, balanced composition, natural framing"
                ],
                "high-angle": [
                    "high angle shot, aerial perspective, bird's eye view",
                    "elevated viewpoint, top-down angle, strategic overview",
                    "drone perspective, overhead shot, tactical view"
                ],
                "low-angle": [
                    "low angle shot, dramatic perspective, heroic view",
                    "ground level up, imposing angle, powerful composition",
                    "worm's eye view, dramatic framing, intimidating perspective"
                ]
            },
            "composition": {
                "rule_of_thirds": [
                    "rule of thirds composition, balanced framing, professional photography",
                    "off-center subject, dynamic composition, visual balance",
                    "thirds grid placement, artistic composition, pleasing arrangement"
                ],
                "centered": [
                    "centered composition, symmetrical framing, focused subject",
                    "central placement, balanced symmetry, formal composition",
                    "middle framing, direct focus, stable composition"
                ],
                "dynamic": [
                    "dynamic composition, action framing, energetic layout",
                    "diagonal composition, movement emphasis, active framing",
                    "asymmetrical balance, tension composition, dramatic layout"
                ]
            },
            "atmosphere": {
                "tense": [
                    "tense atmosphere, dramatic mood, high stakes",
                    "suspenseful environment, charged atmosphere, intense mood",
                    "combat tension, battlefield atmosphere, serious tone"
                ],
                "calm": [
                    "calm atmosphere, peaceful mood, serene environment",
                    "tranquil setting, relaxed atmosphere, quiet moment",
                    "stable environment, controlled situation, peaceful scene"
                ],
                "dramatic": [
                    "dramatic atmosphere, intense lighting, powerful mood",
                    "cinematic drama, emotional intensity, striking scene",
                    "high contrast, dramatic tension, compelling atmosphere"
                ]
            },
            "detail_level": {
                "high": [
                    "highly detailed, intricate details, fine textures",
                    "ultra-detailed, sharp focus, crisp definition",
                    "maximum detail, professional quality, HD rendering"
                ],
                "medium": [
                    "good detail level, clear definition, balanced quality",
                    "moderate detail, standard quality, clear imagery",
                    "adequate detail, decent resolution, acceptable quality"
                ],
                "cinematic": [
                    "cinematic quality, film-like detail, movie production value",
                    "Hollywood quality, professional cinematography, epic detail",
                    "blockbuster quality, premium rendering, theatrical detail"
                ]
            }
        }
    
    def _load_composition_rules(self) -> Dict:
        """加载构图规则"""
        return {
            "target_placement": {
                "tank": ["ground level", "terrain integrated", "tactical position"],
                "aircraft": ["sky dominant", "aerial space", "flight path"],
                "ship": ["water surface", "naval environment", "maritime setting"]
            },
            "weather_lighting": {
                "sunny": ["bright lighting", "clear shadows", "high contrast"],
                "rainy": ["diffused lighting", "wet surfaces", "atmospheric haze"],
                "snowy": ["cold lighting", "white balance", "winter atmosphere"],
                "foggy": ["soft lighting", "reduced contrast", "mysterious mood"],
                "night": ["artificial lighting", "dramatic shadows", "tactical illumination"]
            },
            "terrain_elements": {
                "urban": ["architectural elements", "city infrastructure", "urban textures"],
                "island": ["coastal elements", "water features", "tropical vegetation"],
                "rural": ["natural elements", "open spaces", "countryside features"]
            }
        }
    
    def compose_scene(self, base_elements: Dict) -> Tuple[str, str]:
        """
        根据基础元素合成一个完整的场景描述
        
        Args:
            base_elements: 包含目标、地形等基础信息的字典
                          例如: {"target": "tank", "weather": "sunny", "terrain": "urban"}
        
        Returns:
            一个包含正面和负面提示词的元组 (positive_prompt, negative_prompt)
        """
        try:
            if not isinstance(base_elements, dict):
                # 简单处理非字典输入
                positive_prompt = "a military scene"
                negative_prompt = "blurry, low quality"
                return positive_prompt, negative_prompt
            
            # 提取基础元素
            target = base_elements.get("target", "tank")
            weather = base_elements.get("weather", "sunny")
            terrain = base_elements.get("terrain", "urban")
            
            # 自动选择场景元素
            lighting = self._select_lighting(weather)
            camera_angle = self._select_camera_angle(target)
            composition = self._select_composition()
            atmosphere = self._select_atmosphere(weather, terrain)
            detail_level = self._select_detail_level()
            
            # 构建正面提示词
            positive_parts = [
                f"{target} in {terrain} environment",
                f"{weather} weather conditions",
                lighting,
                camera_angle,
                composition,
                atmosphere,
                detail_level,
                "professional photography, realistic rendering"
            ]
            
            positive_prompt = ", ".join(positive_parts)
            
            # 构建负面提示词
            negative_prompt = self._build_negative_prompt(target, weather, terrain)
            
            return positive_prompt, negative_prompt
            
        except Exception as e:
            logger.error(f"场景合成失败: {e}")
            # 返回基础提示词
            return "military scene, realistic", "blurry, low quality"
    
    def _select_lighting(self, weather: str) -> str:
        """根据天气选择合适的光照"""
        lighting_map = {
            "sunny": "daylight",
            "rainy": "daylight",
            "snowy": "daylight", 
            "foggy": "daylight",
            "night": "night"
        }
        
        lighting_type = lighting_map.get(weather, "daylight")
        return random.choice(self.scene_elements["lighting"][lighting_type])
    
    def _select_camera_angle(self, target: str) -> str:
        """根据目标类型选择合适的相机角度"""
        angle_preferences = {
            "tank": ["eye-level", "low-angle"],
            "aircraft": ["low-angle", "high-angle"],
            "ship": ["eye-level", "high-angle"]
        }
        
        preferred_angles = angle_preferences.get(target, ["eye-level"])
        selected_angle = random.choice(preferred_angles)
        return random.choice(self.scene_elements["camera_angle"][selected_angle])
    
    def _select_composition(self) -> str:
        """选择构图方式"""
        composition_types = list(self.scene_elements["composition"].keys())
        selected_type = random.choice(composition_types)
        return random.choice(self.scene_elements["composition"][selected_type])
    
    def _select_atmosphere(self, weather: str, terrain: str) -> str:
        """根据天气和地形选择氛围"""
        atmosphere_map = {
            ("sunny", "urban"): "calm",
            ("rainy", "urban"): "dramatic",
            ("night", "urban"): "tense",
            ("foggy", "rural"): "dramatic",
            ("snowy", "rural"): "calm"
        }
        
        atmosphere_type = atmosphere_map.get((weather, terrain), "calm")
        return random.choice(self.scene_elements["atmosphere"][atmosphere_type])
    
    def _select_detail_level(self) -> str:
        """选择细节级别"""
        detail_types = ["high", "cinematic"]  # 优先选择高质量
        selected_type = random.choice(detail_types)
        return random.choice(self.scene_elements["detail_level"][selected_type])
    
    def _build_negative_prompt(self, target: str, weather: str, terrain: str) -> str:
        """构建负面提示词"""
        base_negative = [
            "blurry, low quality, pixelated, distorted",
            "cartoon, anime, unrealistic, toy-like",
            "poor lighting, bad composition, amateur"
        ]
        
        # 目标特定的负面词
        target_negative = {
            "tank": "civilian vehicle, car, passenger vehicle",
            "aircraft": "civilian plane, commercial aircraft, passenger jet",
            "ship": "civilian boat, yacht, cruise ship"
        }
        
        # 天气相反条件
        weather_negative = {
            "sunny": "dark, night, storm, rain",
            "rainy": "dry, sunny, desert",
            "snowy": "hot, tropical, summer",
            "foggy": "clear, bright, sharp",
            "night": "daylight, bright, sunny"
        }
        
        # 地形相反条件
        terrain_negative = {
            "urban": "rural, wilderness, forest",
            "island": "landlocked, desert, mountains", 
            "rural": "urban, city, industrial"
        }
        
        negative_parts = [
            random.choice(base_negative),
            target_negative.get(target, ""),
            weather_negative.get(weather, ""),
            terrain_negative.get(terrain, "")
        ]
        
        return ", ".join(filter(None, negative_parts))
    
    def get_available_options(self) -> Dict[str, List[str]]:
        """
        返回可用的场景选项
        """
        return {
            "lighting": list(self.scene_elements["lighting"].keys()),
            "camera_angle": list(self.scene_elements["camera_angle"].keys()),
            "composition": list(self.scene_elements["composition"].keys()),
            "atmosphere": list(self.scene_elements["atmosphere"].keys()),
            "detail_level": list(self.scene_elements["detail_level"].keys())
        }
    
    def add_custom_element(self, category: str, element_type: str, variations: List[str]):
        """
        添加自定义场景元素
        
        Args:
            category: 元素类别 (lighting, camera_angle, composition, etc.)
            element_type: 元素类型
            variations: 变体列表
        """
        try:
            if category not in self.scene_elements:
                self.scene_elements[category] = {}
            
            self.scene_elements[category][element_type] = variations
            logger.info(f"添加自定义场景元素: {category}.{element_type}")
            
        except Exception as e:
            logger.error(f"添加自定义场景元素失败: {e}")

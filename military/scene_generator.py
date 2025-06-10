"""
军事场景生成器
通过组合场景、目标并使用Inpainting技术生成最终图像
"""

import time
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path
from PIL import Image, ImageDraw
import logging

from sd_generator import SDGenerator
from config import Config

logger = logging.getLogger(__name__)

class MilitarySceneGenerator:
    """
    使用图片蒙版（Inpainting）方式，将军事目标合成到场景中。
    """

    def __init__(self, sd_generator: SDGenerator = None):
        """初始化军事场景生成器"""
        self.sd_generator = sd_generator or SDGenerator()
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
            "end_time": None,
        }

    def set_callbacks(self, progress_callback: Callable = None, status_callback: Callable = None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        if self.sd_generator:
            self.sd_generator.set_callbacks(progress_callback, status_callback)

    def _update_status(self, status: str):
        """更新状态"""
        if self.status_callback:
            self.status_callback(status)
        logger.info(status)

    def _create_composition_and_mask(self, 
                                     base_scene: Image.Image, 
                                     targets: List[Dict[str, any]]) -> Tuple[Image.Image, Image.Image]:
        """
        创建合成图和蒙版图
        - base_scene: 背景场景图
        - targets: 目标列表, e.g., [{'image': Image.Image, 'position': (x, y)}]
        """
        self._update_status("正在合成场景与目标...")
        
        # 确保基础场景是RGBA模式，以便粘贴带透明度的图片
        composite_image = base_scene.copy().convert("RGBA")

        # 创建一个纯黑色的蒙版图
        mask_image = Image.new("L", composite_image.size, 0)
        draw = ImageDraw.Draw(mask_image)

        for target in targets:
            target_img = target['image'].convert("RGBA")
            position = target['position']
            
            # 粘贴目标到场景上
            composite_image.paste(target_img, position, target_img)
            
            # 在蒙版上对应位置绘制白色区域
            # 从目标图像的alpha通道创建蒙版
            target_mask = target_img.getchannel('A')
            mask_image.paste(255, position, target_mask)

        self._update_status("✅ 合成完成，蒙版已生成。")
        return composite_image.convert("RGB"), mask_image

    def generate_scene(self,
                       base_scene: Image.Image,
                       targets: List[Dict[str, any]],
                       prompt: str,
                       negative_prompt: str = "low quality, blurry, distorted, text, watermark, signature",
                       output_dir: str = "outputs/military_scenes",
                       **generation_kwargs) -> Optional[str]:
        """
        生成单个军事场景
        """
        self.generation_stats.update({"start_time": time.time(), "total_generated": 1})
        
        try:
            # 1. 创建合成图和蒙版
            composite_image, mask_image = self._create_composition_and_mask(base_scene, targets)

            # 为了调试，可以保存中间图像
            # composite_image.save("debug_composite.png")
            # mask_image.save("debug_mask.png")

            # 2. 调用 Inpainting 生成器
            self._update_status("调用Inpainting模型进行场景融合...")
            final_image = self.sd_generator.generate_inpainting(
                prompt=prompt,
                image=composite_image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                **generation_kwargs
            )

            if not final_image:
                self._update_status("❌ 场景融合失败。")
                self.generation_stats["failed"] = 1
                return None

            self._update_status("✅ 场景生成成功！")
            self.generation_stats["successful"] = 1

            # 3. 保存结果
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            filename = f"scene_{timestamp}.png"
            filepath = output_path / filename
            final_image.save(filepath)
            
            self._update_status(f"图像已保存至: {filepath}")
            
            self.generation_stats["end_time"] = time.time()
            return str(filepath)

        except Exception as e:
            self.generation_stats["failed"] = 1
            self.generation_stats["end_time"] = time.time()
            error_msg = f"生成场景时发生未知错误: {e}"
            self._update_status(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    def get_generation_stats(self) -> Dict:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        if stats.get("start_time") and stats.get("end_time"):
            stats["total_time"] = stats["end_time"] - stats["start_time"]
        return stats 
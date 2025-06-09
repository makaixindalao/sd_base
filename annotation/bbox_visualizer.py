"""
边界框可视化器
提供边界框的绘制和可视化功能
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class BBoxVisualizer:
    """边界框可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.class_colors = {
            0: (255, 0, 0),    # tank - 红色
            1: (0, 255, 0),    # aircraft - 绿色
            2: (0, 0, 255),    # ship - 蓝色
        }
        
        self.class_names = {
            0: "Tank",
            1: "Aircraft", 
            2: "Ship"
        }
        
        self.default_color = (255, 255, 0)  # 黄色
        self.bbox_thickness = 2
        self.font_scale = 0.6
        self.font_thickness = 2
    
    def draw_bboxes_on_image(self, 
                           image: Image.Image, 
                           detections: List[Dict],
                           show_confidence: bool = True,
                           show_class_name: bool = True) -> Image.Image:
        """在图像上绘制边界框"""
        
        # 转换PIL图像为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            bbox = detection["bbox"]  # [x, y, width, height]
            category_id = detection.get("category_id", 0)
            confidence = detection.get("confidence", 1.0)
            
            # 获取边界框坐标
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # 获取颜色
            color = self.class_colors.get(category_id, self.default_color)
            
            # 绘制边界框
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # 准备标签文本
            label_parts = []
            if show_class_name:
                class_name = self.class_names.get(category_id, f"Class_{category_id}")
                label_parts.append(class_name)
            
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                )
                
                # 绘制文本背景
                cv2.rectangle(
                    cv_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    cv_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),  # 白色文本
                    self.font_thickness
                )
        
        # 转换回PIL格式
        result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        return result_image
    
    def create_detection_summary_image(self, 
                                     detections: List[Dict],
                                     image_size: Tuple[int, int] = (400, 300)) -> Image.Image:
        """创建检测结果摘要图像"""
        
        # 创建空白图像
        summary_image = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(summary_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # 统计检测结果
        class_counts = {}
        total_detections = len(detections)
        
        for detection in detections:
            category_id = detection.get("category_id", 0)
            class_name = self.class_names.get(category_id, f"Class_{category_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 绘制标题
        title = f"检测结果摘要 (共{total_detections}个目标)"
        draw.text((10, 10), title, fill=(0, 0, 0), font=font)
        
        # 绘制统计信息
        y_offset = 50
        for class_name, count in class_counts.items():
            # 获取对应的颜色
            category_id = next((k for k, v in self.class_names.items() if v == class_name), 0)
            color = self.class_colors.get(category_id, self.default_color)
            
            # 绘制颜色块
            draw.rectangle([10, y_offset, 30, y_offset + 15], fill=color)
            
            # 绘制文本
            text = f"{class_name}: {count}个"
            draw.text((40, y_offset), text, fill=(0, 0, 0), font=small_font)
            
            y_offset += 25
        
        # 绘制置信度信息
        if detections:
            confidences = [d.get("confidence", 1.0) for d in detections]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            y_offset += 20
            draw.text((10, y_offset), "置信度统计:", fill=(0, 0, 0), font=font)
            y_offset += 25
            
            confidence_info = [
                f"平均: {avg_confidence:.3f}",
                f"最小: {min_confidence:.3f}",
                f"最大: {max_confidence:.3f}"
            ]
            
            for info in confidence_info:
                draw.text((10, y_offset), info, fill=(0, 0, 0), font=small_font)
                y_offset += 20
        
        return summary_image
    
    def visualize_batch_results(self, 
                              results: List[Dict],
                              output_dir: str = "visualizations",
                              save_individual: bool = True,
                              save_summary: bool = True) -> List[str]:
        """批量可视化检测结果"""
        
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        try:
            for i, result in enumerate(results):
                if "error" in result:
                    logger.warning(f"跳过错误结果: {result.get('image_path', 'unknown')}")
                    continue
                
                image_path = result["image_path"]
                detections = result.get("annotations", [])
                
                if save_individual:
                    # 加载原图像
                    original_image = Image.open(image_path)
                    
                    # 绘制边界框
                    visualized_image = self.draw_bboxes_on_image(original_image, detections)
                    
                    # 保存可视化结果
                    image_name = Path(image_path).stem
                    output_file = output_path / f"{image_name}_visualized.png"
                    visualized_image.save(output_file)
                    saved_files.append(str(output_file))
            
            if save_summary and results:
                # 创建总体摘要
                all_detections = []
                for result in results:
                    if "annotations" in result:
                        all_detections.extend(result["annotations"])
                
                summary_image = self.create_detection_summary_image(all_detections)
                summary_file = output_path / "detection_summary.png"
                summary_image.save(summary_file)
                saved_files.append(str(summary_file))
            
            logger.info(f"批量可视化完成，保存了 {len(saved_files)} 个文件")
            return saved_files
            
        except Exception as e:
            logger.error(f"批量可视化失败: {e}")
            return saved_files
    
    def set_class_colors(self, color_mapping: Dict[int, Tuple[int, int, int]]):
        """设置类别颜色映射"""
        self.class_colors.update(color_mapping)
    
    def set_class_names(self, name_mapping: Dict[int, str]):
        """设置类别名称映射"""
        self.class_names.update(name_mapping)
    
    def set_visualization_params(self, 
                               bbox_thickness: int = None,
                               font_scale: float = None,
                               font_thickness: int = None):
        """设置可视化参数"""
        if bbox_thickness is not None:
            self.bbox_thickness = bbox_thickness
        if font_scale is not None:
            self.font_scale = font_scale
        if font_thickness is not None:
            self.font_thickness = font_thickness
    
    def create_comparison_image(self, 
                              original_image: Image.Image,
                              detections: List[Dict],
                              ground_truth: List[Dict] = None) -> Image.Image:
        """创建对比图像（原图 vs 检测结果 vs 真实标注）"""
        
        images = [original_image]
        
        # 添加检测结果图像
        if detections:
            detection_image = self.draw_bboxes_on_image(original_image.copy(), detections)
            images.append(detection_image)
        
        # 添加真实标注图像
        if ground_truth:
            # 使用不同的颜色绘制真实标注
            original_colors = self.class_colors.copy()
            # 使用更亮的颜色表示真实标注
            gt_colors = {k: tuple(min(255, c + 50) for c in v) for k, v in original_colors.items()}
            self.set_class_colors(gt_colors)
            
            gt_image = self.draw_bboxes_on_image(original_image.copy(), ground_truth)
            images.append(gt_image)
            
            # 恢复原始颜色
            self.set_class_colors(original_colors)
        
        # 水平拼接图像
        if len(images) == 1:
            return images[0]
        
        # 计算拼接后的尺寸
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # 创建拼接图像
        comparison_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        x_offset = 0
        for img in images:
            comparison_image.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return comparison_image

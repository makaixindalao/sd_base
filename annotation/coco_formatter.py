"""
COCO格式转换器
将检测结果转换为标准COCO JSON格式
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class COCOFormatter:
    """COCO格式转换器"""
    
    def __init__(self):
        """初始化COCO格式转换器"""
        self.categories = [
            {"id": 0, "name": "tank", "supercategory": "military_vehicle"},
            {"id": 1, "name": "aircraft", "supercategory": "military_vehicle"},
            {"id": 2, "name": "ship", "supercategory": "military_vehicle"}
        ]
        
        self.info = {
            "description": "Military Target Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Military Dataset Generator",
            "date_created": datetime.now().isoformat()
        }
        
        self.licenses = [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ]
    
    def create_coco_dataset(self, 
                           annotations_data: List[Dict],
                           dataset_name: str = "military_dataset") -> Dict[str, Any]:
        """创建COCO格式数据集"""
        
        coco_dataset = {
            "info": self.info.copy(),
            "licenses": self.licenses,
            "categories": self.categories,
            "images": [],
            "annotations": []
        }
        
        # 更新数据集信息
        coco_dataset["info"]["description"] = f"{dataset_name} - Military Target Dataset"
        coco_dataset["info"]["date_created"] = datetime.now().isoformat()
        
        annotation_id = 1
        
        for image_data in annotations_data:
            if "error" in image_data:
                logger.warning(f"跳过错误图像: {image_data.get('image_path', 'unknown')}")
                continue
            
            # 添加图像信息
            image_path = Path(image_data["image_path"])
            image_info = {
                "id": image_data["image_id"],
                "file_name": image_path.name,
                "width": image_data["image_info"]["width"],
                "height": image_data["image_info"]["height"],
                "date_captured": datetime.now().isoformat(),
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            coco_dataset["images"].append(image_info)
            
            # 添加标注信息
            for detection in image_data["annotations"]:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_data["image_id"],
                    "category_id": detection["category_id"],
                    "bbox": detection["bbox"],
                    "area": detection["area"],
                    "iscrowd": detection["iscrowd"],
                    "segmentation": []  # 暂不支持分割
                }
                
                # 添加置信度信息（非标准COCO字段）
                if "confidence" in detection:
                    annotation["confidence"] = detection["confidence"]
                
                coco_dataset["annotations"].append(annotation)
                annotation_id += 1
        
        return coco_dataset
    
    def save_coco_dataset(self, 
                         coco_dataset: Dict[str, Any], 
                         output_path: str,
                         indent: int = 2) -> bool:
        """保存COCO数据集到JSON文件"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_dataset, f, indent=indent, ensure_ascii=False)
            
            logger.info(f"COCO数据集已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存COCO数据集失败: {e}")
            return False
    
    def split_dataset(self, 
                     coco_dataset: Dict[str, Any],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Tuple[Dict, Dict, Dict]:
        """分割数据集为训练集、验证集和测试集"""
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"数据集分割比例总和必须为1.0，当前为: {total_ratio}")
        
        images = coco_dataset["images"]
        annotations = coco_dataset["annotations"]
        
        # 按图像ID分组标注
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 计算分割点
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # 分割图像
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # 创建子数据集
        def create_subset(subset_images):
            subset_annotations = []
            for img in subset_images:
                img_id = img["id"]
                if img_id in annotations_by_image:
                    subset_annotations.extend(annotations_by_image[img_id])
            
            return {
                "info": coco_dataset["info"],
                "licenses": coco_dataset["licenses"],
                "categories": coco_dataset["categories"],
                "images": subset_images,
                "annotations": subset_annotations
            }
        
        train_dataset = create_subset(train_images)
        val_dataset = create_subset(val_images)
        test_dataset = create_subset(test_images)
        
        logger.info(f"数据集分割完成: 训练集({len(train_images)}), 验证集({len(val_images)}), 测试集({len(test_images)})")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataset_statistics(self, coco_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """获取数据集统计信息"""
        images = coco_dataset["images"]
        annotations = coco_dataset["annotations"]
        categories = coco_dataset["categories"]
        
        # 基本统计
        stats = {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "total_categories": len(categories),
            "avg_annotations_per_image": len(annotations) / max(len(images), 1)
        }
        
        # 按类别统计
        category_counts = {}
        for cat in categories:
            category_counts[cat["name"]] = 0
        
        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = next((cat["name"] for cat in categories if cat["id"] == cat_id), "unknown")
            if cat_name in category_counts:
                category_counts[cat_name] += 1
        
        stats["category_distribution"] = category_counts
        
        # 图像尺寸统计
        widths = [img["width"] for img in images]
        heights = [img["height"] for img in images]
        
        if widths and heights:
            stats["image_size_stats"] = {
                "width_range": [min(widths), max(widths)],
                "height_range": [min(heights), max(heights)],
                "avg_width": sum(widths) / len(widths),
                "avg_height": sum(heights) / len(heights)
            }
        
        return stats
    
    def validate_coco_format(self, coco_dataset: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证COCO格式的有效性"""
        errors = []
        
        # 检查必需字段
        required_fields = ["info", "licenses", "categories", "images", "annotations"]
        for field in required_fields:
            if field not in coco_dataset:
                errors.append(f"缺少必需字段: {field}")
        
        if errors:
            return False, errors
        
        # 检查图像字段
        for i, img in enumerate(coco_dataset["images"]):
            required_img_fields = ["id", "file_name", "width", "height"]
            for field in required_img_fields:
                if field not in img:
                    errors.append(f"图像 {i} 缺少字段: {field}")
        
        # 检查标注字段
        for i, ann in enumerate(coco_dataset["annotations"]):
            required_ann_fields = ["id", "image_id", "category_id", "bbox", "area"]
            for field in required_ann_fields:
                if field not in ann:
                    errors.append(f"标注 {i} 缺少字段: {field}")
        
        # 检查类别ID一致性
        category_ids = {cat["id"] for cat in coco_dataset["categories"]}
        for ann in coco_dataset["annotations"]:
            if ann["category_id"] not in category_ids:
                errors.append(f"标注中的类别ID {ann['category_id']} 不存在于类别定义中")
        
        return len(errors) == 0, errors

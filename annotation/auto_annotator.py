"""
自动标注器
基于YOLO等检测模型自动生成军事目标的边界框标注
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from PIL import Image
import logging
from pathlib import Path

try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class AutoAnnotator:
    """自动标注器"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        """初始化自动标注器"""
        self.model_name = model_name
        self.model = None
        self.class_mapping = {
            "tank": 0,
            "aircraft": 1,
            "ship": 2
        }
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载检测模型"""
        if not YOLO_AVAILABLE:
            logger.error("ultralytics未安装，无法使用YOLO模型")
            return False
        
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"成功加载检测模型: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"加载检测模型失败: {e}")
            return False
    
    def set_callbacks(self, progress_callback: Callable = None, status_callback: Callable = None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def _update_progress(self, progress: float):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(progress)
    
    def _update_status(self, status: str):
        """更新状态"""
        if self.status_callback:
            self.status_callback(status)
        logger.info(status)
    
    def detect_objects(self, image: Image.Image) -> List[Dict]:
        """检测图像中的目标对象"""
        if not self.model:
            logger.error("检测模型未加载")
            return []
        
        try:
            # 转换PIL图像为OpenCV格式
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 运行检测
            results = self.model(cv_image, conf=self.confidence_threshold, iou=self.nms_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 转换为COCO格式 (x, y, width, height)
                        x = float(x1)
                        y = float(y1)
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        area = width * height
                        
                        detection = {
                            "bbox": [x, y, width, height],
                            "area": area,
                            "category_id": self._map_class_id(class_id),
                            "confidence": float(confidence),
                            "iscrowd": 0
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return []
    
    def _map_class_id(self, yolo_class_id: int) -> int:
        """映射YOLO类别ID到军事目标类别ID"""
        # YOLO的一些常见类别映射
        yolo_to_military = {
            2: 0,   # car -> tank (近似)
            5: 1,   # airplane -> aircraft
            8: 2,   # boat -> ship
            # 可以根据需要添加更多映射
        }
        return yolo_to_military.get(yolo_class_id, 0)  # 默认为tank
    
    def annotate_image(self, image: Image.Image, target_type: str = None) -> Dict:
        """标注单张图像"""
        self._update_status("正在检测目标...")
        
        detections = self.detect_objects(image)
        
        # 如果指定了目标类型，过滤检测结果
        if target_type and target_type in self.class_mapping:
            target_class_id = self.class_mapping[target_type]
            detections = [d for d in detections if d["category_id"] == target_class_id]
        
        annotation = {
            "image_info": {
                "width": image.width,
                "height": image.height,
                "channels": len(image.getbands())
            },
            "annotations": detections,
            "detection_count": len(detections)
        }
        
        self._update_status(f"检测到 {len(detections)} 个目标")
        return annotation
    
    def annotate_batch(self, image_paths: List[str], target_types: List[str] = None) -> List[Dict]:
        """批量标注图像"""
        results = []
        total_images = len(image_paths)
        
        self._update_status(f"开始批量标注 {total_images} 张图像...")
        
        for i, image_path in enumerate(image_paths):
            try:
                # 加载图像
                image = Image.open(image_path)
                
                # 获取对应的目标类型
                target_type = None
                if target_types and i < len(target_types):
                    target_type = target_types[i]
                
                # 标注图像
                annotation = self.annotate_image(image, target_type)
                annotation["image_path"] = image_path
                annotation["image_id"] = i + 1
                
                results.append(annotation)
                
                # 更新进度
                progress = (i + 1) / total_images * 100
                self._update_progress(progress)
                self._update_status(f"已标注 {i + 1}/{total_images} 张图像")
                
            except Exception as e:
                logger.error(f"标注图像 {image_path} 失败: {e}")
                results.append({
                    "image_path": image_path,
                    "image_id": i + 1,
                    "error": str(e),
                    "annotations": [],
                    "detection_count": 0
                })
        
        self._update_status(f"批量标注完成! 共处理 {total_images} 张图像")
        return results
    
    def set_detection_parameters(self, confidence_threshold: float = None, nms_threshold: float = None):
        """设置检测参数"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
    
    def get_class_mapping(self) -> Dict[str, int]:
        """获取类别映射"""
        return self.class_mapping.copy()
    
    def set_class_mapping(self, mapping: Dict[str, int]):
        """设置类别映射"""
        self.class_mapping = mapping
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

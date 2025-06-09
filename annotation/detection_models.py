"""
检测模型管理器
管理不同的目标检测模型
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DetectionModelManager:
    """检测模型管理器"""
    
    def __init__(self, models_dir: str = "detection_models"):
        """初始化模型管理器"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 可用的预训练模型
        self.available_models = {
            "yolov8n": {
                "name": "YOLOv8 Nano",
                "file": "yolov8n.pt",
                "description": "最小的YOLOv8模型，速度最快",
                "size": "6MB",
                "accuracy": "低",
                "speed": "最快"
            },
            "yolov8s": {
                "name": "YOLOv8 Small", 
                "file": "yolov8s.pt",
                "description": "小型YOLOv8模型，平衡速度和精度",
                "size": "22MB",
                "accuracy": "中等",
                "speed": "快"
            },
            "yolov8m": {
                "name": "YOLOv8 Medium",
                "file": "yolov8m.pt", 
                "description": "中型YOLOv8模型，较好的精度",
                "size": "52MB",
                "accuracy": "高",
                "speed": "中等"
            },
            "yolov8l": {
                "name": "YOLOv8 Large",
                "file": "yolov8l.pt",
                "description": "大型YOLOv8模型，高精度",
                "size": "87MB", 
                "accuracy": "很高",
                "speed": "慢"
            },
            "yolov8x": {
                "name": "YOLOv8 Extra Large",
                "file": "yolov8x.pt",
                "description": "最大的YOLOv8模型，最高精度",
                "size": "136MB",
                "accuracy": "最高", 
                "speed": "最慢"
            }
        }
        
        # 当前加载的模型
        self.current_model = None
        self.current_model_name = None
        
        # 军事目标类别映射
        self.military_class_mapping = {
            # COCO类别ID -> 军事目标类别ID
            2: 0,   # car -> tank (近似)
            3: 0,   # motorcycle -> tank (近似)
            5: 1,   # airplane -> aircraft
            8: 2,   # boat -> ship
            # 可以根据需要添加更多映射
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """获取可用模型列表"""
        return self.available_models.copy()
    
    def download_model(self, model_key: str) -> bool:
        """下载指定模型"""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics未安装，无法下载模型")
            return False
        
        if model_key not in self.available_models:
            logger.error(f"未知模型: {model_key}")
            return False
        
        try:
            model_info = self.available_models[model_key]
            model_file = model_info["file"]
            
            logger.info(f"开始下载模型: {model_info['name']}")
            
            # YOLO会自动下载模型到默认位置
            model = YOLO(model_file)
            
            logger.info(f"模型下载完成: {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            return False
    
    def load_model(self, model_key: str) -> bool:
        """加载指定模型"""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics未安装，无法加载模型")
            return False
        
        if model_key not in self.available_models:
            logger.error(f"未知模型: {model_key}")
            return False
        
        try:
            model_info = self.available_models[model_key]
            model_file = model_info["file"]
            
            logger.info(f"加载模型: {model_info['name']}")
            
            # 加载模型
            self.current_model = YOLO(model_file)
            self.current_model_name = model_key
            
            logger.info(f"模型加载成功: {model_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.current_model = None
            self.current_model_name = None
            return False
    
    def load_custom_model(self, model_path: str) -> bool:
        """加载自定义模型"""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics未安装，无法加载模型")
            return False
        
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            logger.info(f"加载自定义模型: {model_path}")
            
            self.current_model = YOLO(str(model_file))
            self.current_model_name = f"custom_{model_file.stem}"
            
            logger.info("自定义模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载自定义模型失败: {e}")
            self.current_model = None
            self.current_model_name = None
            return False
    
    def is_model_loaded(self) -> bool:
        """检查是否有模型已加载"""
        return self.current_model is not None
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        if not self.is_model_loaded():
            return None
        
        info = {
            "name": self.current_model_name,
            "loaded": True
        }
        
        if self.current_model_name in self.available_models:
            info.update(self.available_models[self.current_model_name])
        
        # 添加设备信息
        if TORCH_AVAILABLE and hasattr(self.current_model, 'device'):
            info["device"] = str(self.current_model.device)
        
        return info
    
    def predict(self, 
                image_path: str,
                conf_threshold: float = 0.5,
                iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """使用当前模型进行预测"""
        if not self.is_model_loaded():
            logger.error("没有加载的模型")
            return []
        
        try:
            # 运行预测
            results = self.current_model(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 映射到军事目标类别
                        military_class_id = self.military_class_mapping.get(class_id, class_id)
                        
                        # 转换为COCO格式 (x, y, width, height)
                        x = float(x1)
                        y = float(y1)
                        width = float(x2 - x1)
                        height = float(y2 - y1)
                        area = width * height
                        
                        detection = {
                            "bbox": [x, y, width, height],
                            "area": area,
                            "category_id": military_class_id,
                            "confidence": float(confidence),
                            "original_class_id": class_id,
                            "iscrowd": 0
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return []
    
    def batch_predict(self, 
                     image_paths: List[str],
                     conf_threshold: float = 0.5,
                     iou_threshold: float = 0.4) -> List[List[Dict[str, Any]]]:
        """批量预测"""
        if not self.is_model_loaded():
            logger.error("没有加载的模型")
            return []
        
        all_detections = []
        
        for image_path in image_paths:
            detections = self.predict(image_path, conf_threshold, iou_threshold)
            all_detections.append(detections)
        
        return all_detections
    
    def set_military_class_mapping(self, mapping: Dict[int, int]):
        """设置军事目标类别映射"""
        self.military_class_mapping.update(mapping)
    
    def get_military_class_mapping(self) -> Dict[int, int]:
        """获取军事目标类别映射"""
        return self.military_class_mapping.copy()
    
    def get_model_performance_info(self) -> Dict[str, Any]:
        """获取模型性能信息"""
        if not self.is_model_loaded():
            return {"error": "没有加载的模型"}
        
        info = {
            "model_name": self.current_model_name,
            "device": "unknown",
            "memory_usage": "unknown"
        }
        
        try:
            if TORCH_AVAILABLE:
                # 获取设备信息
                if hasattr(self.current_model, 'device'):
                    info["device"] = str(self.current_model.device)
                
                # 获取GPU内存使用情况
                if torch.cuda.is_available():
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        except Exception as e:
            logger.warning(f"获取性能信息失败: {e}")
        
        return info
    
    def unload_model(self):
        """卸载当前模型"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            
            # 清理GPU内存
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("模型已卸载")
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的图像格式"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def validate_image_format(self, image_path: str) -> bool:
        """验证图像格式是否支持"""
        file_ext = Path(image_path).suffix.lower()
        return file_ext in self.get_supported_formats()

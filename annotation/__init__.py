"""
自动标注模块
提供军事目标检测、分类和COCO格式标注功能
"""

from .auto_annotator import AutoAnnotator
from .detection_models import DetectionModelManager
from .coco_formatter import COCOFormatter
from .bbox_visualizer import BBoxVisualizer

__all__ = [
    'AutoAnnotator',
    'DetectionModelManager',
    'COCOFormatter',
    'BBoxVisualizer'
]

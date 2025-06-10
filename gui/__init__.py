"""
扩展GUI模块
提供军事目标生成、标注管理、数据集管理等专用界面
"""

from .military_panel import MilitaryPanel
from .annotation_panel import AnnotationPanel
from .dataset_panel import DatasetPanel

__all__ = [
    'MilitaryPanel',
    'AnnotationPanel',
    'DatasetPanel'
]

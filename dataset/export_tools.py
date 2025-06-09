"""
数据集导出工具
支持多种格式的数据集导出
"""

import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ExportTools:
    """数据集导出工具"""
    
    def __init__(self):
        """初始化导出工具"""
        self.supported_formats = ["coco", "yolo", "pascal_voc", "tensorflow"]
    
    def export_to_yolo(self, 
                      coco_file: str,
                      images_dir: str,
                      output_dir: str,
                      class_names: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """导出为YOLO格式"""
        
        try:
            # 加载COCO数据
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            images = coco_data["images"]
            annotations = coco_data["annotations"]
            categories = coco_data["categories"]
            
            # 创建输出目录结构
            output_path = Path(output_dir)
            images_output = output_path / "images"
            labels_output = output_path / "labels"
            
            images_output.mkdir(parents=True, exist_ok=True)
            labels_output.mkdir(parents=True, exist_ok=True)
            
            # 创建类别映射
            if class_names is None:
                class_names = [cat["name"] for cat in sorted(categories, key=lambda x: x["id"])]
            
            category_id_to_yolo = {cat["id"]: i for i, cat in enumerate(sorted(categories, key=lambda x: x["id"]))}
            
            # 按图像分组标注
            annotations_by_image = {}
            for ann in annotations:
                image_id = ann["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # 处理每张图像
            processed_images = 0
            for image_info in images:
                image_id = image_info["id"]
                image_filename = image_info["file_name"]
                image_width = image_info["width"]
                image_height = image_info["height"]
                
                # 复制图像文件
                src_image_path = Path(images_dir) / image_filename
                dst_image_path = images_output / image_filename
                
                if src_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)
                    
                    # 创建对应的标签文件
                    label_filename = Path(image_filename).stem + ".txt"
                    label_path = labels_output / label_filename
                    
                    # 转换标注为YOLO格式
                    yolo_annotations = []
                    if image_id in annotations_by_image:
                        for ann in annotations_by_image[image_id]:
                            bbox = ann["bbox"]  # [x, y, width, height]
                            category_id = ann["category_id"]
                            
                            # 转换为YOLO格式 (相对坐标)
                            x_center = (bbox[0] + bbox[2] / 2) / image_width
                            y_center = (bbox[1] + bbox[3] / 2) / image_height
                            width = bbox[2] / image_width
                            height = bbox[3] / image_height
                            
                            yolo_class_id = category_id_to_yolo.get(category_id, 0)
                            
                            yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    # 保存标签文件
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    processed_images += 1
                else:
                    logger.warning(f"图像文件不存在: {src_image_path}")
            
            # 创建classes.txt文件
            classes_file = output_path / "classes.txt"
            with open(classes_file, 'w') as f:
                f.write('\n'.join(class_names))
            
            # 创建数据集配置文件
            dataset_config = {
                "path": str(output_path),
                "train": "images",
                "val": "images",
                "nc": len(class_names),
                "names": class_names
            }
            
            config_file = output_path / "dataset.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            export_info = {
                "format": "yolo",
                "output_directory": str(output_path),
                "processed_images": processed_images,
                "total_images": len(images),
                "classes": class_names,
                "files": {
                    "images": str(images_output),
                    "labels": str(labels_output),
                    "classes": str(classes_file),
                    "config": str(config_file)
                }
            }
            
            logger.info(f"YOLO格式导出完成: {processed_images}/{len(images)} 张图像")
            return True, export_info
            
        except Exception as e:
            logger.error(f"YOLO格式导出失败: {e}")
            return False, {"error": str(e)}
    
    def export_to_pascal_voc(self,
                            coco_file: str,
                            images_dir: str,
                            output_dir: str) -> Tuple[bool, Dict[str, Any]]:
        """导出为Pascal VOC格式"""
        
        try:
            # 加载COCO数据
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            images = coco_data["images"]
            annotations = coco_data["annotations"]
            categories = coco_data["categories"]
            
            # 创建输出目录结构
            output_path = Path(output_dir)
            images_output = output_path / "JPEGImages"
            annotations_output = output_path / "Annotations"
            imagesets_output = output_path / "ImageSets" / "Main"
            
            for dir_path in [images_output, annotations_output, imagesets_output]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 创建类别映射
            category_map = {cat["id"]: cat["name"] for cat in categories}
            
            # 按图像分组标注
            annotations_by_image = {}
            for ann in annotations:
                image_id = ann["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # 处理每张图像
            processed_images = 0
            image_names = []
            
            for image_info in images:
                image_id = image_info["id"]
                image_filename = image_info["file_name"]
                image_width = image_info["width"]
                image_height = image_info["height"]
                
                # 复制图像文件
                src_image_path = Path(images_dir) / image_filename
                dst_image_path = images_output / image_filename
                
                if src_image_path.exists():
                    shutil.copy2(src_image_path, dst_image_path)
                    
                    # 创建XML标注文件
                    xml_filename = Path(image_filename).stem + ".xml"
                    xml_path = annotations_output / xml_filename
                    
                    # 生成Pascal VOC XML
                    xml_content = self._create_pascal_voc_xml(
                        image_filename, image_width, image_height,
                        annotations_by_image.get(image_id, []),
                        category_map
                    )
                    
                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                    
                    image_names.append(Path(image_filename).stem)
                    processed_images += 1
                else:
                    logger.warning(f"图像文件不存在: {src_image_path}")
            
            # 创建ImageSets文件
            trainval_file = imagesets_output / "trainval.txt"
            with open(trainval_file, 'w') as f:
                f.write('\n'.join(image_names))
            
            export_info = {
                "format": "pascal_voc",
                "output_directory": str(output_path),
                "processed_images": processed_images,
                "total_images": len(images),
                "files": {
                    "images": str(images_output),
                    "annotations": str(annotations_output),
                    "imagesets": str(imagesets_output)
                }
            }
            
            logger.info(f"Pascal VOC格式导出完成: {processed_images}/{len(images)} 张图像")
            return True, export_info
            
        except Exception as e:
            logger.error(f"Pascal VOC格式导出失败: {e}")
            return False, {"error": str(e)}
    
    def _create_pascal_voc_xml(self,
                              filename: str,
                              width: int,
                              height: int,
                              annotations: List[Dict],
                              category_map: Dict[int, str]) -> str:
        """创建Pascal VOC格式的XML标注"""
        
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<annotation>',
            f'    <filename>{filename}</filename>',
            '    <size>',
            f'        <width>{width}</width>',
            f'        <height>{height}</height>',
            '        <depth>3</depth>',
            '    </size>',
            '    <segmented>0</segmented>'
        ]
        
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, width, height]
            category_id = ann["category_id"]
            category_name = category_map.get(category_id, f"class_{category_id}")
            
            # 转换为Pascal VOC格式 (xmin, ymin, xmax, ymax)
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])
            
            xml_lines.extend([
                '    <object>',
                f'        <name>{category_name}</name>',
                '        <pose>Unspecified</pose>',
                '        <truncated>0</truncated>',
                '        <difficult>0</difficult>',
                '        <bndbox>',
                f'            <xmin>{xmin}</xmin>',
                f'            <ymin>{ymin}</ymin>',
                f'            <xmax>{xmax}</xmax>',
                f'            <ymax>{ymax}</ymax>',
                '        </bndbox>',
                '    </object>'
            ])
        
        xml_lines.append('</annotation>')
        return '\n'.join(xml_lines)
    
    def export_to_tensorflow(self,
                           coco_file: str,
                           images_dir: str,
                           output_dir: str) -> Tuple[bool, Dict[str, Any]]:
        """导出为TensorFlow格式"""
        
        try:
            # 这里可以实现TensorFlow Record格式的导出
            # 由于复杂性，暂时返回提示信息
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建说明文件
            readme_file = output_path / "README.txt"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write("TensorFlow格式导出功能正在开发中\n")
                f.write("请使用COCO或YOLO格式进行模型训练\n")
            
            export_info = {
                "format": "tensorflow",
                "output_directory": str(output_path),
                "status": "in_development",
                "message": "TensorFlow格式导出功能正在开发中"
            }
            
            return True, export_info
            
        except Exception as e:
            logger.error(f"TensorFlow格式导出失败: {e}")
            return False, {"error": str(e)}
    
    def export_dataset(self,
                      coco_file: str,
                      images_dir: str,
                      output_dir: str,
                      format_type: str,
                      **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """通用数据集导出接口"""
        
        if format_type not in self.supported_formats:
            return False, {"error": f"不支持的格式: {format_type}"}
        
        if format_type == "coco":
            # 直接复制COCO文件
            return self._export_coco_format(coco_file, images_dir, output_dir)
        elif format_type == "yolo":
            return self.export_to_yolo(coco_file, images_dir, output_dir, **kwargs)
        elif format_type == "pascal_voc":
            return self.export_to_pascal_voc(coco_file, images_dir, output_dir)
        elif format_type == "tensorflow":
            return self.export_to_tensorflow(coco_file, images_dir, output_dir)
        else:
            return False, {"error": f"格式 {format_type} 的导出功能未实现"}
    
    def _export_coco_format(self,
                           coco_file: str,
                           images_dir: str,
                           output_dir: str) -> Tuple[bool, Dict[str, Any]]:
        """导出COCO格式（复制文件）"""
        
        try:
            output_path = Path(output_dir)
            images_output = output_path / "images"
            annotations_output = output_path / "annotations"
            
            images_output.mkdir(parents=True, exist_ok=True)
            annotations_output.mkdir(parents=True, exist_ok=True)
            
            # 复制标注文件
            dst_coco_file = annotations_output / "annotations.json"
            shutil.copy2(coco_file, dst_coco_file)
            
            # 复制图像文件
            images_path = Path(images_dir)
            if images_path.exists():
                for image_file in images_path.glob("*"):
                    if image_file.is_file():
                        shutil.copy2(image_file, images_output / image_file.name)
            
            export_info = {
                "format": "coco",
                "output_directory": str(output_path),
                "files": {
                    "images": str(images_output),
                    "annotations": str(dst_coco_file)
                }
            }
            
            logger.info(f"COCO格式导出完成")
            return True, export_info
            
        except Exception as e:
            logger.error(f"COCO格式导出失败: {e}")
            return False, {"error": str(e)}
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的导出格式"""
        return self.supported_formats.copy()
    
    def validate_export_params(self,
                             coco_file: str,
                             images_dir: str,
                             output_dir: str,
                             format_type: str) -> Tuple[bool, str]:
        """验证导出参数"""
        
        # 检查COCO文件
        if not Path(coco_file).exists():
            return False, f"COCO文件不存在: {coco_file}"
        
        # 检查图像目录
        if not Path(images_dir).exists():
            return False, f"图像目录不存在: {images_dir}"
        
        # 检查格式支持
        if format_type not in self.supported_formats:
            return False, f"不支持的格式: {format_type}"
        
        # 检查输出目录
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"无法创建输出目录: {e}"
        
        return True, "参数验证通过"

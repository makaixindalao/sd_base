"""
数据集分割器
将数据集分割为训练集、验证集和测试集
"""

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DataSplitter:
    """数据集分割器"""
    
    def __init__(self):
        """初始化数据集分割器"""
        self.random_seed = 42
    
    def split_coco_dataset(self, 
                          coco_file: str,
                          output_dir: str,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          random_seed: int = None) -> Tuple[bool, Dict[str, Any]]:
        """分割COCO格式数据集"""
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"数据集分割比例总和必须为1.0，当前为: {total_ratio}")
        
        if random_seed is not None:
            self.random_seed = random_seed
        random.seed(self.random_seed)
        
        try:
            # 加载COCO数据集
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            images = coco_data["images"]
            annotations = coco_data["annotations"]
            
            # 按图像ID分组标注
            annotations_by_image = {}
            for ann in annotations:
                image_id = ann["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)
            
            # 随机打乱图像列表
            random.shuffle(images)
            
            # 计算分割点
            total_images = len(images)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            
            # 分割图像
            train_images = images[:train_count]
            val_images = images[train_count:train_count + val_count]
            test_images = images[train_count + val_count:]
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建子数据集
            def create_subset(subset_images, subset_name):
                subset_annotations = []
                for img in subset_images:
                    img_id = img["id"]
                    if img_id in annotations_by_image:
                        subset_annotations.extend(annotations_by_image[img_id])
                
                subset_data = {
                    "info": coco_data["info"],
                    "licenses": coco_data["licenses"],
                    "categories": coco_data["categories"],
                    "images": subset_images,
                    "annotations": subset_annotations
                }
                
                # 保存子数据集
                subset_file = output_path / f"{subset_name}.json"
                with open(subset_file, 'w', encoding='utf-8') as f:
                    json.dump(subset_data, f, indent=2, ensure_ascii=False)
                
                return str(subset_file), len(subset_images), len(subset_annotations)
            
            # 创建训练集、验证集和测试集
            train_file, train_img_count, train_ann_count = create_subset(train_images, "train")
            val_file, val_img_count, val_ann_count = create_subset(val_images, "val")
            test_file, test_img_count, test_ann_count = create_subset(test_images, "test")
            
            # 创建分割信息
            split_info = {
                "original_file": coco_file,
                "output_directory": output_dir,
                "random_seed": self.random_seed,
                "ratios": {
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio
                },
                "counts": {
                    "total_images": total_images,
                    "total_annotations": len(annotations),
                    "train": {"images": train_img_count, "annotations": train_ann_count},
                    "val": {"images": val_img_count, "annotations": val_ann_count},
                    "test": {"images": test_img_count, "annotations": test_ann_count}
                },
                "files": {
                    "train": train_file,
                    "val": val_file,
                    "test": test_file
                }
            }
            
            # 保存分割信息
            split_info_file = output_path / "split_info.json"
            with open(split_info_file, 'w', encoding='utf-8') as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"数据集分割完成: 训练集({train_img_count}), 验证集({val_img_count}), 测试集({test_img_count})")
            
            return True, split_info
            
        except Exception as e:
            logger.error(f"数据集分割失败: {e}")
            return False, {"error": str(e)}
    
    def split_image_folder(self,
                          images_dir: str,
                          output_dir: str,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          copy_files: bool = True,
                          random_seed: int = None) -> Tuple[bool, Dict[str, Any]]:
        """分割图像文件夹"""
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"数据集分割比例总和必须为1.0，当前为: {total_ratio}")
        
        if random_seed is not None:
            self.random_seed = random_seed
        random.seed(self.random_seed)
        
        try:
            images_path = Path(images_dir)
            if not images_path.exists():
                raise FileNotFoundError(f"图像目录不存在: {images_dir}")
            
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(images_path.glob(f"*{ext}"))
                image_files.extend(images_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                raise ValueError(f"在目录中未找到图像文件: {images_dir}")
            
            # 随机打乱文件列表
            random.shuffle(image_files)
            
            # 计算分割点
            total_images = len(image_files)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            
            # 分割文件
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]
            
            # 创建输出目录结构
            output_path = Path(output_dir)
            train_dir = output_path / "train"
            val_dir = output_path / "val"
            test_dir = output_path / "test"
            
            for dir_path in [train_dir, val_dir, test_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 复制或移动文件
            def process_files(file_list, target_dir, subset_name):
                processed_files = []
                for file_path in file_list:
                    target_file = target_dir / file_path.name
                    
                    if copy_files:
                        shutil.copy2(file_path, target_file)
                    else:
                        shutil.move(str(file_path), str(target_file))
                    
                    processed_files.append(str(target_file))
                
                logger.info(f"{subset_name}集: {len(processed_files)} 个文件")
                return processed_files
            
            # 处理各个子集
            train_processed = process_files(train_files, train_dir, "训练")
            val_processed = process_files(val_files, val_dir, "验证")
            test_processed = process_files(test_files, test_dir, "测试")
            
            # 创建分割信息
            split_info = {
                "source_directory": images_dir,
                "output_directory": output_dir,
                "operation": "copy" if copy_files else "move",
                "random_seed": self.random_seed,
                "ratios": {
                    "train": train_ratio,
                    "val": val_ratio,
                    "test": test_ratio
                },
                "counts": {
                    "total_images": total_images,
                    "train": len(train_processed),
                    "val": len(val_processed),
                    "test": len(test_processed)
                },
                "directories": {
                    "train": str(train_dir),
                    "val": str(val_dir),
                    "test": str(test_dir)
                }
            }
            
            # 保存分割信息
            split_info_file = output_path / "split_info.json"
            with open(split_info_file, 'w', encoding='utf-8') as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"图像文件夹分割完成: 训练集({len(train_processed)}), 验证集({len(val_processed)}), 测试集({len(test_processed)})")
            
            return True, split_info
            
        except Exception as e:
            logger.error(f"图像文件夹分割失败: {e}")
            return False, {"error": str(e)}
    
    def validate_split_ratios(self, train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
        """验证分割比例"""
        total = train_ratio + val_ratio + test_ratio
        return abs(total - 1.0) < 0.001
    
    def get_recommended_splits(self) -> Dict[str, Dict[str, float]]:
        """获取推荐的分割比例"""
        return {
            "standard": {"train": 0.8, "val": 0.1, "test": 0.1},
            "large_dataset": {"train": 0.85, "val": 0.1, "test": 0.05},
            "small_dataset": {"train": 0.7, "val": 0.15, "test": 0.15},
            "research": {"train": 0.6, "val": 0.2, "test": 0.2}
        }
    
    def merge_splits(self, 
                    split_dirs: List[str],
                    output_dir: str,
                    new_ratios: Dict[str, float] = None) -> Tuple[bool, Dict[str, Any]]:
        """合并已分割的数据集并重新分割"""
        
        try:
            # 收集所有文件
            all_files = []
            for split_dir in split_dirs:
                split_path = Path(split_dir)
                if split_path.exists():
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                    for ext in image_extensions:
                        all_files.extend(split_path.glob(f"*{ext}"))
                        all_files.extend(split_path.glob(f"*{ext.upper()}"))
            
            if not all_files:
                raise ValueError("未找到可合并的图像文件")
            
            # 使用默认比例或指定比例
            if new_ratios is None:
                new_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
            
            # 创建临时目录并复制所有文件
            temp_dir = Path(output_dir) / "temp_merged"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for i, file_path in enumerate(all_files):
                temp_file = temp_dir / f"image_{i:06d}{file_path.suffix}"
                shutil.copy2(file_path, temp_file)
            
            # 重新分割
            success, split_info = self.split_image_folder(
                str(temp_dir),
                output_dir,
                new_ratios["train"],
                new_ratios["val"],
                new_ratios["test"]
            )
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if success:
                split_info["merged_from"] = split_dirs
                logger.info(f"数据集合并和重新分割完成")
            
            return success, split_info
            
        except Exception as e:
            logger.error(f"数据集合并失败: {e}")
            return False, {"error": str(e)}

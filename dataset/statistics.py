"""
数据集统计分析模块
提供数据集的详细统计分析功能
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class DatasetStatistics:
    """数据集统计分析器"""
    
    def __init__(self):
        """初始化统计分析器"""
        self.category_names = {
            0: "Tank",
            1: "Aircraft", 
            2: "Ship"
        }
    
    def analyze_coco_dataset(self, coco_file: str) -> Dict[str, Any]:
        """分析COCO格式数据集"""
        
        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            images = coco_data.get("images", [])
            annotations = coco_data.get("annotations", [])
            categories = coco_data.get("categories", [])
            
            # 基本统计
            basic_stats = {
                "total_images": len(images),
                "total_annotations": len(annotations),
                "total_categories": len(categories),
                "avg_annotations_per_image": len(annotations) / max(len(images), 1)
            }
            
            # 类别统计
            category_stats = self._analyze_categories(annotations, categories)
            
            # 图像尺寸统计
            image_size_stats = self._analyze_image_sizes(images)
            
            # 边界框统计
            bbox_stats = self._analyze_bboxes(annotations)
            
            # 分布统计
            distribution_stats = self._analyze_distribution(annotations, images)
            
            return {
                "basic": basic_stats,
                "categories": category_stats,
                "image_sizes": image_size_stats,
                "bboxes": bbox_stats,
                "distribution": distribution_stats,
                "dataset_info": coco_data.get("info", {}),
                "analysis_summary": self._generate_summary(basic_stats, category_stats, bbox_stats)
            }
            
        except Exception as e:
            logger.error(f"分析COCO数据集失败: {e}")
            return {"error": str(e)}
    
    def _analyze_categories(self, annotations: List[Dict], categories: List[Dict]) -> Dict[str, Any]:
        """分析类别分布"""
        
        # 统计每个类别的数量
        category_counts = Counter()
        category_areas = defaultdict(list)
        
        for ann in annotations:
            cat_id = ann.get("category_id", 0)
            category_counts[cat_id] += 1
            
            if "area" in ann:
                category_areas[cat_id].append(ann["area"])
        
        # 创建类别映射
        category_map = {cat["id"]: cat["name"] for cat in categories}
        
        # 计算统计信息
        category_stats = {}
        total_annotations = sum(category_counts.values())
        
        for cat_id, count in category_counts.items():
            cat_name = category_map.get(cat_id, f"Category_{cat_id}")
            areas = category_areas[cat_id]
            
            stats = {
                "count": count,
                "percentage": (count / total_annotations) * 100 if total_annotations > 0 else 0,
                "avg_area": np.mean(areas) if areas else 0,
                "min_area": np.min(areas) if areas else 0,
                "max_area": np.max(areas) if areas else 0,
                "std_area": np.std(areas) if areas else 0
            }
            
            category_stats[cat_name] = stats
        
        return {
            "category_distribution": category_stats,
            "most_common": category_counts.most_common(3),
            "least_common": category_counts.most_common()[-3:] if len(category_counts) >= 3 else category_counts.most_common(),
            "balance_score": self._calculate_balance_score(list(category_counts.values()))
        }
    
    def _analyze_image_sizes(self, images: List[Dict]) -> Dict[str, Any]:
        """分析图像尺寸分布"""
        
        if not images:
            return {"error": "没有图像数据"}
        
        widths = [img.get("width", 0) for img in images]
        heights = [img.get("height", 0) for img in images]
        aspects = [w/h if h > 0 else 0 for w, h in zip(widths, heights)]
        
        # 计算尺寸统计
        size_stats = {
            "width": {
                "min": np.min(widths),
                "max": np.max(widths),
                "mean": np.mean(widths),
                "std": np.std(widths),
                "median": np.median(widths)
            },
            "height": {
                "min": np.min(heights),
                "max": np.max(heights),
                "mean": np.mean(heights),
                "std": np.std(heights),
                "median": np.median(heights)
            },
            "aspect_ratio": {
                "min": np.min(aspects),
                "max": np.max(aspects),
                "mean": np.mean(aspects),
                "std": np.std(aspects),
                "median": np.median(aspects)
            }
        }
        
        # 常见尺寸统计
        size_combinations = [(img.get("width", 0), img.get("height", 0)) for img in images]
        size_counter = Counter(size_combinations)
        
        return {
            "statistics": size_stats,
            "common_sizes": size_counter.most_common(5),
            "unique_sizes": len(size_counter),
            "size_diversity": len(size_counter) / len(images) if images else 0
        }
    
    def _analyze_bboxes(self, annotations: List[Dict]) -> Dict[str, Any]:
        """分析边界框统计"""
        
        if not annotations:
            return {"error": "没有标注数据"}
        
        # 提取边界框信息
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        
        for ann in annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                aspect_ratios.append(w / h if h > 0 else 0)
        
        if not widths:
            return {"error": "没有有效的边界框数据"}
        
        # 计算统计信息
        bbox_stats = {
            "width": {
                "min": np.min(widths),
                "max": np.max(widths),
                "mean": np.mean(widths),
                "std": np.std(widths),
                "median": np.median(widths)
            },
            "height": {
                "min": np.min(heights),
                "max": np.max(heights),
                "mean": np.mean(heights),
                "std": np.std(heights),
                "median": np.median(heights)
            },
            "area": {
                "min": np.min(areas),
                "max": np.max(areas),
                "mean": np.mean(areas),
                "std": np.std(areas),
                "median": np.median(areas)
            },
            "aspect_ratio": {
                "min": np.min(aspect_ratios),
                "max": np.max(aspect_ratios),
                "mean": np.mean(aspect_ratios),
                "std": np.std(aspect_ratios),
                "median": np.median(aspect_ratios)
            }
        }
        
        # 尺寸分类
        small_objects = sum(1 for area in areas if area < 32*32)
        medium_objects = sum(1 for area in areas if 32*32 <= area < 96*96)
        large_objects = sum(1 for area in areas if area >= 96*96)
        
        return {
            "statistics": bbox_stats,
            "size_distribution": {
                "small": {"count": small_objects, "percentage": small_objects/len(areas)*100},
                "medium": {"count": medium_objects, "percentage": medium_objects/len(areas)*100},
                "large": {"count": large_objects, "percentage": large_objects/len(areas)*100}
            },
            "total_objects": len(areas)
        }
    
    def _analyze_distribution(self, annotations: List[Dict], images: List[Dict]) -> Dict[str, Any]:
        """分析数据分布"""
        
        # 每张图像的标注数量分布
        image_annotation_counts = defaultdict(int)
        for ann in annotations:
            image_id = ann.get("image_id", 0)
            image_annotation_counts[image_id] += 1
        
        annotation_counts = list(image_annotation_counts.values())
        
        # 没有标注的图像数量
        total_images = len(images)
        images_with_annotations = len(image_annotation_counts)
        images_without_annotations = total_images - images_with_annotations
        
        distribution_stats = {
            "annotations_per_image": {
                "min": np.min(annotation_counts) if annotation_counts else 0,
                "max": np.max(annotation_counts) if annotation_counts else 0,
                "mean": np.mean(annotation_counts) if annotation_counts else 0,
                "std": np.std(annotation_counts) if annotation_counts else 0,
                "median": np.median(annotation_counts) if annotation_counts else 0
            },
            "image_coverage": {
                "images_with_annotations": images_with_annotations,
                "images_without_annotations": images_without_annotations,
                "coverage_percentage": (images_with_annotations / total_images) * 100 if total_images > 0 else 0
            }
        }
        
        return distribution_stats
    
    def _calculate_balance_score(self, counts: List[int]) -> float:
        """计算类别平衡分数 (0-1, 1表示完全平衡)"""
        if not counts or len(counts) <= 1:
            return 1.0
        
        total = sum(counts)
        expected_count = total / len(counts)
        
        # 计算每个类别与期望值的偏差
        deviations = [abs(count - expected_count) / expected_count for count in counts]
        avg_deviation = np.mean(deviations)
        
        # 转换为0-1分数，0表示完全不平衡，1表示完全平衡
        balance_score = max(0, 1 - avg_deviation)
        return balance_score
    
    def _generate_summary(self, basic_stats: Dict, category_stats: Dict, bbox_stats: Dict) -> Dict[str, Any]:
        """生成分析摘要"""
        
        summary = {
            "dataset_size": "small" if basic_stats["total_images"] < 1000 else "medium" if basic_stats["total_images"] < 10000 else "large",
            "annotation_density": "sparse" if basic_stats["avg_annotations_per_image"] < 1 else "normal" if basic_stats["avg_annotations_per_image"] < 5 else "dense",
            "class_balance": "balanced" if category_stats.get("balance_score", 0) > 0.8 else "imbalanced",
            "object_sizes": "mixed" if bbox_stats.get("size_distribution", {}).get("medium", {}).get("percentage", 0) > 40 else "mostly_small" if bbox_stats.get("size_distribution", {}).get("small", {}).get("percentage", 0) > 60 else "mostly_large"
        }
        
        # 生成建议
        recommendations = []
        
        if basic_stats["total_images"] < 500:
            recommendations.append("数据集较小，建议增加更多图像以提高模型性能")
        
        if basic_stats["avg_annotations_per_image"] < 0.5:
            recommendations.append("标注密度较低，建议检查标注质量或增加标注")
        
        if category_stats.get("balance_score", 0) < 0.6:
            recommendations.append("类别分布不平衡，建议平衡各类别的样本数量")
        
        summary["recommendations"] = recommendations
        
        return summary
    
    def compare_datasets(self, dataset_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较多个数据集的统计信息"""
        
        if len(dataset_stats_list) < 2:
            return {"error": "需要至少两个数据集进行比较"}
        
        comparison = {
            "dataset_count": len(dataset_stats_list),
            "basic_comparison": {},
            "category_comparison": {},
            "size_comparison": {}
        }
        
        # 基本信息比较
        total_images = [stats["basic"]["total_images"] for stats in dataset_stats_list]
        total_annotations = [stats["basic"]["total_annotations"] for stats in dataset_stats_list]
        
        comparison["basic_comparison"] = {
            "total_images": {"min": min(total_images), "max": max(total_images), "sum": sum(total_images)},
            "total_annotations": {"min": min(total_annotations), "max": max(total_annotations), "sum": sum(total_annotations)}
        }
        
        return comparison
    
    def export_statistics_report(self, stats: Dict[str, Any], output_file: str) -> bool:
        """导出统计报告"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("数据集统计分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本信息
                basic = stats.get("basic", {})
                f.write("基本信息:\n")
                f.write(f"  图像总数: {basic.get('total_images', 0)}\n")
                f.write(f"  标注总数: {basic.get('total_annotations', 0)}\n")
                f.write(f"  类别总数: {basic.get('total_categories', 0)}\n")
                f.write(f"  平均每张图像标注数: {basic.get('avg_annotations_per_image', 0):.2f}\n\n")
                
                # 类别分布
                categories = stats.get("categories", {}).get("category_distribution", {})
                if categories:
                    f.write("类别分布:\n")
                    for cat_name, cat_stats in categories.items():
                        f.write(f"  {cat_name}: {cat_stats['count']}个 ({cat_stats['percentage']:.1f}%)\n")
                    f.write("\n")
                
                # 分析摘要
                summary = stats.get("analysis_summary", {})
                if summary:
                    f.write("分析摘要:\n")
                    f.write(f"  数据集规模: {summary.get('dataset_size', 'unknown')}\n")
                    f.write(f"  标注密度: {summary.get('annotation_density', 'unknown')}\n")
                    f.write(f"  类别平衡: {summary.get('class_balance', 'unknown')}\n")
                    f.write(f"  目标尺寸: {summary.get('object_sizes', 'unknown')}\n\n")
                
                # 建议
                recommendations = summary.get("recommendations", [])
                if recommendations:
                    f.write("改进建议:\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"  {i}. {rec}\n")
            
            logger.info(f"统计报告已导出到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出统计报告失败: {e}")
            return False

"""
数据集管理器
提供数据集的创建、读取、更新、删除等CRUD操作
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, datasets_root: str = "datasets"):
        """初始化数据集管理器"""
        self.datasets_root = Path(datasets_root)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        
        # 数据集元数据文件
        self.metadata_file = self.datasets_root / "datasets_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载数据集元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载数据集元数据失败: {e}")
        
        return {"datasets": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_metadata(self):
        """保存数据集元数据"""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存数据集元数据失败: {e}")
    
    def create_dataset(self, 
                      name: str, 
                      description: str = "",
                      target_types: List[str] = None,
                      weather_conditions: List[str] = None,
                      terrain_types: List[str] = None) -> bool:
        """创建新数据集"""
        
        if name in self.metadata["datasets"]:
            logger.error(f"数据集 '{name}' 已存在")
            return False
        
        # 创建数据集目录
        dataset_path = self.datasets_root / name
        try:
            dataset_path.mkdir(parents=True, exist_ok=False)
            
            # 创建子目录
            (dataset_path / "images").mkdir()
            (dataset_path / "annotations").mkdir()
            (dataset_path / "splits").mkdir()
            
            # 创建数据集配置文件
            dataset_config = {
                "name": name,
                "description": description,
                "created_date": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "target_types": target_types or [],
                "weather_conditions": weather_conditions or [],
                "terrain_types": terrain_types or [],
                "image_count": 0,
                "annotation_count": 0,
                "splits": {
                    "train": {"image_count": 0, "annotation_count": 0},
                    "val": {"image_count": 0, "annotation_count": 0},
                    "test": {"image_count": 0, "annotation_count": 0}
                },
                "statistics": {}
            }
            
            config_file = dataset_path / "dataset_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_config, f, indent=2, ensure_ascii=False)
            
            # 更新元数据
            self.metadata["datasets"][name] = {
                "path": str(dataset_path),
                "created_date": dataset_config["created_date"],
                "last_modified": dataset_config["last_modified"],
                "image_count": 0,
                "status": "active"
            }
            self._save_metadata()
            
            logger.info(f"成功创建数据集: {name}")
            return True
            
        except Exception as e:
            logger.error(f"创建数据集失败: {e}")
            # 清理可能创建的目录
            if dataset_path.exists():
                shutil.rmtree(dataset_path, ignore_errors=True)
            return False
    
    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """获取数据集列表"""
        datasets = []
        for name, info in self.metadata["datasets"].items():
            if info["status"] == "active":
                dataset_info = info.copy()
                dataset_info["name"] = name
                
                # 获取详细信息
                config = self.get_dataset_config(name)
                if config:
                    dataset_info.update({
                        "description": config.get("description", ""),
                        "target_types": config.get("target_types", []),
                        "annotation_count": config.get("annotation_count", 0)
                    })
                
                datasets.append(dataset_info)
        
        return datasets
    
    def get_dataset_config(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据集配置"""
        if name not in self.metadata["datasets"]:
            return None
        
        dataset_path = Path(self.metadata["datasets"][name]["path"])
        config_file = dataset_path / "dataset_config.json"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取数据集配置失败: {e}")
            return None
    
    def update_dataset(self, 
                      name: str, 
                      description: str = None,
                      target_types: List[str] = None,
                      weather_conditions: List[str] = None,
                      terrain_types: List[str] = None) -> bool:
        """更新数据集信息"""
        
        config = self.get_dataset_config(name)
        if not config:
            logger.error(f"数据集 '{name}' 不存在")
            return False
        
        # 更新配置
        if description is not None:
            config["description"] = description
        if target_types is not None:
            config["target_types"] = target_types
        if weather_conditions is not None:
            config["weather_conditions"] = weather_conditions
        if terrain_types is not None:
            config["terrain_types"] = terrain_types
        
        config["last_modified"] = datetime.now().isoformat()
        
        # 保存配置
        dataset_path = Path(self.metadata["datasets"][name]["path"])
        config_file = dataset_path / "dataset_config.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 更新元数据
            self.metadata["datasets"][name]["last_modified"] = config["last_modified"]
            self._save_metadata()
            
            logger.info(f"成功更新数据集: {name}")
            return True
            
        except Exception as e:
            logger.error(f"更新数据集失败: {e}")
            return False
    
    def delete_dataset(self, name: str, permanent: bool = False) -> bool:
        """删除数据集"""
        
        if name not in self.metadata["datasets"]:
            logger.error(f"数据集 '{name}' 不存在")
            return False
        
        try:
            if permanent:
                # 永久删除
                dataset_path = Path(self.metadata["datasets"][name]["path"])
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                
                del self.metadata["datasets"][name]
                logger.info(f"永久删除数据集: {name}")
            else:
                # 标记为已删除
                self.metadata["datasets"][name]["status"] = "deleted"
                self.metadata["datasets"][name]["deleted_date"] = datetime.now().isoformat()
                logger.info(f"标记删除数据集: {name}")
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            return False
    
    def restore_dataset(self, name: str) -> bool:
        """恢复已删除的数据集"""
        
        if name not in self.metadata["datasets"]:
            logger.error(f"数据集 '{name}' 不存在")
            return False
        
        if self.metadata["datasets"][name]["status"] != "deleted":
            logger.error(f"数据集 '{name}' 未被删除")
            return False
        
        try:
            self.metadata["datasets"][name]["status"] = "active"
            if "deleted_date" in self.metadata["datasets"][name]:
                del self.metadata["datasets"][name]["deleted_date"]
            
            self._save_metadata()
            logger.info(f"恢复数据集: {name}")
            return True
            
        except Exception as e:
            logger.error(f"恢复数据集失败: {e}")
            return False
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """获取数据集路径"""
        if name not in self.metadata["datasets"]:
            return None
        return Path(self.metadata["datasets"][name]["path"])
    
    def dataset_exists(self, name: str) -> bool:
        """检查数据集是否存在"""
        return (name in self.metadata["datasets"] and 
                self.metadata["datasets"][name]["status"] == "active")
    
    def get_dataset_statistics(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据集统计信息"""
        config = self.get_dataset_config(name)
        if not config:
            return None
        
        dataset_path = self.get_dataset_path(name)
        if not dataset_path:
            return None
        
        # 实时统计图像数量
        images_dir = dataset_path / "images"
        image_count = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        
        # 统计标注文件
        annotations_dir = dataset_path / "annotations"
        annotation_files = list(annotations_dir.glob("*.json"))
        
        stats = {
            "name": name,
            "image_count": image_count,
            "annotation_files": len(annotation_files),
            "target_types": config.get("target_types", []),
            "weather_conditions": config.get("weather_conditions", []),
            "terrain_types": config.get("terrain_types", []),
            "created_date": config.get("created_date", ""),
            "last_modified": config.get("last_modified", ""),
            "splits": config.get("splits", {}),
            "disk_usage": self._calculate_disk_usage(dataset_path)
        }
        
        return stats
    
    def _calculate_disk_usage(self, path: Path) -> Dict[str, int]:
        """计算磁盘使用量"""
        total_size = 0
        file_count = 0
        
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        except Exception as e:
            logger.error(f"计算磁盘使用量失败: {e}")
        
        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count
        }

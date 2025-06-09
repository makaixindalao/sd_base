"""
标注管理面板
提供自动标注和标注管理的界面
"""

import sys
from typing import List, Dict, Optional
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QTextEdit, QProgressBar, 
                                QGroupBox, QListWidget, QListWidgetItem, 
                                QMessageBox, QFileDialog, QTabWidget, 
                                QScrollArea, QFrame, QSpinBox, QDoubleSpinBox,
                                QCheckBox, QComboBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

if PYQT_AVAILABLE:
    from annotation.auto_annotator import AutoAnnotator
    from annotation.coco_formatter import COCOFormatter
    from annotation.bbox_visualizer import BBoxVisualizer
    import logging

    logger = logging.getLogger(__name__)

    class AnnotationThread(QThread):
        """标注处理线程"""
        progress_updated = pyqtSignal(float)
        status_updated = pyqtSignal(str)
        annotation_completed = pyqtSignal(list)
        
        def __init__(self, annotator: AutoAnnotator, image_paths: List[str], target_types: List[str] = None):
            super().__init__()
            self.annotator = annotator
            self.image_paths = image_paths
            self.target_types = target_types
            self.is_cancelled = False
        
        def run(self):
            """运行标注任务"""
            try:
                # 设置回调
                self.annotator.set_callbacks(
                    progress_callback=self.progress_updated.emit,
                    status_callback=self.status_updated.emit
                )
                
                # 执行批量标注
                results = self.annotator.annotate_batch(self.image_paths, self.target_types)
                
                if not self.is_cancelled:
                    self.annotation_completed.emit(results)
                    
            except Exception as e:
                self.status_updated.emit(f"标注失败: {str(e)}")
                logger.error(f"自动标注失败: {e}", exc_info=True)
        
        def cancel(self):
            """取消标注"""
            self.is_cancelled = True

    class AnnotationPanel(QWidget):
        """标注管理面板"""
        
        def __init__(self):
            super().__init__()
            
            if not PYQT_AVAILABLE:
                raise ImportError("PyQt5不可用")
            
            self.annotator = AutoAnnotator()
            self.coco_formatter = COCOFormatter()
            self.bbox_visualizer = BBoxVisualizer()
            
            self.annotation_thread = None
            self.current_results = []
            self.selected_images = []
            
            self.setup_ui()
        
        def setup_ui(self):
            """设置用户界面"""
            layout = QVBoxLayout(self)
            
            # 创建选项卡
            tab_widget = QTabWidget()
            
            # 批量标注选项卡
            batch_tab = self.create_batch_annotation_tab()
            tab_widget.addTab(batch_tab, "批量标注")
            
            # 标注设置选项卡
            settings_tab = self.create_settings_tab()
            tab_widget.addTab(settings_tab, "标注设置")
            
            # 结果查看选项卡
            results_tab = self.create_results_tab()
            tab_widget.addTab(results_tab, "标注结果")
            
            layout.addWidget(tab_widget)
            
            # 状态栏
            self.create_status_bar(layout)
        
        def create_batch_annotation_tab(self) -> QWidget:
            """创建批量标注选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 图像选择组
            image_group = QGroupBox("图像选择")
            image_layout = QVBoxLayout(image_group)
            
            # 选择按钮
            button_layout = QHBoxLayout()
            
            self.select_images_btn = QPushButton("选择图像文件")
            self.select_images_btn.clicked.connect(self.select_images)
            button_layout.addWidget(self.select_images_btn)
            
            self.select_folder_btn = QPushButton("选择图像文件夹")
            self.select_folder_btn.clicked.connect(self.select_image_folder)
            button_layout.addWidget(self.select_folder_btn)
            
            self.clear_selection_btn = QPushButton("清空选择")
            self.clear_selection_btn.clicked.connect(self.clear_image_selection)
            button_layout.addWidget(self.clear_selection_btn)
            
            image_layout.addLayout(button_layout)
            
            # 选中的图像列表
            self.selected_images_list = QListWidget()
            self.selected_images_list.setMaximumHeight(150)
            image_layout.addWidget(self.selected_images_list)
            
            layout.addWidget(image_group)
            
            # 标注选项组
            options_group = QGroupBox("标注选项")
            options_layout = QGridLayout(options_group)
            
            options_layout.addWidget(QLabel("目标类型过滤:"), 0, 0)
            self.target_filter_combo = QComboBox()
            self.target_filter_combo.addItems(["全部", "tank", "aircraft", "ship"])
            options_layout.addWidget(self.target_filter_combo, 0, 1)
            
            options_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
            self.confidence_spinbox = QDoubleSpinBox()
            self.confidence_spinbox.setRange(0.1, 1.0)
            self.confidence_spinbox.setValue(0.5)
            self.confidence_spinbox.setSingleStep(0.1)
            options_layout.addWidget(self.confidence_spinbox, 1, 1)
            
            options_layout.addWidget(QLabel("NMS阈值:"), 2, 0)
            self.nms_spinbox = QDoubleSpinBox()
            self.nms_spinbox.setRange(0.1, 1.0)
            self.nms_spinbox.setValue(0.4)
            self.nms_spinbox.setSingleStep(0.1)
            options_layout.addWidget(self.nms_spinbox, 2, 1)
            
            layout.addWidget(options_group)
            
            # 输出设置组
            output_group = QGroupBox("输出设置")
            output_layout = QGridLayout(output_group)
            
            output_layout.addWidget(QLabel("输出目录:"), 0, 0)
            self.output_dir_layout = QHBoxLayout()
            self.output_dir_label = QLabel("annotations/")
            self.output_dir_btn = QPushButton("选择")
            self.output_dir_btn.clicked.connect(self.select_output_directory)
            self.output_dir_layout.addWidget(self.output_dir_label)
            self.output_dir_layout.addWidget(self.output_dir_btn)
            output_layout.addLayout(self.output_dir_layout, 0, 1)
            
            self.save_coco_cb = QCheckBox("保存COCO格式")
            self.save_coco_cb.setChecked(True)
            output_layout.addWidget(self.save_coco_cb, 1, 0)
            
            self.save_visualization_cb = QCheckBox("保存可视化结果")
            self.save_visualization_cb.setChecked(False)
            output_layout.addWidget(self.save_visualization_cb, 1, 1)
            
            layout.addWidget(output_group)
            
            # 控制按钮
            button_layout = QHBoxLayout()
            
            self.annotate_btn = QPushButton("开始标注")
            self.annotate_btn.clicked.connect(self.start_annotation)
            self.annotate_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
            button_layout.addWidget(self.annotate_btn)
            
            self.cancel_btn = QPushButton("取消标注")
            self.cancel_btn.clicked.connect(self.cancel_annotation)
            self.cancel_btn.setEnabled(False)
            button_layout.addWidget(self.cancel_btn)
            
            layout.addLayout(button_layout)
            
            return widget
        
        def create_settings_tab(self) -> QWidget:
            """创建标注设置选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 检测模型设置
            model_group = QGroupBox("检测模型设置")
            model_layout = QGridLayout(model_group)
            
            model_layout.addWidget(QLabel("检测模型:"), 0, 0)
            self.model_combo = QComboBox()
            self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
            model_layout.addWidget(self.model_combo, 0, 1)
            
            self.reload_model_btn = QPushButton("重新加载模型")
            self.reload_model_btn.clicked.connect(self.reload_detection_model)
            model_layout.addWidget(self.reload_model_btn, 0, 2)
            
            layout.addWidget(model_group)
            
            # 类别映射设置
            mapping_group = QGroupBox("类别映射")
            mapping_layout = QGridLayout(mapping_group)
            
            mapping_layout.addWidget(QLabel("Tank ID:"), 0, 0)
            self.tank_id_spinbox = QSpinBox()
            self.tank_id_spinbox.setRange(0, 100)
            self.tank_id_spinbox.setValue(0)
            mapping_layout.addWidget(self.tank_id_spinbox, 0, 1)
            
            mapping_layout.addWidget(QLabel("Aircraft ID:"), 1, 0)
            self.aircraft_id_spinbox = QSpinBox()
            self.aircraft_id_spinbox.setRange(0, 100)
            self.aircraft_id_spinbox.setValue(1)
            mapping_layout.addWidget(self.aircraft_id_spinbox, 1, 1)
            
            mapping_layout.addWidget(QLabel("Ship ID:"), 2, 0)
            self.ship_id_spinbox = QSpinBox()
            self.ship_id_spinbox.setRange(0, 100)
            self.ship_id_spinbox.setValue(2)
            mapping_layout.addWidget(self.ship_id_spinbox, 2, 1)
            
            self.apply_mapping_btn = QPushButton("应用映射")
            self.apply_mapping_btn.clicked.connect(self.apply_class_mapping)
            mapping_layout.addWidget(self.apply_mapping_btn, 3, 0, 1, 2)
            
            layout.addWidget(mapping_group)
            
            # 模型状态显示
            status_group = QGroupBox("模型状态")
            status_layout = QVBoxLayout(status_group)
            
            self.model_status_label = QLabel("检查模型状态...")
            status_layout.addWidget(self.model_status_label)
            
            layout.addWidget(status_group)
            
            layout.addStretch()
            
            # 检查模型状态
            QTimer.singleShot(1000, self.check_model_status)
            
            return widget
        
        def create_results_tab(self) -> QWidget:
            """创建结果查看选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 结果列表
            self.results_list = QListWidget()
            layout.addWidget(self.results_list)
            
            # 结果统计
            stats_group = QGroupBox("标注统计")
            stats_layout = QGridLayout(stats_group)
            
            self.stats_labels = {
                "total_images": QLabel("图像总数: 0"),
                "total_detections": QLabel("检测总数: 0"),
                "avg_detections": QLabel("平均检测数: 0"),
                "processing_time": QLabel("处理时间: 0秒")
            }
            
            row = 0
            for key, label in self.stats_labels.items():
                stats_layout.addWidget(label, row // 2, row % 2)
                row += 1
            
            layout.addWidget(stats_group)
            
            # 导出按钮
            export_layout = QHBoxLayout()
            
            self.export_coco_btn = QPushButton("导出COCO格式")
            self.export_coco_btn.clicked.connect(self.export_coco_format)
            self.export_coco_btn.setEnabled(False)
            export_layout.addWidget(self.export_coco_btn)
            
            self.export_summary_btn = QPushButton("导出统计报告")
            self.export_summary_btn.clicked.connect(self.export_summary_report)
            self.export_summary_btn.setEnabled(False)
            export_layout.addWidget(self.export_summary_btn)
            
            layout.addLayout(export_layout)
            
            return widget
        
        def create_status_bar(self, layout: QVBoxLayout):
            """创建状态栏"""
            status_frame = QFrame()
            status_frame.setFrameStyle(QFrame.StyledPanel)
            status_layout = QVBoxLayout(status_frame)
            
            # 进度条
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            status_layout.addWidget(self.progress_bar)
            
            # 状态标签
            self.status_label = QLabel("就绪")
            status_layout.addWidget(self.status_label)
            
            layout.addWidget(status_frame)
        
        def select_images(self):
            """选择图像文件"""
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图像文件", "", 
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            if files:
                self.selected_images.extend(files)
                self.update_selected_images_list()
        
        def select_image_folder(self):
            """选择图像文件夹"""
            folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if folder:
                folder_path = Path(folder)
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                images = []
                for ext in image_extensions:
                    images.extend(folder_path.glob(f"*{ext}"))
                    images.extend(folder_path.glob(f"*{ext.upper()}"))
                
                self.selected_images.extend([str(img) for img in images])
                self.update_selected_images_list()
        
        def clear_image_selection(self):
            """清空图像选择"""
            self.selected_images.clear()
            self.update_selected_images_list()
        
        def update_selected_images_list(self):
            """更新选中图像列表显示"""
            self.selected_images_list.clear()
            for image_path in self.selected_images:
                item = QListWidgetItem(Path(image_path).name)
                item.setToolTip(image_path)
                self.selected_images_list.addItem(item)
            
            # 更新按钮状态
            self.annotate_btn.setEnabled(len(self.selected_images) > 0)
        
        def select_output_directory(self):
            """选择输出目录"""
            directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if directory:
                self.output_dir_label.setText(directory)
        
        def check_model_status(self):
            """检查模型状态"""
            if self.annotator.is_model_loaded():
                self.model_status_label.setText("✅ 检测模型已加载")
                self.model_status_label.setStyleSheet("color: green;")
            else:
                self.model_status_label.setText("❌ 检测模型未加载")
                self.model_status_label.setStyleSheet("color: red;")
        
        def reload_detection_model(self):
            """重新加载检测模型"""
            model_name = self.model_combo.currentText()
            self.annotator.model_name = model_name
            success = self.annotator._load_model()
            
            if success:
                QMessageBox.information(self, "成功", f"模型 {model_name} 加载成功")
            else:
                QMessageBox.warning(self, "失败", f"模型 {model_name} 加载失败")
            
            self.check_model_status()
        
        def apply_class_mapping(self):
            """应用类别映射"""
            mapping = {
                "tank": self.tank_id_spinbox.value(),
                "aircraft": self.aircraft_id_spinbox.value(),
                "ship": self.ship_id_spinbox.value()
            }
            self.annotator.set_class_mapping(mapping)
            QMessageBox.information(self, "成功", "类别映射已更新")
        
        def start_annotation(self):
            """开始标注"""
            if not self.selected_images:
                QMessageBox.warning(self, "警告", "请先选择要标注的图像")
                return
            
            if not self.annotator.is_model_loaded():
                QMessageBox.warning(self, "警告", "检测模型未加载，请先加载模型")
                return
            
            # 更新检测参数
            self.annotator.set_detection_parameters(
                confidence_threshold=self.confidence_spinbox.value(),
                nms_threshold=self.nms_spinbox.value()
            )
            
            # 获取目标类型过滤
            target_filter = self.target_filter_combo.currentText()
            target_types = None if target_filter == "全部" else [target_filter] * len(self.selected_images)
            
            # 启动标注线程
            self.annotation_thread = AnnotationThread(
                self.annotator, self.selected_images, target_types
            )
            self.annotation_thread.progress_updated.connect(self.update_progress)
            self.annotation_thread.status_updated.connect(self.update_status)
            self.annotation_thread.annotation_completed.connect(self.on_annotation_completed)
            
            self.annotation_thread.start()
            
            # 更新UI状态
            self.annotate_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        
        def cancel_annotation(self):
            """取消标注"""
            if self.annotation_thread:
                self.annotation_thread.cancel()
                self.annotation_thread.wait()
            
            self.reset_ui_state()
            self.update_status("标注已取消")
        
        def update_progress(self, progress: float):
            """更新进度"""
            self.progress_bar.setValue(int(progress))
        
        def update_status(self, status: str):
            """更新状态"""
            self.status_label.setText(status)
        
        def on_annotation_completed(self, results: List[Dict]):
            """标注完成处理"""
            self.current_results = results
            self.update_results_display()
            self.update_statistics()
            self.reset_ui_state()
            
            # 自动保存结果
            if self.save_coco_cb.isChecked():
                self.save_coco_results()
            
            # 显示完成消息
            total_detections = sum(r.get("detection_count", 0) for r in results)
            QMessageBox.information(
                self, "标注完成", 
                f"批量标注完成!\n处理图像: {len(results)}\n检测目标: {total_detections}"
            )
        
        def update_results_display(self):
            """更新结果显示"""
            self.results_list.clear()
            
            for result in self.current_results:
                if "error" in result:
                    item_text = f"❌ {Path(result['image_path']).name} - 错误: {result['error']}"
                else:
                    detection_count = result.get("detection_count", 0)
                    item_text = f"✅ {Path(result['image_path']).name} - 检测到 {detection_count} 个目标"
                
                item = QListWidgetItem(item_text)
                item.setToolTip(result["image_path"])
                self.results_list.addItem(item)
        
        def update_statistics(self):
            """更新统计信息"""
            if not self.current_results:
                return
            
            total_images = len(self.current_results)
            total_detections = sum(r.get("detection_count", 0) for r in self.current_results)
            avg_detections = total_detections / max(total_images, 1)
            
            self.stats_labels["total_images"].setText(f"图像总数: {total_images}")
            self.stats_labels["total_detections"].setText(f"检测总数: {total_detections}")
            self.stats_labels["avg_detections"].setText(f"平均检测数: {avg_detections:.1f}")
            
            # 启用导出按钮
            self.export_coco_btn.setEnabled(True)
            self.export_summary_btn.setEnabled(True)
        
        def save_coco_results(self):
            """保存COCO格式结果"""
            if not self.current_results:
                return
            
            try:
                output_dir = Path(self.output_dir_label.text())
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建COCO数据集
                coco_dataset = self.coco_formatter.create_coco_dataset(
                    self.current_results, "auto_annotation_dataset"
                )
                
                # 保存COCO文件
                coco_file = output_dir / "annotations.json"
                self.coco_formatter.save_coco_dataset(coco_dataset, str(coco_file))
                
                self.update_status(f"COCO格式已保存到: {coco_file}")
                
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存COCO格式失败: {str(e)}")
        
        def export_coco_format(self):
            """导出COCO格式"""
            if not self.current_results:
                QMessageBox.warning(self, "警告", "没有可导出的标注结果")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存COCO格式", "annotations.json", "JSON文件 (*.json)"
            )
            
            if file_path:
                try:
                    coco_dataset = self.coco_formatter.create_coco_dataset(
                        self.current_results, "exported_dataset"
                    )
                    self.coco_formatter.save_coco_dataset(coco_dataset, file_path)
                    QMessageBox.information(self, "成功", f"COCO格式已导出到: {file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "导出失败", f"导出COCO格式失败: {str(e)}")
        
        def export_summary_report(self):
            """导出统计报告"""
            if not self.current_results:
                QMessageBox.warning(self, "警告", "没有可导出的统计数据")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存统计报告", "annotation_report.txt", "文本文件 (*.txt)"
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("自动标注统计报告\n")
                        f.write("=" * 50 + "\n\n")
                        
                        total_images = len(self.current_results)
                        total_detections = sum(r.get("detection_count", 0) for r in self.current_results)
                        
                        f.write(f"处理图像总数: {total_images}\n")
                        f.write(f"检测目标总数: {total_detections}\n")
                        f.write(f"平均每张图像检测数: {total_detections / max(total_images, 1):.2f}\n\n")
                        
                        f.write("详细结果:\n")
                        f.write("-" * 30 + "\n")
                        
                        for result in self.current_results:
                            image_name = Path(result["image_path"]).name
                            if "error" in result:
                                f.write(f"{image_name}: 错误 - {result['error']}\n")
                            else:
                                detection_count = result.get("detection_count", 0)
                                f.write(f"{image_name}: {detection_count} 个目标\n")
                    
                    QMessageBox.information(self, "成功", f"统计报告已导出到: {file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "导出失败", f"导出统计报告失败: {str(e)}")
        
        def reset_ui_state(self):
            """重置UI状态"""
            self.annotate_btn.setEnabled(len(self.selected_images) > 0)
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setVisible(False)

else:
    # PyQt5不可用时的占位类
    class AnnotationPanel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5不可用，无法创建标注面板")

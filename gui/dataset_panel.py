"""
数据集管理面板
提供数据集的CRUD操作和管理界面
"""

import sys
from typing import List, Dict, Optional
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QTextEdit, QProgressBar, 
                                QGroupBox, QListWidget, QListWidgetItem, 
                                QMessageBox, QFileDialog, QTabWidget, 
                                QScrollArea, QFrame, QLineEdit, QSpinBox,
                                QTableWidget, QTableWidgetItem, QHeaderView,
                                QDialog, QDialogButtonBox, QComboBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

if PYQT_AVAILABLE:
    from dataset.dataset_manager import DatasetManager
    from annotation.coco_formatter import COCOFormatter
    import logging

    logger = logging.getLogger(__name__)

    class CreateDatasetDialog(QDialog):
        """创建数据集对话框"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("创建新数据集")
            self.setModal(True)
            self.resize(400, 300)
            
            self.setup_ui()
        
        def setup_ui(self):
            """设置界面"""
            layout = QVBoxLayout(self)
            
            # 基本信息
            info_group = QGroupBox("基本信息")
            info_layout = QGridLayout(info_group)
            
            info_layout.addWidget(QLabel("数据集名称:"), 0, 0)
            self.name_edit = QLineEdit()
            info_layout.addWidget(self.name_edit, 0, 1)
            
            info_layout.addWidget(QLabel("描述:"), 1, 0)
            self.description_edit = QTextEdit()
            self.description_edit.setMaximumHeight(80)
            info_layout.addWidget(self.description_edit, 1, 1)
            
            layout.addWidget(info_group)
            
            # 目标类型
            targets_group = QGroupBox("目标类型")
            targets_layout = QVBoxLayout(targets_group)
            
            self.target_checkboxes = {}
            for target in ["tank", "aircraft", "ship"]:
                checkbox = QCheckBox(target.upper())
                checkbox.setChecked(True)
                self.target_checkboxes[target] = checkbox
                targets_layout.addWidget(checkbox)
            
            layout.addWidget(targets_group)
            
            # 按钮
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)
        
        def get_dataset_info(self) -> Dict:
            """获取数据集信息"""
            selected_targets = [target for target, cb in self.target_checkboxes.items() if cb.isChecked()]
            
            return {
                "name": self.name_edit.text().strip(),
                "description": self.description_edit.toPlainText().strip(),
                "target_types": selected_targets,
                "weather_conditions": ["sunny", "rainy", "snowy", "foggy", "night"],
                "terrain_types": ["urban", "island", "rural"]
            }

    class DatasetPanel(QWidget):
        """数据集管理面板"""
        
        def __init__(self):
            super().__init__()
            
            if not PYQT_AVAILABLE:
                raise ImportError("PyQt5不可用")
            
            self.dataset_manager = DatasetManager()
            self.coco_formatter = COCOFormatter()
            
            self.current_datasets = []
            
            self.setup_ui()
            self.refresh_dataset_list()
        
        def setup_ui(self):
            """设置用户界面"""
            layout = QVBoxLayout(self)
            
            # 创建选项卡
            tab_widget = QTabWidget()
            
            # 数据集列表选项卡
            list_tab = self.create_dataset_list_tab()
            tab_widget.addTab(list_tab, "数据集列表")
            
            # 数据集详情选项卡
            details_tab = self.create_dataset_details_tab()
            tab_widget.addTab(details_tab, "数据集详情")
            
            # 数据集操作选项卡
            operations_tab = self.create_operations_tab()
            tab_widget.addTab(operations_tab, "数据集操作")
            
            layout.addWidget(tab_widget)
            
            # 状态栏
            self.create_status_bar(layout)
        
        def create_dataset_list_tab(self) -> QWidget:
            """创建数据集列表选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 工具栏
            toolbar_layout = QHBoxLayout()
            
            self.create_dataset_btn = QPushButton("创建数据集")
            self.create_dataset_btn.clicked.connect(self.create_dataset)
            self.create_dataset_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            toolbar_layout.addWidget(self.create_dataset_btn)
            
            self.refresh_btn = QPushButton("刷新列表")
            self.refresh_btn.clicked.connect(self.refresh_dataset_list)
            toolbar_layout.addWidget(self.refresh_btn)
            
            self.delete_dataset_btn = QPushButton("删除数据集")
            self.delete_dataset_btn.clicked.connect(self.delete_dataset)
            self.delete_dataset_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            self.delete_dataset_btn.setEnabled(False)
            toolbar_layout.addWidget(self.delete_dataset_btn)
            
            toolbar_layout.addStretch()
            layout.addLayout(toolbar_layout)
            
            # 数据集表格
            self.dataset_table = QTableWidget()
            self.dataset_table.setColumnCount(6)
            self.dataset_table.setHorizontalHeaderLabels([
                "名称", "描述", "图像数量", "目标类型", "创建时间", "状态"
            ])
            
            # 设置表格属性
            header = self.dataset_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            
            self.dataset_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.dataset_table.itemSelectionChanged.connect(self.on_dataset_selection_changed)
            
            layout.addWidget(self.dataset_table)
            
            return widget
        
        def create_dataset_details_tab(self) -> QWidget:
            """创建数据集详情选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 数据集选择
            selection_layout = QHBoxLayout()
            selection_layout.addWidget(QLabel("选择数据集:"))
            
            self.dataset_combo = QComboBox()
            self.dataset_combo.currentTextChanged.connect(self.on_dataset_combo_changed)
            selection_layout.addWidget(self.dataset_combo)
            
            self.view_details_btn = QPushButton("查看详情")
            self.view_details_btn.clicked.connect(self.view_dataset_details)
            selection_layout.addWidget(self.view_details_btn)
            
            selection_layout.addStretch()
            layout.addLayout(selection_layout)
            
            # 详情显示区域
            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            self.details_layout = QVBoxLayout(scroll_widget)
            
            # 基本信息组
            self.basic_info_group = QGroupBox("基本信息")
            self.basic_info_layout = QGridLayout(self.basic_info_group)
            self.details_layout.addWidget(self.basic_info_group)
            
            # 统计信息组
            self.stats_group = QGroupBox("统计信息")
            self.stats_layout = QGridLayout(self.stats_group)
            self.details_layout.addWidget(self.stats_group)
            
            # 文件信息组
            self.files_group = QGroupBox("文件信息")
            self.files_layout = QVBoxLayout(self.files_group)
            self.details_layout.addWidget(self.files_group)
            
            self.details_layout.addStretch()
            
            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)
            
            return widget
        
        def create_operations_tab(self) -> QWidget:
            """创建数据集操作选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 数据集选择
            selection_layout = QHBoxLayout()
            selection_layout.addWidget(QLabel("操作数据集:"))
            
            self.operation_dataset_combo = QComboBox()
            selection_layout.addWidget(self.operation_dataset_combo)
            
            selection_layout.addStretch()
            layout.addLayout(selection_layout)
            
            # 导入操作组
            import_group = QGroupBox("导入操作")
            import_layout = QVBoxLayout(import_group)
            
            import_buttons_layout = QHBoxLayout()
            
            self.import_images_btn = QPushButton("导入图像")
            self.import_images_btn.clicked.connect(self.import_images)
            import_buttons_layout.addWidget(self.import_images_btn)
            
            self.import_annotations_btn = QPushButton("导入标注")
            self.import_annotations_btn.clicked.connect(self.import_annotations)
            import_buttons_layout.addWidget(self.import_annotations_btn)
            
            import_layout.addLayout(import_buttons_layout)
            layout.addWidget(import_group)
            
            # 导出操作组
            export_group = QGroupBox("导出操作")
            export_layout = QVBoxLayout(export_group)
            
            export_buttons_layout = QHBoxLayout()
            
            self.export_coco_btn = QPushButton("导出COCO格式")
            self.export_coco_btn.clicked.connect(self.export_coco_dataset)
            export_buttons_layout.addWidget(self.export_coco_btn)
            
            self.export_yolo_btn = QPushButton("导出YOLO格式")
            self.export_yolo_btn.clicked.connect(self.export_yolo_dataset)
            export_buttons_layout.addWidget(self.export_yolo_btn)
            
            export_layout.addLayout(export_buttons_layout)
            layout.addWidget(export_group)
            
            # 数据集分割组
            split_group = QGroupBox("数据集分割")
            split_layout = QGridLayout(split_group)
            
            split_layout.addWidget(QLabel("训练集比例:"), 0, 0)
            self.train_ratio_spinbox = QSpinBox()
            self.train_ratio_spinbox.setRange(50, 90)
            self.train_ratio_spinbox.setValue(80)
            self.train_ratio_spinbox.setSuffix("%")
            split_layout.addWidget(self.train_ratio_spinbox, 0, 1)
            
            split_layout.addWidget(QLabel("验证集比例:"), 1, 0)
            self.val_ratio_spinbox = QSpinBox()
            self.val_ratio_spinbox.setRange(5, 25)
            self.val_ratio_spinbox.setValue(10)
            self.val_ratio_spinbox.setSuffix("%")
            split_layout.addWidget(self.val_ratio_spinbox, 1, 1)
            
            split_layout.addWidget(QLabel("测试集比例:"), 2, 0)
            self.test_ratio_spinbox = QSpinBox()
            self.test_ratio_spinbox.setRange(5, 25)
            self.test_ratio_spinbox.setValue(10)
            self.test_ratio_spinbox.setSuffix("%")
            split_layout.addWidget(self.test_ratio_spinbox, 2, 1)
            
            self.split_dataset_btn = QPushButton("执行分割")
            self.split_dataset_btn.clicked.connect(self.split_dataset)
            split_layout.addWidget(self.split_dataset_btn, 3, 0, 1, 2)
            
            layout.addWidget(split_group)
            
            layout.addStretch()
            
            return widget
        
        def create_status_bar(self, layout: QVBoxLayout):
            """创建状态栏"""
            status_frame = QFrame()
            status_frame.setFrameStyle(QFrame.StyledPanel)
            status_layout = QVBoxLayout(status_frame)
            
            # 状态标签
            self.status_label = QLabel("就绪")
            status_layout.addWidget(self.status_label)
            
            layout.addWidget(status_frame)
        
        def refresh_dataset_list(self):
            """刷新数据集列表"""
            try:
                self.current_datasets = self.dataset_manager.get_dataset_list()
                self.update_dataset_table()
                self.update_dataset_combos()
                self.update_status(f"已加载 {len(self.current_datasets)} 个数据集")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"刷新数据集列表失败: {str(e)}")
                logger.error(f"刷新数据集列表失败: {e}", exc_info=True)
        
        def update_dataset_table(self):
            """更新数据集表格"""
            self.dataset_table.setRowCount(len(self.current_datasets))
            
            for row, dataset in enumerate(self.current_datasets):
                # 名称
                self.dataset_table.setItem(row, 0, QTableWidgetItem(dataset["name"]))
                
                # 描述
                description = dataset.get("description", "")[:50]
                if len(dataset.get("description", "")) > 50:
                    description += "..."
                self.dataset_table.setItem(row, 1, QTableWidgetItem(description))
                
                # 图像数量
                self.dataset_table.setItem(row, 2, QTableWidgetItem(str(dataset.get("image_count", 0))))
                
                # 目标类型
                target_types = ", ".join(dataset.get("target_types", []))
                self.dataset_table.setItem(row, 3, QTableWidgetItem(target_types))
                
                # 创建时间
                created_date = dataset.get("created_date", "")[:10]  # 只显示日期部分
                self.dataset_table.setItem(row, 4, QTableWidgetItem(created_date))
                
                # 状态
                status = dataset.get("status", "unknown")
                self.dataset_table.setItem(row, 5, QTableWidgetItem(status))
        
        def update_dataset_combos(self):
            """更新数据集下拉框"""
            dataset_names = [ds["name"] for ds in self.current_datasets]
            
            self.dataset_combo.clear()
            self.dataset_combo.addItems(dataset_names)
            
            self.operation_dataset_combo.clear()
            self.operation_dataset_combo.addItems(dataset_names)
        
        def on_dataset_selection_changed(self):
            """数据集选择改变处理"""
            selected_rows = set()
            for item in self.dataset_table.selectedItems():
                selected_rows.add(item.row())
            
            self.delete_dataset_btn.setEnabled(len(selected_rows) > 0)
        
        def on_dataset_combo_changed(self, dataset_name: str):
            """数据集下拉框改变处理"""
            if dataset_name:
                self.view_dataset_details()
        
        def create_dataset(self):
            """创建数据集"""
            dialog = CreateDatasetDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                dataset_info = dialog.get_dataset_info()
                
                if not dataset_info["name"]:
                    QMessageBox.warning(self, "警告", "请输入数据集名称")
                    return
                
                try:
                    success = self.dataset_manager.create_dataset(**dataset_info)
                    if success:
                        QMessageBox.information(self, "成功", f"数据集 '{dataset_info['name']}' 创建成功")
                        self.refresh_dataset_list()
                    else:
                        QMessageBox.warning(self, "失败", "数据集创建失败")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"创建数据集时出错: {str(e)}")
        
        def delete_dataset(self):
            """删除数据集"""
            selected_rows = set()
            for item in self.dataset_table.selectedItems():
                selected_rows.add(item.row())
            
            if not selected_rows:
                return
            
            selected_datasets = [self.current_datasets[row]["name"] for row in selected_rows]
            
            reply = QMessageBox.question(
                self, "确认删除", 
                f"确定要删除以下数据集吗？\n{', '.join(selected_datasets)}\n\n注意：这将标记数据集为已删除状态，不会立即删除文件。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    for dataset_name in selected_datasets:
                        self.dataset_manager.delete_dataset(dataset_name, permanent=False)
                    
                    QMessageBox.information(self, "成功", f"已删除 {len(selected_datasets)} 个数据集")
                    self.refresh_dataset_list()
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"删除数据集时出错: {str(e)}")
        
        def view_dataset_details(self):
            """查看数据集详情"""
            dataset_name = self.dataset_combo.currentText()
            if not dataset_name:
                return
            
            try:
                stats = self.dataset_manager.get_dataset_statistics(dataset_name)
                if not stats:
                    self.update_status("无法获取数据集详情")
                    return
                
                # 清空现有内容
                self.clear_details_layout()
                
                # 基本信息
                self.add_detail_item(self.basic_info_layout, "名称:", stats["name"])
                self.add_detail_item(self.basic_info_layout, "图像数量:", str(stats["image_count"]))
                self.add_detail_item(self.basic_info_layout, "标注文件:", str(stats["annotation_files"]))
                self.add_detail_item(self.basic_info_layout, "创建时间:", stats["created_date"][:19])
                self.add_detail_item(self.basic_info_layout, "最后修改:", stats["last_modified"][:19])
                
                # 统计信息
                self.add_detail_item(self.stats_layout, "目标类型:", ", ".join(stats["target_types"]))
                self.add_detail_item(self.stats_layout, "天气条件:", ", ".join(stats["weather_conditions"]))
                self.add_detail_item(self.stats_layout, "地形类型:", ", ".join(stats["terrain_types"]))
                
                # 磁盘使用
                disk_usage = stats["disk_usage"]
                self.add_detail_item(self.stats_layout, "磁盘使用:", f"{disk_usage['total_mb']} MB ({disk_usage['file_count']} 文件)")
                
                self.update_status(f"已显示数据集 '{dataset_name}' 的详情")
                
            except Exception as e:
                QMessageBox.warning(self, "错误", f"获取数据集详情失败: {str(e)}")
        
        def clear_details_layout(self):
            """清空详情布局"""
            for layout in [self.basic_info_layout, self.stats_layout]:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        
        def add_detail_item(self, layout: QGridLayout, label: str, value: str):
            """添加详情项"""
            row = layout.rowCount()
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(QLabel(value), row, 1)
        
        def import_images(self):
            """导入图像"""
            dataset_name = self.operation_dataset_combo.currentText()
            if not dataset_name:
                QMessageBox.warning(self, "警告", "请选择数据集")
                return
            
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图像文件", "", 
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            
            if files:
                try:
                    dataset_path = self.dataset_manager.get_dataset_path(dataset_name)
                    images_dir = dataset_path / "images"
                    
                    imported_count = 0
                    for file_path in files:
                        src_path = Path(file_path)
                        dst_path = images_dir / src_path.name
                        
                        # 复制文件
                        import shutil
                        shutil.copy2(src_path, dst_path)
                        imported_count += 1
                    
                    QMessageBox.information(self, "成功", f"已导入 {imported_count} 张图像")
                    self.refresh_dataset_list()
                    
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"导入图像失败: {str(e)}")
        
        def import_annotations(self):
            """导入标注"""
            dataset_name = self.operation_dataset_combo.currentText()
            if not dataset_name:
                QMessageBox.warning(self, "警告", "请选择数据集")
                return
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择标注文件", "", "JSON文件 (*.json)"
            )
            
            if file_path:
                try:
                    dataset_path = self.dataset_manager.get_dataset_path(dataset_name)
                    annotations_dir = dataset_path / "annotations"
                    
                    # 复制标注文件
                    import shutil
                    src_path = Path(file_path)
                    dst_path = annotations_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    
                    QMessageBox.information(self, "成功", "标注文件导入成功")
                    
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"导入标注失败: {str(e)}")
        
        def export_coco_dataset(self):
            """导出COCO格式数据集"""
            dataset_name = self.operation_dataset_combo.currentText()
            if not dataset_name:
                QMessageBox.warning(self, "警告", "请选择数据集")
                return
            
            directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if directory:
                try:
                    # 这里需要实现COCO格式导出逻辑
                    QMessageBox.information(self, "提示", "COCO格式导出功能正在开发中")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"导出COCO格式失败: {str(e)}")
        
        def export_yolo_dataset(self):
            """导出YOLO格式数据集"""
            dataset_name = self.operation_dataset_combo.currentText()
            if not dataset_name:
                QMessageBox.warning(self, "警告", "请选择数据集")
                return
            
            directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if directory:
                try:
                    # 这里需要实现YOLO格式导出逻辑
                    QMessageBox.information(self, "提示", "YOLO格式导出功能正在开发中")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"导出YOLO格式失败: {str(e)}")
        
        def split_dataset(self):
            """分割数据集"""
            dataset_name = self.operation_dataset_combo.currentText()
            if not dataset_name:
                QMessageBox.warning(self, "警告", "请选择数据集")
                return
            
            # 验证比例
            train_ratio = self.train_ratio_spinbox.value() / 100
            val_ratio = self.val_ratio_spinbox.value() / 100
            test_ratio = self.test_ratio_spinbox.value() / 100
            
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
                QMessageBox.warning(self, "警告", "训练集、验证集和测试集比例之和必须为100%")
                return
            
            try:
                # 这里需要实现数据集分割逻辑
                QMessageBox.information(self, "提示", "数据集分割功能正在开发中")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"分割数据集失败: {str(e)}")
        
        def update_status(self, status: str):
            """更新状态"""
            self.status_label.setText(status)

else:
    # PyQt5不可用时的占位类
    class DatasetPanel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5不可用，无法创建数据集面板")

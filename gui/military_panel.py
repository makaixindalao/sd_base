"""
军事目标生成面板
提供军事目标图像生成的专用界面
"""

import sys
import random
from typing import List, Dict, Optional
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QComboBox, QCheckBox, QSpinBox, QPushButton,
                                QTextEdit, QProgressBar, QGroupBox, QListWidget,
                                QListWidgetItem, QMessageBox, QFileDialog,
                                QTabWidget, QScrollArea, QFrame)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

if PYQT_AVAILABLE:
    from military.target_generator import MilitaryTargetGenerator
    from military.prompt_templates import PromptTemplateManager
    from config import Config
    import logging

    logger = logging.getLogger(__name__)

    class MilitaryGenerationThread(QThread):
        """军事目标生成线程"""
        progress_updated = pyqtSignal(float)
        status_updated = pyqtSignal(str)
        generation_completed = pyqtSignal(list)
        
        def __init__(self, generator: MilitaryTargetGenerator, generation_params: Dict):
            super().__init__()
            self.generator = generator
            self.params = generation_params
            self.is_cancelled = False
        
        def run(self):
            """运行生成任务"""
            try:
                # 设置回调
                self.generator.set_callbacks(
                    progress_callback=self.progress_updated.emit,
                    status_callback=self.status_updated.emit
                )
                
                # 执行批量生成
                results = self.generator.generate_batch_targets(**self.params)
                
                if not self.is_cancelled:
                    self.generation_completed.emit(results)
                    
            except Exception as e:
                self.status_updated.emit(f"生成失败: {str(e)}")
                logger.error(f"军事目标生成失败: {e}", exc_info=True)
        
        def cancel(self):
            """取消生成"""
            self.is_cancelled = True

    class MilitaryGenerationPanel(QWidget):
        """军事目标生成面板"""
        
        def __init__(self, sd_generator=None):
            super().__init__()
            
            if not PYQT_AVAILABLE:
                raise ImportError("PyQt5不可用")
            
            self.sd_generator = sd_generator
            self.military_generator = MilitaryTargetGenerator(sd_generator)
            self.prompt_manager = PromptTemplateManager()
            self.config = Config()
            
            self.generation_thread = None
            self.current_results = []
            
            self.setup_ui()
            self.load_settings()
        
        def setup_ui(self):
            """设置用户界面"""
            layout = QVBoxLayout(self)
            
            # 创建选项卡
            tab_widget = QTabWidget()
            
            # 生成配置选项卡
            config_tab = self.create_config_tab()
            tab_widget.addTab(config_tab, "生成配置")
            
            # 批量生成选项卡
            batch_tab = self.create_batch_tab()
            tab_widget.addTab(batch_tab, "批量生成")
            
            # 结果查看选项卡
            results_tab = self.create_results_tab()
            tab_widget.addTab(results_tab, "生成结果")
            
            layout.addWidget(tab_widget)
            
            # 状态栏
            self.create_status_bar(layout)
        
        def create_config_tab(self) -> QWidget:
            """创建生成配置选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 目标类型选择
            target_group = QGroupBox("军事目标类型")
            target_layout = QVBoxLayout(target_group)
            
            self.target_checkboxes = {}
            available_targets = self.prompt_manager.get_available_options()["targets"]
            for target in available_targets:
                checkbox = QCheckBox(target.upper())
                checkbox.setChecked(True)
                self.target_checkboxes[target] = checkbox
                target_layout.addWidget(checkbox)
            
            layout.addWidget(target_group)
            
            # 天气条件选择
            weather_group = QGroupBox("天气条件")
            weather_layout = QVBoxLayout(weather_group)
            
            self.weather_checkboxes = {}
            available_weather = self.prompt_manager.get_available_options()["weather"]
            for weather in available_weather:
                checkbox = QCheckBox(weather.upper())
                checkbox.setChecked(True)
                self.weather_checkboxes[weather] = checkbox
                weather_layout.addWidget(checkbox)
            
            layout.addWidget(weather_group)
            
            # 地形类型选择
            terrain_group = QGroupBox("地形类型")
            terrain_layout = QVBoxLayout(terrain_group)
            
            self.terrain_checkboxes = {}
            available_terrain = self.prompt_manager.get_available_options()["terrain"]
            for terrain in available_terrain:
                checkbox = QCheckBox(terrain.upper())
                checkbox.setChecked(True)
                self.terrain_checkboxes[terrain] = checkbox
                terrain_layout.addWidget(checkbox)
            
            layout.addWidget(terrain_group)
            
            # 生成模式选择
            mode_group = QGroupBox("生成模式")
            mode_layout = QVBoxLayout(mode_group)
            
            self.mixed_targets_cb = QCheckBox("混合目标随机生成")
            self.mixed_targets_cb.setChecked(True)
            mode_layout.addWidget(self.mixed_targets_cb)
            
            self.mixed_scenes_cb = QCheckBox("混合场景随机生成")
            self.mixed_scenes_cb.setChecked(True)
            mode_layout.addWidget(self.mixed_scenes_cb)
            
            layout.addWidget(mode_group)
            
            return widget
        
        def create_batch_tab(self) -> QWidget:
            """创建批量生成选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 生成数量设置
            count_group = QGroupBox("生成设置")
            count_layout = QGridLayout(count_group)
            
            count_layout.addWidget(QLabel("生成数量:"), 0, 0)
            self.count_spinbox = QSpinBox()
            self.count_spinbox.setRange(1, 10000)
            self.count_spinbox.setValue(10)
            count_layout.addWidget(self.count_spinbox, 0, 1)
            
            # 快速设置按钮
            quick_buttons_layout = QHBoxLayout()
            for count in [10, 50, 100, 500, 1000]:
                btn = QPushButton(f"{count}张")
                btn.clicked.connect(lambda checked, c=count: self.count_spinbox.setValue(c))
                quick_buttons_layout.addWidget(btn)
            count_layout.addLayout(quick_buttons_layout, 1, 0, 1, 2)
            
            count_layout.addWidget(QLabel("输出目录:"), 2, 0)
            self.output_dir_layout = QHBoxLayout()
            self.output_dir_label = QLabel("outputs/military")
            self.output_dir_btn = QPushButton("选择")
            self.output_dir_btn.clicked.connect(self.select_output_directory)
            self.output_dir_layout.addWidget(self.output_dir_label)
            self.output_dir_layout.addWidget(self.output_dir_btn)
            count_layout.addLayout(self.output_dir_layout, 2, 1)
            
            layout.addWidget(count_group)
            
            # 生成参数设置
            params_group = QGroupBox("生成参数")
            params_layout = QGridLayout(params_group)
            
            params_layout.addWidget(QLabel("图像宽度:"), 0, 0)
            self.width_spinbox = QSpinBox()
            self.width_spinbox.setRange(512, 1024)
            self.width_spinbox.setValue(512)
            self.width_spinbox.setSingleStep(64)
            params_layout.addWidget(self.width_spinbox, 0, 1)
            
            params_layout.addWidget(QLabel("图像高度:"), 1, 0)
            self.height_spinbox = QSpinBox()
            self.height_spinbox.setRange(512, 1024)
            self.height_spinbox.setValue(512)
            self.height_spinbox.setSingleStep(64)
            params_layout.addWidget(self.height_spinbox, 1, 1)
            
            params_layout.addWidget(QLabel("采样步数:"), 2, 0)
            self.steps_spinbox = QSpinBox()
            self.steps_spinbox.setRange(10, 50)
            self.steps_spinbox.setValue(20)
            params_layout.addWidget(self.steps_spinbox, 2, 1)
            
            params_layout.addWidget(QLabel("引导系数:"), 3, 0)
            self.guidance_spinbox = QSpinBox()
            self.guidance_spinbox.setRange(1, 20)
            self.guidance_spinbox.setValue(7)
            params_layout.addWidget(self.guidance_spinbox, 3, 1)
            
            layout.addWidget(params_group)
            
            # 控制按钮
            button_layout = QHBoxLayout()
            
            self.generate_btn = QPushButton("开始生成")
            self.generate_btn.clicked.connect(self.start_generation)
            self.generate_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            button_layout.addWidget(self.generate_btn)
            
            self.cancel_btn = QPushButton("取消生成")
            self.cancel_btn.clicked.connect(self.cancel_generation)
            self.cancel_btn.setEnabled(False)
            button_layout.addWidget(self.cancel_btn)
            
            layout.addLayout(button_layout)
            
            return widget
        
        def create_results_tab(self) -> QWidget:
            """创建结果查看选项卡"""
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            # 结果列表
            self.results_list = QListWidget()
            layout.addWidget(self.results_list)
            
            # 结果统计
            stats_group = QGroupBox("生成统计")
            stats_layout = QGridLayout(stats_group)
            
            self.stats_labels = {
                "total": QLabel("总数: 0"),
                "successful": QLabel("成功: 0"),
                "failed": QLabel("失败: 0"),
                "time": QLabel("耗时: 0秒")
            }
            
            row = 0
            for key, label in self.stats_labels.items():
                stats_layout.addWidget(label, row // 2, row % 2)
                row += 1
            
            layout.addWidget(stats_group)
            
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
        
        def get_selected_targets(self) -> List[str]:
            """获取选中的目标类型"""
            return [target for target, checkbox in self.target_checkboxes.items() 
                   if checkbox.isChecked()]
        
        def get_selected_weather(self) -> List[str]:
            """获取选中的天气条件"""
            return [weather for weather, checkbox in self.weather_checkboxes.items() 
                   if checkbox.isChecked()]
        
        def get_selected_terrain(self) -> List[str]:
            """获取选中的地形类型"""
            return [terrain for terrain, checkbox in self.terrain_checkboxes.items() 
                   if checkbox.isChecked()]
        
        def select_output_directory(self):
            """选择输出目录"""
            directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if directory:
                self.output_dir_label.setText(directory)
        
        def start_generation(self):
            """开始生成"""
            # 验证选择
            targets = self.get_selected_targets()
            if not targets:
                QMessageBox.warning(self, "警告", "请至少选择一种军事目标类型")
                return
            
            # 准备生成参数
            generation_params = {
                "target_types": targets,
                "weather_conditions": self.get_selected_weather(),
                "terrain_types": self.get_selected_terrain(),
                "count": self.count_spinbox.value(),
                "mixed_targets": self.mixed_targets_cb.isChecked(),
                "mixed_scenes": self.mixed_scenes_cb.isChecked(),
                "output_dir": self.output_dir_label.text(),
                "width": self.width_spinbox.value(),
                "height": self.height_spinbox.value(),
                "num_inference_steps": self.steps_spinbox.value(),
                "guidance_scale": self.guidance_spinbox.value()
            }
            
            # 启动生成线程
            self.generation_thread = MilitaryGenerationThread(
                self.military_generator, generation_params
            )
            self.generation_thread.progress_updated.connect(self.update_progress)
            self.generation_thread.status_updated.connect(self.update_status)
            self.generation_thread.generation_completed.connect(self.on_generation_completed)
            
            self.generation_thread.start()
            
            # 更新UI状态
            self.generate_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        
        def cancel_generation(self):
            """取消生成"""
            if self.generation_thread:
                self.generation_thread.cancel()
                self.generation_thread.wait()
            
            self.reset_ui_state()
            self.update_status("生成已取消")
        
        def update_progress(self, progress: float):
            """更新进度"""
            self.progress_bar.setValue(int(progress))
        
        def update_status(self, status: str):
            """更新状态"""
            self.status_label.setText(status)
        
        def on_generation_completed(self, results: List[Dict]):
            """生成完成处理"""
            self.current_results = results
            self.update_results_display()
            self.update_statistics()
            self.reset_ui_state()
            
            # 显示完成消息
            successful_count = sum(1 for r in results if r["success"])
            QMessageBox.information(
                self, "生成完成", 
                f"批量生成完成!\n成功生成: {successful_count}/{len(results)} 张图像"
            )
        
        def update_results_display(self):
            """更新结果显示"""
            self.results_list.clear()
            
            for result in self.current_results:
                status_icon = "✅" if result["success"] else "❌"
                item_text = (f"{status_icon} {result['target_type']} - "
                           f"{result.get('weather', 'default')} - "
                           f"{result.get('terrain', 'default')}")
                
                item = QListWidgetItem(item_text)
                self.results_list.addItem(item)
        
        def update_statistics(self):
            """更新统计信息"""
            if not self.current_results:
                return
            
            total = len(self.current_results)
            successful = sum(1 for r in self.current_results if r["success"])
            failed = total - successful
            
            stats = self.military_generator.get_generation_stats()
            total_time = stats.get("total_time", 0)
            
            self.stats_labels["total"].setText(f"总数: {total}")
            self.stats_labels["successful"].setText(f"成功: {successful}")
            self.stats_labels["failed"].setText(f"失败: {failed}")
            self.stats_labels["time"].setText(f"耗时: {total_time:.1f}秒")
        
        def reset_ui_state(self):
            """重置UI状态"""
            self.generate_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
        
        def load_settings(self):
            """加载设置"""
            military_config = self.config.get("military", {})
            
            # 加载默认分辨率
            default_resolution = military_config.get("default_resolution", [512, 512])
            self.width_spinbox.setValue(default_resolution[0])
            self.height_spinbox.setValue(default_resolution[1])
        
        def save_settings(self):
            """保存设置"""
            # 可以在这里保存用户的选择偏好
            pass

else:
    # PyQt5不可用时的占位类
    class MilitaryGenerationPanel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5不可用，无法创建军事生成面板")

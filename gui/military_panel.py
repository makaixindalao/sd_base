"""
军事场景生成面板
提供基于图片蒙版（Inpainting）的军事场景生成交互界面
"""

import sys
from typing import List, Dict, Optional
from pathlib import Path

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QListWidget, QListWidgetItem,
                                QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                                QAbstractItemView, QSplitter, QGroupBox, QSpinBox,
                                QTextEdit, QMessageBox, QProgressDialog)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
    from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QColor, QBrush, QPen
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Define placeholders for PyQt classes if not available
    QWidget = object
    pyqtSignal = object
    QThread = object


if PYQT_AVAILABLE:
    from military.scene_generator import MilitarySceneGenerator
    from config import Config
    import logging

    logger = logging.getLogger(__name__)

    # --- Generation Thread ---
    class SceneGenerationThread(QThread):
        """在新线程中运行军事场景生成任务"""
        generation_completed = pyqtSignal(str)
        status_updated = pyqtSignal(str)
        progress_updated = pyqtSignal(float)

        def __init__(self, generator: MilitarySceneGenerator, params: dict):
            super().__init__()
            self.generator = generator
            self.params = params

        def run(self):
            try:
                self.generator.set_callbacks(
                    status_callback=self.status_updated.emit,
                    progress_callback=self.progress_updated.emit
                )
                filepath = self.generator.generate_scene(**self.params)
                self.generation_completed.emit(filepath or "")
            except Exception as e:
                error_message = f"场景生成时发生意外错误: {e}"
                self.status_updated.emit(error_message)
                logger.error(error_message, exc_info=True)
                self.generation_completed.emit("")

    # --- Draggable Target Item ---
    class DraggableTarget(QGraphicsPixmapItem):
        """可拖拽的军事目标图元"""
        def __init__(self, pixmap: QPixmap, path: str):
            super().__init__(pixmap)
            self.path = path
            self.setFlags(QGraphicsPixmapItem.ItemIsSelectable | QGraphicsPixmapItem.ItemIsMovable)
            self.setAcceptHoverEvents(True)
        
        def hoverEnterEvent(self, event):
            self.setPen(QPen(QColor("yellow"), 2))
            super().hoverEnterEvent(event)
        
        def hoverLeaveEvent(self, event):
            self.setPen(QPen(Qt.NoPen))
            super().hoverLeaveEvent(event)

    # --- Main Panel ---
    class MilitaryPanel(QWidget):
        def __init__(self, sd_generator=None):
            super().__init__()
            if not PYQT_AVAILABLE:
                self.setLayout(QVBoxLayout())
                self.layout().addWidget(QLabel("错误: PyQt5 未安装或加载失败。"))
                return
            
            self.sd_generator = sd_generator
            self.scene_generator = MilitarySceneGenerator(self.sd_generator)
            self.config = Config()
            
            self.base_scene_path: Optional[str] = None
            self.target_items: Dict[str, DraggableTarget] = {} # path: item

            self.setup_ui()
            self.apply_stylesheet()

        def setup_ui(self):
            main_layout = QHBoxLayout(self)
            splitter = QSplitter(Qt.Horizontal)

            # --- Left Panel (Controls) ---
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(10, 10, 10, 10)
            left_layout.setSpacing(15)

            # Scene Group
            scene_group = QGroupBox("1. 选择背景场景")
            scene_layout = QVBoxLayout(scene_group)
            self.scene_path_label = QLabel("未选择场景文件")
            self.scene_path_label.setWordWrap(True)
            self.select_scene_btn = QPushButton("浏览...")
            self.select_scene_btn.clicked.connect(self.select_base_scene)
            scene_layout.addWidget(self.scene_path_label)
            scene_layout.addWidget(self.select_scene_btn)
            
            # Target Group
            target_group = QGroupBox("2. 添加并拖拽目标")
            target_layout = QVBoxLayout(target_group)
            self.target_list_widget = QListWidget()
            self.target_list_widget.setToolTip("已添加的目标列表")
            target_buttons_layout = QHBoxLayout()
            self.add_target_btn = QPushButton("添加")
            self.add_target_btn.clicked.connect(self.add_targets)
            self.remove_target_btn = QPushButton("移除")
            self.remove_target_btn.clicked.connect(self.remove_selected_target)
            target_buttons_layout.addWidget(self.add_target_btn)
            target_buttons_layout.addWidget(self.remove_target_btn)
            target_layout.addWidget(self.target_list_widget)
            target_layout.addLayout(target_buttons_layout)

            # Prompt Group
            prompt_group = QGroupBox("3. 输入融合指令")
            prompt_layout = QVBoxLayout(prompt_group)
            self.prompt_input = QTextEdit()
            self.prompt_input.setPlaceholderText("描述天气、光照和整体风格，例如：\n- a heavy rain, night time, cinematic lighting\n- a sandstorm in desert, photorealistic\n- snowy weather, soft light")
            self.prompt_input.setFixedHeight(100)
            prompt_layout.addWidget(self.prompt_input)

            # Generation Group
            gen_group = QGroupBox("4. 开始生成")
            gen_layout = QVBoxLayout(gen_group)
            self.generate_btn = QPushButton("生成场景")
            self.generate_btn.setObjectName("GenerateButton")
            self.generate_btn.clicked.connect(self.start_generation)
            gen_layout.addWidget(self.generate_btn)

            left_layout.addWidget(scene_group)
            left_layout.addWidget(target_group)
            left_layout.addWidget(prompt_group)
            left_layout.addWidget(gen_group)
            left_layout.addStretch()

            # --- Right Panel (Visual Editor) ---
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            
            self.graphics_scene = QGraphicsScene()
            self.graphics_view = QGraphicsView(self.graphics_scene)
            self.graphics_view.setRenderHint(QPainter.Antialiasing)
            self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
            self.graphics_view.setBackgroundBrush(QBrush(QColor(50, 50, 50)))

            self.scene_pixmap_item = QGraphicsPixmapItem()
            self.graphics_scene.addItem(self.scene_pixmap_item)

            right_layout.addWidget(self.graphics_view)

            # Add panels to splitter
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 3) # Give more space to the visual editor
            main_layout.addWidget(splitter)
        
        def apply_stylesheet(self):
            self.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #4A4A4A;
                    border-radius: 5px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 5px;
                }
                QPushButton {
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #0078D7;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
                QPushButton:disabled {
                    background-color: #555555;
                }
                #GenerateButton {
                    background-color: #107C10; /* Green */
                }
                #GenerateButton:hover {
                    background-color: #0B530B;
                }
                QListWidget {
                    border: 1px solid #4A4A4A;
                    border-radius: 4px;
                }
                QTextEdit, QSpinBox {
                    border: 1px solid #4A4A4A;
                    border-radius: 4px;
                    padding: 5px;
                }
            """)

        def select_base_scene(self):
            filepath, _ = QFileDialog.getOpenFileName(self, "选择背景场景图片", "", "图片文件 (*.png *.jpg *.jpeg)")
            if not filepath: return
            
            self.base_scene_path = filepath
            pixmap = QPixmap(filepath)
            self.scene_pixmap_item.setPixmap(pixmap)
            self.graphics_view.setSceneRect(pixmap.rect())
            self.scene_path_label.setText(Path(filepath).name)

        def add_targets(self):
            filepaths, _ = QFileDialog.getOpenFileNames(self, "选择一个或多个目标图片 (PNG格式)", "", "PNG 图片 (*.png)")
            if not filepaths: return

            for path in filepaths:
                if path in self.target_items: continue # Avoid duplicates
                
                pixmap = QPixmap(path)
                item = DraggableTarget(pixmap, path)
                self.graphics_scene.addItem(item)
                self.target_items[path] = item
                
                list_item = QListWidgetItem(Path(path).name)
                list_item.setData(Qt.UserRole, path)
                self.target_list_widget.addItem(list_item)
        
        def remove_selected_target(self):
            selected_items = self.target_list_widget.selectedItems()
            if not selected_items: return

            for item in selected_items:
                path = item.data(Qt.UserRole)
                if path in self.target_items:
                    self.graphics_scene.removeItem(self.target_items[path])
                    del self.target_items[path]
                self.target_list_widget.takeItem(self.target_list_widget.row(item))

        def start_generation(self):
            if not self.base_scene_path or not self.target_items:
                QMessageBox.warning(self, "信息不完整", "请先选择一个背景场景并添加至少一个目标。")
                return
            
            prompt = self.prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(self, "信息不完整", "请输入融合指令（Prompt）。")
                return

            try:
                base_scene = Image.open(self.base_scene_path)
                targets_to_generate = []
                for path, item in self.target_items.items():
                    target_img = Image.open(path)
                    pos = item.pos()
                    targets_to_generate.append({
                        "image": target_img,
                        "position": (int(pos.x()), int(pos.y()))
                    })

                params = {
                    "base_scene": base_scene,
                    "targets": targets_to_generate,
                    "prompt": prompt
                }

                # Setup and start thread
                self.progress_dialog = QProgressDialog("正在生成场景...", "取消", 0, 100, self)
                self.progress_dialog.setWindowTitle("请稍候")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                
                self.gen_thread = SceneGenerationThread(self.scene_generator, params)
                self.gen_thread.status_updated.connect(self.progress_dialog.setLabelText)
                self.gen_thread.progress_updated.connect(lambda p: self.progress_dialog.setValue(int(p)))
                self.gen_thread.generation_completed.connect(self.on_generation_finished)
                self.progress_dialog.canceled.connect(self.gen_thread.terminate) # Force stop
                
                self.generate_btn.setEnabled(False)
                self.gen_thread.start()
                self.progress_dialog.show()

            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法开始生成: {e}")

        def on_generation_finished(self, filepath: str):
            self.progress_dialog.close()
            self.generate_btn.setEnabled(True)

            if filepath:
                QMessageBox.information(self, "生成完成", f"场景已成功生成并保存至:\n{filepath}")
                # Display the image
                pixmap = QPixmap(filepath)
                self.scene_pixmap_item.setPixmap(pixmap) # Replace scene with result
                # Clear targets
                for item in list(self.target_items.values()):
                    self.graphics_scene.removeItem(item)
                self.target_items.clear()
                self.target_list_widget.clear()
            else:
                QMessageBox.warning(self, "生成失败", "无法生成场景，请查看日志获取详细信息。")

else:
    # PyQt5不可用时的占位类
    class MilitaryPanel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt5不可用，无法创建军事生成面板")

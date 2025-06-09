"""
PyQt5图形用户界面模块
提供现代化的Stable Diffusion图片生成界面
"""

import sys
import os
import threading
import random
from datetime import datetime
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox,
        QDoubleSpinBox, QProgressBar, QMenuBar, QMenu, QAction, QStatusBar,
        QGroupBox, QComboBox, QCheckBox, QSlider, QScrollArea, QDialog,
        QDialogButtonBox, QRadioButton, QButtonGroup, QMessageBox, QFileDialog,
        QSplitter, QFrame, QTabWidget, QListWidget, QListWidgetItem
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
    from PyQt5.QtGui import (QPixmap, QFont, QIcon, QPalette, QColor, QImage)
    PYQT_AVAILABLE = True
except ImportError as e:
    PYQT_AVAILABLE = False
    IMPORT_ERROR = str(e)

from PIL import Image
from sd_generator import SDGenerator
from config import config
from utils import (logger, validate_prompt, generate_filename, 
                  save_image_with_metadata, get_system_info, check_disk_space)


class ModelLoadThread(QThread):
    """模型加载线程"""
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)  # 进度更新 0-100
    load_completed = pyqtSignal(bool)  # 加载是否成功

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.is_cancelled = False

    def run(self):
        """运行模型加载任务"""
        try:
            # 设置进度回调
            original_status_callback = self.generator.status_callback
            self.generator.status_callback = self.emit_status

            # 模拟加载进度
            self.progress_updated.emit(10)
            self.status_updated.emit("开始加载模型...")

            self.progress_updated.emit(30)
            success = self.generator.load_model()

            if success:
                self.progress_updated.emit(100)
                self.status_updated.emit("模型加载完成")
            else:
                self.progress_updated.emit(0)
                self.status_updated.emit("模型加载失败")

            # 恢复原始回调
            self.generator.status_callback = original_status_callback

            self.load_completed.emit(success)
        except Exception as e:
            logger.error(f"模型加载线程异常: {e}")
            self.status_updated.emit(f"模型加载失败: {str(e)}")
            self.progress_updated.emit(0)
            self.load_completed.emit(False)

    def emit_status(self, message):
        """发送状态更新"""
        self.status_updated.emit(message)

        # 根据状态消息更新进度
        if "正在检测设备" in message:
            self.progress_updated.emit(20)
        elif "正在检查模型" in message:
            self.progress_updated.emit(40)
        elif "正在加载" in message:
            self.progress_updated.emit(60)
        elif "加载成功" in message or "加载完成" in message:
            self.progress_updated.emit(90)

    def cancel(self):
        """取消加载"""
        self.is_cancelled = True


class ImageGenerationThread(QThread):
    """图片生成线程"""
    progress_updated = pyqtSignal(float)
    status_updated = pyqtSignal(str)
    generation_completed = pyqtSignal(object)  # PIL Image or None

    def __init__(self, generator, params):
        super().__init__()
        self.generator = generator
        self.params = params
        self.is_cancelled = False

    def run(self):
        """运行生成任务"""
        try:
            # 确保模型已加载
            if not self.generator.model_loaded:
                self.status_updated.emit("正在加载模型...")
                success = self.generator.load_model()
                if not success:
                    self.generation_completed.emit(None)
                    return

            if self.is_cancelled:
                return

            # 生成图片
            image = self.generator.generate_image(**self.params)

            if not self.is_cancelled:
                self.generation_completed.emit(image)

        except Exception as e:
            logger.error(f"生成线程异常: {e}")
            self.status_updated.emit(f"生成失败: {str(e)}")
            self.generation_completed.emit(None)

    def cancel(self):
        """取消生成"""
        self.is_cancelled = True


class ModelSelectionDialog(QDialog):
    """模型选择对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择模型")
        self.setFixedSize(600, 500)
        self.setModal(True)
        
        # 预设模型列表
        self.preset_models = [
            ("Stable Diffusion v1.5", "stable-diffusion-v1-5/stable-diffusion-v1-5", "经典模型，兼容性好，推荐使用"),
            ("Stable Diffusion v1.5 (RunwayML)", "runwayml/stable-diffusion-v1-5", "经典模型，RunwayML版本"),
            ("Stable Diffusion v2.1", "stabilityai/stable-diffusion-2-1", "改进版本，更好的文本理解"),
            ("Stable Diffusion XL Base", "stabilityai/stable-diffusion-xl-base-1.0", "高分辨率专用模型"),
            ("Stable Diffusion 3.5 Large", "stabilityai/stable-diffusion-3.5-large", "最新模型，质量最高"),
        ]
        
        self.setup_ui()
        self.load_current_selection()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 当前模型显示
        current_model = config.get("model.name")
        current_label = QLabel(f"当前模型: {current_model}")
        current_label.setFont(QFont("", 10, QFont.Bold))
        layout.addWidget(current_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        
        # 预设模型组
        preset_group = QGroupBox("预设模型")
        preset_layout = QVBoxLayout(preset_group)
        
        self.model_group = QButtonGroup(self)
        
        for i, (display_name, model_id, description) in enumerate(self.preset_models):
            radio = QRadioButton(display_name)
            radio.setProperty("model_id", model_id)
            self.model_group.addButton(radio, i)
            preset_layout.addWidget(radio)
            
            # 添加描述
            desc_label = QLabel(f"    {description}")
            desc_label.setStyleSheet("color: gray; margin-left: 20px;")
            preset_layout.addWidget(desc_label)
        
        layout.addWidget(preset_group)
        
        # 自定义模型组
        custom_group = QGroupBox("自定义模型")
        custom_layout = QVBoxLayout(custom_group)

        self.custom_radio = QRadioButton("使用自定义模型ID:")
        self.model_group.addButton(self.custom_radio, len(self.preset_models))
        custom_layout.addWidget(self.custom_radio)

        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("输入Hugging Face模型ID，例如: huggingface/model-name")
        custom_layout.addWidget(self.custom_input)

        layout.addWidget(custom_group)

        # 本地模型组
        local_group = QGroupBox("本地模型")
        local_layout = QVBoxLayout(local_group)

        self.local_radio = QRadioButton("使用本地模型文件:")
        self.model_group.addButton(self.local_radio, len(self.preset_models) + 1)
        local_layout.addWidget(self.local_radio)

        # 本地模型路径输入和浏览
        local_path_layout = QHBoxLayout()
        self.local_path_input = QLineEdit()
        self.local_path_input.setPlaceholderText("选择本地模型文件或目录...")
        self.local_path_input.setReadOnly(True)
        local_path_layout.addWidget(self.local_path_input)

        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_local_model)
        local_path_layout.addWidget(self.browse_button)

        local_layout.addLayout(local_path_layout)

        # 本地模型信息显示
        self.local_model_info = QLabel("选择本地模型后将显示模型信息")
        self.local_model_info.setStyleSheet("color: gray; font-style: italic;")
        self.local_model_info.setWordWrap(True)
        local_layout.addWidget(self.local_model_info)

        layout.addWidget(local_group)
        
        # 说明文本
        help_text = QLabel(
            "说明:\n"
            "• 预设模型: 已经过测试，推荐使用\n"
            "• 自定义模型: 输入完整的Hugging Face模型ID\n"
            "• 本地模型: 支持已下载的模型文件或目录\n"
            "• 首次使用新模型会自动下载（可能需要较长时间）\n"
            "• SD 3.5 Large 提供最佳质量但需要更多资源\n"
            "• 本地模型可以离线使用，加载速度更快"
        )
        help_text.setStyleSheet("color: gray;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def load_current_selection(self):
        """加载当前选择"""
        current_model = config.get("model.name")

        # 检查是否是预设模型
        for i, (_, model_id, _) in enumerate(self.preset_models):
            if model_id == current_model:
                self.model_group.button(i).setChecked(True)
                return

        # 检查是否是本地模型（路径包含斜杠或反斜杠，或者以models/开头）
        if current_model and (
            current_model.startswith('/') or
            current_model.startswith('\\') or
            ':' in current_model or
            current_model.startswith('models/')
        ):
            self.local_radio.setChecked(True)
            self.local_path_input.setText(current_model)
            self.validate_local_model(current_model)
            return

        # 如果不是预设模型或本地模型，设置为自定义
        if current_model:
            self.custom_radio.setChecked(True)
            self.custom_input.setText(current_model)
        else:
            # 如果没有配置模型，默认选择第一个预设模型（SD1.5）
            self.model_group.button(0).setChecked(True)

    def browse_local_model(self):
        """浏览本地模型"""
        from PyQt5.QtWidgets import QMessageBox

        # 询问用户选择类型
        reply = QMessageBox.question(
            self, "选择模型类型",
            "请选择要加载的模型类型：\n\n"
            "• 选择「是」浏览模型文件(.safetensors, .bin等)\n"
            "• 选择「否」浏览模型目录\n"
            "• 选择「取消」返回",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )

        model_path = ""

        if reply == QMessageBox.Yes:
            # 选择模型文件
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件",
                config.get("model.cache_dir", ""),
                "Stable Diffusion模型 (*.safetensors *.bin *.ckpt *.pth);;SafeTensors文件 (*.safetensors);;所有文件 (*)"
            )
        elif reply == QMessageBox.No:
            # 选择模型目录
            model_path = QFileDialog.getExistingDirectory(
                self, "选择模型目录",
                config.get("model.cache_dir", ""),
                QFileDialog.ShowDirsOnly
            )

        if model_path:
            self.local_path_input.setText(model_path)
            self.local_radio.setChecked(True)
            self.validate_local_model(model_path)

    def validate_local_model(self, model_path):
        """验证本地模型"""
        try:
            from pathlib import Path
            path = Path(model_path)

            if not path.exists():
                self.local_model_info.setText("❌ 路径不存在")
                self.local_model_info.setStyleSheet("color: red;")
                return False

            if path.is_file():
                # 检查文件扩展名
                valid_extensions = ['.safetensors', '.bin', '.ckpt', '.pth']
                if path.suffix.lower() in valid_extensions:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    file_type = "SafeTensors" if path.suffix.lower() == '.safetensors' else "模型"

                    info_text = f"✅ {file_type}文件: {path.name}\n"
                    info_text += f"大小: {size_mb:.1f} MB\n"
                    info_text += f"类型: {path.suffix.upper()}\n"
                    info_text += "状态: 可直接加载"

                    self.local_model_info.setText(info_text)
                    self.local_model_info.setStyleSheet("color: green; font-weight: bold;")
                    return True
                else:
                    self.local_model_info.setText("❌ 不支持的文件格式\n支持: .safetensors, .bin, .ckpt, .pth")
                    self.local_model_info.setStyleSheet("color: red;")
                    return False

            elif path.is_dir():
                # 检查目录中是否包含模型文件
                model_files = []
                config_files = []
                safetensors_files = []

                for file_path in path.iterdir():
                    if file_path.is_file():
                        if file_path.suffix.lower() == '.safetensors':
                            safetensors_files.append(file_path.name)
                            model_files.append(file_path.name)
                        elif file_path.suffix.lower() in ['.bin', '.ckpt', '.pth']:
                            model_files.append(file_path.name)
                        elif file_path.name in ['model_index.json', 'config.json', 'tokenizer_config.json']:
                            config_files.append(file_path.name)

                if model_files or config_files:
                    info_text = f"✅ 模型目录: {path.name}\n"
                    if safetensors_files:
                        info_text += f"SafeTensors文件: {len(safetensors_files)}个\n"
                    if model_files:
                        info_text += f"模型文件总数: {len(model_files)}个\n"
                    if config_files:
                        info_text += f"配置文件: {len(config_files)}个\n"
                    info_text += "状态: 可加载"

                    self.local_model_info.setText(info_text)
                    self.local_model_info.setStyleSheet("color: green; font-weight: bold;")
                    return True
                else:
                    self.local_model_info.setText("❌ 目录中未找到模型文件\n请选择包含.safetensors或其他模型文件的目录")
                    self.local_model_info.setStyleSheet("color: red;")
                    return False

        except Exception as e:
            self.local_model_info.setText(f"❌ 验证失败: {str(e)}")
            self.local_model_info.setStyleSheet("color: red;")
            return False
    
    def get_selected_model(self):
        """获取选择的模型"""
        checked_button = self.model_group.checkedButton()
        if not checked_button:
            return None

        button_id = self.model_group.id(checked_button)

        if button_id < len(self.preset_models):
            # 预设模型
            return self.preset_models[button_id][1]
        elif button_id == len(self.preset_models):
            # 自定义模型
            custom_model = self.custom_input.text().strip()
            if not custom_model or custom_model == "huggingface/model-name":
                return None
            return custom_model
        else:
            # 本地模型
            local_path = self.local_path_input.text().strip()
            if not local_path:
                return None
            return local_path


class SDGeneratorMainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 检查PyQt5可用性
        if not PYQT_AVAILABLE:
            raise ImportError(f"PyQt5不可用: {IMPORT_ERROR}")
        
        # 初始化组件
        self.generator = SDGenerator()
        self.current_image = None
        self.generation_thread = None
        self.is_generating = False
        
        # 设置回调
        self.generator.set_callbacks(
            progress_callback=self.update_progress,
            status_callback=self.update_status
        )
        
        # 设置界面
        self.setup_ui()
        self.load_config()

        # 启动时检查模型状态，但不自动加载
        QTimer.singleShot(1000, self.check_initial_model_status)
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("Stable Diffusion 图片生成器 v1.2")
        self.setMinimumSize(1000, 700)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        self.create_control_panel(splitter)
        
        # 右侧预览面板
        self.create_preview_panel(splitter)
        
        # 设置分割器比例
        splitter.setSizes([400, 600])
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_widget = QWidget()
        parent.addWidget(control_widget)
        
        layout = QVBoxLayout(control_widget)
        
        # 提示词组
        self.create_prompt_group(layout)
        
        # 参数组
        self.create_parameters_group(layout)
        
        # 控制按钮组
        self.create_control_buttons(layout)
        
        # 添加弹性空间
        layout.addStretch()
    
    def create_preview_panel(self, parent):
        """创建预览面板"""
        preview_widget = QWidget()
        parent.addWidget(preview_widget)
        
        layout = QVBoxLayout(preview_widget)
        
        # 预览标签
        preview_label = QLabel("图片预览")
        preview_label.setFont(QFont("", 12, QFont.Bold))
        layout.addWidget(preview_label)
        
        # 图片显示区域
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "QLabel { border: 2px dashed #aaa; background-color: #f5f5f5; }"
        )
        self.preview_label.setText("生成的图片将在这里显示")
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.preview_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # 保存按钮
        self.save_button = QPushButton("保存图片")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)

    def create_prompt_group(self, layout):
        """创建提示词组"""
        group = QGroupBox("提示词设置")
        group_layout = QVBoxLayout(group)

        # 正向提示词
        group_layout.addWidget(QLabel("描述 (Prompt):"))
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(80)
        self.prompt_text.setPlaceholderText("输入想要生成的图片描述...")
        group_layout.addWidget(self.prompt_text)

        # 负向提示词
        group_layout.addWidget(QLabel("负向提示词 (Negative Prompt):"))
        self.negative_prompt_text = QTextEdit()
        self.negative_prompt_text.setMaximumHeight(60)
        self.negative_prompt_text.setPlaceholderText("输入不想要的内容...")
        group_layout.addWidget(self.negative_prompt_text)

        layout.addWidget(group)

    def create_parameters_group(self, layout):
        """创建参数组"""
        group = QGroupBox("生成参数")
        group_layout = QGridLayout(group)

        # 尺寸设置
        group_layout.addWidget(QLabel("宽度:"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setSingleStep(8)
        self.width_spin.setValue(512)  # SD1.5默认512x512
        group_layout.addWidget(self.width_spin, 0, 1)

        group_layout.addWidget(QLabel("高度:"), 0, 2)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setSingleStep(8)
        self.height_spin.setValue(512)  # SD1.5默认512x512
        group_layout.addWidget(self.height_spin, 0, 3)

        # 采样步数
        group_layout.addWidget(QLabel("采样步数:"), 1, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(20)  # SD1.5默认20步
        group_layout.addWidget(self.steps_spin, 1, 1)

        # 引导系数
        group_layout.addWidget(QLabel("引导系数:"), 1, 2)
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(7.5)  # SD1.5默认7.5
        group_layout.addWidget(self.guidance_spin, 1, 3)

        # 随机种子
        group_layout.addWidget(QLabel("随机种子:"), 2, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31-1)
        self.seed_spin.setValue(-1)
        group_layout.addWidget(self.seed_spin, 2, 1)

        # 随机种子按钮
        random_seed_btn = QPushButton("随机")
        random_seed_btn.clicked.connect(self.randomize_seed)
        group_layout.addWidget(random_seed_btn, 2, 2)

        layout.addWidget(group)

    def create_control_buttons(self, layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()

        # 生成按钮
        self.generate_button = QPushButton("生成图片")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self.generate_image)
        button_layout.addWidget(self.generate_button)

        # 停止按钮
        self.stop_button = QPushButton("停止生成")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_generation)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        save_action = QAction("保存图片", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        save_config_action = QAction("保存配置", self)
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 模型菜单
        model_menu = menubar.addMenu("模型")

        select_model_action = QAction("选择模型", self)
        select_model_action.triggered.connect(self.select_model)
        model_menu.addAction(select_model_action)

        model_menu.addSeparator()

        load_model_action = QAction("加载模型", self)
        load_model_action.triggered.connect(self.load_model)
        model_menu.addAction(load_model_action)

        unload_model_action = QAction("卸载模型", self)
        unload_model_action.triggered.connect(self.unload_model)
        model_menu.addAction(unload_model_action)

        model_info_action = QAction("模型信息", self)
        model_info_action.triggered.connect(self.show_model_info)
        model_menu.addAction(model_info_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        system_info_action = QAction("系统信息", self)
        system_info_action.triggered.connect(self.show_system_info)
        help_menu.addAction(system_info_action)

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

    def load_config(self):
        """加载配置"""
        gen_config = config.get_generation_config()

        self.width_spin.setValue(gen_config["width"])
        self.height_spin.setValue(gen_config["height"])
        self.steps_spin.setValue(gen_config["num_inference_steps"])
        self.guidance_spin.setValue(gen_config["guidance_scale"])
        self.seed_spin.setValue(gen_config["seed"])

        if gen_config.get("negative_prompt"):
            self.negative_prompt_text.setPlainText(gen_config["negative_prompt"])

    def save_config(self):
        """保存配置"""
        config.update_generation_config(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            num_inference_steps=self.steps_spin.value(),
            guidance_scale=self.guidance_spin.value(),
            seed=self.seed_spin.value(),
            negative_prompt=self.negative_prompt_text.toPlainText().strip()
        )
        config.save_config()
        QMessageBox.information(self, "保存配置", "配置已保存")

    def randomize_seed(self):
        """随机化种子"""
        self.seed_spin.setValue(random.randint(0, 2**31 - 1))

    def check_initial_model_status(self):
        """检查初始模型状态"""
        current_model = config.get("model.name")
        if current_model:
            # 检查是否是本地模型
            if self.generator._is_local_model(current_model):
                # 本地模型，检查是否存在
                if self.generator._check_model_cached(current_model):
                    self.update_status(f"本地模型已准备: {Path(current_model).name}")
                    self.generate_button.setEnabled(True)
                else:
                    self.update_status(f"本地模型不存在: {Path(current_model).name}")
                    self.generate_button.setEnabled(False)
            else:
                # 在线模型，检查是否已缓存
                if self.generator._check_model_cached(current_model):
                    self.update_status(f"模型已缓存: {current_model}")
                    self.generate_button.setEnabled(True)
                else:
                    self.update_status(f"模型未缓存: {current_model} (需要下载)")
                    self.generate_button.setEnabled(True)  # 允许生成，会自动下载
        else:
            self.update_status("未配置模型，请选择模型")
            self.generate_button.setEnabled(False)

    def update_status(self, message):
        """更新状态"""
        self.status_bar.showMessage(message)

    def update_progress(self, progress):
        """更新进度"""
        self.progress_bar.setValue(int(progress))

    def select_model(self):
        """选择模型"""
        if self.is_generating:
            QMessageBox.warning(self, "警告", "正在生成图片，请稍后再试")
            return

        dialog = ModelSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_model = dialog.get_selected_model()
            if selected_model:
                # 更新配置
                config.set("model.name", selected_model)
                config.save_config()

                # 卸载当前模型
                if self.generator.model_loaded:
                    self.generator.unload_model()
                    self.generate_button.setEnabled(False)

                QMessageBox.information(
                    self, "模型已更换",
                    f"模型已更换为: {selected_model}\n\n"
                    "请点击'模型 -> 加载模型'来加载新模型"
                )
            else:
                QMessageBox.warning(self, "警告", "请选择有效的模型")

    def load_model(self):
        """加载模型"""
        if self.is_generating:
            QMessageBox.warning(self, "警告", "正在生成图片，请稍后再试")
            return

        # 创建模型加载线程
        self.model_load_thread = ModelLoadThread(self.generator)
        self.model_load_thread.status_updated.connect(self.update_status)
        self.model_load_thread.progress_updated.connect(self.update_model_load_progress)
        self.model_load_thread.load_completed.connect(self.on_model_load_completed)

        # 禁用生成按钮并显示进度条
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.update_status("准备加载模型...")

        # 启动线程
        self.model_load_thread.start()

    def update_model_load_progress(self, progress):
        """更新模型加载进度"""
        self.progress_bar.setValue(progress)

    def on_model_load_completed(self, success):
        """模型加载完成回调"""
        # 隐藏进度条
        self.progress_bar.setVisible(False)

        if success:
            self.generate_button.setEnabled(True)
            self.update_status("模型加载完成，可以开始生成图片")
            QMessageBox.information(self, "成功", "模型加载成功！现在可以生成图片了。")
        else:
            self.generate_button.setEnabled(False)
            self.update_status("模型加载失败")

            # 获取当前模型信息以提供更好的错误提示
            current_model = config.get("model.name")
            is_local = self.generator._is_local_model(current_model) if current_model else False

            if is_local:
                # 检查是否是单个文件
                from pathlib import Path
                path = Path(current_model)

                if path.is_file() and path.suffix.lower() == '.safetensors':
                    # 单个SafeTensors文件的特殊提示
                    QMessageBox.critical(self, "SafeTensors文件加载失败",
                        f"SafeTensors文件加载失败: {path.name}\n\n"
                        "单个.safetensors文件需要配置文件支持。\n\n"
                        "解决方案:\n"
                        "• 选择包含完整模型的目录（包含config.json等文件）\n"
                        "• 或下载完整的模型包而不是单个权重文件\n"
                        "• 推荐使用Hugging Face上的完整模型\n\n"
                        "提示: .safetensors文件只是模型权重，还需要配置文件才能正常加载。")
                else:
                    QMessageBox.critical(self, "本地模型加载失败",
                        f"本地模型加载失败: {current_model}\n\n"
                        "请检查:\n"
                        "• 模型文件/目录是否存在\n"
                        "• 是否包含必要的配置文件(config.json等)\n"
                        "• 模型格式是否正确\n"
                        "• 模型是否完整")
            else:
                QMessageBox.critical(self, "在线模型加载失败",
                    f"在线模型加载失败: {current_model}\n\n"
                    "请检查:\n"
                    "• 网络连接是否正常\n"
                    "• 模型ID是否正确\n"
                    "• 磁盘空间是否充足")

    def unload_model(self):
        """卸载模型"""
        if self.is_generating:
            QMessageBox.warning(self, "警告", "正在生成图片，请稍后再试")
            return

        self.generator.unload_model()
        self.generate_button.setEnabled(False)
        self.update_status("模型已卸载")

    def generate_image(self):
        """生成图片"""
        if self.is_generating:
            return

        # 验证提示词
        prompt = self.prompt_text.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请输入提示词")
            return

        # 验证参数
        valid, msg = self.generator.validate_parameters(
            self.width_spin.value(),
            self.height_spin.value(),
            self.steps_spin.value()
        )
        if not valid:
            QMessageBox.warning(self, "参数错误", msg)
            return

        # 准备参数
        params = {
            'prompt': prompt,
            'negative_prompt': self.negative_prompt_text.toPlainText().strip(),
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'num_inference_steps': self.steps_spin.value(),
            'guidance_scale': self.guidance_spin.value(),
            'seed': self.seed_spin.value() if self.seed_spin.value() != -1 else None
        }

        # 开始生成
        self.start_generation(params)

    def start_generation(self, params):
        """开始生成"""
        self.is_generating = True
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 创建生成线程
        self.generation_thread = ImageGenerationThread(self.generator, params)
        self.generation_thread.progress_updated.connect(self.update_progress)
        self.generation_thread.status_updated.connect(self.update_status)
        self.generation_thread.generation_completed.connect(self.on_generation_completed)
        self.generation_thread.start()

    def stop_generation(self):
        """停止生成"""
        if self.generation_thread:
            self.generation_thread.cancel()
            self.generation_thread.wait()

        self.is_generating = False
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.update_status("生成已停止")

    def on_generation_completed(self, image):
        """生成完成回调"""
        self.is_generating = False
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

        if image:
            self.display_image(image)
            self.current_image = image
            self.save_button.setEnabled(True)
            self.update_status("图片生成完成")
        else:
            self.update_status("图片生成失败")

    def display_image(self, image):
        """在预览区域显示图片"""
        if not image:
            self.update_status("图片生成失败，无法显示")
            return
        
        try:
            # 检查并转换图像模式为RGBA
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # 将PIL Image转换为QImage
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            
            # 从QImage创建QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            # 缩放图片以适应标签大小
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.update_status("图片显示完成")
            self.current_pixmap = pixmap  # 保存当前pixmap以便保存
        except Exception as e:
            logger.error(f"显示图片时出错: {e}", exc_info=True)
            self.update_status(f"显示图片时出错: {e}")

    def save_image(self):
        """保存当前预览的图片"""
        if not self.current_pixmap:
            QMessageBox.warning(self, "警告", "没有可保存的图片")
            return

        # 生成默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"sd_generated_{timestamp}.png"

        # 选择保存路径
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_filename,
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*)"
        )

        if filepath:
            try:
                # 保存图片和元数据
                prompt = self.prompt_text.toPlainText().strip()
                negative_prompt = self.negative_prompt_text.toPlainText().strip()

                metadata = {
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'width': self.width_spin.value(),
                    'height': self.height_spin.value(),
                    'steps': self.steps_spin.value(),
                    'guidance_scale': self.guidance_spin.value(),
                    'seed': self.seed_spin.value(),
                    'model': config.get("model.name"),
                    'timestamp': datetime.now().isoformat()
                }

                save_image_with_metadata(self.current_image, filepath, metadata)
                QMessageBox.information(self, "保存成功", f"图片已保存到: {filepath}")

            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图片时出错: {str(e)}")

    def show_model_info(self):
        """显示模型信息"""
        info = self.generator.get_model_info()
        current_model = config.get("model.name")

        info_text = f"配置模型: {current_model}\n"
        info_text += f"模型状态: {'已加载' if info['loaded'] else '未加载'}\n"

        if info['loaded']:
            info_text += f"设备: {info['device']}\n"
            info_text += f"已加载模型: {info['model_name']}\n"

            if info['device'] == 'cuda':
                allocated = info.get('gpu_memory_allocated', 0)
                reserved = info.get('gpu_memory_reserved', 0)
                info_text += f"GPU内存使用: {allocated / 1024**3:.1f}GB\n"
                info_text += f"GPU内存预留: {reserved / 1024**3:.1f}GB"
        else:
            info_text += "\n提示: 点击'模型 -> 加载模型'来加载配置的模型"

        QMessageBox.information(self, "模型信息", info_text)

    def show_system_info(self):
        """显示系统信息"""
        info = get_system_info()
        info_text = f"系统: {info['platform']}\n"
        info_text += f"处理器: {info['processor']}\n"
        info_text += f"Python版本: {info['python_version']}\n"
        info_text += f"CPU核心数: {info['cpu_count']}\n"
        info_text += f"内存: {info['memory_total'] / 1024**3:.1f}GB\n"

        if info['cuda_available']:
            info_text += f"CUDA: 可用 (版本 {info['cuda_version']})\n"
            info_text += f"GPU数量: {info['gpu_count']}\n"
            for i, name in enumerate(info['gpu_names']):
                memory_gb = info['gpu_memory'][i] / 1024**3
                info_text += f"GPU {i}: {name} ({memory_gb:.1f}GB)\n"
        else:
            info_text += "CUDA: 不可用"

        QMessageBox.information(self, "系统信息", info_text)

    def show_about(self):
        """显示关于信息"""
        about_text = """Stable Diffusion 图片生成器 v1.2

基于Hugging Face Diffusers库开发的跨平台AI图片生成应用

功能特性:
• 支持多种Stable Diffusion模型
• 手动选择和切换模型
• 自动模型下载功能
• 现代化PyQt5界面
• 丰富的参数调整选项
• 跨平台兼容 (Windows/Linux)
• 自动GPU/CPU检测和优化

开发: AI助手
技术栈: Python + PyQt5 + PyTorch + Diffusers"""

        QMessageBox.about(self, "关于", about_text)

    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_generating:
            reply = QMessageBox.question(
                self, "退出", "正在生成图片，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.generation_thread:
                    self.generation_thread.cancel()
                    self.generation_thread.wait()
                self.generator.unload_model()
                event.accept()
            else:
                event.ignore()
        else:
            # 直接退出，不保存配置
            self.generator.unload_model()
            event.accept()


def main():
    """主函数"""
    if not PYQT_AVAILABLE:
        print(f"错误: PyQt5不可用: {IMPORT_ERROR}")
        print("请安装PyQt5: pip install PyQt5")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Stable Diffusion 图片生成器")
    app.setApplicationVersion("1.2")

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建主窗口
    try:
        window = SDGeneratorMainWindow()
        window.show()

        return app.exec_()

    except Exception as e:
        QMessageBox.critical(None, "启动错误", f"应用程序启动失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

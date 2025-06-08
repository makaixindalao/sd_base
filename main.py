#!/usr/bin/env python3
"""
Stable Diffusion图片生成器 - 主程序入口
基于PyQt5的现代化AI图片生成应用
"""

import sys
import traceback
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函数"""
    try:
        # 检查PyQt5可用性
        try:
            from PyQt5.QtWidgets import QApplication
            from gui_qt import main as qt_main
            print("🎨 启动 Stable Diffusion 图片生成器...")
            return qt_main()
        except ImportError as e:
            print(f"❌ PyQt5不可用: {e}")
            print("\n🔧 解决方案:")
            print("1. 安装PyQt5: pip install PyQt5")
            print("2. 或运行启动脚本: python start.py")
            print("3. 启动脚本会自动安装所需依赖")
            return 1

    except Exception as e:
        error_msg = f"应用程序启动失败: {e}"
        print(f"❌ 错误: {error_msg}")
        print("\n详细错误信息:")
        traceback.print_exc()
        print("\n🔧 故障排除建议:")
        print("1. 确保Python版本为3.8或更高")
        print("2. 运行启动脚本: python start.py")
        print("3. 检查依赖安装: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Stable Diffusion GPU配置指南

## 📋 当前状态

根据检测结果，您的系统当前使用的是**CPU版本的PyTorch**，这意味着Stable Diffusion将在CPU模式下运行，生成速度较慢。

## 🎯 优化目标

将系统配置为使用**GPU模式**，以获得显著的性能提升：
- **CPU模式**: 生成一张图片需要 30-60秒
- **GPU模式**: 生成一张图片只需要 3-10秒（提升5-10倍）

## 🔧 升级步骤

### 第一步：检查GPU硬件

1. **确认您有NVIDIA GPU**
   ```bash
   # 在命令行运行以下命令检查GPU
   nvidia-smi
   ```
   
2. **如果命令不存在或报错**：
   - 您可能没有NVIDIA GPU，或者驱动未安装
   - 请先安装NVIDIA驱动程序

### 第二步：卸载CPU版本PyTorch

```bash
# 卸载现有的CPU版本PyTorch
pip uninstall torch torchvision torchaudio -y
```

### 第三步：安装CUDA版本PyTorch

根据您的CUDA版本选择合适的安装命令：

#### 选项1：CUDA 12.1（推荐）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 选项2：CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 选项3：自动检测（使用启动脚本）
```bash
# 重新运行启动脚本，它会自动检测并安装合适的CUDA版本
python start.py
```

### 第四步：验证安装

运行测试脚本验证GPU配置：
```bash
python test_gpu_config.py
```

成功的输出应该显示：
```
🎮 CUDA可用: True
✅ 检测到最优设备: cuda
✅ 应用程序将使用GPU模式，性能最佳
```

## 🚀 性能优化配置

安装CUDA版本PyTorch后，应用程序会自动应用以下优化：

### 根据GPU内存自动优化

| GPU内存 | 优化策略 | 预期性能 |
|---------|----------|----------|
| **12GB+** | 最高质量模式 | 最佳性能 |
| **8-12GB** | 高质量模式 | 优秀性能 |
| **6-8GB** | 平衡模式 | 良好性能 |
| **4-6GB** | 低显存模式 | 可用性能 |
| **2-4GB** | 极限模式 | 基础性能 |

### 自动应用的优化设置

- **精度优化**: 自动选择BFloat16/Float16精度
- **内存优化**: 启用attention slicing和CPU offload
- **性能优化**: 启用xformers加速（如果可用）

## 🔍 故障排除

### 问题1：nvidia-smi命令不存在
**解决方案**: 安装NVIDIA驱动
1. 访问 [NVIDIA官网](https://www.nvidia.com/drivers/)
2. 下载并安装适合您GPU的驱动程序
3. 重启计算机

### 问题2：CUDA版本不匹配
**解决方案**: 检查CUDA版本
```bash
nvidia-smi  # 查看右上角的CUDA Version
```
然后选择对应的PyTorch版本安装。

### 问题3：安装后仍显示CPU模式
**解决方案**: 
1. 重启Python环境
2. 重新运行测试脚本
3. 检查是否有多个Python环境冲突

### 问题4：GPU内存不足
**解决方案**: 
- 应用程序会自动启用低显存模式
- 可以手动调整图片尺寸（如512x512）
- 减少采样步数

## 📊 性能对比

### 生成时间对比（512x512图片，20步）

| 硬件配置 | CPU模式 | GPU模式 | 提升倍数 |
|----------|---------|---------|----------|
| **RTX 4090** | 45秒 | 3秒 | **15倍** |
| **RTX 3080** | 50秒 | 5秒 | **10倍** |
| **RTX 3060** | 55秒 | 8秒 | **7倍** |
| **GTX 1660** | 60秒 | 12秒 | **5倍** |

## 🎉 完成验证

配置完成后，您应该看到：

1. **启动时的设备检测**：
   ```
   🎮 检测到GPU: NVIDIA GeForce RTX XXXX
   💾 GPU内存: X.XGB
   ✅ GPU内存充足，使用CUDA加速，性能最佳
   ```

2. **生成图片时的GPU信息**：
   ```
   使用设备: cuda
   使用BFloat16精度生成
   初始GPU内存使用: XXX MB
   生成完成 (耗时: X.XX秒, GPU内存: XXX MB)
   ```

## 📞 需要帮助？

如果在配置过程中遇到问题：

1. **运行诊断脚本**: `python test_gpu_config.py`
2. **查看详细日志**: 检查 `logs/sd_generator.log` 文件
3. **重新运行安装**: `python start.py`

---

**注意**: GPU模式需要NVIDIA GPU和相应的CUDA驱动支持。如果您使用的是AMD GPU或Intel集成显卡，请继续使用CPU模式。

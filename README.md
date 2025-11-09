# 相机位姿估计系统 (Camera PnP Pose Estimation)

基于 OpenCV PnP 算法的相机位姿估计系统，用于计算平面方形标记（黑色正方形）在相机坐标系下的 3D 姿态。

## 📋 项目简介

本项目实现了从图像、视频或实时摄像头中检测黑色正方形标记，并通过 PnP（Perspective-n-Point）算法计算其在相机坐标系下的位置和旋转姿态。

### 主要特性

- ✅ **自动检测黑色正方形标记**：基于 Otsu 阈值、形态学操作和轮廓分析
- ✅ **高精度姿态估计**：使用 ITERATIVE 和 LM 精炼的 PnP 求解器
- ✅ **多种输入源支持**：图片、视频、实时摄像头
- ✅ **分辨率自适应**：自动根据图像分辨率缩放相机内参
- ✅ **可视化输出**：叠加 3D 坐标轴、角点标注、姿态信息
- ✅ **遮挡鲁棒性**：支持部分遮挡情况下的检测

## 🔧 系统要求

### 硬件要求
- 摄像头（用于实时检测，可选）
- 黑色正方形标记（推荐尺寸：100-200mm）

### 软件要求
- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd camera-pnp
```

### 2. 安装依赖
```bash
# 使用 conda（推荐）
conda create -n camera-pnp python=3.9
conda activate camera-pnp
conda install opencv numpy

# 或使用 pip
pip install opencv-python opencv-contrib-python numpy
```

### 3. 相机标定（首次使用必须）
在使用前，需要先标定您的相机以获得内参矩阵和畸变系数：

```bash
python scripts/CameraCalibration/camera_calibration.py
```

标定结果将保存为 `calibration_results.npz`。

> **注意**：标定时使用的分辨率应与实际使用时的图像分辨率一致。本项目当前标定分辨率为 **640×480**。

## 🚀 快速开始

### 基本用法

#### 1. 处理单张图片
```bash
python main.py --input ./input/images/img.png
```

#### 2. 处理视频文件
```bash
python main.py --input ./input/videos/video.mp4
```

#### 3. 实时摄像头检测
```bash
python main.py
```

### 高级参数

```bash
python main.py \
  --input ./input/images/img.png \
  --marker-size-mm 127 \
  --save ./output/result.jpg \
  --target black-square \
  --margin-pixels 20
```

## 📝 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--input`, `-i` | str | None | 输入图片/视频路径；不提供则使用摄像头 |
| `--marker-size-mm` | float | 127.0 | 黑色正方形的实际边长（毫米）⚠️ 必须准确 |
| `--save` | str | None | 结果保存路径（图片或视频） |
| `--target` | str | black-square | 目标类型：`auto` 或 `black-square` |
| `--intrinsics` | str | None | 相机内参 npz 文件路径 |
| `--K` | str | None | 手动指定 3×3 相机内参矩阵 |
| `--dist` | str | None | 手动指定畸变系数（4/5/8个） |
| `--no-dist` | flag | False | 忽略畸变系数（零畸变） |
| `--margin-pixels` | int | 20 | 边界容差（像素），触边的检测将被过滤 |

## 📐 相机标定说明

### 标定流程

1. **准备棋盘格标定板**（推荐 9×6 或 7×5 内角点）
2. **拍摄多张标定图片**（15-30张，不同角度和距离）
3. **运行标定脚本**：
   ```bash
   python scripts/CameraCalibration/camera_calibration.py
   ```
4. **检查标定结果**：
   - 重投影误差应 < 0.5 像素
   - 成功标定后生成 `calibration_results.npz`

### 标定文件内容
```python
calibration_results.npz:
  - camera_matrix: 3×3 内参矩阵 K
  - dist_coeffs: 1×5 畸变系数
  - resolution: (width, height) 标定时的分辨率
```

### 当前标定参数（640×480）
```
相机内参矩阵 K:
  fx = 456.39
  fy = 455.91
  cx = 327.80
  cy = 239.71

畸变系数 dist:
  k1 = 0.0776
  k2 = 0.0043
  p1 = -0.0016
  p2 = 0.0023
  k3 = -0.3766
```

## 📊 输出说明

### 姿态输出格式

程序会在控制台输出以下信息：
```
相机坐标下的位姿:
tvec (mm): [-28.19  -4.69  280.15]  # 平移向量 (X, Y, Z)
rvec: [-0.471  0.065  0.020]        # 旋转向量（罗德里格斯）
```

### 可视化元素

叠加图像包含：
- 🟢 **绿色 X 轴**：指向右侧
- 🔴 **红色 Y 轴**：指向下方
- 🔵 **蓝色 Z 轴**：垂直纸面向外
- 🟡 **黄色角点**：TL（左上）、TR（右上）、BR（右下）、BL（左下）
- 🔵 **青色 C 点**：几何中心
- 🟣 **紫色 O 点**：坐标系原点（与 C 重合表示检测正确）
- 📊 **文字信息**：平移向量、深度、旋转角度、重投影误差

### 坐标系定义

- **相机坐标系**：原点在相机光心
  - X 轴：向右
  - Y 轴：向下
  - Z 轴：垂直相机指向前方
  
- **标记坐标系**：原点在正方形中心
  - X 轴：指向右边缘
  - Y 轴：指向下边缘
  - Z 轴：垂直纸面向外

## 🎯 使用示例

### 示例 1：单张图片处理
```bash
python main.py \
  --input ./input/images/img.png \
  --marker-size-mm 127 \
  --save ./output/result.jpg
```

**输出**：
```
[DEBUG] Black-square via Otsu eps=0.02, area=38990.5
[DEBUG] Detected corners: [[197. 149.] [384. 148.] [389. 336.] [159. 333.]]
相机坐标下的位姿:
tvec (mm): [-28.19  -4.69  280.15]
rvec: [-0.471  0.065  0.020]
[DEBUG] Result saved to: ./output/result.jpg
```

### 示例 2：视频处理并保存
```bash
python main.py \
  --input ./input/videos/demo.mp4 \
  --save ./output/demo_output.mp4
```

### 示例 3：实时摄像头
```bash
python main.py --marker-size-mm 150
```

按 `ESC` 或 `Q` 键退出。

## 🔍 故障排查

### 1. 无法检测到标记
**可能原因**：
- ❌ 标记太小或太大
- ❌ 光照不均匀
- ❌ 标记不是纯黑色
- ❌ 标记触碰图像边界

**解决方法**：
- ✅ 使用 100-200mm 的黑色正方形
- ✅ 改善光照条件，避免强烈阴影
- ✅ 确保标记在图像中完整可见（不触边）
- ✅ 调整 `--margin-pixels` 参数

### 2. 姿态估计不准确（RMSE > 20px）
**可能原因**：
- ❌ `--marker-size-mm` 参数不正确
- ❌ 相机标定不准确
- ❌ 图像分辨率与标定分辨率不匹配

**解决方法**：
- ✅ 用尺子精确测量标记边长
- ✅ 重新标定相机
- ✅ 使用与标定相同的分辨率（640×480）

### 3. 提示 "Pose INVALID"
**含义**：重投影误差过大或深度不合理

**解决方法**：
```bash
# 检查控制台输出的建议 marker_size
[HINT] 检测到方块像素尺寸 ~250px；若深度 ~500mm，建议 --marker-size-mm 135

# 使用建议值重新运行
python main.py --input ./input/images/img.png --marker-size-mm 135
```

### 4. 相机标定失败
**可能原因**：
- ❌ 标定图片数量不足（< 10张）
- ❌ 棋盘格角点检测失败
- ❌ 图片模糊或光照不佳

**解决方法**：
- ✅ 拍摄 20-30 张清晰的标定图片
- ✅ 覆盖不同角度、距离、位置
- ✅ 确保棋盘格在每张图中完整可见
- ✅ 避免过曝或欠曝

## 📂 项目结构

```
camera-pnp/
├── main.py                          # 主程序
├── calibration_results.npz          # 相机标定结果
├── README.md                        # 项目文档（本文件）
├── input/                           # 输入文件夹
│   ├── images/                      # 测试图片
│   │   ├── img (1).png
│   │   ├── img (2).png
│   │   ├── img (3).png
│   │   └── img (4).png
│   └── videos/                      # 测试视频
├── output/                          # 输出结果
│   ├── images/                      # 处理后的图片
│   └── videos/                      # 处理后的视频
└── scripts/                         # 工具脚本
    └── CameraCalibration/
        ├── camera_calibration.py    # 相机标定脚本
        └── Color/                   # 标定用棋盘格图片
```

## 🔬 技术细节

### 检测算法流程

1. **预处理**：
   - 灰度转换
   - 高斯模糊去噪（5×5 核）
   - Otsu 自适应阈值二值化

2. **轮廓检测**：
   - 形态学闭运算（7×7 核）连接断裂边缘
   - 提取外部轮廓
   - 按面积排序

3. **筛选与验证**：
   - 过滤触边轮廓（`margin_pixels=20`）
   - 多层次多边形逼近（eps=0.02, 0.04, 0.08）
   - 凸性检查
   - 长宽比验证（ratio > 0.75 为类正方形）

4. **角点排序**：
   - TL（左上）：x+y 最小
   - BR（右下）：x+y 最大
   - TR（右上）：x-y 最小
   - BL（左下）：x-y 最大

### PnP 求解策略

1. **初值求解**：
   - 优先使用 `SOLVEPNP_IPPE_SQUARE`（快速但需验证）
   - 若 RMSE > 50px，退回到 `SOLVEPNP_ITERATIVE`

2. **精炼优化**：
   - 使用 `solvePnPRefineLM` 进行 Levenberg-Marquardt 优化
   - 降低重投影误差

3. **多顺序尝试**：
   - 若 RMSE > 20px，尝试 8 种角点循环顺序
   - 选择 RMSE 最小的解

### 姿态质量指标

- **重投影 RMSE**：
  - < 3px：优秀
  - 3-10px：良好
  - 10-20px：一般
  - \> 20px：较低（需检查）

- **深度合理范围**：50mm - 8000mm

## 📖 参考资料

- [OpenCV solvePnP 文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- [相机标定原理](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [PnP 算法综述](https://en.wikipedia.org/wiki/Perspective-n-Point)

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 邮件联系项目维护者

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

**最后更新**：2025年11月9日  
**版本**：v1.0  
**作者**：M-Sir-zhou

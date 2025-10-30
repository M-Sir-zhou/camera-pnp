# Camera-PnP 项目目录结构

## 📁 项目概述

本项目实现基于 PnP (Perspective-n-Point) 算法的相机姿态估计系统，支持从图片、视频或摄像头实时检测平面标记（如黑色方块）并计算其在相机坐标系下的 6 自由度位姿（位置 + 姿态）。

---

## 📂 目录结构

```
camera-pnp/
│
├── main.py                          # 主程序入口（姿态估计核心代码）
├── answer1.py                       # 辅助脚本或测试代码
│
├── calibration_results.npz          # 相机标定结果（内参矩阵 K 和畸变系数 dist）
│
├── readme.md                        # 项目说明文档
├── PROBLEM_ANALYSIS.md              # 问题诊断与解决方案文档
├── PROJECT_STRUCTURE.md             # 本文件：项目目录结构说明
│
├── input/                           # 输入文件目录
│   ├── images/                      # 测试图片
│   │   └── 1.jpg                    # 示例图片（1279x1706 分辨率）
│   └── videos/                      # 测试视频
│       └── 1.mp4                    # 示例视频（720x1280 分辨率）
│
├── output/                          # 输出结果目录
│   ├── images/                      # 处理后的图片输出
│   │   ├── 1_out.jpg               # 标注后的图片结果
│   │   ├── 1_out_fixed.jpg         # 修复后的输出
│   │   └── result.jpg              # 最终结果
│   └── videos/                      # 处理后的视频输出
│       ├── result.mp4              # 标注后的视频结果
│       ├── result_300mm.mp4        # 使用不同参数的输出
│       └── result_nodist.mp4       # 无畸变模式输出
│
├── scripts/                         # 脚本工具目录
│   ├── camera_find.py              # 相机检测相关工具
│   └── CameraCalibration/          # 相机标定模块
│       ├── camera_calibration.py   # 相机标定主程序
│       └── Color/                  # 标定板颜色检测（可能）
│
├── diagnose_problem.py              # 问题诊断脚本（深度分析 RMSE）
├── compare_rmse.py                  # RMSE 计算方法对比脚本
├── test_solve_pose.py               # PnP 求解方法测试脚本
└── diagnose_video.py                # 视频处理问题诊断脚本
```

---

## 📄 核心文件说明

### 1. **main.py** 🌟
**主程序**，包含完整的姿态估计流水线：

#### 核心功能模块
```python
# 相机标定参数管理
load_intrinsics()              # 加载相机内参
adapt_intrinsics_to_frame()    # 分辨率自适应缩放

# 图像处理与检测
find_black_square_corners()    # 黑色方块检测（Otsu阈值）
find_quad_corners()            # 通用四边形检测
order_corners()                # 角点排序（TL/TR/BR/BL）

# 姿态估计（PnP）
solve_pose()                   # PnP 求解（支持多种算法）
pose_diagnostics()             # 位姿质量评估（RMSE/深度）

# 可视化
draw_axes()                    # 绘制 3D 坐标轴
process_frame()                # 单帧处理+可视化

# 主函数
main()                         # 支持图片/视频/摄像头输入
```

#### 支持的输入模式
- 📷 **图片**：`.jpg`, `.png`, `.bmp`, `.tiff`
- 🎥 **视频**：`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- 📹 **实时摄像头**：不指定 `--input` 时自动打开

#### 关键修复
✅ **已修复 SOLVEPNP_IPPE_SQUARE 失效问题**（详见 `PROBLEM_ANALYSIS.md`）

---

### 2. **calibration_results.npz**
相机标定结果文件（NumPy 压缩格式）

**包含数据**：
- `camera_matrix`：3×3 相机内参矩阵 K
  ```
  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
  ```
- `dist_coeffs`：畸变系数 [k1, k2, p1, p2, k3]

**标定分辨率**：1279×1706（基准分辨率）

---

### 3. **诊断脚本**

#### **diagnose_problem.py**
深度诊断脚本，用于分析：
- 重投影误差（RMSE）计算
- 不同 marker-size 的效果对比
- 相机标定质量检查
- 畸变校正效果分析

**用法**：
```bash
python diagnose_problem.py
```

#### **compare_rmse.py**
对比 `main.py` 和独立测试的 RMSE 计算差异，用于发现算法问题。

#### **test_solve_pose.py**
测试不同 PnP 求解方法的效果：
- `ITERATIVE` ✅ 推荐
- `EPNP`
- `IPPE_SQUARE` ⚠️ 可能失效
- `SQPNP`
- `P3P`

#### **diagnose_video.py**
专门诊断视频处理问题，检查分辨率自适应是否正确。

---

## 🛠️ 使用方法

### 基本命令

#### 1. 处理图片
```bash
# 基础用法
python main.py --input ./input/images/1.jpg --target black-square --marker-size-mm 312

# 保存结果
python main.py --input ./input/images/1.jpg --target black-square --marker-size-mm 312 --save_output ./output/images/result.jpg

# 设置边界容差
python main.py --input ./input/images/1.jpg --target black-square --marker-size-mm 312 --margin-pixels 20
```

#### 2. 处理视频
```bash
# 处理视频并保存
python main.py --input ./input/videos/1.mp4 --target black-square --marker-size-mm 250 --save ./output/videos/result.mp4

# 使用无畸变模式（分辨率不匹配时）
python main.py --input ./input/videos/1.mp4 --target black-square --marker-size-mm 250 --no-dist --save ./output/videos/result_nodist.mp4
```

#### 3. 实时摄像头
```bash
# 打开摄像头实时检测
python main.py --target black-square --marker-size-mm 124

# 按 ESC 或 Q 键退出
```

---

## ⚙️ 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--input`, `-i` | str | None | 图片/视频路径（不指定则打开摄像头） |
| `--marker-size-mm` | float | 124.0 | 方块实际边长（毫米）⚠️ **关键参数** |
| `--save`, `--save_output` | str | None | 保存输出路径 |
| `--target` | str | `black-square` | 目标类型：`auto` 或 `black-square` |
| `--margin-pixels` | int | 20 | 边界容差（像素） |
| `--intrinsics` | str | None | 自定义标定文件路径 |
| `--K` | str | None | 直接指定相机矩阵 |
| `--dist` | str | None | 直接指定畸变系数 |
| `--no-dist` | flag | False | 忽略畸变（零畸变模式） |
| `--adapt-k` | flag | False | 强制分辨率自适应 |

---

## 📊 输出数据说明

### 1. 终端输出
```
[DEBUG] Detected corners: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
相机坐标下的位姿:
tvec (mm): [X, Y, Z]        # 平移向量（位置）
rvec: [rx, ry, rz]          # 旋转向量（姿态）
```

### 2. 可视化输出（图片/视频）
- 🔵 **蓝点 + TL/TR/BR/BL**：四个角点
- 🟡 **黄点 + C**：几何中心
- 🟣 **紫点 + O**：坐标系原点
- 🟢 **绿线 + X**：X 轴（方块平面，向右）
- 🔴 **红线 + Y**：Y 轴（方块平面，向下）
- 🔵 **蓝线 + Z**：Z 轴（垂直方块，向外）
- 📝 **文本信息**：tvec, 深度, 欧拉角, RMSE

---

## 🔧 相机标定

### 使用内置标定工具
```bash
python scripts/CameraCalibration/camera_calibration.py
```

### 标定步骤
1. 准备棋盘格标定板（推荐 9×6 内角点）
2. 在不同角度和距离拍摄 15-20 张标定图
3. 运行标定脚本生成 `calibration_results.npz`
4. 确保标定图分辨率与使用时一致

---

## ⚠️ 常见问题

### 1. **Pose rejected: rmse > 10px**
**原因**：
- marker-size-mm 参数不正确
- 相机标定参数不匹配当前分辨率
- IPPE_SQUARE 算法失效

**解决**：
- 测量方块实际尺寸
- 参考系统建议值调整 `--marker-size-mm`
- 使用 `--no-dist` 测试

### 2. **视频处理 RMSE 过高**
**原因**：
- 视频分辨率与标定分辨率差异过大
- 畸变系数不适用

**解决**：
- 为该分辨率重新标定
- 使用 `--no-dist` 模式
- 或转换视频分辨率至 1279×1706

### 3. **检测不到方块**
**原因**：
- 光照不均匀
- 对比度不足
- 方块触碰边界

**解决**：
- 改善光照条件
- 调整 `--margin-pixels` 参数
- 确保方块完全在画面内

---

## 📚 相关文档

- **[README.md](./readme.md)**：快速入门指南
- **[PROBLEM_ANALYSIS.md](./PROBLEM_ANALYSIS.md)**：问题诊断与解决方案
- **[本文件](./PROJECT_STRUCTURE.md)**：项目结构说明

---

## 🎯 技术栈

- **Python** 3.10+
- **OpenCV** 4.x（cv2）
- **NumPy**
- **视觉算法**：PnP (Perspective-n-Point)
- **图像处理**：Otsu 阈值、Canny 边缘检测、形态学操作

---

## 📈 项目特点

✅ **鲁棒性高**：多种检测算法备份  
✅ **自适应**：支持不同分辨率自动调整  
✅ **实时性好**：支持摄像头实时处理  
✅ **易用性强**：命令行参数丰富  
✅ **可调试**：详细的 DEBUG 输出  
✅ **已修复**：IPPE_SQUARE 算法失效问题

---

*最后更新：2025年10月30日*

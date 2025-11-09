# 快速操作指南

## 🚀 5分钟快速上手

### 第一步：安装依赖
```bash
pip install opencv-python opencv-contrib-python numpy
```

### 第二步：相机标定（首次必须）
```bash
python scripts/CameraCalibration/camera_calibration.py
```
> 确保生成 `calibration_results.npz` 文件

### 第三步：运行检测

#### 方式1：处理图片
```bash
python main.py --input ./input/images/img.png --marker-size-mm 127
```

#### 方式2：实时摄像头
```bash
python main.py --marker-size-mm 127
```
> 按 ESC 或 Q 退出

---

## 📋 常用命令

### 1. 基本使用
```bash
# 检测图片并显示
python main.py --input 图片路径.png

# 检测并保存结果
python main.py --input 图片.png --save 输出.jpg

# 处理视频
python main.py --input 视频.mp4 --save 输出.mp4
```

### 2. 指定标记尺寸（重要！）
```bash
# 如果黑色正方形边长是 150mm
python main.py --input 图片.png --marker-size-mm 150
```

⚠️ **必须准确测量标记的实际边长（毫米）**

### 3. 实时摄像头
```bash
# 默认摄像头
python main.py

# 指定标记尺寸
python main.py --marker-size-mm 127
```

---

## 🎯 参数速查表

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--input` | 输入文件路径 | `--input ./img.png` |
| `--marker-size-mm` | 标记边长(mm) | `--marker-size-mm 127` |
| `--save` | 保存路径 | `--save ./output.jpg` |
| `--target` | 目标类型 | `--target black-square` |
| `--margin-pixels` | 边界容差 | `--margin-pixels 20` |

---

## 🔧 故障速查

### ❌ 无法检测到标记
**解决**：
1. 确保标记是纯黑色正方形
2. 检查光照是否均匀
3. 标记不要太小（推荐 100-200mm）
4. 标记不要触碰图像边界

### ❌ 提示 "Pose INVALID"
**解决**：
```bash
# 查看控制台输出的建议值
[HINT] 建议 --marker-size-mm 135

# 使用建议值重新运行
python main.py --input 图片.png --marker-size-mm 135
```

### ❌ 重投影误差 RMSE > 20px
**解决**：
1. ✅ 用尺子精确测量标记边长
2. ✅ 检查标定文件是否存在
3. ✅ 确认图像分辨率与标定分辨率一致

---

## 📊 输出解读

### 控制台输出
```
相机坐标下的位姿:
tvec (mm): [-28.19  -4.69  280.15]  # 位置 (X, Y, Z)
rvec: [-0.471  0.065  0.020]        # 旋转向量
```

- **tvec[2]（Z）**：标记到相机的距离（深度）
- **tvec[0]（X）**：左右偏移（正值=右侧，负值=左侧）
- **tvec[1]（Y）**：上下偏移（正值=下方，负值=上方）

### 可视化图像
- 🟢 **X轴（绿色）**：指向右侧
- 🔴 **Y轴（红色）**：指向下方
- 🔵 **Z轴（蓝色）**：垂直向外
- 🟡 **角点（黄色）**：TL, TR, BR, BL
- 📊 **文字信息**：位置、深度、角度、误差

---

## 📁 文件位置

```
项目根目录/
├── main.py                    # 主程序（运行这个）
├── calibration_results.npz    # 标定文件（首次使用前生成）
├── input/images/              # 放测试图片
├── output/images/             # 结果保存位置
└── scripts/CameraCalibration/ # 标定脚本
```

---

## ⚡ 快捷技巧

### 1. 批量处理多张图片
```bash
# Windows PowerShell
Get-ChildItem ./input/images/*.png | ForEach-Object {
    python main.py --input $_.FullName --save "./output/images/$($_.BaseName)_result.jpg"
}
```

### 2. 只保存姿态数据（不显示图像）
修改 `main.py`，注释掉这些行：
```python
# cv2.imshow('Pose', vis)
# cv2.waitKey(0)
```

### 3. 调整检测灵敏度
```bash
# 增大边界容差（允许更靠近边缘）
python main.py --input 图片.png --margin-pixels 10

# 减小边界容差（更严格）
python main.py --input 图片.png --margin-pixels 30
```

---

## 📞 需要帮助？

1. 📖 查看完整文档：`README.md`
2. 🐛 检查常见问题：README.md 的"故障排查"部分
3. 💬 提交 Issue 或联系维护者

---

**提示**：第一次使用请务必先运行相机标定！

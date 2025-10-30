"""
诊断视频中的 PnP 问题
"""
import cv2
import numpy as np

# 加载视频
cap = cv2.VideoCapture('./input/videos/1.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取视频")
    exit(1)

h, w = frame.shape[:2]
print(f"视频分辨率: {w}x{h}")

# 加载相机内参
data = np.load('calibration_results.npz')
K_base = data['camera_matrix']
dist = data['dist_coeffs']

print(f"\n基准分辨率: 1279x1706")
print(f"视频分辨率: {w}x{h}")
print(f"缩放比例: sx={w/1279:.3f}, sy={h/1706:.3f}")

# 自适应K
sx = w / 1279.0
sy = h / 1706.0
K = K_base.copy()
K[0, 0] *= sx  # fx
K[1, 1] *= sy  # fy
K[0, 2] *= sx  # cx
K[1, 2] *= sy  # cy

print(f"\n调整后的相机内参:")
print(K)

# 模拟第一帧的角点
corners = np.array([[107., 311.],
                    [545., 322.],
                    [527., 748.],
                    [113., 745.]], dtype=np.double)

print(f"\n检测到的角点:")
print(corners)

# 计算像素尺寸
w_px = np.linalg.norm(corners[1] - corners[0])
h_px = np.linalg.norm(corners[2] - corners[1])
avg_px = (w_px + h_px) / 2.0

print(f"\n像素尺寸:")
print(f"  宽度: {w_px:.1f}px")
print(f"  高度: {h_px:.1f}px")
print(f"  平均: {avg_px:.1f}px")

# 测试不同 marker_size 的结果
def test_pnp(size_mm, K, dist, corners):
    half = size_mm / 2.0
    pts3d = np.array([[-half, -half, 0],
                      [ half, -half, 0],
                      [ half,  half, 0],
                      [-half,  half, 0]], dtype=np.double)
    
    # 使用 ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(pts3d, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # 计算RMSE
    reproj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    errors = np.linalg.norm(corners - reproj, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    depth = tvec[2, 0]
    
    return rmse, depth, tvec, rvec

print(f"\n{'Size(mm)':<12} {'RMSE(px)':<12} {'Depth(mm)':<12} {'Status'}")
print("-" * 60)

for size in [100, 124, 150, 200, 250, 300, 350, 400]:
    rmse, depth, tvec, rvec = test_pnp(size, K, dist, corners)
    status = "✓" if rmse <= 10 and 50 <= depth <= 8000 else "✗"
    print(f"{size:<12.0f} {rmse:<12.2f} {depth:<12.1f} {status}")

print(f"\n=== 问题分析 ===")
print(f"如果所有 marker_size 的 RMSE 都很高且接近，说明：")
print(f"1. 相机标定参数不准确（对于这个分辨率）")
print(f"2. 角点检测有系统性偏差")
print(f"3. 畸变系数不适用于这个分辨率")

# 测试无畸变情况
print(f"\n=== 测试无畸变情况 ===")
print(f"{'Size(mm)':<12} {'RMSE(px)':<12} {'Depth(mm)':<12} {'Status'}")
print("-" * 60)

for size in [200, 250, 300]:
    rmse, depth, tvec, rvec = test_pnp(size, K, np.zeros(5), corners)
    status = "✓" if rmse <= 10 and 50 <= depth <= 8000 else "✗"
    print(f"{size:<12.0f} {rmse:<12.2f} {depth:<12.1f} {status}")

print(f"\n如果无畸变时 RMSE 明显降低，说明畸变系数不适合这个分辨率")

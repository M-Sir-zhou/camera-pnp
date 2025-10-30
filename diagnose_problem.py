"""
深度诊断脚本：分析姿态估计失败的根本原因
"""
import cv2
import numpy as np

# 加载图像
img = cv2.imread('./input/images/1.jpg')
h, w = img.shape[:2]
print(f"图像分辨率: {w}x{h}")

# 加载相机内参
data = np.load('calibration_results.npz')
K = data['camera_matrix']
dist = data['dist_coeffs']

print(f"\n相机内参 K:")
print(K)
print(f"\n畸变系数:")
print(dist)

# 检测到的角点
corners = np.array([[ 202.,  344.],
                    [1016.,  376.],
                    [ 959., 1146.],
                    [ 209., 1122.]], dtype=np.double)

print(f"\n检测到的角点:")
for i, (x, y) in enumerate(corners):
    print(f"  {['TL','TR','BR','BL'][i]}: ({x:.1f}, {y:.1f})")

# 计算像素尺寸
w_px = np.linalg.norm(corners[1] - corners[0])
h_px = np.linalg.norm(corners[2] - corners[1])
avg_px = (w_px + h_px) / 2.0
print(f"\n像素尺寸:")
print(f"  宽度: {w_px:.1f}px")
print(f"  高度: {h_px:.1f}px")
print(f"  平均: {avg_px:.1f}px")

# 测试不同的 marker_size 和深度组合
print(f"\n=== 深度与尺寸关系分析 ===")
print(f"如果方块实际深度为 Z mm，那么真实边长应该是:")

fx_avg = (K[0, 0] + K[1, 1]) / 2.0
print(f"平均焦距 fx: {fx_avg:.1f}px")

for depth_mm in [300, 400, 500, 600, 700, 800]:
    size_mm = (avg_px / fx_avg) * depth_mm
    print(f"  深度 {depth_mm}mm → 边长 {size_mm:.1f}mm")

# 测试不同 marker_size_mm 的 PnP 结果
print(f"\n=== 测试不同 marker_size_mm 的 PnP 结果 ===")

def test_pnp(size_mm):
    half = size_mm / 2.0
    points_3d = np.array([[-half, -half, 0],
                          [ half, -half, 0],
                          [ half,  half, 0],
                          [-half,  half, 0]], dtype=np.double)
    
    ok, rvec, tvec = cv2.solvePnP(points_3d, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # 计算重投影误差
    reproj_pts, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    reproj_pts = reproj_pts.reshape(-1, 2)
    errors = np.linalg.norm(corners - reproj_pts, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # 深度
    depth_z = tvec[2, 0]
    
    return rvec, tvec, rmse, depth_z

print(f"{'Size(mm)':<12} {'Depth(mm)':<12} {'RMSE(px)':<12} {'tvec'}")
print("-" * 80)

for size in [100, 124, 200, 250, 312, 400, 500, 600]:
    rvec, tvec, rmse, depth = test_pnp(size)
    tvec_str = f"[{tvec[0,0]:7.1f}, {tvec[1,0]:7.1f}, {tvec[2,0]:7.1f}]"
    status = "✓" if rmse <= 10 and 50 <= depth <= 8000 else "✗"
    print(f"{size:<12.0f} {depth:<12.1f} {rmse:<12.2f} {tvec_str} {status}")

# 检查相机标定质量
print(f"\n=== 相机标定质量检查 ===")

# 测试畸变校正
corners_undist = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=K)
corners_undist = corners_undist.reshape(-1, 2)

dist_diff = np.linalg.norm(corners - corners_undist, axis=1)
print(f"畸变校正偏移量:")
for i, diff in enumerate(dist_diff):
    print(f"  {['TL','TR','BR','BL'][i]}: {diff:.2f}px")

# 分析角点排列
print(f"\n=== 角点几何分析 ===")
# 计算对角线
diag1 = np.linalg.norm(corners[2] - corners[0])
diag2 = np.linalg.norm(corners[3] - corners[1])
diag_ratio = max(diag1, diag2) / min(diag1, diag2)
print(f"对角线长度: {diag1:.1f}px, {diag2:.1f}px")
print(f"对角线比例: {diag_ratio:.3f} (理想=1.000)")

# 边长比例
side1 = np.linalg.norm(corners[1] - corners[0])
side2 = np.linalg.norm(corners[2] - corners[1])
side3 = np.linalg.norm(corners[3] - corners[2])
side4 = np.linalg.norm(corners[0] - corners[3])
print(f"边长: {side1:.1f}, {side2:.1f}, {side3:.1f}, {side4:.1f} px")
side_ratio = max([side1, side2, side3, side4]) / min([side1, side2, side3, side4])
print(f"边长比例: {side_ratio:.3f} (理想=1.000)")

print(f"\n=== 建议 ===")
print(f"1. 如果方块实际尺寸未知，需要测量实际边长")
print(f"2. 如果 RMSE 无法降低，可能是相机标定不准确")
print(f"3. 如果角点几何异常（对角线/边长比例偏离1），检查检测算法")
print(f"4. 尝试重新标定相机，使用更多标定图像和更好的光照条件")

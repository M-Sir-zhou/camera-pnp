"""
对比 main.py 和诊断脚本的 RMSE 计算差异
"""
import cv2
import numpy as np

# 加载数据
data = np.load('calibration_results.npz')
K = data['camera_matrix']
dist = data['dist_coeffs']

corners = np.array([[ 202.,  344.],
                    [1016.,  376.],
                    [ 959., 1146.],
                    [ 209., 1122.]], dtype=np.double)

marker_size_mm = 312.0
half = marker_size_mm / 2.0

# 3D 点
pts3d = np.array([[-half, -half, 0],
                  [ half, -half, 0],
                  [ half,  half, 0],
                  [-half,  half, 0]], dtype=np.double)

# 求解PnP
ok, rvec, tvec = cv2.solvePnP(pts3d, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

print(f"rvec: {rvec.ravel()}")
print(f"tvec: {tvec.ravel()}")

# 方法1: 标准重投影RMSE（诊断脚本的方法）
reproj_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
reproj_pts = reproj_pts.reshape(-1, 2)

print(f"\n原始角点:")
print(corners)
print(f"\n重投影角点:")
print(reproj_pts)

errors = np.linalg.norm(corners - reproj_pts, axis=1)
print(f"\n每个点的误差:")
for i, e in enumerate(errors):
    print(f"  {['TL','TR','BR','BL'][i]}: {e:.3f}px")

rmse_method1 = np.sqrt(np.mean(errors ** 2))
print(f"\n方法1 RMSE (标准): {rmse_method1:.3f}px")

# 方法2: main.py 中 pose_diagnostics 的方法
err = reproj_pts - corners
rmse_method2 = float(np.sqrt(np.mean(err[:, 0]**2 + err[:, 1]**2)))
print(f"方法2 RMSE (main.py): {rmse_method2:.3f}px")

# 测试是否使用了未畸变校正的角点
corners_undist = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=K)
corners_undist = corners_undist.reshape(-1, 2)

print(f"\n--- 使用未畸变校正角点测试 ---")
ok2, rvec2, tvec2 = cv2.solvePnP(pts3d, corners_undist, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
reproj_pts2, _ = cv2.projectPoints(pts3d, rvec2, tvec2, K, np.zeros(5))
reproj_pts2 = reproj_pts2.reshape(-1, 2)

errors2 = np.linalg.norm(corners_undist - reproj_pts2, axis=1)
rmse_undist = np.sqrt(np.mean(errors2 ** 2))
print(f"RMSE (无畸变): {rmse_undist:.3f}px")

# 测试用毫米单位计算是否会得到606
print(f"\n--- 测试单位问题 ---")
# 如果误算成毫米
tvec_norm = np.linalg.norm(tvec)
print(f"tvec 模长: {tvec_norm:.2f}mm")

# 看看是否某个计算使用了错误的参数
R, _ = cv2.Rodrigues(rvec)
n_cam = R[:, 2].reshape(3)
depth = float(abs(n_cam.dot(tvec.reshape(3))))
print(f"深度: {depth:.2f}mm")

# 检查是否用了像素单位的K计算
print(f"\n--- 检查是否误用了参数 ---")
# 如果把 dist 当成 tvec 或其他错误...
print(f"dist 系数: {dist}")

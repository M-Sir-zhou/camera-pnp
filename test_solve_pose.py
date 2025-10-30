"""
测试 solve_pose 函数的实际行为
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

# 复制 main.py 中的 solve_pose 函数
def solve_pose(corners_2d, marker_size_mm, K, dist):
    half = marker_size_mm / 2.0
    points_3d = np.array([[-half, -half, 0],
                          [ half, -half, 0],
                          [ half,  half, 0],
                          [-half,  half, 0]], dtype=np.double)

    rvec = np.zeros((3, 1), dtype=np.double)
    tvec = np.zeros((3, 1), dtype=np.double)
    ok = False
    try:
        if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
            ok, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        else:
            ok = False
    except Exception:
        ok = False
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    try:
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(points_3d, corners_2d, K, dist, rvec, tvec)
        else:
            ok2, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        pass
    return rvec, tvec

# 测试
print("=== 使用 main.py 的 solve_pose 函数 ===")
rvec, tvec = solve_pose(corners, marker_size_mm, K, dist)
print(f"rvec: {rvec.ravel()}")
print(f"tvec: {tvec.ravel()}")

# 计算 RMSE
pts3d = np.array([[-half, -half, 0],
                  [ half, -half, 0],
                  [ half,  half, 0],
                  [-half,  half, 0]], dtype=np.double)

reproj_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
reproj_pts = reproj_pts.reshape(-1, 2)

print(f"\n重投影角点:")
print(reproj_pts)

errors = np.linalg.norm(corners - reproj_pts, axis=1)
rmse = np.sqrt(np.mean(errors ** 2))
print(f"\nRMSE: {rmse:.3f}px")

# 测试不同求解方法
print(f"\n=== 测试不同的 PnP 求解方法 ===")

methods = [
    ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE),
    ('EPNP', cv2.SOLVEPNP_EPNP),
    ('P3P', cv2.SOLVEPNP_P3P) if len(corners) >= 4 else None,
]

if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
    methods.append(('IPPE_SQUARE', cv2.SOLVEPNP_IPPE_SQUARE))
if hasattr(cv2, 'SOLVEPNP_IPPE'):
    methods.append(('IPPE', cv2.SOLVEPNP_IPPE))
if hasattr(cv2, 'SOLVEPNP_SQPNP'):
    methods.append(('SQPNP', cv2.SOLVEPNP_SQPNP))

for name, flag in methods:
    if flag is None:
        continue
    try:
        ok, rv, tv = cv2.solvePnP(pts3d, corners, K, dist, flags=flag)
        if ok:
            rp, _ = cv2.projectPoints(pts3d, rv, tv, K, dist)
            rp = rp.reshape(-1, 2)
            errs = np.linalg.norm(corners - rp, axis=1)
            rms = np.sqrt(np.mean(errs ** 2))
            print(f"{name:20s}: RMSE={rms:8.3f}px, tvec={tv.ravel()}")
    except Exception as e:
        print(f"{name:20s}: Failed - {e}")

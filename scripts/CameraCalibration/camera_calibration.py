import cv2
import numpy as np
import glob
import os

# 棋盘格参数
CHECKERBOARD = (11, 8)  # 棋盘格内角点数量 (rows, columns)
square_size = 174.0  # 棋盘格方块的实际尺寸（单位：毫米）

# 存储棋盘格角点的世界坐标和图像坐标
obj_points = []  # 3D点
img_points = []  # 2D点

# 生成棋盘格的世界坐标
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 读取棋盘格图像（默认读取与本脚本同级目录下的 Color 文件夹）
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'Color')
# 支持常见的图片扩展名
images = sorted([p for ext in ('*.jpg', '*.jpeg', '*.png')
                 for p in glob.glob(os.path.join(images_dir, ext))])

if not images:
    print(f"未找到棋盘格图像，请检查 '{images_dir}' 路径和图像格式（支持 jpg/jpeg/png）。")
    exit()

# 检测角点
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        obj_points.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners_refined)

        # 可视化角点
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)  # 显示500毫秒

cv2.destroyAllWindows()

# 标定相机
if len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # 输出标定结果
    print("相机内参矩阵 (camera_matrix):")
    print(camera_matrix)
    print("\n畸变系数 (dist_coeffs):")
    print(dist_coeffs)

    # 保存标定结果
    np.savez('calibration_results.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("标定结果已保存到 'calibration_results.npz'")
else:
    print("未检测到足够的棋盘格角点，请检查图像质量或棋盘格参数。")
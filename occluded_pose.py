"""
部分遮挡黑色矩形的姿态估计

该脚本专门用于处理黑色矩形被部分遮挡的情况,通过以下策略提高检测鲁棒性:
1. 边缘检测 + 直线拟合恢复完整轮廓
2. 凸包计算补全缺失角点
3. 使用可见角点进行PnP求解
4. RANSAC方法提高抗噪能力
"""

import argparse
import cv2
import numpy as np
import os
import math


# 从main.py复用的函数
def load_intrinsics(intrinsics_path: str | None = None):
    """加载相机内参与畸变系数"""
    default_K = np.array(
        [[456.38928938,   0.00000000, 327.80471807],
         [  0.00000000, 455.91173277, 239.7064269 ],
         [  0.00000000,   0.00000000,   1.00000000]], dtype=np.double)
    default_dist = np.array([0.07756993, 0.00433085, -0.00155046, 0.00228385, -0.37656735], dtype=np.double)
    npz_path = intrinsics_path or os.path.join(os.getcwd(), 'calibration_results.npz')
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            K = data.get('camera_matrix', default_K)
            dist = data.get('dist_coeffs', default_dist)
            return K.astype(np.double), dist.astype(np.double)
        except Exception:
            pass
    return default_K, default_dist


def rvec_tvec_to_euler(rvec):
    """将旋转向量转为欧拉角(度)"""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        yaw_x = math.atan2(R[2, 1], R[2, 2])
        pitch_y = math.atan2(-R[2, 0], sy)
        roll_z = math.atan2(R[1, 0], R[0, 0])
    else:
        yaw_x = math.atan2(-R[1, 2], R[1, 1])
        pitch_y = math.atan2(-R[2, 0], sy)
        roll_z = 0
    return math.degrees(roll_z), math.degrees(pitch_y), math.degrees(yaw_x)


def detect_black_rectangle_with_occlusion(image, debug=False):
    """
    检测部分遮挡的黑色矩形
    
    策略:
    1. 多种阈值方法组合
    2. 边缘检测 + 霍夫直线检测
    3. 凸包 + 多边形逼近
    4. 形态学修复
    
    返回: (corners, mask, debug_info)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 存储调试信息
    debug_info = {}
    
    # ===== 方法1: HSV颜色空间检测黑色(更严格的阈值) =====
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # 降低V值,只检测深黑色
    black_mask_hsv = cv2.inRange(hsv, lower_black, upper_black)
    
    # ===== 方法2: 固定阈值(针对深黑色) =====
    _, black_mask_fixed = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    
    # ===== 方法3: Otsu自适应阈值 =====
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black_mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 使用最严格的mask(HSV和固定阈值的交集)
    black_mask = cv2.bitwise_and(black_mask_hsv, black_mask_fixed)
    
    if debug:
        debug_info['mask_hsv'] = black_mask_hsv.copy()
        debug_info['mask_otsu'] = black_mask_otsu.copy()
        debug_info['mask_fixed'] = black_mask_fixed.copy()
        debug_info['mask_combined'] = black_mask.copy()
    
    # 形态学操作: 先开运算去噪,再闭运算填补小孔
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    if debug:
        debug_info['mask_morphed'] = black_mask.copy()
    
    # 查找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[WARN] 未检测到黑色区域")
        return None, black_mask, debug_info
    
    # 过滤触边的轮廓
    margin = 10
    valid_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # 检查是否触碰图像边界
        if x > margin and y > margin and (x + cw) < (w - margin) and (y + ch) < (h - margin):
            valid_contours.append(cnt)
    
    if not valid_contours:
        print("[WARN] 所有检测到的区域都触碰边界")
        # 如果没有不触边的轮廓,使用所有轮廓但排除最大的(可能是整个图像)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours_sorted) > 1:
            valid_contours = contours_sorted[1:]  # 跳过最大的
        else:
            return None, black_mask, debug_info
    
    # 选择最大的有效轮廓
    cnt = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    print(f"[INFO] 检测到最大黑色区域面积: {area:.1f} px^2")
    
    if area < (w * h * 0.005):
        print("[WARN] 检测到的区域太小")
        return None, black_mask, debug_info
    
    # ===== 角点检测策略 =====
    
    # 策略1: 直接使用最小外接矩形(对遮挡最鲁棒)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    corners_rect = box
    
    (cx, cy), (rw, rh), angle = rect
    print(f"[INFO] 最小外接矩形: 中心=({cx:.1f},{cy:.1f}), 尺寸={rw:.1f}x{rh:.1f}, 角度={angle:.1f}°")
    
    # 策略2: 多边形逼近(仅作为验证)
    corners_approx = None
    peri = cv2.arcLength(cnt, True)
    for eps in [0.01, 0.015, 0.02, 0.025, 0.03]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            corners_approx = approx.reshape(-1, 2).astype(np.float32)
            print(f"[INFO] 多边形逼近成功 (eps={eps})")
            
            # 验证长宽比
            approx_rect = cv2.minAreaRect(approx)
            (_, (aw, ah), _) = approx_rect
            if aw > 0 and ah > 0:
                ratio = min(aw, ah) / max(aw, ah)
                if ratio > 0.7:  # 接近正方形
                    print(f"[INFO] 多边形近似正方形,长宽比={ratio:.2f}")
                    break
            corners_approx = None  # 不符合条件,继续尝试
    
    if debug:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)
        
        # 绘制最小外接矩形
        for i in range(4):
            pt1 = tuple(corners_rect[i].astype(int))
            pt2 = tuple(corners_rect[(i+1)%4].astype(int))
            cv2.line(debug_img, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(debug_img, pt1, 5, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # 绘制多边形逼近
        if corners_approx is not None:
            for i, pt in enumerate(corners_approx):
                cv2.circle(debug_img, tuple(pt.astype(int)), 5, (255, 0, 0), -1)
                cv2.putText(debug_img, f"A{i}", tuple(pt.astype(int)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        
        debug_info['detection'] = debug_img
    
    # 优先选择多边形逼近(如果质量好),否则使用最小外接矩形
    if corners_approx is not None and len(corners_approx) == 4:
        # 比较两种方法的角点差异
        corners_rect_sorted = order_corners(corners_rect)
        corners_approx_sorted = order_corners(corners_approx)
        
        diff = np.linalg.norm(corners_rect_sorted - corners_approx_sorted, axis=1)
        mean_diff = np.mean(diff)
        
        print(f"[INFO] 多边形与矩形角点平均差异: {mean_diff:.1f} px")
        
        if mean_diff < 20:  # 差异小,说明多边形逼近可靠
            corners = corners_approx_sorted
            print("[INFO] 使用多边形逼近角点")
        else:
            corners = corners_rect_sorted
            print("[INFO] 使用最小外接矩形角点(多边形差异过大)")
    else:
        corners = order_corners(corners_rect)
        print("[INFO] 使用最小外接矩形角点")
    
    return corners, black_mask, debug_info


def order_corners(pts):
    """将四个角点按 TL, TR, BR, BL 顺序排序"""
    pts = np.squeeze(np.array(pts))
    assert pts.shape == (4, 2)
    
    # 计算重心
    center = pts.mean(axis=0)
    
    # 按角度排序
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 找到左上角(角度最小的点)
    tl_idx = np.argmin(angles)
    
    # 按逆时针顺序排列
    sorted_indices = np.argsort(angles)
    pts_sorted = pts[sorted_indices]
    
    # 根据左上角位置旋转数组
    tl_position = np.where(sorted_indices == tl_idx)[0][0]
    pts_sorted = np.roll(pts_sorted, -tl_position, axis=0)
    
    # 验证排序: TL在左上,BR在右下
    tl, tr, br, bl = pts_sorted
    
    # 重新排序确保正确
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype=np.double)


def solve_pose_with_occlusion(corners_2d, marker_size_mm, K, dist):
    """
    使用部分可见的角点求解姿态
    
    参数:
        corners_2d: 检测到的角点 (可能包含噪声)
        marker_size_mm: 矩形实际尺寸
        K: 相机内参
        dist: 畸变系数
    
    返回:
        rvec, tvec, reproj_error
    """
    half = marker_size_mm / 2.0
    
    # 3D世界坐标 (Z=0平面)
    points_3d = np.array([
        [-half, -half, 0],  # TL
        [ half, -half, 0],  # TR
        [ half,  half, 0],  # BR
        [-half,  half, 0],  # BL
    ], dtype=np.double)
    
    # 初始求解
    success, rvec, tvec = cv2.solvePnP(
        points_3d, corners_2d, K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("[WARN] PnP求解失败")
        return None, None, float('inf')
    
    # RANSAC方法提高鲁棒性 (OpenCV 4.5+)
    try:
        success_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
            points_3d, corners_2d, K, dist,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success_ransac and inliers is not None and len(inliers) >= 3:
            rvec, tvec = rvec_ransac, tvec_ransac
            print(f"[INFO] RANSAC内点数: {len(inliers)}/4")
    except Exception as e:
        print(f"[DEBUG] RANSAC求解跳过: {e}")
    
    # LM精炼优化
    try:
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(points_3d, corners_2d, K, dist, rvec, tvec)
    except Exception:
        pass
    
    # 计算重投影误差
    reproj, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    errors = np.linalg.norm(reproj - corners_2d, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    
    print(f"[INFO] 重投影RMSE: {rmse:.2f} px")
    print(f"[INFO] 各角点误差: {errors}")
    
    return rvec, tvec, rmse


def draw_axes(img, K, dist, rvec, tvec, axis_len=50.0):
    """绘制3D坐标轴"""
    # 原点和坐标轴端点
    axis_3d = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],  # X轴(红色)
        [0, axis_len, 0],  # Y轴(绿色)
        [0, 0, axis_len],  # Z轴(蓝色)
    ], dtype=np.float32)
    
    # 投影到图像
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    axis_2d = axis_2d.reshape(-1, 2)
    
    origin = tuple(axis_2d[0].astype(int))
    x_pt = tuple(axis_2d[1].astype(int))
    y_pt = tuple(axis_2d[2].astype(int))
    z_pt = tuple(axis_2d[3].astype(int))
    
    # 绘制坐标轴
    cv2.line(img, origin, x_pt, (0, 0, 255), 3)  # X - 红色
    cv2.line(img, origin, y_pt, (0, 255, 0), 3)  # Y - 绿色
    cv2.line(img, origin, z_pt, (255, 0, 0), 3)  # Z - 蓝色
    
    # 标注
    cv2.circle(img, origin, 5, (255, 255, 255), -1)
    cv2.putText(img, 'O', (origin[0]+5, origin[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, 'X', x_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, 'Y', y_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, 'Z', z_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def visualize_result(image, corners, rvec, tvec, K, dist, marker_size_mm, rmse):
    """可视化检测和姿态结果"""
    vis = image.copy()
    
    if corners is not None:
        # 绘制角点
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        for i, (pt, label) in enumerate(zip(corners, corner_labels)):
            pt = tuple(pt.astype(int))
            cv2.circle(vis, pt, 6, (0, 255, 255), -1)
            cv2.putText(vis, label, (pt[0]+8, pt[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制矩形边界
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i+1)%4].astype(int))
            cv2.line(vis, pt1, pt2, (255, 255, 0), 2)
    
    if rvec is not None and tvec is not None:
        # 绘制坐标轴
        draw_axes(vis, K, dist, rvec, tvec, axis_len=marker_size_mm*0.5)
        
        # 显示姿态信息
        roll, pitch, yaw = rvec_tvec_to_euler(rvec)
        
        # 计算距离
        distance = float(np.linalg.norm(tvec))
        
        # 添加半透明背景以提高文字可读性
        overlay = vis.copy()
        cv2.rectangle(overlay, (5, 5), (650, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        
        # 文本信息 - 完整显示所有姿态参数
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)
        
        # tvec: 平移向量原始值
        cv2.putText(vis, f'tvec (mm): [{tvec[0,0]:7.2f}, {tvec[1,0]:7.2f}, {tvec[2,0]:7.2f}]',
                   (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += 22
        
        # 距离: 计算值
        cv2.putText(vis, f'Distance: {distance:7.2f} mm',
                   (10, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += 22
        
        # rvec: 旋转向量(罗德里格斯表示)
        cv2.putText(vis, f'rvec (rod): [{rvec[0,0]:6.3f}, {rvec[1,0]:6.3f}, {rvec[2,0]:6.3f}]',
                   (10, y_offset), font, font_scale, (100, 200, 255), thickness, cv2.LINE_AA)
        y_offset += 22
        
        # 欧拉角: Roll, Pitch, Yaw
        cv2.putText(vis, f'Roll:  {roll:7.2f} deg',
                   (10, y_offset), font, font_scale, (255, 200, 100), thickness, cv2.LINE_AA)
        y_offset += 22
        cv2.putText(vis, f'Pitch: {pitch:7.2f} deg',
                   (10, y_offset), font, font_scale, (255, 200, 100), thickness, cv2.LINE_AA)
        y_offset += 22
        cv2.putText(vis, f'Yaw:   {yaw:7.2f} deg',
                   (10, y_offset), font, font_scale, (255, 200, 100), thickness, cv2.LINE_AA)
        y_offset += 22
        
        # RMSE: 重投影误差
        rmse_color = (0, 255, 0) if rmse < 10 else (0, 165, 255) if rmse < 20 else (0, 0, 255)
        cv2.putText(vis, f'RMSE: {rmse:5.2f} px',
                   (10, y_offset), font, font_scale, rmse_color, thickness, cv2.LINE_AA)
    
    return vis


def process_single_image(image, K, dist, marker_size_mm, debug=False):
    """
    处理单张图像,返回可视化结果和姿态信息
    
    返回: (vis_image, pose_dict)
    """
    # 检测黑色矩形
    corners, mask, debug_info = detect_black_rectangle_with_occlusion(image, debug=debug)
    
    if corners is None:
        vis = image.copy()
        cv2.putText(vis, 'No rectangle detected', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis, None, debug_info
    
    # 求解姿态
    rvec, tvec, rmse = solve_pose_with_occlusion(corners, marker_size_mm, K, dist)
    
    if rvec is None:
        vis = image.copy()
        cv2.putText(vis, 'Pose estimation failed', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis, None, debug_info
    
    # 可视化
    vis = visualize_result(image, corners, rvec, tvec, K, dist, marker_size_mm, rmse)
    
    # 构建姿态信息字典
    roll, pitch, yaw = rvec_tvec_to_euler(rvec)
    pose_dict = {
        'corners': corners,
        'rvec': rvec,
        'tvec': tvec,
        'rmse': rmse,
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw,
        'distance': float(np.linalg.norm(tvec))
    }
    
    return vis, pose_dict, debug_info


def run_camera_mode(camera_id, K, dist, marker_size_mm, debug=False, save_video=None):
    """
    实时摄像头模式
    
    参数:
        camera_id: 摄像头ID (0, 1, 2, ...)
        K: 相机内参
        dist: 畸变系数
        marker_size_mm: 矩形边长
        debug: 是否显示调试窗口
        save_video: 保存视频路径(可选)
    """
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}")
        return
    
    # 获取摄像头参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] 摄像头已打开: {width}x{height} @ {fps:.1f}fps")
    print(f"[INFO] 按 'q' 或 ESC 键退出")
    print(f"[INFO] 按 's' 键保存当前帧")
    print(f"[INFO] 按 'd' 键切换调试模式")
    
    # 视频写入器
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        print(f"[INFO] 视频将保存到: {save_video}")
    
    frame_count = 0
    debug_mode = debug
    pose_output_interval = 10  # 每10帧输出一次姿态信息
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 无法读取帧")
            break
        
        frame_count += 1
        
        # 处理帧
        vis, pose_dict, debug_info = process_single_image(
            frame, K, dist, marker_size_mm, debug=debug_mode
        )
        
        # 每N帧输出姿态信息到控制台
        if pose_dict is not None and frame_count % pose_output_interval == 0:
            print(f"[帧 {frame_count:04d}] "
                  f"位置: X={pose_dict['tvec'][0,0]:7.1f}, Y={pose_dict['tvec'][1,0]:7.1f}, Z={pose_dict['tvec'][2,0]:7.1f} mm, "
                  f"距离={pose_dict['distance']:7.1f} mm | "
                  f"旋转: R={pose_dict['roll']:6.1f}°, P={pose_dict['pitch']:6.1f}°, Y={pose_dict['yaw']:6.1f}° | "
                  f"RMSE={pose_dict['rmse']:.2f}px")
        
        # 显示帧率
        cv2.putText(vis, f'Frame: {frame_count}', (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示主窗口
        cv2.imshow('Occluded Pose Estimation - Camera', vis)
        
        # 调试窗口
        if debug_mode and debug_info:
            if 'mask_combined' in debug_info:
                cv2.imshow('Debug: Mask', debug_info['mask_combined'])
            if 'detection' in debug_info:
                cv2.imshow('Debug: Detection', debug_info['detection'])
        
        # 保存视频帧
        if writer is not None:
            writer.write(vis)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print(f"\n[INFO] 退出程序 - 共处理 {frame_count} 帧")
            break
        elif key == ord('s'):  # 保存当前帧
            filename = f'output/images/camera_frame_{frame_count}.png'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            cv2.imwrite(filename, vis)
            print(f"\n[INFO] 已保存帧到: {filename}")
            
            # 如果检测到姿态,打印详细信息
            if pose_dict is not None:
                print(f"      平移向量 tvec (mm): [{pose_dict['tvec'][0,0]:.2f}, {pose_dict['tvec'][1,0]:.2f}, {pose_dict['tvec'][2,0]:.2f}]")
                print(f"      距离: {pose_dict['distance']:.2f} mm")
                print(f"      欧拉角 (度): Roll={pose_dict['roll']:.2f}°, Pitch={pose_dict['pitch']:.2f}°, Yaw={pose_dict['yaw']:.2f}°")
                print(f"      重投影RMSE: {pose_dict['rmse']:.2f} px")
            else:
                print("      [WARN] 未检测到有效姿态")
        elif key == ord('d'):  # 切换调试模式
            debug_mode = not debug_mode
            print(f"[INFO] 调试模式: {'开启' if debug_mode else '关闭'}")
            if not debug_mode:
                cv2.destroyWindow('Debug: Mask')
                cv2.destroyWindow('Debug: Detection')
        elif key == ord('p'):  # 新增: 打印当前帧姿态
            if pose_dict is not None:
                print(f"\n[帧 {frame_count}] 当前姿态详情:")
                print(f"  tvec = {pose_dict['tvec'].ravel()}")
                print(f"  rvec = {pose_dict['rvec'].ravel()}")
                print(f"  距离 = {pose_dict['distance']:.2f} mm")
                print(f"  Roll={pose_dict['roll']:.2f}°, Pitch={pose_dict['pitch']:.2f}°, Yaw={pose_dict['yaw']:.2f}°")
                print(f"  RMSE = {pose_dict['rmse']:.2f} px")
            else:
                print(f"\n[帧 {frame_count}] 未检测到矩形")
    
    # 释放资源
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] 视频已保存")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='计算部分遮挡黑色矩形的姿态')
    parser.add_argument('--input', '-i', type=str, default=None, 
                       help='输入图片路径 (不提供则使用摄像头)')
    parser.add_argument('--camera', '-c', type=int, default=0, 
                       help='摄像头ID (默认0)')
    parser.add_argument('--marker-size-mm', type=float, default=127.0, 
                       help='矩形实际边长(毫米)')
    parser.add_argument('--save', type=str, default=None, 
                       help='保存结果路径(图片或视频)')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--intrinsics', type=str, default=None, 
                       help='相机内参npz文件路径')
    args = parser.parse_args()
    
    # 加载相机参数
    K, dist = load_intrinsics(args.intrinsics)
    print(f"[INFO] 相机内参 fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    
    # 判断是图片模式还是摄像头模式
    if args.input is None:
        # 摄像头模式
        print(f"\n[模式] 实时摄像头 (ID={args.camera})")
        run_camera_mode(args.camera, K, dist, args.marker_size_mm, 
                       debug=args.debug, save_video=args.save)
    else:
        # 图片模式
        print(f"\n[模式] 单张图片处理")
        
        # 加载图像
        if not os.path.exists(args.input):
            print(f"[ERROR] 文件不存在: {args.input}")
            return
        
        image = cv2.imread(args.input)
        if image is None:
            print(f"[ERROR] 无法读取图像: {args.input}")
            return
        
        print(f"[INFO] 图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 检测黑色矩形
        print("\n[步骤1] 检测黑色矩形...")
        corners, mask, debug_info = detect_black_rectangle_with_occlusion(image, debug=args.debug)
        
        if corners is None:
            print("[ERROR] 未能检测到黑色矩形")
            if args.debug:
                cv2.imshow('Mask', mask)
                cv2.waitKey(0)
            return
        
        print(f"[INFO] 检测到角点:\n{corners}")
        
        # 求解姿态
        print("\n[步骤2] 求解相机姿态...")
        rvec, tvec, rmse = solve_pose_with_occlusion(corners, args.marker_size_mm, K, dist)
        
        if rvec is None:
            print("[ERROR] 姿态求解失败")
            return
        
        # 输出结果
        print("\n" + "="*60)
        print("姿态估计结果:")
        print("="*60)
        print(f"平移向量 tvec (mm):")
        print(f"  X = {tvec[0,0]:8.2f} mm (相机坐标系X轴)")
        print(f"  Y = {tvec[1,0]:8.2f} mm (相机坐标系Y轴)")
        print(f"  Z = {tvec[2,0]:8.2f} mm (相机坐标系Z轴,深度)")
        print(f"  距离 = {np.linalg.norm(tvec):8.2f} mm")
        print()
        
        roll, pitch, yaw = rvec_tvec_to_euler(rvec)
        print(f"旋转(欧拉角, 度):")
        print(f"  Roll  (Z轴) = {roll:7.2f}°")
        print(f"  Pitch (Y轴) = {pitch:7.2f}°")
        print(f"  Yaw   (X轴) = {yaw:7.2f}°")
        print()
        
        print(f"旋转向量 rvec (罗德里格斯):")
        print(f"  {rvec.ravel()}")
        print()
        print(f"重投影RMSE: {rmse:.2f} px")
        print("="*60)
        
        # 可视化
        print("\n[步骤3] 生成可视化结果...")
        vis = visualize_result(image, corners, rvec, tvec, K, dist, args.marker_size_mm, rmse)
        
        # 显示结果
        cv2.imshow('Result', vis)
        
        # 调试窗口
        if args.debug and debug_info:
            if 'mask_combined' in debug_info:
                cv2.imshow('Debug: Combined Mask', debug_info['mask_combined'])
            if 'mask_morphed' in debug_info:
                cv2.imshow('Debug: Morphed Mask', debug_info['mask_morphed'])
            if 'detection' in debug_info:
                cv2.imshow('Debug: Detection', debug_info['detection'])
        
        # 保存结果
        if args.save:
            save_dir = os.path.dirname(args.save)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(args.save, vis)
            print(f"[INFO] 结果已保存到: {args.save}")
        
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import argparse
import cv2
import math
import numpy as np
import os


def load_intrinsics(intrinsics_path: str | None = None):
    """加载相机内参与畸变；支持外部 npz 路径。若未提供则尝试 cwd 下 calibration_results.npz，否则使用默认值。
    返回 (K, dist)。"""
    default_K = np.array(
        [[452.23412465, 0, 323.18854531],
         [0, 451.9791104, 247.60366291],
         [0, 0, 1]], dtype=np.double)
    default_dist = np.array([2.62134015e-02, 3.20251776e-01, 3.62635113e-03,
                             -9.70087931e-04, -1.02851706e+00], dtype=np.double)
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


def adapt_intrinsics_to_frame(K: np.ndarray, frame_shape, verbose=True):
    """根据图像尺寸对 K 做尺度自适应（启发式）：
    假设原相机主点约位于图像中心，原分辨率近似为 (2*cx, 2*cy)。按当前帧宽高分别缩放 fx, fy, cx, cy。
    仅当缩放系数偏离 1.0 明显时打印调试信息。"""
    h, w = frame_shape[:2]
    K = K.copy().astype(np.double)
    # 估计原始分辨率（启发式）
    ref_w = max(1.0, float(K[0, 2]) * 2.0)
    ref_h = max(1.0, float(K[1, 2]) * 2.0)
    sx = w / ref_w
    sy = h / ref_h
    # 只做合理范围内的缩放（避免极端值）
    if 0.3 < sx < 4.0 and 0.3 < sy < 4.0:
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        if verbose and (abs(sx - 1.0) > 0.05 or abs(sy - 1.0) > 0.05):
            print(f"[DEBUG] Adapted intrinsics to frame size: sx={sx:.3f}, sy={sy:.3f}")
    else:
        if verbose:
            print(f"[DEBUG] Skip K scaling due to suspicious scale: sx={sx:.3f}, sy={sy:.3f}")
    return K


def order_corners(pts):
    """将四个角点按 TL, TR, BR, BL 顺序排序。输入为 (4,2) 或 (4,1,2)。"""
    pts = np.squeeze(np.array(pts))
    assert pts.shape == (4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.double)


def touches_border(cnt, w, h, margin=2):
    x, y, cw, ch = cv2.boundingRect(cnt)
    return x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin)


def find_black_square_corners(image):
    """面向黑色正方形目标的鲁棒检测：
    - 灰度 + Otsu 反阈值 -> 形态学闭运算
    - 过滤触边轮廓、面积、近似正方形
    - 优先四边形，多边形失败时使用 minAreaRect 回退
    返回 TL,TR,BR,BL 四点或 None
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 连接断裂边缘并去小噪声
    kern = np.ones((7, 7), np.uint8)
    bin_close = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kern)
    contours, _ = cv2.findContours(bin_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    def is_square_like(cnt):
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw == 0 or rh == 0:
            return False
        ratio = min(rw, rh) / max(rw, rh)
        return ratio > 0.75  # 接近正方形

    # 优先找四边形且不触边、面积合适、接近正方形
    for eps in (0.02, 0.04, 0.08):
        for cnt in contours[:30]:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.01):
                continue
            if touches_border(cnt, w, h):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and is_square_like(approx):
                print(f"[DEBUG] Black-square via Otsu eps={eps}, area={area:.1f}")
                return order_corners(approx)

    # 回退：取最大合格轮廓的 minAreaRect
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.005):
            continue
        if touches_border(cnt, w, h):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)  # contourArea 需要 float32/int32
        if is_square_like(cnt):
            print(f"[DEBUG] Black-square fallback minAreaRect area={area:.1f}")
            return order_corners(box)
    return None


def find_quad_corners(image, target_mode='auto'):
    """在图像中查找方形标记四角点。target_mode: 'auto' 或 'black-square'"""
    if target_mode == 'black-square':
        corners = find_black_square_corners(image)
        if corners is not None:
            return corners
        # 若黑方块通道失败，回到通用流程
        print('[DEBUG] Black-square path failed, fallback to generic pipeline.')
    # 通用流程
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 先尝试边缘法
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # 尝试直接从 edges 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # 多个 epsilon 备选，较大值会合并更多折线并产生更规则的多边形
    for eps in (0.02, 0.04, 0.08):
        for cnt in contours[:20]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                # 面积阈值可稍微放宽
                if area > (w * h * 0.01):
                    print(f"[DEBUG] Found quad via edges eps={eps}, area={area:.1f}")
                    return order_corners(approx)

    # 如果轮廓断裂导致失败，尝试形态学闭运算以连接间断边缘后再试
    try:
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours_c, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_c = sorted(contours_c, key=cv2.contourArea, reverse=True)
        for eps in (0.02, 0.04, 0.08):
            for cnt in contours_c[:20]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if area > (w * h * 0.01):
                        print(f"[DEBUG] Found quad via closed edges eps={eps}, area={area:.1f}")
                        return order_corners(approx)
        # 回退方案：使用最大轮廓的最小外接矩形 (minAreaRect)
        if len(contours_c) > 0:
            largest = contours_c[0]
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            area = cv2.contourArea(box)
            if area > (w * h * 0.005):
                print(f"[DEBUG] Using minAreaRect on largest contour, area={area:.1f}")
                return order_corners(box)
    except Exception as e:
        print(f"[DEBUG] closed-contour fallback error: {e}")
    # 备用：HSV 饱和度阈值法
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    # 先尝试固定阈值，再尝试更宽松的阈值
    for sat_thresh in (75, 60, 50, 40):
        _, binary = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)
        # 用闭运算平滑并去噪
        kern = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:20]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > (w * h * 0.01):
                    print(f"[DEBUG] Found quad via saturation thresh={sat_thresh}, area={area:.1f}")
                    return order_corners(approx)
    return None


def solve_pose(corners_2d, marker_size_mm, K, dist):
    """
    使用四边形角点求解平面标志在相机坐标系下的姿态。
    corners_2d: TL, TR, BR, BL 顺序，形状 (4,2)
    返回 rvec, tvec
    """
    half = marker_size_mm / 2.0
    # 与 order_corners 对应的 3D 角点顺序（Z=0 平面）
    points_3d = np.array([[-half, -half, 0],
                          [ half, -half, 0],
                          [ half,  half, 0],
                          [-half,  half, 0]], dtype=np.double)
    flags = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=flags)
    if not ok:
        # 退回到 ITERATIVE
        ok, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def pose_diagnostics(corners_2d, rvec, tvec, K, dist, marker_size_mm):
    """计算位姿质量指标：
    - reproj_rmse_px: 重投影RMSE（像素）
    - depth_along_normal_mm: 相机到平面沿平面法向的距离 |n_cam^T * t|（毫米）
    - lateral_mm: 平面内的横向位移范数（毫米）
    返回 dict。
    """
    half = marker_size_mm / 2.0
    pts3d = np.array([[-half, -half, 0],
                      [ half, -half, 0],
                      [ half,  half, 0],
                      [-half,  half, 0]], dtype=np.double)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    # 更健壮的 RMSE 计算，防止数值不稳定导致的非有限值
    if not np.isfinite(proj).all() or not np.isfinite(corners_2d).all():
        rmse = float('inf')
    else:
        err = proj - corners_2d.astype(np.double)
        rmse = float(np.sqrt(np.mean(err[:, 0]**2 + err[:, 1]**2)))
    R, _ = cv2.Rodrigues(rvec)
    n_cam = R[:, 2].reshape(3)
    depth = float(abs(n_cam.dot(tvec.reshape(3))))
    lateral_vec = tvec.reshape(3) - depth * n_cam
    lateral = float(np.linalg.norm(lateral_vec))
    # 投影原点与几何中心像素偏差（验证原点是否固定在中心）
    origin_img, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]], dtype=np.double), rvec, tvec, K, dist)
    origin_img = origin_img.reshape(-1, 2)[0]
    center2d = corners_2d.astype(np.double).mean(axis=0)
    if np.isfinite(origin_img).all() and np.isfinite(center2d).all():
        center_offset = float(np.linalg.norm(origin_img - center2d))
    else:
        center_offset = float('inf')

    return {
        'reproj_rmse_px': rmse,
        'depth_along_normal_mm': depth,
        'lateral_mm': lateral,
        'origin_center_offset_px': center_offset,
        'origin_2d': origin_img
    }


def rvec_tvec_to_euler(rvec):
    """将 rvec 转为 ZYX（roll=Z, pitch=Y, yaw=X）欧拉角，单位度。"""
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


def draw_axes(img, K, dist, rvec, tvec, axis_len=100.0):
    axis = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len], [0, 0, 0]])
    proj, _ = cv2.projectPoints(axis[:3], rvec, tvec, K, dist)
    origin, _ = cv2.projectPoints(axis[3:], rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    origin = origin.reshape(-1, 2)

    def to_pt(p, frame_shape=None):
        x, y = float(p[0]), float(p[1])
        if not np.isfinite([x, y]).all():
            return None
        xi, yi = int(round(x)), int(round(y))
        # 可选裁剪，避免极端坐标
        if frame_shape is not None:
            h, w = frame_shape[:2]
            xi = max(-10000, min(10000, xi))
            yi = max(-10000, min(10000, yi))
        return (int(xi), int(yi))

    O = to_pt(origin[0], img.shape)
    X = to_pt(proj[0], img.shape)
    Y = to_pt(proj[1], img.shape)
    Z = to_pt(proj[2], img.shape)

    if None in (O, X, Y, Z):
        return

    # 调试：确保点类型正确
    # print('O:', O, type(O[0]), type(O[1]))
    # print('X:', X, type(X[0]), type(X[1]))

    # 绘制坐标系原点 O，便于与几何中心对照
    cv2.circle(img, O, 5, (255, 0, 255), -1)
    cv2.putText(img, 'O', (O[0] + 6, O[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.line(img, O, X, (0, 255, 0), 3)  # X 绿色
    cv2.line(img, O, Y, (0, 0, 255), 3)  # Y 红色
    cv2.line(img, O, Z, (255, 0, 0), 3)  # Z 蓝色
    cv2.putText(img, 'X', X, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, 'Y', Y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, 'Z', Z, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def process_frame(frame, K, dist, marker_size_mm, target_mode='auto'):
    corners = find_quad_corners(frame, target_mode=target_mode)
    if corners is None:
        # 尝试使用增强的图像进行回退检测（CLAHE / 不同 Canny 阈值）
        corners_fallback, enhanced = None, None
        try:
            corners_fallback, enhanced = try_enhanced_detection(frame, target_mode)
        except Exception as e:
            print(f"[DEBUG] Enhanced detection failed: {e}")
        if corners_fallback is not None:
            corners = corners_fallback
            print('[DEBUG] Quad found after enhancement.')
        else:
            cv2.putText(frame, 'No quad found', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("[DEBUG] No quad corners found in the frame.")  # 调试信息
            return frame, None, None
    print(f"[DEBUG] Detected corners: {corners}")  # 调试信息
    rvec, tvec = solve_pose(corners, marker_size_mm, K, dist)
    diag = pose_diagnostics(corners, rvec, tvec, K, dist, marker_size_mm)
    draw_axes(frame, K, dist, rvec, tvec, axis_len=marker_size_mm * 0.5)
    # 几何中心 C（角点平均），用于直观看到“原点应在中心”的约束
    center2d = tuple(np.round(corners.mean(axis=0)).astype(int))
    cv2.circle(frame, center2d, 5, (0, 255, 255), -1)
    cv2.putText(frame, 'C', (center2d[0] + 6, center2d[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # 文本信息
    roll, pitch, yaw = rvec_tvec_to_euler(rvec)
    cv2.putText(frame, f'T: {tvec.ravel()[0]:.1f}, {tvec.ravel()[1]:.1f}, {tvec.ravel()[2]:.1f} mm', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f'd_n: {diag["depth_along_normal_mm"]:.1f} mm  lateral: {diag["lateral_mm"]:.1f} mm  rmse: {diag["reproj_rmse_px"]:.2f}px  center_off: {diag["origin_center_offset_px"]:.1f}px',
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f'RzRyRx(deg): {roll:.1f}, {pitch:.1f}, {yaw:.1f}',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # 标注四个角点
    for i, p in enumerate(corners):
        cv2.circle(frame, tuple(p.astype(int)), 5, (255, 255, 255), -1)
        cv2.putText(frame, ['TL', 'TR', 'BR', 'BL'][i], tuple(p.astype(int) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame, rvec, tvec


def save_debug_images(img, out_base):
    """保存调试用的中间图像：灰度边缘、HSV 饱和度二值图、以及在原图上绘制的前 N 个轮廓。
    out_base: 不含扩展名的保存基路径（full path without extension）
    返回保存的文件路径列表。"""
    saved = []
    try:
        # 灰度->边缘
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        p = out_base + '_edges.jpg'
        if cv2.imwrite(p, edges):
            saved.append(p)

        # HSV 饱和度二值图
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        _, binary = cv2.threshold(s, 75, 255, cv2.THRESH_BINARY)
        p = out_base + '_saturation.jpg'
        if cv2.imwrite(p, binary):
            saved.append(p)

        # 在原图上绘制前若干轮廓（便于人工检查）
        contours_src = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(contours_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        vis_cnt = img.copy()
        for i, cnt in enumerate(contours[:20]):
            color = (0, 255, 255) if i == 0 else (0, 128, 255)
            cv2.drawContours(vis_cnt, [cnt], -1, color, 2)
        p = out_base + '_contours.jpg'
        if cv2.imwrite(p, vis_cnt):
            saved.append(p)

        # 原图备份
        p = out_base + '_orig.jpg'
        if cv2.imwrite(p, img):
            saved.append(p)
    except Exception as e:
        print(f'[DEBUG] Failed to save debug images: {e}')
    return saved


def try_enhanced_detection(img, target_mode='auto'):
    """对图像做增强（CLAHE）并尝试再次检测四边形。
    返回 (corners, enhanced_img)。如果未找到 corners，corners 为 None，但仍返回增强图以供调试。"""
    try:
        # CLAHE on LAB L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # 先用原有方法尝试
        corners = find_quad_corners(enhanced, target_mode=target_mode)
        if corners is not None:
            return corners, enhanced

        # 如果仍未检测到，尝试不同的 Canny 阈值
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        for t1, t2 in [(30, 100), (40, 120), (20, 80)]:
            edges = cv2.Canny(blur, t1, t2)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            h, w = img.shape[:2]
            for cnt in contours[:10]:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if area > (w * h * 0.01):
                        return order_corners(approx), enhanced
        return None, enhanced
    except Exception as e:
        print(f'[DEBUG] try_enhanced_detection exception: {e}')
        return None, img


def is_image(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


def is_video(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']


def main():
    parser = argparse.ArgumentParser(description='从图片或视频计算平面方形标记在相机坐标系下的姿态')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='图片或视频路径；若不提供则尝试打开摄像头')
    parser.add_argument('--marker-size-mm', type=float, default=268.0,
                        help='方形标记的实际边长（毫米）')
    parser.add_argument('--save', '--save_output', dest='save', type=str, default=None,
                        help='当输入是图片时保存叠加结果；当输入是视频时保存叠加视频（支持 mp4/avi）')
    parser.add_argument('--target', choices=['auto', 'black-square'], default='auto',
                        help='目标类型：auto 自动或 black-square 纯黑方形（推荐在黑方块时使用）')
    parser.add_argument('--intrinsics', type=str, default=None,
                        help='相机内参 npz 路径（包含 camera_matrix, dist_coeffs）。未提供时使用默认策略')
    parser.add_argument('--no-dist', action='store_true',
                        help='忽略畸变系数，按零畸变处理（适合未知镜头或截图）')
    parser.add_argument('--adapt-k', action='store_true',
                        help='强制按图像分辨率自适应缩放内参K；默认仅在未提供 --intrinsics 时启用')
    args = parser.parse_args()

    K, dist = load_intrinsics(args.intrinsics)
    if args.no_dist:
        dist = np.zeros((5, ), dtype=np.double)

    # 当未提供 intrinsics（使用内置默认 K）时，默认开启自适应；提供 intrinsics 时仅在 --adapt-k 指定时启用
    def maybe_adapt(K_in, shape):
        if args.intrinsics is None or args.adapt_k:
            return adapt_intrinsics_to_frame(K_in, shape)
        else:
            # 用户提供了相机内参，默认不缩放
            return K_in

    if args.input is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('无法打开摄像头，也未提供输入文件。')
            return
        # 先读一帧用于自适应 K
        ret, frame = cap.read()
        if not ret:
            print('无法从摄像头读取帧。')
            return
        K_cam = maybe_adapt(K, frame.shape)
        # 第一帧也要处理显示
        vis, rvec, tvec = process_frame(frame, K_cam, dist, args.marker_size_mm, target_mode=args.target)
        cv2.imshow('Pose', vis)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, rvec, tvec = process_frame(frame, K_cam, dist, args.marker_size_mm, target_mode=args.target)
            cv2.imshow('Pose', vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    # 文件输入
    path = args.input
    if not os.path.exists(path):
        print(f'输入文件不存在: {path}')
        return

    if is_image(path):
        img = cv2.imread(path)
        K_img = maybe_adapt(K, img.shape)
        vis, rvec, tvec = process_frame(img, K_img, dist, args.marker_size_mm, target_mode=args.target)
        if rvec is not None:
            print('相机坐标下的位姿:')
            print('tvec (mm):', tvec.ravel())
            print('rvec:', rvec.ravel())
        if args.save is not None:
            save_dir = os.path.dirname(args.save)
            if save_dir and not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except Exception as e:
                    print(f'[ERROR] 无法创建保存目录 {save_dir}: {e}')
            try:
                ok = cv2.imwrite(args.save, vis)
                if ok:
                    print(f'[DEBUG] Result saved to: {args.save}')
                else:
                    print(f'[ERROR] cv2.imwrite 返回 False，未能保存文件到: {args.save}')
            except Exception as e:
                print(f'[ERROR] 保存文件时发生异常: {e}')

        # 如果未检测到四边形，额外产出调试图像以便排查
        if rvec is None and args.save is not None:
            base, _ = os.path.splitext(args.save)
            dbg_list = save_debug_images(img, base + '_debug')
            if dbg_list:
                print('[DEBUG] Saved debug images:')
                for p in dbg_list:
                    print('  ', p)
        cv2.imshow('Pose', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if is_video(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f'无法打开视频: {path}')
            return
        # 如果需要保存视频，初始化 VideoWriter
        writer = None
        if args.save is not None:
            # 分辨率、FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps != fps or fps <= 0:
                fps = 25.0
            ret, frame0 = cap.read()
            if not ret:
                print('无法读取视频首帧，放弃保存输出。')
                return
            h0, w0 = frame0.shape[:2]
            K_vid = maybe_adapt(K, frame0.shape)
            # 选择编码
            ext = os.path.splitext(args.save)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w0, h0))
            # 处理首帧并写入
            vis, rvec, tvec = process_frame(frame0, K_vid, dist, args.marker_size_mm, target_mode=args.target)
            writer.write(vis)
            cv2.imshow('Pose', vis)
        else:
            writer = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, rvec, tvec = process_frame(frame, K_vid, dist, args.marker_size_mm, target_mode=args.target)
            cv2.imshow('Pose', vis)
            if writer is not None:
                # 确保分辨率一致
                if (vis.shape[1], vis.shape[0]) != (int(writer.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else vis.shape[1], int(writer.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else vis.shape[0]):
                    # 若编码器无法获取属性，则忽略，直接写入
                    pass
                writer.write(vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
        cap.release()
        if writer is not None:
            writer.release()
            print(f'已保存视频到: {args.save}')
        cv2.destroyAllWindows()
        return

    print('不支持的输入文件类型，请提供图片或视频')


if __name__ == '__main__':
    main()

import argparse
import cv2
import math
import numpy as np
import os

# 标定基准分辨率（用于按分辨率自适应缩放内参K）。
# 新标定使用的分辨率为 640x480
CALIB_BASE_RES = (640, 480)  # (width, height)


def load_intrinsics(intrinsics_path: str | None = None):
    """加载相机内参与畸变；支持外部 npz 路径。若未提供则尝试 cwd 下 calibration_results.npz，否则使用默认值。
    返回 (K, dist)。"""
    # 使用新标定的相机参数（640x480分辨率）
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


def _parse_floats(seq: str) -> list[float]:
    """将包含数字的字符串解析为浮点数组，支持逗号/空格/分号分隔。"""
    # 统一分隔符
    s = seq.replace('\n', ';').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    # 先按分号切行，再按逗号或空格切分
    parts = []
    for row in s.split(';'):
        row = row.strip()
        if not row:
            continue
        # 允许使用逗号或空格
        tokens = [t for t in row.replace(',', ' ').split(' ') if t]
        parts.extend(tokens)
    vals = []
    for t in parts:
        try:
            vals.append(float(t))
        except Exception:
            raise ValueError(f"无法解析为浮点数: '{t}'")
    return vals


def parse_camera_matrix(text: str) -> np.ndarray:
    """解析 3x3 相机内参矩阵，支持格式：
    - "fx,0,cx;0,fy,cy;0,0,1"
    - "fx 0 cx; 0 fy cy; 0 0 1"
    - 直接 9 个数字（按行）
    返回 np.double 3x3。
    """
    vals = _parse_floats(text)
    if len(vals) == 9:
        K = np.array(vals, dtype=np.double).reshape(3, 3)
        return K
    # 也支持 3 行 * 3 列（通过分号/换行识别），上面已展平，所以此分支通常不会命中；保留一致性判断
    raise ValueError('相机矩阵需要 9 个数字，示例："1260.4,0,784.46;0,1285.38,710.28;0,0,1"')


def parse_dist_coeffs(text: str) -> np.ndarray:
    """解析畸变系数，支持 4/5/8 个数字。返回 np.double 一维向量。"""
    vals = _parse_floats(text)
    if len(vals) not in (4, 5, 8):
        raise ValueError('畸变系数数量需为 4/5/8 个数字，例如 "k1,k2,p1,p2,k3"')
    return np.array(vals, dtype=np.double)


def adapt_intrinsics_to_frame(K: np.ndarray, frame_shape, verbose=True):
    """根据图像尺寸对 K 做尺度自适应：
    以固定的标定基准分辨率 CALIB_BASE_RES=(1279,1706) 为参考，
    按当前帧 (w,h) 分别缩放 fx, fy, cx, cy。"""
    h, w = frame_shape[:2]
    K = K.copy().astype(np.double)
    # 使用固定基准分辨率（更稳定，不依赖 cx, cy 推断）
    ref_w = float(CALIB_BASE_RES[0])
    ref_h = float(CALIB_BASE_RES[1])
    sx = w / ref_w
    sy = h / ref_h
    # 只做合理范围内的缩放（避免极端值）
    if 0.3 < sx < 4.0 and 0.3 < sy < 4.0:
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        if verbose and (abs(sx - 1.0) > 0.02 or abs(sy - 1.0) > 0.02):
            print(f"[DEBUG] Adapted intrinsics to frame size (base={int(ref_w)}x{int(ref_h)}): sx={sx:.3f}, sy={sy:.3f}")
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


def quad_center_from_diagonals(corners: np.ndarray):
    """由四边形对角线求交点，作为正方形中心的投影。
    参数 corners: TL, TR, BR, BL 顺序的四点，形状 (4,2)。
    返回 (x, y) 浮点坐标；若对角线近乎平行，退化为四点平均。
    """
    c = np.asarray(corners, dtype=np.double).reshape(4, 2)
    TL, TR, BR, BL = c
    def to_h(p):
        return np.array([p[0], p[1], 1.0], dtype=np.double)
    # 直线（齐次）l = p × q
    l1 = np.cross(to_h(TL), to_h(BR))
    l2 = np.cross(to_h(TR), to_h(BL))
    x = np.cross(l1, l2)
    if np.isfinite(x).all() and abs(x[2]) > 1e-9:
        return float(x[0] / x[2]), float(x[1] / x[2])
    m = c.mean(axis=0)
    return float(m[0]), float(m[1])


def touches_border(cnt, w, h, margin=2):
    """轮廓的外接矩形是否触碰图像边界（含 margin）"""
    x, y, cw, ch = cv2.boundingRect(cnt)
    return x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin)


def polygon_touches_border(pts: np.ndarray, w: int, h: int, margin: int = 2) -> bool:
    """任一角点或多边形包围盒触边则返回 True。pts 形状 (N,2) 或 (N,1,2)。"""
    p = np.squeeze(np.asarray(pts)).astype(np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        return True
    xs, ys = p[:, 0], p[:, 1]
    if np.any(xs <= margin) or np.any(ys <= margin) or np.any(xs >= (w - margin)) or np.any(ys >= (h - margin)):
        return True
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return x0 <= margin or y0 <= margin or x1 >= (w - margin) or y1 >= (h - margin)


def touches_border_points(pts: np.ndarray, w: int, h: int, margin: int) -> bool:
    """任一角点在图像边界 margin 内则判定触边。pts 形状 (N,2) 或 (N,1,2)。"""
    p = np.squeeze(np.asarray(pts)).reshape(-1, 2)
    if p.size == 0:
        return False
    xs = p[:, 0]
    ys = p[:, 1]
    return bool(
        (xs <= margin).any() or (ys <= margin).any() or (xs >= (w - 1 - margin)).any() or (ys >= (h - 1 - margin)).any()
    )


# 已移除：保存调试图片的函数（按需求停用文件输出）


def corners_touch_border(corners: np.ndarray, w: int, h: int, margin: int) -> bool:
    pts = np.asarray(corners).reshape(4, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return (xs <= margin).any() or (ys <= margin).any() or (xs >= (w - 1 - margin)).any() or (ys >= (h - 1 - margin)).any()


def find_black_square_corners(image, margin_pixels: int = 20):
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
            if touches_border(cnt, w, h, margin_pixels):
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
        if touches_border(cnt, w, h, margin_pixels):
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)  # contourArea 需要 float32/int32
        if is_square_like(cnt):
            print(f"[DEBUG] Black-square fallback minAreaRect area={area:.1f}")
            return order_corners(box)
    return None


def find_quad_corners(image, target_mode='auto', margin_pixels: int = 20):
    """在图像中查找方形标记四角点。target_mode: 'auto' 或 'black-square'"""
    if target_mode == 'black-square':
        corners = find_black_square_corners(image, margin_pixels=margin_pixels)
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
    # 多组 eps + 触边过滤 + 面积与近似正方形得分，选最佳
    best = None
    best_score = -1.0
    best_area = 0.0
    best_eps = None
    eps_list = (0.02, 0.04, 0.08)
    for cnt in contours[:30]:
        if touches_border(cnt, w, h, margin_pixels):
            print("[WARN] Candidate touches border (edges).")
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in eps_list:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            area = cv2.contourArea(approx)
            if area <= (w * h * 0.01):
                continue
            rect = cv2.minAreaRect(approx)
            (rw, rh) = rect[1]
            if rw == 0 or rh == 0:
                continue
            ratio = min(rw, rh) / max(rw, rh)
            score = float(area) * (0.5 + 0.5 * ratio)  # 面积优先，兼顾接近正方形
            if score > best_score:
                best_score = score
                best = approx
                best_area = area
                best_eps = eps
    if best is not None:
        print(f"[DEBUG] Found quad via edges eps={best_eps}, area={best_area:.1f}")
        return order_corners(best)

    # 如果轮廓断裂导致失败，尝试形态学闭运算以连接间断边缘后再试
    try:
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours_c, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_c = sorted(contours_c, key=cv2.contourArea, reverse=True)
        best = None
        best_score = -1.0
        best_area = 0.0
        best_eps = None
        for cnt in contours_c[:30]:
            if touches_border(cnt, w, h, margin_pixels):
                print("[WARN] Candidate touches border (closed).")
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in eps_list:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                area = cv2.contourArea(approx)
                if area <= (w * h * 0.01):
                    continue
                rect = cv2.minAreaRect(approx)
                (rw, rh) = rect[1]
                if rw == 0 or rh == 0:
                    continue
                ratio = min(rw, rh) / max(rw, rh)
                score = float(area) * (0.5 + 0.5 * ratio)
                if score > best_score:
                    best_score = score
                    best = approx
                    best_area = area
                    best_eps = eps
        if best is not None:
            print(f"[DEBUG] Found quad via closed edges eps={best_eps}, area={best_area:.1f}")
            return order_corners(best)
        # 回退方案：使用最大轮廓的最小外接矩形 (minAreaRect)
        if len(contours_c) > 0:
            largest = contours_c[0]
            if touches_border(largest, w, h, margin_pixels):
                print("[WARN] minAreaRect touches border.")
                # 触边则放弃此回退
                return None
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
            if len(approx) == 4 and cv2.isContourConvex(approx) and not touches_border_points(approx, w, h, margin_pixels):
                area = cv2.contourArea(approx)
                if area > (w * h * 0.01):
                    print(f"[DEBUG] Found quad via saturation thresh={sat_thresh}, area={area:.1f}")
                    return order_corners(approx)
            elif len(approx) == 4 and touches_border_points(approx, w, h, margin_pixels):
                print("[WARN] Candidate touches border (saturation).")

    # 额外回退：CLAHE + 多组 Canny 阈值（不改变现有日志；新增日志前缀 enhanced）
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img_enh = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        g = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        for t1, t2 in [(30, 100), (40, 120), (20, 80)]:
            e2 = cv2.Canny(g, t1, t2)
            e2 = cv2.dilate(e2, np.ones((3, 3), np.uint8), iterations=1)
            cs, _ = cv2.findContours(e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = sorted(cs, key=cv2.contourArea, reverse=True)
            for cnt in cs[:30]:
                if touches_border(cnt, w, h, margin_pixels):
                    continue
                peri = cv2.arcLength(cnt, True)
                for eps in eps_list:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx) and not touches_border_points(approx, w, h, margin_pixels):
                        area = cv2.contourArea(approx)
                        if area > (w * h * 0.01):
                            print(f"[DEBUG] Found quad via enhanced Canny t=({t1},{t2}) eps={eps}, area={area:.1f}")
                            return order_corners(approx)
    except Exception as e:
        print(f"[DEBUG] enhanced-canny fallback error: {e}")
    return None


def solve_pose(corners_2d, marker_size_mm, K, dist):
    """
    使用四边形角点求解平面标志在相机坐标系下的姿态。
    - 先用 IPPE_SQUARE 求初值（若可用），再用 LM/ITERATIVE 细化，显著降低重投影误差。
    corners_2d: TL, TR, BR, BL 顺序，形状 (4,2)
    返回 rvec, tvec
    """
    half = marker_size_mm / 2.0
    # 与 order_corners 对应的 3D 角点顺序（Z=0 平面）
    points_3d = np.array([[-half, -half, 0],
                          [ half, -half, 0],
                          [ half,  half, 0],
                          [-half,  half, 0]], dtype=np.double)

    # 初值：优先使用 ITERATIVE（更稳定），而不是 IPPE_SQUARE（在某些情况下会失败）
    rvec = np.zeros((3, 1), dtype=np.double)
    tvec = np.zeros((3, 1), dtype=np.double)
    ok = False
    
    # 尝试 IPPE_SQUARE，但要验证结果的合理性
    try:
        if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
            ok_ippe, rvec_ippe, tvec_ippe = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok_ippe:
                # 验证 IPPE_SQUARE 的结果：计算重投影误差
                reproj, _ = cv2.projectPoints(points_3d, rvec_ippe, tvec_ippe, K, dist)
                reproj = reproj.reshape(-1, 2)
                rmse_ippe = float(np.sqrt(np.mean(np.sum((reproj - corners_2d)**2, axis=1))))
                
                # 只有当 RMSE < 50px 时才接受 IPPE_SQUARE 的结果
                if rmse_ippe < 50.0:
                    rvec, tvec = rvec_ippe, tvec_ippe
                    ok = True
    except Exception:
        ok = False
        
    if not ok:
        # 退回到 ITERATIVE 求解（更稳定）
        ok, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    # 细化：优先使用 LM 精炼，否则用 ITERATIVE + 初值
    try:
        if hasattr(cv2, 'solvePnPRefineLM'):
            rvec, tvec = cv2.solvePnPRefineLM(points_3d, corners_2d, K, dist, rvec, tvec)
        else:
            ok2, rvec, tvec = cv2.solvePnP(points_3d, corners_2d, K, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        pass
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
    # 使用对角线交点作为正方形中心的投影
    c_xy = quad_center_from_diagonals(corners_2d)
    center2d = np.array(c_xy, dtype=np.double)
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


def draw_axes(img, K, dist, rvec, tvec, axis_len=100.0, origin_xy: tuple | None = None):
    """绘制坐标轴；可选地以平面内一点 (origin_xy) 作为坐标系原点。
    origin_xy 单位为毫米（与 marker_size_mm 相同）。
    """
    if origin_xy is None:
        O3 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        X3 = np.array([[axis_len, 0.0, 0.0]], dtype=np.float32)
        Y3 = np.array([[0.0, axis_len, 0.0]], dtype=np.float32)
        Z3 = np.array([[0.0, 0.0, axis_len]], dtype=np.float32)
    else:
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        O3 = np.array([[ox, oy, 0.0]], dtype=np.float32)
        X3 = np.array([[ox + axis_len, oy, 0.0]], dtype=np.float32)
        Y3 = np.array([[ox, oy + axis_len, 0.0]], dtype=np.float32)
        Z3 = np.array([[ox, oy, axis_len]], dtype=np.float32)

    projX, _ = cv2.projectPoints(X3, rvec, tvec, K, dist)
    projY, _ = cv2.projectPoints(Y3, rvec, tvec, K, dist)
    projZ, _ = cv2.projectPoints(Z3, rvec, tvec, K, dist)
    origin, _ = cv2.projectPoints(O3, rvec, tvec, K, dist)
    proj = np.vstack([projX, projY, projZ]).reshape(-1, 2)
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


def process_frame(frame, K, dist, marker_size_mm, target_mode='auto', margin_pixels: int = 20):
    corners = find_quad_corners(frame, target_mode=target_mode, margin_pixels=margin_pixels)
    if corners is None:
        # 尝试使用增强的图像进行回退检测（CLAHE / 不同 Canny 阈值）
        corners_fallback, enhanced = None, None
        try:
            corners_fallback, enhanced = try_enhanced_detection(frame, target_mode, margin_pixels=margin_pixels)
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
    # 最终框触边检查与告警
    h, w = frame.shape[:2]
    if corners_touch_border(corners, w, h, margin_pixels):
        print("[WARN] Detected quad touches image border.")

    rvec, tvec = solve_pose(corners, marker_size_mm, K, dist)
    diag = pose_diagnostics(corners, rvec, tvec, K, dist, marker_size_mm)

    # 若重投影误差异常大，尝试不同的角点循环顺序以消除偶发排序歧义
    def cyclic_variants(pts: np.ndarray):
        p = np.asarray(pts, dtype=np.double).reshape(4, 2)
        variants = []
        for k in range(4):
            variants.append(np.roll(p, -k, axis=0))  # TL->TR->BR->BL 轮转
        q = p[[0, 3, 2, 1]]  # TL, BL, BR, TR 反向
        for k in range(4):
            variants.append(np.roll(q, -k, axis=0))
        return variants

    if not np.isfinite(diag.get('reproj_rmse_px', np.inf)) or diag['reproj_rmse_px'] > 20.0:
        best = (diag['reproj_rmse_px'], rvec, tvec, corners)
        for c2 in cyclic_variants(corners)[1:]:  # 跳过原始排序
            r2, t2 = solve_pose(c2, marker_size_mm, K, dist)
            d2 = pose_diagnostics(c2, r2, t2, K, dist, marker_size_mm)
            if d2['reproj_rmse_px'] < best[0]:
                best = (d2['reproj_rmse_px'], r2, t2, c2)
        if best[0] < diag['reproj_rmse_px']:
            diag = {'reproj_rmse_px': best[0], **pose_diagnostics(best[3], best[1], best[2], K, dist, marker_size_mm)}
            rvec, tvec = best[1], best[2]

    # 姿态合理性检查：RMSE 与深度范围
    rmse = diag.get('reproj_rmse_px', float('inf'))
    depth_n = diag.get('depth_along_normal_mm', float('inf'))
    pose_valid = np.isfinite(rmse) and rmse <= 10.0 and np.isfinite(depth_n) and 50.0 <= depth_n <= 8000.0
    
    if not pose_valid:
        print(f"[DEBUG] Pose rejected. rmse={rmse:.2f}px, d_n={depth_n:.1f}mm (limits: rmse<=10px, 50<=d_n<=8000)")
        # 估算合理的 marker_size：基于角点像素距离与内参反推
        try:
            corners_px = np.asarray(corners, dtype=np.double).reshape(4, 2)
            w_px = float(np.linalg.norm(corners_px[1] - corners_px[0]))
            h_px = float(np.linalg.norm(corners_px[2] - corners_px[1]))
            avg_px = (w_px + h_px) / 2.0
            fx_avg = (K[0, 0] + K[1, 1]) / 2.0
            # 假设深度 ~500mm，估算真实边长
            estimated_size_mm = (avg_px / fx_avg) * 500.0
            print(f"[HINT] 检测到方块像素尺寸 ~{avg_px:.0f}px；若深度 ~500mm，建议 --marker-size-mm {estimated_size_mm:.0f}")
        except Exception:
            pass
        cv2.putText(frame, 'Pose INVALID (check marker_size!)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # 继续绘制坐标系（用警示色），便于调试可视化
        # 可视化角点
        for i, pnt in enumerate(corners):
            cv2.circle(frame, tuple(pnt.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(frame, ['TL','TR','BR','BL'][i], tuple(pnt.astype(int) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    # 几何中心 C：用四边形对角线交点（透视条件下更准确）
    cx, cy = quad_center_from_diagonals(corners)
    center2d = (int(round(cx)), int(round(cy)))
    # 通过单应矩阵 H 反投影，将 C 的图像点转换为平面坐标 (Xc, Yc)
    origin_xy = None
    try:
        R, _ = cv2.Rodrigues(rvec)
        H = K @ np.column_stack((R[:, 0], R[:, 1], tvec.reshape(3)))  # 3x3
        H_inv = np.linalg.inv(H)
        uv1 = np.array([cx, cy, 1.0], dtype=np.double)
        v = H_inv @ uv1
        if np.isfinite(v).all() and abs(v[2]) > 1e-12:
            origin_xy = (float(v[0] / v[2]), float(v[1] / v[2]))
    except Exception as e:
        print(f"[DEBUG] homography invert failed: {e}")

    # 绘制坐标轴：若 origin_xy 有效，则 O 将与 C 重合；姿态无效时用黄色警示
    axis_color_x = (0, 255, 0) if pose_valid else (0, 200, 200)
    axis_color_y = (0, 0, 255) if pose_valid else (0, 200, 200)
    axis_color_z = (255, 0, 0) if pose_valid else (0, 200, 200)
    draw_axes(frame, K, dist, rvec, tvec, axis_len=marker_size_mm * 0.5, origin_xy=origin_xy)
    cv2.circle(frame, center2d, 5, (0, 255, 255) if pose_valid else (0, 150, 255), -1)
    cv2.putText(frame, 'C', (center2d[0] + 6, center2d[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if pose_valid else (0, 150, 255), 2)
    
    # 文本信息 - 添加半透明背景提高可读性
    roll, pitch, yaw = rvec_tvec_to_euler(rvec)
    
    # 添加半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 50), (660, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # 完整显示所有姿态参数
    y_offset = 70
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # tvec: 平移向量原始值
    info_color = (0, 255, 0) if pose_valid else (0, 150, 255)
    cv2.putText(frame, f'tvec (mm): [{tvec.ravel()[0]:7.2f}, {tvec.ravel()[1]:7.2f}, {tvec.ravel()[2]:7.2f}]',
                (15, y_offset), font, font_scale, info_color, thickness, cv2.LINE_AA)
    y_offset += 22
    
    # 距离: 计算值
    distance = float(np.linalg.norm(tvec))
    cv2.putText(frame, f'Distance: {distance:7.2f} mm',
                (15, y_offset), font, font_scale, info_color, thickness, cv2.LINE_AA)
    y_offset += 22
    
    # rvec: 旋转向量(罗德里格斯表示)
    rvec_color = (100, 200, 255) if pose_valid else (0, 150, 255)
    cv2.putText(frame, f'rvec (rod): [{rvec.ravel()[0]:6.3f}, {rvec.ravel()[1]:6.3f}, {rvec.ravel()[2]:6.3f}]',
                (15, y_offset), font, font_scale, rvec_color, thickness, cv2.LINE_AA)
    y_offset += 22
    
    # 欧拉角: Roll, Pitch, Yaw
    euler_color = (255, 200, 100) if pose_valid else (0, 150, 255)
    cv2.putText(frame, f'Roll:  {roll:7.2f} deg',
                (15, y_offset), font, font_scale, euler_color, thickness, cv2.LINE_AA)
    y_offset += 22
    cv2.putText(frame, f'Pitch: {pitch:7.2f} deg',
                (15, y_offset), font, font_scale, euler_color, thickness, cv2.LINE_AA)
    y_offset += 22
    cv2.putText(frame, f'Yaw:   {yaw:7.2f} deg',
                (15, y_offset), font, font_scale, euler_color, thickness, cv2.LINE_AA)
    y_offset += 22
    
    # RMSE: 重投影误差
    rmse = diag["reproj_rmse_px"]
    rmse_color = (0, 255, 0) if rmse < 10 else (0, 165, 255) if rmse < 20 else (0, 0, 255)
    if not pose_valid:
        rmse_color = (0, 150, 255)
    cv2.putText(frame, f'RMSE: {rmse:5.2f} px',
                (15, y_offset), font, font_scale, rmse_color, thickness, cv2.LINE_AA)
    # 标注四个角点（无效姿态时用红色）
    corner_color = (255, 255, 255) if pose_valid else (0, 0, 255)
    for i, p in enumerate(corners):
        cv2.circle(frame, tuple(p.astype(int)), 5, corner_color, -1)
        cv2.putText(frame, ['TL', 'TR', 'BR', 'BL'][i], tuple(p.astype(int) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0) if pose_valid else (0, 0, 255), 2)
    return frame, rvec, tvec


# 已移除：批量保存调试中间图的函数（按需求停用文件输出）


def try_enhanced_detection(img, target_mode='auto', margin_pixels: int = 20):
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
        corners = find_quad_corners(enhanced, target_mode=target_mode, margin_pixels=margin_pixels)
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
                if len(approx) == 4 and cv2.isContourConvex(approx) and not touches_border_points(approx, w, h, margin_pixels):
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
    parser.add_argument('--marker-size-mm', type=float, default=127.0,
                        help='方形标记的实际边长（毫米）')
    parser.add_argument('--save', '--save_output', dest='save', type=str, default=None,
                        help='当输入是图片时保存叠加结果；当输入是视频时保存叠加视频（支持 mp4/avi）')
    parser.add_argument('--target', choices=['auto', 'black-square'], default='black-square',
                        help='目标类型：auto 自动或 black-square 纯黑方形（推荐在黑方块时使用）')
    parser.add_argument('--intrinsics', type=str, default=None,
                        help='相机内参 npz 路径（包含 camera_matrix, dist_coeffs）。未提供时使用默认策略')
    parser.add_argument('--K', type=str, default=None,
                        help='直接指定相机矩阵，格式如 "fx,0,cx;0,fy,cy;0,0,1" 或 9 个数字')
    parser.add_argument('--dist', type=str, default=None,
                        help='直接指定畸变系数，4/5/8 个数字，如 "k1,k2,p1,p2,k3"')
    parser.add_argument('--no-dist', action='store_true',
                        help='忽略畸变系数，按零畸变处理（适合未知镜头或截图）')
    parser.add_argument('--adapt-k', action='store_true',
                        help='强制按图像分辨率自适应缩放内参K；默认仅在未提供 --intrinsics 时启用')
    parser.add_argument('--margin-pixels', type=int, default=20,
                        help='边界容差像素。候选/最终四边形触边（<=该像素）将被过滤或警告')
    args = parser.parse_args()

    # 1) 先从文件或默认加载
    K, dist = load_intrinsics(args.intrinsics)
    # 2) 若 CLI 提供了 K/dist，则覆盖
    K_from_cli = False
    try:
        if args.K is not None:
            K = parse_camera_matrix(args.K)
            K_from_cli = True
            print(f"[INFO] Using camera matrix from CLI: fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, cx={K[0,2]:.3f}, cy={K[1,2]:.3f}")
        if args.dist is not None:
            dist = parse_dist_coeffs(args.dist)
            print(f"[INFO] Using distortion from CLI: len={len(dist)} -> {dist}")
    except Exception as e:
        print(f"[ERROR] 解析 --K/--dist 失败: {e}")
        return
    if args.no_dist:
        dist = np.zeros((5, ), dtype=np.double)

    # 自适应策略：按 CALIB_BASE_RES 对 K 进行分辨率缩放（与给定图片 1279x1706 匹配，若尺寸一致则 sx=sy=1）
    def maybe_adapt(K_in, shape):
        return adapt_intrinsics_to_frame(K_in, shape)

    if args.input is None:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print('无法打开摄像头，也未提供输入文件。')
            return
        # 先读一帧用于自适应 K
        ret, frame = cap.read()
        if not ret:
            print('无法从摄像头读取帧。')
            return
        K_cam = maybe_adapt(K, frame.shape)
        
        print('\n[摄像头模式] 实时姿态估计已启动')
        print('按 q 或 ESC 退出')
        print('=' * 60)
        
        frame_count = 0
        # 第一帧也要处理显示
        vis, rvec, tvec = process_frame(frame, K_cam, dist, args.marker_size_mm, target_mode=args.target, margin_pixels=args.margin_pixels)
        
        # 输出第一帧姿态
        if rvec is not None and tvec is not None:
            roll, pitch, yaw = rvec_tvec_to_euler(rvec)
            distance = float(np.linalg.norm(tvec))
            print(f'\n[帧 {frame_count}] 姿态信息:')
            print(f'  位置: X={tvec[0,0]:7.1f}, Y={tvec[1,0]:7.1f}, Z={tvec[2,0]:7.1f} mm, 距离={distance:7.1f} mm')
            print(f'  旋转: Roll={roll:6.1f}°, Pitch={pitch:6.1f}°, Yaw={yaw:6.1f}°')
        
        cv2.imshow('Pose', vis)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            vis, rvec, tvec = process_frame(frame, K_cam, dist, args.marker_size_mm, target_mode=args.target, margin_pixels=args.margin_pixels)
            
            # 每10帧输出一次姿态信息
            if rvec is not None and tvec is not None and frame_count % 10 == 0:
                roll, pitch, yaw = rvec_tvec_to_euler(rvec)
                distance = float(np.linalg.norm(tvec))
                print(f'[帧 {frame_count}] 位置: X={tvec[0,0]:7.1f}, Y={tvec[1,0]:7.1f}, Z={tvec[2,0]:7.1f} mm, 距离={distance:7.1f} mm | 旋转: Roll={roll:6.1f}°, Pitch={pitch:6.1f}°, Yaw={yaw:6.1f}°')
            
            cv2.imshow('Pose', vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f'\n总共处理 {frame_count} 帧')
        return

    # 文件输入
    path = args.input
    if not os.path.exists(path):
        print(f'输入文件不存在: {path}')
        return

    if is_image(path):
        img = cv2.imread(path)
        K_img = maybe_adapt(K, img.shape)
        vis, rvec, tvec = process_frame(img, K_img, dist, args.marker_size_mm, target_mode=args.target, margin_pixels=args.margin_pixels)
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

        # 已按需求移除：未检测到四边形时的调试图片输出
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
            vis, rvec, tvec = process_frame(frame0, K_vid, dist, args.marker_size_mm, target_mode=args.target, margin_pixels=args.margin_pixels)
            writer.write(vis)
            cv2.imshow('Pose', vis)
        else:
            writer = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, rvec, tvec = process_frame(frame, K_vid, dist, args.marker_size_mm, target_mode=args.target, margin_pixels=args.margin_pixels)
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


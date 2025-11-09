import cv2
import numpy as np

for i in range(1, 5):
    img = cv2.imread(f'./input/Shield/shield ({i}).png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HSV颜色空间检测黑色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 找轮廓
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        print(f"\nshield ({i}).png:")
        print(f"  包围盒: ({x}, {y}) 尺寸 {w}×{h}")
        print(f"  旋转矩形: {rect[1][0]:.0f}×{rect[1][1]:.0f}, 角度 {rect[2]:.1f}°")
        print(f"  角点:")
        for j, pt in enumerate(box):
            print(f"    {j+1}: ({pt[0]:.1f}, {pt[1]:.1f})")

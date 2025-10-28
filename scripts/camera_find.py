import cv2

# 尝试打开不同索引的摄像头
for i in range(0, 10):  # 通常索引0-9足够
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头索引 {i} 可用")
        cap.release()
    else:
        print(f"摄像头索引 {i} 不可用")
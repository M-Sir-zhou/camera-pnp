计算物体在图像/视频中的位置与姿态。
支持黑方块标记（black-square）和 ArUco 标记（aruco-5x5-1000）。
支持图像和视频输入，支持相机内参与真实边长以获得更精确的位姿估计，支持保存结果。
安装依赖
pip install -r requirements.txt
用法
图像    

仅显示（黑方块检测）
python .\main.py --input .\input\images\1.jpg --target black-square

显示并保存叠加结果
python .\main.py --input .\input\images\1.jpg --target black-square --save_output .\output\images\1_output.jpg

使用相机内参与真实边长（推荐更准），并保存
python .\main.py --input .\input\images\1.jpg --target black-square --intrinsics .\calibration_results.npz --marker-size-mm 147 --save_output .\output\images\1_output.jpg

若是屏幕截图或畸变未知，也可忽略畸变
python .\main.py --input .\input\images\1.jpg --target black-square --no-dist --save_output .\output\images\1_output.jpg


视频


处理并保存视频（编码依据扩展名：.mp4 用 mp4v，其它默认 XVID）
python .\main.py --input .\input\videos\1.mp4 --target black-square --save_output .\output\videos\1_output.mp4

使用相机内参与真实边长
python .\main.py --input .\input\videos\1.mp4 --target black-square --intrinsics .\calibration_results.npz --marker-size-mm 147 --save_output .\output\videos\1_output.mp4

忽略畸变
python .\main.py --input .\input\videos\1.mp4 --target black
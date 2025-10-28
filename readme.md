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
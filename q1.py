import os
import cv2

video_path = 'data/hw-4.mp4'  # 视频文件路径
output_folder = 'data/frame'  # 保存视频帧的文件夹

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 打开视频文件
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 保存帧
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()

print("视频帧提取完成。")

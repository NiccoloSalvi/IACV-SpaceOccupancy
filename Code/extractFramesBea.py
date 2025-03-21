import cv2
import os
import numpy as np


video_path = 'video.mp4'
output_folder = 'outputFolder'
os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


qty = 20
frame_indices = np.linspace(0, total_frames - 1, qty, dtype=int)


frame_num = 0
saved_count = 0

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    success, frame = cap.read()
    if success:
        print(f'Extracting frame {frame_num}/{qty}')
        filename = os.path.join(output_folder, f'frame_{saved_count:02d}.png')
        cv2.imwrite(filename, frame)
        saved_count += 1

cap.release()
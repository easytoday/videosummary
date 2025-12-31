# check_videos.py
import cv2
import numpy as np
import os

def check_video_quality(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad_frames = 0
    
    for i in range(min(100, total_frames)):  # Vérifier les 100 premières frames
        ret, frame = cap.read()
        if ret:
            if frame is not None:
                if frame.shape == (1, 1, 3) or frame.dtype == np.object_:
                    bad_frames += 1
                    print(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
    
    cap.release()
    
    if bad_frames > 0:
        print(f"❌ {video_path}: {bad_frames}/{min(100, total_frames)} frames corrompues")
        return False
    else:
        print(f"✅ {video_path}: OK")
        return True

# Vérifier toutes les vidéos
video_folder = "videos"
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        check_video_quality(os.path.join(video_folder, video_file))

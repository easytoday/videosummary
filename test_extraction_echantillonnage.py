# test_simple_extraction.py
import cv2
import numpy as np
from PIL import Image
import os

def test_video_reading(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    error_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 25 == 0:  # Échantillonner toutes les 25 frames
            print(f"\nFrame {frame_count}:")
            print(f"  Shape: {frame.shape}")
            print(f"  Dtype: {frame.dtype}")
            print(f"  Min/Max: {frame.min()}/{frame.max()}")
            
            # Vérifier la conversion
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                print(f"  PIL conversion: OK")
            except Exception as e:
                print(f"  PIL conversion: ERROR - {e}")
                error_count += 1
        
        frame_count += 1
        if frame_count > 100:  # Limiter aux 100 premières frames
            break
    
    cap.release()
    print(f"\nTotal frames: {frame_count}, Errors: {error_count}")
    return error_count == 0

# Tester quelques vidéos
video_files = [f for f in os.listdir("videos") if f.endswith('.mp4')][:5]

for video_file in video_files:
    print(f"\n{'='*50}")
    print(f"Testing: {video_file}")
    print(f"{'='*50}")
    test_video_reading(os.path.join("videos", video_file))

import h5py
import numpy as np
import shutil
import os

'''
Permet de faire un mapping entre les pics (forte proba)  et les images auxquelles elle correspondent.
'''

# Configuration
h5_path = 'log/alt_run_kts/result.h5'
frames_dir = 'videos/v00072_frames/' # images JPG extraites par ffmpeg
output_dir = 'analysis_peaks_v72/'
video_index = 13 # Index de v00072
subsample_rate = 15 # 1 image sur 15 pour les features

os.makedirs(output_dir, exist_ok=True)

with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())
    key = keys[video_index]
    scores = f[key]['score'][()]

    # Trouver les indices des 5 plus hauts pics
    # On trie les indices par valeur de score décroissante
    top_indices = np.argsort(scores)[-5:][::-1]

    print(f"Analyse des pics pour {key}...")
    for rank, idx in enumerate(top_indices):
        score_val = scores[idx]
        # Calcul de la vraie frame dans le dossier JPG
        real_frame_num = idx * subsample_rate
        # Formatage du nom (ex: 000150.jpg)
        frame_name = f"{real_frame_num:06d}.jpg"
        src_path = os.path.join(frames_dir, frame_name)
        
        if os.path.exists(src_path):
            dest_name = f"peak_{rank+1}_score_{score_val:.2f}_frame_{real_frame_num}.jpg"
            shutil.copy(src_path, os.path.join(output_dir, dest_name))
            print(f"Pic {rank+1} identifié : Frame {real_frame_num} (Score: {score_val:.4f})")

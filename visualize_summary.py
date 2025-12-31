import torch
import h5py
import cv2
import os
from model import DSN

# Config
video_id = "v1" # La vidéo que vous voulez voir
video_path = f"videos/{video_id}.mp4"
features_path = "dataset_cpu/features.h5"
model_path = "outputs/models/dsn_model.pth"
output_dir = f"outputs/visualisation/{video_id}"
os.makedirs(output_dir, exist_ok=True)

def visualize():
    # 1. Charger le modèle
    model = DSN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Obtenir les probabilités
    with h5py.File(features_path, 'r') as hdf:
        features = torch.tensor(hdf[video_id][:]).unsqueeze(0)
        with torch.no_grad():
            probs = model(features).squeeze().tolist()

    # 3. Sélectionner les frames (Top 15% des meilleures probabilités)
    n_to_extract = int(len(probs) * 0.15)
    # On trie par probabilité décroissante et on prend les index
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:n_to_extract]
    top_indices.sort() # Remettre dans l'ordre chronologique

    # 4. Extraire les frames du fichier MP4 original
    cap = cv2.VideoCapture(video_path)
    count = 0
    extracted = 0
    
    print(f"Extraction de {n_to_extract} frames pour le résumé...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Si cette frame (seconde) est dans notre top_indices
        if count in top_indices:
            cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", frame)
            extracted += 1
        
        count += 1
    cap.release()
    print(f"Terminé ! Les images sont dans : {output_dir}")

if __name__ == "__main__":
    visualize()

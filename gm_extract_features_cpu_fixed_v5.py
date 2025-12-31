# scripts/extract_features_cpu_v5.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class CPUOptimizedGoogLeNetExtractor:
    def __init__(self, batch_size=16):
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        logger.info("Initialisation de GoogLeNet...")
        
        from torchvision.models import googlenet, GoogLeNet_Weights
        model_raw = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        self.model = nn.Sequential(*list(model_raw.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_video(self, video_path, target_fps=1):
        # Utilisation du chemin absolu pour éviter les erreurs de caractères spéciaux
        abs_path = os.path.abspath(video_path)
        cap = cv2.VideoCapture(abs_path)
        
        if not cap.isOpened():
            return None

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps_orig / target_fps))

        frames_batch = []
        features_list = []
        
        pbar = tqdm(total=total_frames, desc="Frames", unit="fr", leave=False)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame is not None and frame_idx % frame_interval == 0:
                try:
                    # Conversion et préparation
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames_batch.append(self.transform(img))

                    if len(frames_batch) >= self.batch_size:
                        features_list.append(self._process_batch(frames_batch))
                        frames_batch = []
                except Exception as e:
                    continue # On passe à la suivante en cas d'erreur isolée

            frame_idx += 1
            pbar.update(1)

        if frames_batch:
            features_list.append(self._process_batch(frames_batch))

        cap.release()
        pbar.close()

        if not features_list:
            return None
            
        return np.vstack(features_list).astype(np.float32)

    def _process_batch(self, batch):
        with torch.no_grad():
            tensors = torch.stack(batch).to(self.device)
            out = self.model(tensors)
            # Global Average Pooling pour obtenir un vecteur de 1024
            out = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(out), 1)
            return out.cpu().numpy()

class VideoProcessor:
    def __init__(self, output_dir="./dataset_cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = CPUOptimizedGoogLeNetExtractor()

    def run(self, video_folder, force=False):
        # Recherche récursive des vidéos
        video_files = []
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.extend(list(Path(video_folder).rglob(f"*{ext}")))
        
        video_files = sorted(video_files)
        hdf5_path = self.output_dir / "features_googlenet.h5"
        
        if force and hdf5_path.exists():
            os.remove(hdf5_path)

        with h5py.File(hdf5_path, "a") as hf:
            for i, v_path in enumerate(video_files):
                v_name = v_path.name
                # Nettoyage du nom pour HDF5 (évite les erreurs de clés)
                v_key = "".join(c for c in v_name if c.isalnum() or c in "._- ").strip()
                
                if v_key in hf:
                    logger.info(f"[{i+1}/{len(video_files)}] Déjà traité: {v_name}")
                    continue

                logger.info(f"[{i+1}/{len(video_files)}] 📹 Traitement: {v_name}")
                features = self.extractor.extract_features_from_video(str(v_path))
                
                if features is not None:
                    hf.create_dataset(v_key, data=features, compression="gzip")
                    logger.info(f"   ✅ OK: {features.shape}")
                else:
                    logger.error(f"   ❌ Erreur de lecture pour cette vidéo.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    proc = VideoProcessor()
    proc.run(args.video_folder, force=args.force)

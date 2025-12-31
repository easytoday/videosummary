# scripts/gm_extract_features_cpu_fixed_v6.py
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

# Configuration du logging
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
        
        logger.info(f"Initialisation de GoogLeNet (Batch Size: {batch_size})")
        torch.set_grad_enabled(False)
        
        from torchvision.models import googlenet, GoogLeNet_Weights
        model_raw = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        
        # Extracteur : Couches de convolution + Global Average Pooling
        self.backbone = nn.Sequential(*list(model_raw.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.backbone.eval()
        self.backbone.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, video_path, target_fps=1):
        # Utilisation du chemin absolu pour OpenCV
        abs_path = os.path.abspath(video_path)
        cap = cv2.VideoCapture(abs_path)
        
        if not cap.isOpened():
            return None

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps_orig / target_fps))

        frames_batch = []
        features_list = []
        
        # Barre de progression interne
        pbar = tqdm(total=total_frames, desc=" > Frames", unit="fr", leave=False)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0 and frame is not None:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames_batch.append(self.transform(img))

                    if len(frames_batch) >= self.batch_size:
                        features_list.append(self._process_batch(frames_batch))
                        frames_batch = []
                except Exception:
                    pass

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
            # Passage dans le réseau
            features = self.backbone(tensors)
            # Réduction de dimension (Pooling) -> (Batch, 1024)
            features = self.pool(features)
            return torch.flatten(features, 1).cpu().numpy()

class VideoProcessor:
    def __init__(self, output_dir, batch_size):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = CPUOptimizedGoogLeNetExtractor(batch_size=batch_size)

    def run(self, video_folder, target_fps, hdf5_name, force):
        video_files = []
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.extend(list(Path(video_folder).rglob(f"*{ext}")))
        video_files = sorted(video_files)

        hdf5_path = self.output_dir / (hdf5_name or "features_googlenet.h5")
        if force and hdf5_path.exists():
            os.remove(hdf5_path)
            logger.info(f"Fichier {hdf5_path.name} supprimé.")

        with h5py.File(hdf5_path, "a") as hf:
            for i, v_path in enumerate(video_files):
                v_name = v_path.name
                # Nettoyage de la clé pour HDF5
                v_key = "".join(c for c in v_name if c.isalnum() or c in "._- ").strip()
                
                if v_key in hf:
                    logger.info(f"[{i+1}/{len(video_files)}] Déjà présent : {v_name}")
                    continue

                logger.info(f"[{i+1}/{len(video_files)}] 📹 Extraction : {v_name}")
                features = self.extractor.extract_features(str(v_path), target_fps)
                
                if features is not None:
                    hf.create_dataset(v_key, data=features, compression="gzip")
                    logger.info(f"   ✅ Succès : {features.shape[0]} vecteurs.")
                else:
                    logger.error(f"   ❌ Échec lecture : {v_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extracteur de features GoogLeNet")
    parser.add_argument("--video_folder", type=str, required=True, help="Dossier contenant les vidéos")
    parser.add_argument("--output_dir", type=str, default="./dataset_cpu", help="Dossier de sortie")
    parser.add_argument("--fps", type=int, default=1, help="Images par seconde à extraire")
    parser.add_argument("--batch_size", type=int, default=16, help="Taille du batch CPU")
    parser.add_argument("--hdf5_name", type=str, default=None, help="Nom du fichier HDF5")
    parser.add_argument("--force", action="store_true", help="Écrase le fichier existant")
    
    args = parser.parse_args()

    proc = VideoProcessor(output_dir=args.output_dir, batch_size=args.batch_size)
    proc.run(args.video_folder, target_fps=args.fps, hdf5_name=args.hdf5_name, force=args.force)

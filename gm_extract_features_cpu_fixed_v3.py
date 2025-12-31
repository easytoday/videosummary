# scripts/extract_features_cpu_fixed_v3.py
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
import time
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
        
        logger.info("Initialisation de GoogLeNet sur CPU...")
        torch.set_grad_enabled(False)
        
        from torchvision.models import googlenet, GoogLeNet_Weights
        model_raw = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        
        # Extracteur de features (avant-dernière couche)
        backbone = nn.Sequential(*list(model_raw.children())[:-1])
        self.model = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_video_optimized(self, video_path, target_fps=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps_orig / target_fps))

        frames_batch = []
        features_list = []
        consecutive_errors = 0

        pbar = tqdm(total=total_frames, desc=f"📹 {os.path.basename(video_path)[:15]}", unit="fr")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Si la frame est vide, on essaie de passer à la suivante au lieu de tout couper
                if frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 20: break
                    frame_idx += 1
                    continue

                try:
                    # On s'assure que c'est du uint8 pour PIL
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames_batch.append(self.transform(img))
                    consecutive_errors = 0 

                except Exception:
                    consecutive_errors += 1
                    if consecutive_errors > 20: break

                if len(frames_batch) >= self.batch_size:
                    feat = self._process_batch(frames_batch)
                    if feat is not None: features_list.append(feat)
                    frames_batch = []

            frame_idx += 1
            pbar.update(1)

        if frames_batch:
            feat = self._process_batch(frames_batch)
            if feat is not None: features_list.append(feat)

        cap.release()
        pbar.close()

        if not features_list: return None, None
        return np.vstack(features_list).astype(np.float32), {"video": video_path}

    def _process_batch(self, batch):
        try:
            tensors = torch.stack(batch).to(self.device)
            with torch.no_grad():
                return self.model(tensors).cpu().numpy()
        except:
            return None

class VideoDatasetProcessorCPU:
    def __init__(self, output_dir="./dataset_cpu", batch_size=16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = CPUOptimizedGoogLeNetExtractor(batch_size=batch_size)

    def process(self, video_folder, target_fps=1, hdf5_name=None, force=False):
        video_files = sorted([p for p in Path(video_folder).glob("**/*") if p.suffix.lower() in [".mp4", ".avi", ".mov"]])
        
        hdf5_path = self.output_dir / (hdf5_name or "features.h5")
        if hdf5_path.exists() and force:
            os.remove(hdf5_path)
            logger.info("Fichier HDF5 précédent supprimé (--force).")

        with h5py.File(hdf5_path, "a") as hf:
            if "features" not in hf: hf.create_group("features")
            
            for i, v_path in enumerate(video_files):
                v_id = f"video_{i:03d}"
                if v_id in hf["features"]: continue

                logger.info(f"Traitement {i+1}/{len(video_files)}: {v_path.name}")
                features, _ = self.extractor.extract_features_from_video_optimized(str(v_path), target_fps)
                
                if features is not None:
                    hf["features"].create_dataset(v_id, data=features, compression="gzip")
                    logger.info(f"✅ Terminé : {features.shape[0]} frames extraites.")
                else:
                    logger.error(f"❌ Échec : Aucune donnée lue pour {v_path.name}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dataset_cpu")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hdf5_name", type=str, default=None)
    
    args = parser.parse_args()

    processor = VideoDatasetProcessorCPU(output_dir=args.output_dir, batch_size=args.batch_size)
    processor.process(args.video_folder, target_fps=args.fps, hdf5_name=args.hdf5_name, force=args.force)

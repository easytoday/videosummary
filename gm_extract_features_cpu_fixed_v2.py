# scripts/extract_features_cpu_fixed.py
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
import traceback
from functools import lru_cache

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class CPUOptimizedGoogLeNetExtractor:
    def __init__(self, batch_size=8):
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        logger.info("Initialisation de GoogLeNet sur CPU...")

        torch.set_grad_enabled(False)
        from torchvision.models import googlenet, GoogLeNet_Weights

        # Chargement du modèle
        model_raw = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        self.model = self._create_feature_extractor(model_raw)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"✅ GoogLeNet initialisé | Batch size: {batch_size}")

    def _create_feature_extractor(self, model):
        # On récupère tout sauf la dernière couche de classification
        backbone = nn.Sequential(*list(model.children())[:-1])
        return nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))

    def extract_features_from_video_optimized(self, video_path, target_fps=1):
        logger.info(f"📹 Traitement de: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir le fichier vidéo.")
            return None, None

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps_orig / target_fps))

        frames_batch = []
        features_list = []
        consecutive_errors = 0
        max_errors = 15 

        pbar = tqdm(total=total_frames, desc="Extraction", unit="fr")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # --- VALIDATION ASSOUPLIE ---
                if frame is None or not hasattr(frame, "shape"):
                    consecutive_errors += 1
                    if consecutive_errors > max_errors: break
                    frame_idx += 1
                    pbar.update(1)
                    continue

                try:
                    # Conversion forcée en uint8 si nécessaire
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                    # Passage en RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Transformation PIL
                    img = Image.fromarray(frame_rgb)
                    tensor = self.transform(img)
                    frames_batch.append(tensor)
                    consecutive_errors = 0 # Succès : on réinitialise

                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors > max_errors: break

                # Traitement du batch
                if len(frames_batch) >= self.batch_size:
                    feat = self._process_batch(frames_batch)
                    if feat is not None: features_list.append(feat)
                    frames_batch = []

            frame_idx += 1
            pbar.update(1)

        # Dernier batch
        if frames_batch:
            feat = self._process_batch(frames_batch)
            if feat is not None: features_list.append(feat)

        cap.release()
        pbar.close()

        if not features_list:
            return None, None

        all_feats = np.vstack(features_list)
        
        # Generation des métadonnées simplifiées
        meta = {
            "video_path": video_path,
            "extracted_frames": all_feats.shape[0],
            "feature_dim": all_feats.shape[1],
            "target_fps": target_fps
        }
        return all_feats.astype(np.float32), meta

    def _process_batch(self, batch):
        try:
            tensors = torch.stack(batch).to(self.device)
            with torch.no_grad():
                out = self.model(tensors)
            return out.cpu().numpy()
        except:
            return None

class VideoDatasetProcessorCPU:
    def __init__(self, output_dir="./dataset_cpu", resume=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.extractor = CPUOptimizedGoogLeNetExtractor(batch_size=16)
        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def get_video_files(self, folder):
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        return sorted([p for p in Path(folder).glob("**/*") if p.suffix.lower() in exts])

    def process_videos_to_hdf5(self, video_folder, target_fps=1, hdf5_name=None, force=False):
        video_files = self.get_video_files(video_folder)
        if not video_files: return None

        hdf5_path = self.output_dir / (hdf5_name or "features_googlenet.h5")
        if hdf5_path.exists() and force: os.remove(hdf5_path)

        with h5py.File(hdf5_path, "a") as hf:
            if "features" not in hf: hf.create_group("features")
            
            for i, v_path in enumerate(video_files):
                v_name = v_path.stem
                v_id = f"video_{i:03d}"
                
                if v_id in hf["features"]:
                    continue

                features, meta = self.extractor.extract_features_from_video_optimized(str(v_path), target_fps)
                
                if features is not None:
                    ds = hf["features"].create_dataset(v_id, data=features, compression="gzip")
                    for k, v in meta.items(): ds.attrs[k] = str(v)
                    logger.info(f"✅ {v_name} sauvegardé.")
                else:
                    logger.warning(f"❌ Saut de {v_name} (aucune donnée).")

        return hdf5_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    processor = VideoDatasetProcessorCPU()
    processor.process_videos_to_hdf5(args.video_folder, target_fps=args.fps)

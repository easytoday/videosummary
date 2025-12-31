# scripts/extract_features_cpu_v4.py
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
        
        logger.info("Initialisation de GoogLeNet sur CPU...")
        torch.set_grad_enabled(False)
        
        from torchvision.models import googlenet, GoogLeNet_Weights
        model_raw = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        
        # Extracteur de features : on s'arrête juste avant la classification
        self.model = nn.Sequential(*list(model_raw.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_video_optimized(self, video_path, target_fps=1):
        cap = cv2.VideoCapture(video_path)
        
        # --- TEST DE LECTURE CRITIQUE ---
        ret_test, frame_test = cap.read()
        if not ret_test or frame_test is None:
            logger.error(f"❌ OpenCV ne peut pas décoder les images de: {os.path.basename(video_path)}. Vérifiez vos codecs (ffmpeg).")
            cap.release()
            return None, None

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps_orig / target_fps))

        frames_batch = []
        features_list = []
        
        # Re-préparer la capture (revenir au début après le test)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        pbar = tqdm(total=total_frames, desc="Extraction", unit="fr", leave=False)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval == 0:
                try:
                    # Conversion BGR -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames_batch.append(self.transform(img))

                    if len(frames_batch) >= self.batch_size:
                        features_list.append(self._process_batch(frames_batch))
                        frames_batch = []
                except Exception as e:
                    logger.debug(f"Erreur frame {frame_idx}: {e}")

            frame_idx += 1
            pbar.update(1)

        # Dernier batch
        if frames_batch:
            features_list.append(self._process_batch(frames_batch))

        cap.release()
        pbar.close()

        if not features_list:
            return None, None
            
        return np.vstack(features_list).astype(np.float32), {"fps": target_fps}

    def _process_batch(self, batch):
        with torch.no_grad():
            tensors = torch.stack(batch).to(self.device)
            # Passage dans le modèle + pooling final pour avoir un vecteur plat
            out = self.model(tensors)
            out = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(out), 1)
            return out.cpu().numpy()

class VideoDatasetProcessorCPU:
    def __init__(self, output_dir="./dataset_cpu", batch_size=16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = CPUOptimizedGoogLeNetExtractor(batch_size=batch_size)

    def process(self, video_folder, target_fps=1, force=False):
        video_files = sorted([p for p in Path(video_folder).glob("**/*") if p.suffix.lower() in [".mp4", ".avi", ".mov"]])
        hdf5_path = self.output_dir / "features_googlenet.h5"
        
        if hdf5_path.exists() and force:
            os.remove(hdf5_path)

        with h5py.File(hdf5_path, "a") as hf:
            if "features" not in hf: hf.create_group("features")
            
            for i, v_path in enumerate(video_files):
                v_name = v_path.stem
                if v_name in hf["features"]:
                    continue

                logger.info(f"[{i+1}/{len(video_files)}] 📹 {v_path.name}")
                features, _ = self.extractor.extract_features_from_video_optimized(str(v_path), target_fps)
                
                if features is not None:
                    hf["features"].create_dataset(v_name, data=features, compression="gzip")
                    logger.info(f"   ✅ Succès: {len(features)} vecteurs extraits.")
                else:
                    logger.error(f"   ❌ Échec: Vidéo illisible.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dataset_cpu")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    processor = VideoDatasetProcessorCPU(output_dir=args.output_dir, batch_size=args.batch_size)
    processor.process(args.video_folder, target_fps=args.fps, force=args.force)

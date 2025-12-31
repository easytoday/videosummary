# scripts/gm_extract_features_cpu_fixed_v13.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, batch_size=16):
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        
        from torchvision.models import googlenet, GoogLeNet_Weights
        
        # 1. On charge avec aux_logits=True (obligatoire selon votre erreur)
        full_model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        
        # 2. CHIRURGIE : On extrait manuellement les blocs dans l'ordre SANS les branches auxiliaires
        # GoogLeNet chez torchvision est structuré ainsi :
        # conv1, maxpool1, conv2, conv3, maxpool2, inception3a, inception3b, maxpool3...
        # Les couches 'aux1' et 'aux2' sont des attributs séparés, pas dans les enfants directs si on filtre bien.
        
        layers = [
            full_model.conv1, full_model.maxpool1,
            full_model.conv2, full_model.conv3, full_model.maxpool2,
            full_model.inception3a, full_model.inception3b, full_model.maxpool3,
            full_model.inception4a, full_model.inception4b, full_model.inception4c, 
            full_model.inception4d, full_model.inception4e, full_model.maxpool4,
            full_model.inception5a, full_model.inception5b,
            full_model.avgpool,
            nn.Flatten(1)
        ]
        
        self.model = nn.Sequential(*layers)
        self.model.eval().to(self.device)
        
        self.resize_transform = transforms.Resize((224, 224))
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def process_video(self, path, fps_target=1):
        cap = cv2.VideoCapture(str(path))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps_in / fps_target))
        
        frames_batch, all_features = [], []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_resized = self.resize_transform(img_pil)
                
                # Correction Numpy 2.0 (Copie via torch.tensor)
                img_np = np.array(img_resized, dtype=np.float32) / 255.0
                tensor = torch.tensor(img_np).permute(2, 0, 1)
                tensor = self.normalize_transform(tensor)
                
                frames_batch.append(tensor)
                
                if len(frames_batch) >= self.batch_size:
                    all_features.append(self._run_model(frames_batch))
                    frames_batch = []
        
        if frames_batch:
            all_features.append(self._run_model(frames_batch))
            
        cap.release()
        return np.vstack(all_features) if all_features else None

    def _run_model(self, batch):
        with torch.no_grad():
            # Conversion en batch tenseur
            input_tensor = torch.stack(batch).to(self.device)
            return self.model(input_tensor).cpu().numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset_cpu")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    extractor = FeatureExtractor(batch_size=args.batch_size)
    video_paths = sorted([p for p in Path(args.video_folder).glob("v*") if p.suffix.lower() in ['.mp4', '.avi']])
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / "features.h5"
    
    if args.force and hdf5_path.exists():
        os.remove(hdf5_path)

    with h5py.File(hdf5_path, "a") as hf:
        for v_path in tqdm(video_paths, desc="Extraction"):
            if v_path.stem in hf: continue
            features = extractor.process_video(v_path, args.fps)
            if features is not None:
                hf.create_dataset(v_path.stem, data=features, compression="gzip")
                logger.info(f"✅ {v_path.name} : {features.shape}")

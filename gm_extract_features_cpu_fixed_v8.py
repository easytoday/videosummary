# gm_extract_features_v8.py
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

class FeatureExtractor:
    def __init__(self, batch_size=16):
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        
        # Modèle GoogLeNet
        from torchvision.models import googlenet, GoogLeNet_Weights
        weights = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=weights, aux_logits=True)
        
        # On garde les couches de convolution et on ajoute un pooling global
        self.model = nn.Sequential(*list(model.children())[:-1], 
                                   nn.AdaptiveAvgPool2d((1, 1)), 
                                   nn.Flatten(1))
        self.model.eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_video(self, path, fps_target=1):
        cap = cv2.VideoCapture(str(path))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps_in / fps_target))
        
        frames_batch, all_features = [], []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # On ne traite que la frame au bon intervalle
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_batch.append(self.transform(img))
                
                if len(frames_batch) >= self.batch_size:
                    all_features.append(self._run_model(frames_batch))
                    frames_batch = []
        
        if frames_batch:
            all_features.append(self._run_model(frames_batch))
            
        cap.release()
        return np.vstack(all_features) if all_features else None

    def _run_model(self, batch):
        with torch.no_grad():
            return self.model(torch.stack(batch)).cpu().numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="dataset_cpu/features.h5")
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    extractor = FeatureExtractor()
    video_paths = sorted(Path(args.video_folder).glob("v*.mp4")) # On cherche les v1, v2...
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with h5py.File(args.output_file, "w") as hf:
        for v_path in tqdm(video_paths, desc="Vidéos"):
            features = extractor.process_video(v_path, args.fps)
            if features is not None:
                hf.create_dataset(v_path.stem, data=features)
                print(f" ✅ {v_path.name} : {features.shape}")

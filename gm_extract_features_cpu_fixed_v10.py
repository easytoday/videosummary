# scripts/gm_extract_features_cpu_fixed_v10.py
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
        
        # Chargement du modèle
        from torchvision.models import googlenet, GoogLeNet_Weights
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT, aux_logits=True)
        
        # On garde le backbone + pooling + flatten
        self.model = nn.Sequential(*list(model.children())[:-1], 
                                   nn.AdaptiveAvgPool2d((1, 1)), 
                                   nn.Flatten(1))
        self.model.eval().to(self.device)
        
        # NOTE : On garde uniquement Resize et Normalize ici.
        # La conversion ToTensor est faite manuellement pour éviter le bug numpy.
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
            
            # On ne prend que les frames à l'intervalle voulu
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
                # 1. BGR vers RGB (OpenCV vers standard couleur)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Conversion en Image PIL pour le redimensionnement propre
                img_pil = Image.fromarray(frame_rgb)
                img_resized = self.resize_transform(img_pil)
                
                # 3. CONVERSION MANUELLE (Contournement du bug numpy/torch)
                # On convertit en array numpy float32
                img_np = np.array(img_resized, dtype=np.float32)
                
                # On normalise entre 0 et 1 (division par 255)
                img_np /= 255.0
                
                # On convertit en Tensor PyTorch
                tensor = torch.from_numpy(img_np)
                
                # On change l'ordre des dimensions : (H, W, C) -> (C, H, W)
                tensor = tensor.permute(2, 0, 1)
                
                # 4. Normalisation finale (ImageNet stats)
                tensor = self.normalize_transform(tensor)
                
                frames_batch.append(tensor)
                
                # Traitement par lot
                if len(frames_batch) >= self.batch_size:
                    all_features.append(self._run_model(frames_batch))
                    frames_batch = []
        
        # Traiter le reste du dernier batch
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
    # Arguments complets réintégrés
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset_cpu")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()

    # Initialisation
    extractor = FeatureExtractor(batch_size=args.batch_size)
    
    # Recherche des fichiers simplifiés (v1.mp4, etc.)
    video_paths = sorted([p for p in Path(args.video_folder).glob("v*") if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])
    
    # Préparation dossier sortie
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / "features.h5"
    
    # Gestion du --force
    if args.force and hdf5_path.exists():
        os.remove(hdf5_path)
        logger.info(f"♻️  Fichier précédent supprimé : {hdf5_path}")

    # Boucle principale
    with h5py.File(hdf5_path, "a") as hf:
        # On utilise le nom 'features' comme groupe racine si vous voulez, 
        # ou directement les datasets à la racine (plus simple pour lecture ultérieure)
        
        for v_path in tqdm(video_paths, desc="Extraction"):
            # Si déjà traité, on saute
            if v_path.stem in hf:
                continue
                
            features = extractor.process_video(v_path, args.fps)
            
            if features is not None:
                hf.create_dataset(v_path.stem, data=features, compression="gzip")
                logger.info(f"✅ {v_path.name} : {features.shape[0]} frames extraites")
            else:
                logger.warning(f"⚠️ {v_path.name} : Aucune frame extraite")

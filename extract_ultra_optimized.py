# scripts/extract_ultra_optimized.py
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

# Ajouter au début de votre script
import os
import torch

# Optimisations PyTorch pour CPU
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Utiliser MKL si disponible
if torch.backends.mkl.is_available():
    torch.backends.mkl.enabled = True

# Désactiver le gradient pour économiser de la mémoire
torch.set_grad_enabled(False)

# Forcer PyTorch à utiliser CPU
device = torch.device('cpu')

class ParallelVideoProcessor:
    """
    Traite plusieurs vidéos en parallèle sur CPU multicœur
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or os.cpu_count() // 2
        self.model = self._load_optimized_model()
        
    def _load_optimized_model(self):
        """Charge GoogLeNet optimisé pour CPU"""
        model = models.googlenet(pretrained=True)
        
        # Geler toutes les couches sauf la dernière
        for param in model.parameters():
            param.requires_grad = False
        
        # Extraire seulement les couches nécessaires
        layers = list(model.children())[:-1]  # Exclure la dernière couche FC
        model = nn.Sequential(*layers)
        
        # Compiler avec TorchScript pour accélérer
        model.eval()
        model = torch.jit.script(model)
        
        return model
    
    def process_video_parallel(self, video_path, target_fps=1):
        """Traite une vidéo avec parallélisation"""
        # Lire toutes les frames d'un coup (si mémoire suffisante)
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / target_fps))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        # Traiter les frames en parallèle
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Préparer les transformations
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Fonction pour traiter un batch de frames
            def process_batch(batch_frames):
                batch_tensors = []
                for frame in batch_frames:
                    tensor = transform(frame).unsqueeze(0)
                    batch_tensors.append(tensor)
                
                batch = torch.cat(batch_tensors, 0)
                with torch.no_grad():
                    features = self.model(batch)
                    features = features.view(features.size(0), -1)
                
                return features.numpy()
            
            # Diviser en batchs et traiter en parallèle
            batch_size = 32
            results = []
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                future = executor.submit(process_batch, batch)
                results.append(future)
            
            # Collecter les résultats
            all_features = []
            for future in results:
                all_features.append(future.result())
        
        return np.vstack(all_features) if all_features else np.array([])

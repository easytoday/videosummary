import h5py
import numpy as np
import cv2
from tqdm import tqdm
import os
from pathlib import Path # ajouter

class VideoToHDF5Converter:
    """Convertit un dossier de vidéos en HDF5 optimisé"""
    
    def __init__(self, hdf5_path='dataset.h5', feature_extractor=None):
        self.hdf5_path = hdf5_path
        self.feature_extractor = feature_extractor
        
    def create_hdf5_dataset(self, video_folder, target_fps=2):
        """
        Crée un dataset HDF5 à partir d'un dossier de vidéos
        
        Args:
            video_folder: dossier contenant les vidéos
            target_fps: fps cible pour l'extraction des frames
        """
        # Ajout 
        # video folder
        #cwd = Path.cwd()
        #parent = cwd.parent
        #video_folder = os.path.join(parent, "court")

        # Liste des vidéos
        video_files = [f for f in os.listdir(video_folder) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Trouvé {len(video_files)} vidéos")
        
        with h5py.File(self.hdf5_path, 'w') as hf:
            # Groupe principal pour les vidéos
            videos_group = hf.create_group('videos')
            metadata_group = hf.create_group('metadata')
            
            # Stocker les métadonnées
            video_names = []
            video_lengths = []
            
            for idx, video_file in enumerate(tqdm(video_files, desc="Processing videos")):
                video_path = os.path.join(video_folder, video_file)
                video_name = os.path.splitext(video_file)[0]
                
                # Extraction des features
                features, metadata = self.extract_video_features(
                    video_path, target_fps
                )
                
                if features is not None:
                    # Créer un groupe pour cette vidéo
                    vid_group = videos_group.create_group(video_name)
                    
                    # Stocker les features avec compression
                    vid_group.create_dataset(
                        'features',
                        data=features,
                        dtype='float32',
                        compression='gzip',  # Compression pour économiser l'espace
                        compression_opts=9    # Niveau de compression (1-9)
                    )
                    
                    # Stocker les métadonnées
                    vid_group.create_dataset('fps', data=target_fps)
                    vid_group.create_dataset('original_fps', data=metadata['original_fps'])
                    vid_group.create_dataset('duration', data=metadata['duration'])
                    vid_group.create_dataset('resolution', data=metadata['resolution'])
                    vid_group.create_dataset('num_frames', data=len(features))
                    
                    # Pour un accès rapide aux indices temporels
                    frame_timestamps = np.arange(len(features)) / target_fps
                    vid_group.create_dataset('timestamps', data=frame_timestamps)
                    
                    # Mettre à jour les métadonnées globales
                    video_names.append(video_name.encode('utf-8'))  # HDF5 nécessite bytes
                    video_lengths.append(len(features))
            
            # Stocker les métadonnées globales
            metadata_group.create_dataset('video_names', data=video_names)
            metadata_group.create_dataset('video_lengths', data=video_lengths)
            metadata_group.create_dataset('total_videos', data=len(video_names))
            
            # Ajouter des attributs descriptifs
            hf.attrs['dataset_name'] = 'MyVideoDataset'
            hf.attrs['created_date'] = str(datetime.now())
            hf.attrs['feature_dim'] = features.shape[1] if len(features) > 0 else 0
            hf.attrs['target_fps'] = target_fps
            
        print(f"Dataset HDF5 créé : {self.hdf5_path}")
        print(f"Taille du fichier : {os.path.getsize(self.hdf5_path) / (1024**3):.2f} GB")
        
    def extract_video_features(self, video_path, target_fps):
        """Extrait les features d'une vidéo"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Impossible d'ouvrir {video_path}")
                return None, None
            
            # Métadonnées de la vidéo
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calcul de l'intervalle d'échantillonnage
            frame_interval = int(original_fps / target_fps)
            
            # Liste pour stocker les features
            all_features = []
            
            # Pour accélérer, vous pouvez utiliser batch processing
            frames_batch = []
            batch_size = 32
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Échantillonnage
                if frame_count % frame_interval == 0:
                    # Prétraitement de l'image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    frame_normalized = frame_resized / 255.0
                    
                    frames_batch.append(frame_normalized)
                    
                    # Extraire les features quand le batch est plein
                    if len(frames_batch) == batch_size:
                        batch_features = self.extract_batch_features(frames_batch)
                        all_features.extend(batch_features)
                        frames_batch = []
                
                frame_count += 1
            
            # Extraire les features du dernier batch
            if frames_batch:
                batch_features = self.extract_batch_features(frames_batch)
                all_features.extend(batch_features)
            
            cap.release()
            
            if len(all_features) == 0:
                return None, None
            
            # Convertir en numpy array
            features_array = np.array(all_features, dtype=np.float32)
            
            metadata = {
                'original_fps': original_fps,
                'duration': duration,
                'resolution': [height, width],
                'total_frames_original': total_frames,
                'total_frames_sampled': len(features_array)
            }
            
            return features_array, metadata
            
        except Exception as e:
            print(f"Erreur avec {video_path}: {e}")
            return None, None
    
    def extract_batch_features(self, frames_batch):
        """Extrait les features d'un batch de frames"""
        # Ici, utilisez votre modèle d'extraction de features
        # Exemple avec un modèle pré-entraîné
        frames_array = np.array(frames_batch)
        # Simulé - à remplacer par votre vrai extracteur
        return [np.random.randn(1024) for _ in range(len(frames_batch))]

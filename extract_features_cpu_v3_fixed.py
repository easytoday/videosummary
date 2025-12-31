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
    """
    Extracteur GoogLeNet optimisé pour CPU
    Utilise des techniques pour accélérer l'extraction sans GPU
    """

    def __init__(self, batch_size=8):
        self.device = torch.device("cpu")
        self.batch_size = batch_size

        logger.info("Initialisation de GoogLeNet sur CPU...")

        # Désactiver les gradients pour économiser de la mémoire
        torch.set_grad_enabled(False)

        # Charger GoogLeNet avec les poids ImageNet
        from torchvision.models import googlenet, GoogLeNet_Weights

        self.model = googlenet(
            weights=GoogLeNet_Weights.DEFAULT,
            aux_logits=True  # CORRECTION: True pour les poids pré-entraînés
        )

        # Modifier pour extraire les features (couche avant-dernière)
        self.model = self._create_feature_extractor(self.model)

        # Passer en mode évaluation
        self.model.eval()

        # Déplacer sur CPU
        self.model.to(self.device)

        # Transformations (attendent une PIL Image)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Taille fixe pour GoogLeNet
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"✅ GoogLeNet initialisé sur CPU | Batch size: {batch_size}")

    def _create_feature_extractor(self, model):
        """
        Crée un extracteur de features à partir de GoogLeNet
        Retourne un module qui renvoie un vecteur par image
        """
        # Utiliser le backbone puis un AdaptiveAvgPool2d pour assurer la forme (B, C)
        backbone = nn.Sequential(*list(model.children())[:-1])
        extractor = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        return extractor

    def extract_features_from_video_optimized(self, video_path, target_fps=1):
        """
        Version optimisée pour CPU avec batch processing intelligent
        Retourne (features: np.ndarray shape (n_frames, feature_dim), metadata: dict)
        """
        logger.info(f"📹 Traitement de: {os.path.basename(video_path)}")

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir: {video_path}")
            return None, None

        # Métadonnées de la vidéo
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / original_fps if original_fps > 0 else 0

        # Ajuster le FPS cible si nécessaire
        if original_fps > 0 and original_fps < target_fps:
            logger.warning(
                f"⚠️  FPS original ({original_fps}) < FPS cible ({target_fps})"
            )
            target_fps = int(original_fps)

        # Calculer l'intervalle d'échantillonnage (garder au moins 1)
        frame_interval = max(1, int((original_fps or target_fps) / max(1, target_fps)))

        # Pré-allouer des listes pour les batchs
        frames_batch = []
        features_list = []

        # Statistiques
        start_time = time.time()
        processed_frames = 0
        extracted_frames = 0

        # Sauter les vidéos problématiques après trop d'erreurs
        consecutive_errors = 0
        max_consecutive_errors = 10

        # Barre de progression
        pbar_total = total_frames if total_frames > 0 else None
        pbar = tqdm(
            total=pbar_total,
            desc=f"{os.path.basename(video_path)[:20]:20}",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Échantillonnage des frames
            if frame_idx % frame_interval == 0:
                # Vérifications de robustesse avant toute conversion
                if frame is None:
                    logger.warning(f"Frame {frame_idx} lue comme None, ignorée")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Frame doit être HxWxC
                if not hasattr(frame, "shape") or frame.ndim != 3:
                    logger.warning(f"Frame {frame_idx} invalide (ndim != 3), ignorée")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                h, w, c = frame.shape

                # Vérification explicite pour les frames 1x1
                if frame.shape == (1, 1, 3):
                    logger.warning(f"Frame {frame_idx} ignorée (forme 1x1x3 explicite)")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                if c != 3:
                    logger.warning(f"Frame {frame_idx} ignorée (canaux={c}), attendu 3")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Filtrer les très petites images
                if h <= 1 or w <= 1:
                    logger.warning(f"Frame {frame_idx} ignorée (shape={frame.shape})")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # CORRECTION: Vérifier et convertir le dtype de manière robuste
                try:
                    # Récupérer le dtype de manière safe - CORRECTION CRITIQUE
                    frame_dtype = None
                    if hasattr(frame, 'dtype'):
                        frame_dtype = frame.dtype
                    elif isinstance(frame, np.ndarray):
                        frame_dtype = frame.dtype
                    else:
                        logger.warning(f"Frame {frame_idx}: objet sans dtype, tentative de conversion directe")
                        try:
                            frame = np.array(frame, dtype=np.uint8)
                            frame_dtype = frame.dtype
                        except Exception as e:
                            logger.warning(f"Frame {frame_idx} conversion échouée: {e}")
                            consecutive_errors += 1
                            if consecutive_errors > max_consecutive_errors:
                                logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt")
                                break
                            pbar.update(1)
                            frame_idx += 1
                            continue
                    
                    # Si le dtype n'est pas uint8, convertir
                    if frame_dtype != np.uint8:
                        try:
                            if isinstance(frame, np.ndarray):
                                # Conversion simple pour les ndarray
                                frame = frame.astype(np.uint8)
                                logger.info(f"Frame {frame_idx} convertie de {frame_dtype} vers uint8")
                            else:
                                # Pour les autres types, utiliser array()
                                frame = np.array(frame, dtype=np.uint8)
                                logger.info(f"Frame {frame_idx} convertie de {type(frame).__name__} vers uint8")
                        except Exception as e:
                            logger.warning(f"Frame {frame_idx} conversion dtype échouée: {e}")
                            consecutive_errors += 1
                            if consecutive_errors > max_consecutive_errors:
                                logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt")
                                break
                            pbar.update(1)
                            frame_idx += 1
                            continue
                            
                except Exception as e:
                    logger.warning(f"Erreur vérification dtype frame {frame_idx}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Convertir BGR à RGB
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.warning(f"Erreur conversion BGR->RGB frame {frame_idx}: {e}")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Vérification supplémentaire après conversion
                if frame_rgb.shape[0] <= 1 or frame_rgb.shape[1] <= 1:
                    logger.warning(f"Frame {frame_idx} ignorée après conversion (shape={frame_rgb.shape})")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Conversion vers PIL puis transform
                try:
                    image = Image.fromarray(frame_rgb)
                    tensor = self.transform(image)
                    frames_batch.append(tensor)
                    consecutive_errors = 0  # Réinitialisation après succès
                except Exception as e:
                    logger.warning(f"Erreur transformation frame {frame_idx}: {e}")
                    logger.error(f"Détails frame: shape={frame_rgb.shape if 'frame_rgb' in locals() else 'N/A'}, dtype={frame_rgb.dtype if 'frame_rgb' in locals() else 'N/A'}")
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du traitement de cette vidéo")
                        break
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Traiter le batch quand il est plein
                if len(frames_batch) >= self.batch_size:
                    batch_features = self._process_batch(frames_batch)
                    if batch_features is not None and batch_features.size:
                        features_list.append(batch_features)
                        extracted_frames += batch_features.shape[0]
                    frames_batch = []

            pbar.update(1)
            frame_idx += 1
            processed_frames += 1

        # Traiter le dernier batch
        if frames_batch:
            batch_features = self._process_batch(frames_batch)
            if batch_features is not None and batch_features.size:
                features_list.append(batch_features)
                extracted_frames += batch_features.shape[0]

        pbar.close()
        cap.release()

        # Calculer le temps d'extraction
        extraction_time = time.time() - start_time

        if not features_list:
            logger.error(f"❌ Aucune feature extraite de: {video_path}")
            return None, None

        # Concaténer toutes les features
        try:
            all_features = np.vstack(features_list)
        except Exception as e:
            logger.error(f"Erreur concaténation features: {e}")
            return None, None

        logger.info(
            f"✅ Extraction réussie: {all_features.shape[0]} frames en {extraction_time:.1f}s "
            f"({all_features.shape[0]/extraction_time:.1f} fps)"
        )

        # Métadonnées
        metadata = {
            "video_path": video_path,
            "original_fps": float(original_fps),
            "target_fps": int(target_fps),
            "original_frames": int(total_frames),
            "extracted_frames": int(all_features.shape[0]),
            "duration": float(duration),
            "resolution": f"{width}x{height}",
            "feature_dim": int(all_features.shape[1]),
            "extraction_time_seconds": float(extraction_time),
            "extraction_speed_fps": (
                float(all_features.shape[0] / extraction_time)
                if extraction_time > 0
                else 0.0
            ),
            "frame_interval": int(frame_interval),
            "batch_size_used": int(self.batch_size),
        }

        return all_features.astype(np.float32), metadata

    def _process_batch(self, frames_batch):
        """Traite un batch de frames de manière optimisée"""
        try:
            # Convertir en tenseur (B, C, H, W)
            batch_tensor = torch.stack(frames_batch).to(self.device)

            # Forward pass
            with torch.no_grad():
                features = self.model(batch_tensor)
                # Assurer une forme 2D (batch, feature_dim)
                features = torch.flatten(features, start_dim=1)

            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Erreur lors du traitement du batch: {e}")
            return np.array([])


class VideoDatasetProcessorCPU:
    """
    Processeur de dataset optimisé pour CPU
    Gère l'extraction et le stockage HDF5
    """

    def __init__(self, output_dir="./dataset_cpu", resume=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

        # Initialiser l'extracteur avec un batch size adapté au CPU
        self.extractor = CPUOptimizedGoogLeNetExtractor(batch_size=16)

        # Fichier de checkpoint pour reprendre en cas d'arrêt
        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def get_video_files(self, video_folder):
        """Récupère tous les fichiers vidéo d'un dossier"""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(video_folder).glob(f"*{ext}"))
            video_files.extend(Path(video_folder).glob(f"*{ext.upper()}"))

        return sorted(video_files)

    def load_checkpoint(self):
        """Charge l'état d'avancement"""
        if not self.checkpoint_file.exists():
            return {"processed_videos": [], "failed_videos": []}

        import json

        with open(self.checkpoint_file, "r") as f:
            return json.load(f)

    def save_checkpoint(self, checkpoint):
        """Sauvegarde l'état d'avancement"""
        import json

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def process_videos_to_hdf5(
        self, video_folder, target_fps=1, hdf5_name=None, force=False
    ):
        """
        Traite toutes les vidéos et crée un HDF5 avec les features
        """
        video_files = self.get_video_files(video_folder)

        if not video_files:
            logger.error(f"❌ Aucune vidéo trouvée dans: {video_folder}")
            return None

        logger.info(f"🎬 {len(video_files)} vidéos trouvées")

        # Gestion de la reprise
        checkpoint = self.load_checkpoint()
        processed_videos = set(checkpoint.get("processed_videos", []))

        # Créer le fichier HDF5
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if hdf5_name:
            hdf5_filename = hdf5_name
        else:
            hdf5_filename = f"googlenet_features_cpu_{timestamp}.h5"

        hdf5_path = self.output_dir / hdf5_filename

        # Si le fichier existe et que --force est demandé, le supprimer d'abord
        if hdf5_path.exists() and force:
            logger.info(f"Supprimer ancien fichier HDF5: {hdf5_path}")
            os.remove(hdf5_path)

        # Si le fichier existe et qu'on ne demande pas la reprise, refuser
        if hdf5_path.exists() and not self.resume:
            logger.error(
                f"⚠️  {hdf5_path} existe déjà. Utilisez --force pour le recréer ou --resume pour reprendre."
            )
            return None

        total_start_time = time.time()

        # Ouvrir en mode append/création selon la situation
        mode = "a" if hdf5_path.exists() and self.resume else "w"

        with h5py.File(hdf5_path, mode) as hf:
            # Initialiser les groupes si nouveau fichier
            if "features" not in hf:
                features_group = hf.create_group("features")
            else:
                features_group = hf["features"]

            if "metadata" not in hf:
                metadata_group = hf.create_group("metadata")
                
                metadata_group.create_dataset(
                    "video_ids",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

                metadata_group.create_dataset(
                    "video_names",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

                metadata_group.create_dataset(
                    "frame_counts",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="i4",
                )
            else:
                metadata_group = hf["metadata"]

            # Compter les vidéos déjà traitées
            existing_videos = list(features_group.keys())
            logger.info(f"📊 {len(existing_videos)} vidéos déjà dans le HDF5")

            # Traiter chaque vidéo
            successful_videos = []
            failed_videos = []

            for i, video_path in enumerate(video_files, 1):
                video_name = video_path.stem
                video_id = f"video_{i:03d}"

                # Vérifier si déjà traitée (par id ou nom)
                if video_id in existing_videos or video_name in processed_videos:
                    logger.info(f"⏭️  Déjà traitée: {video_name}")
                    continue

                logger.info(f"\n{'='*50}")
                logger.info(f"📼 Vidéo {i}/{len(video_files)}: {video_name}")
                logger.info(f"{'='*50}")

                try:
                    # Extraire les features
                    features, metadata = (
                        self.extractor.extract_features_from_video_optimized(
                            str(video_path), target_fps
                        )
                    )

                    if features is None:
                        logger.error(f"❌ Échec extraction: {video_name}")
                        failed_videos.append(video_name)
                        checkpoint["failed_videos"].append(video_name)
                        self.save_checkpoint(checkpoint)
                        continue

                    # Stocker dans HDF5
                    vid_group = features_group.create_group(video_id)

                    # Stocker les features avec compression
                    vid_group.create_dataset(
                        "features",
                        data=features,
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(
                            (min(1024, features.shape[0]), features.shape[1])
                            if features.shape[0] > 0
                            else (1, features.shape[1])
                        ),
                    )

                    # Stocker les métadonnées dans des datasets
                    for key, value in metadata.items():
                        try:
                            if isinstance(value, str):
                                vid_group.create_dataset(
                                    key,
                                    data=np.array(
                                        [value],
                                        dtype=h5py.string_dtype(encoding="utf-8"),
                                    ),
                                )
                            elif isinstance(value, (int, float)):
                                vid_group.create_dataset(key, data=np.array([value]))
                            else:
                                vid_group.create_dataset(key, data=value)
                        except Exception:
                            logger.debug(
                                f"Impossible de stocker metadata {key} pour {video_name}"
                            )

                    # Mettre à jour les métadonnées globales
                    self._update_global_metadata(
                        metadata_group, video_id, video_name, len(features)
                    )

                    # Marquer comme traitée
                    successful_videos.append(video_name)
                    processed_videos.add(video_name)
                    checkpoint["processed_videos"] = list(processed_videos)

                    # Sauvegarder le checkpoint
                    self.save_checkpoint(checkpoint)

                    # Estimation du temps restant
                    elapsed = time.time() - total_start_time
                    avg_time_per_video = (
                        elapsed / len(successful_videos) if successful_videos else 0
                    )
                    remaining = avg_time_per_video * (len(video_files) - i)

                    logger.info(f"⏱️  Temps estimé restant: {remaining/60:.1f} minutes")

                except Exception as e:
                    logger.error(f"❌ Erreur avec {video_name}")
                    logger.error(traceback.format_exc())  # CORRECTION: format_exc() au lieu de format_exec()
                    failed_videos.append(video_name)
                    checkpoint["failed_videos"].append(video_name)
                    self.save_checkpoint(checkpoint)
                    continue

            # Finaliser les statistiques
            total_time = time.time() - total_start_time

            # Calculer les statistiques finales
            all_frame_counts = []
            total_size_bytes = 0
            for vid_id in features_group.keys():
                features = features_group[f"{vid_id}/features"]
                all_frame_counts.append(features.shape[0])
                total_size_bytes += features.size * features.dtype.itemsize

            if all_frame_counts:
                total_frames = sum(all_frame_counts)
                total_size_mb = total_size_bytes / (1024 * 1024)

                # Ajouter/mettre à jour les statistiques finales
                if "total_videos" in metadata_group:
                    del metadata_group["total_videos"]
                metadata_group.create_dataset(
                    "total_videos", data=np.array([len(features_group)], dtype=np.int32)
                )

                if "total_frames" in metadata_group:
                    del metadata_group["total_frames"]
                metadata_group.create_dataset(
                    "total_frames", data=np.array([total_frames], dtype=np.int32)
                )

                if "feature_dim" in metadata_group:
                    del metadata_group["feature_dim"]
                metadata_group.create_dataset(
                    "feature_dim", data=np.array([1024], dtype=np.int32)
                )

                if "avg_frames_per_video" in metadata_group:
                    del metadata_group["avg_frames_per_video"]
                metadata_group.create_dataset(
                    "avg_frames_per_video",
                    data=np.array(
                        [total_frames / len(features_group)], dtype=np.float32
                    ),
                )

                if "extraction_time_hours" in metadata_group:
                    del metadata_group["extraction_time_hours"]
                metadata_group.create_dataset(
                    "extraction_time_hours",
                    data=np.array([total_time / 3600], dtype=np.float32),
                )

                logger.info(f"\n{'='*50}")
                logger.info("📊 STATISTIQUES FINALES")
                logger.info(f"{'='*50}")
                logger.info(f"✅ Vidéos réussies: {len(successful_videos)}")
                logger.info(f"❌ Vidéos échouées: {len(failed_videos)}")
                logger.info(f"📁 Frames totales: {total_frames}")
                logger.info(f"💾 Taille features: {total_size_mb:.2f} MB")
                logger.info(f"⏱️  Temps total: {total_time/3600:.2f} heures")
                logger.info(f"🚀 Vitesse moyenne: {total_frames/total_time:.1f} fps")

            # Sauvegarder la liste des échecs
            if failed_videos:
                failed_path = self.output_dir / "failed_videos.txt"
                with open(failed_path, "w") as f:
                    for vid in failed_videos:
                        f.write(f"{vid}\n")
                logger.info(f"📝 Liste des échecs sauvegardée: {failed_path}")

        return hdf5_path

    def _update_global_metadata(
        self, metadata_group, video_id, video_name, frame_count
    ):
        """Met à jour les métadonnées globales"""
        # Étendre les datasets
        for dset_name, value in [
            ("video_ids", video_id),
            ("video_names", video_name),
            ("frame_counts", frame_count),
        ]:

            if dset_name in metadata_group:
                dset = metadata_group[dset_name]
                dset.resize((dset.shape[0] + 1,))
                if isinstance(value, str):
                    dset[-1] = value
                else:
                    dset[-1] = value
            else:
                if isinstance(value, str):
                    data = np.array([value], dtype=h5py.string_dtype(encoding="utf-8"))
                elif isinstance(value, int):
                    data = np.array([value], dtype=np.int32)
                elif isinstance(value, float):
                    data = np.array([value], dtype=np.float32)
                else:
                    raise TypeError(f"Type non supporté pour metadata: {type(value)}")

                metadata_group.create_dataset(
                    dset_name,
                    data=data,
                    maxshape=(None,),
                )

    def estimate_extraction_time(self, video_folder, target_fps=1):
        """Estime le temps d'extraction total"""
        video_files = self.get_video_files(video_folder)

        if not video_files:
            return 0

        # Temps moyen par minute de vidéo (empirique, basé sur des tests)
        avg_time_per_frame_seconds = 0.3

        total_frames_estimate = 0
        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                duration = frame_count / fps if fps > 0 else 0
                estimated_frames = duration * target_fps
                total_frames_estimate += estimated_frames
                cap.release()

        total_time_seconds = total_frames_estimate * avg_time_per_frame_seconds

        logger.info(f"⏱️  Estimation du temps d'extraction:")
        logger.info(f"   - Vidéos: {len(video_files)}")
        logger.info(f"   - Frames estimées: {total_frames_estimate:.0f}")
        logger.info(f"   - Temps estimé: {total_time_seconds/3600:.1f} heures")
        logger.info(
            f"   - (~{total_time_seconds/60/len(video_files):.1f} minutes/vidéo)"
        )

        return total_time_seconds


# Script principal avec interface en ligne de commande
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extraction de features GoogLeNet optimisée pour CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python extract_cpu.py --video_folder ./videos --fps 2
  python extract_cpu.py --video_folder ./videos --fps 1 --resume
  python extract_cpu.py --video_folder ./videos --estimate_only
        """,
    )

    parser.add_argument(
        "--video_folder", type=str, required=True, help="Dossier contenant les vidéos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset_cpu",
        help="Dossier de sortie pour le HDF5",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="FPS cible pour extraction (défaut: 1 pour CPU)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Taille des batchs (défaut: 16)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre l'extraction là où elle s'est arrêtée",
    )
    parser.add_argument(
        "--estimate_only",
        action="store_true",
        help="Estimer seulement le temps sans extraire",
    )
    parser.add_argument(
        "--hdf5_name", type=str, help="Nom personnalisé pour le fichier HDF5"
    )

    parser.add_argument(
        "--force", action="store_true", help="Forcer recréation HDF5"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🎬 EXTRACTION GOOGLENET - VERSION CPU OPTIMISÉE")
    print("=" * 60)
    print(f"📁 Dossier vidéos: {args.video_folder}")
    print(f"📂 Dossier sortie: {args.output_dir}")
    print(f"🎯 FPS cible: {args.fps}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"🔁 Reprise: {'Oui' if args.resume else 'Non'}")
    print("=" * 60)

    # Initialiser le processeur
    processor = VideoDatasetProcessorCPU(output_dir=args.output_dir, resume=args.resume)

    # Ajuster le batch size
    processor.extractor.batch_size = args.batch_size

    # Estimation du temps
    if args.estimate_only:
        processor.estimate_extraction_time(args.video_folder, args.fps)
        print("\n💡 Astuce: Pour accélérer l'extraction:")
        print("   - Utilisez --fps 1 au lieu de 2")
        print("   - Traitez les vidéos par lots")
        print("   - Lancez l'extraction pendant la nuit")
    else:
        # Confirmation
        confirm = (
            input(f"\n⚠️  L'extraction sur CPU peut être longue. Continuer? [O/n]: ")
            .strip()
            .lower()
        )

        if confirm in ["", "o", "oui", "y", "yes"]:
            print("\n🚀 Lancement de l'extraction...")
            print("📝 Les logs détaillés sont sauvegardés dans 'extraction.log'")

            # Lancer l'extraction
            hdf5_path = processor.process_videos_to_hdf5(
                video_folder=args.video_folder,
                target_fps=args.fps,
                hdf5_name=args.hdf5_name,
                force=args.force,
            )

            if hdf5_path:
                print(f"\n✅ Extraction terminée avec succès!")
                print(f"📁 HDF5 créé: {hdf5_path}")

                # Afficher des informations sur l'utilisation
                import psutil

                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                print(f"\n💻 Utilisation système finale:")
                print(f"   CPU: {cpu_percent}%")
                print(f"   Mémoire: {memory.percent}%")
        else:
            print("❌ Extraction annulée.")
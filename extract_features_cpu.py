# scripts/extract_features_cpu.py
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

        # Charger GoogLeNet avec des paramètres optimisés pour CPU
        self.model = models.googlenet(pretrained=True)

        # Modifier pour extraire les features (couche avant-dernière)
        self.model = self._create_feature_extractor(self.model)

        # Passer en mode évaluation
        self.model.eval()

        # Déplacer sur CPU
        self.model.to(self.device)

        # Transformations optimisées (SANS ToPILImage - cause du bug)
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
        Retourne les features de la couche penultimate (1024 dimensions)
        """
        # Prendre toutes les couches sauf la dernière (fc)
        layers = list(model.children())[:-1]
        return nn.Sequential(*layers)

    def extract_features_from_video_optimized(self, video_path, target_fps=1):
        """
        Version optimisée pour CPU avec batch processing intelligent
        """
        logger.info(f"📹 Traitement de: {os.path.basename(video_path)}")

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Impossible d'ouvrir: {video_path}")
            return None, None

        # Métadonnées de la vidéo
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # Ajuster le FPS cible si nécessaire
        if original_fps < target_fps:
            logger.warning(
                f"⚠️  FPS original ({original_fps}) < FPS cible ({target_fps})"
            )
            target_fps = original_fps

        # Calculer l'intervalle d'échantillonnage
        frame_interval = max(1, int(original_fps / target_fps))

        # Pré-allouer des listes pour les batchs
        frames_batch = []
        features_list = []

        # Statistiques
        start_time = time.time()
        processed_frames = 0

        # Barre de progression
        pbar = tqdm(
            total=total_frames,
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
                # Convertir BGR à RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convertir en PIL Image
                try:
                    # Assurer que c'est un uint8
                    if frame_rgb.dtype != np.uint8:
                        frame_rgb = frame_rgb.astype(np.uint8)

                    # Conversion directe sans ToPILImage dans le transform
                    image = Image.fromarray(frame_rgb)
                    tensor = self.transform(image)
                    frames_batch.append(tensor)
                except Exception as e:
                    logger.warning(f"Erreur transformation frame {frame_idx}: {e}")
                    pbar.update(1)
                    frame_idx += 1
                    continue

                # Traiter le batch quand il est plein
                if len(frames_batch) >= self.batch_size:
                    batch_features = self._process_batch(frames_batch)
                    features_list.append(batch_features)
                    frames_batch = []  # Réinitialiser

            pbar.update(1)
            frame_idx += 1
            processed_frames += 1

        # Traiter le dernier batch
        if frames_batch:
            batch_features = self._process_batch(frames_batch)
            features_list.append(batch_features)

        pbar.close()
        cap.release()

        # Calculer le temps d'extraction
        extraction_time = time.time() - start_time

        if not features_list:
            logger.error(f"❌ Aucune feature extraite de: {video_path}")
            return None, None

        # Concaténer toutes les features
        all_features = np.vstack(features_list)  # (n_frames, 1024)

        logger.info(
            f"✅ Extraction réussie: {all_features.shape[0]} frames en {extraction_time:.1f}s "
            f"({all_features.shape[0]/extraction_time:.1f} fps)"
        )

        # Métadonnées
        metadata = {
            "video_path": video_path,
            "original_fps": original_fps,
            "target_fps": target_fps,
            "original_frames": total_frames,
            "extracted_frames": all_features.shape[0],
            "duration": duration,
            "resolution": f"{width}x{height}",
            "feature_dim": all_features.shape[1],
            "extraction_time_seconds": extraction_time,
            "extraction_speed_fps": all_features.shape[0] / extraction_time,
            "frame_interval": frame_interval,
            "batch_size_used": self.batch_size,
        }

        return all_features.astype(np.float32), metadata

    def _process_batch(self, frames_batch):
        """Traite un batch de frames de manière optimisée"""
        try:
            # Convertir en tenseur
            batch_tensor = torch.stack(frames_batch).to(self.device)

            # Forward pass
            with torch.no_grad():
                features = self.model(batch_tensor)
                # Reshape: (batch, 1024, 1, 1) -> (batch, 1024)
                features = features.view(features.size(0), -1)

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

    def process_videos_to_hdf5(self, video_folder, target_fps=1, hdf5_name=None):
        """
        Traite toutes les vidéos et crée un HDF5 avec les features
        """
        mode = "w"  # Aziz toujours 'w' si --force

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

        # Mode d'ouverture du HDF5
        mode = "a" if self.resume and hdf5_path.exists() else "w"

        total_start_time = time.time()

        with h5py.File(hdf5_path, mode) as hf:
            if hdf5_path.exists() and not args.force:
                print(f"⚠️  {hdf5_path} existe déjà. Utilisez --force pour recréer.")
                return None

            # Initialiser les groupes si nouveau fichier
            if mode == "w":
                features_group = hf.create_group("features")
                metadata_group = hf.create_group("metadata")

                # Initialiser les datasets pour les métadonnées (sans .attrs pour éviter bug)
                # metadata_group.create_dataset('video_ids', maxshape=(None,),
                #                            dtype=h5py.string_dtype())
                # metadata_group.create_dataset('video_names', maxshape=(None,),
                #                            dtype=h5py.string_dtype())
                # metadata_group.create_dataset('frame_counts', maxshape=(None,), dtype=int)

                # Stocker quelques métadonnées essentielles
                # metadata_group.create_dataset('created_date', data=[str(timestamp)],
                #                            dtype=h5py.string_dtype())
                # metadata_group.create_dataset('extractor', data=['GoogLeNet (CPU optimized)'],
                #                            dtype=h5py.string_dtype())
                # metadata_group.create_dataset('target_fps', data=[float(target_fps)], dtype=float)
                # metadata_group.create_dataset('feature_dim', data=[1024], dtype=int)

                # Initialiser les datasets pour les métadonnées - Aziz correction1 bug
                # Créer d'abord avec une donnée vide, puis on étendra
                # metadata_group.create_dataset('video_ids',
                #                            data=np.array([], dtype=h5py.string_dtype()),
                #                            maxshape=(None,))
                # metadata_group.create_dataset('video_names',
                #                            data=np.array([], dtype=h5py.string_dtype()),
                #                            maxshape=(None,))
                # metadata_group.create_dataset('frame_counts',
                #                            data=np.array([], dtype=int),
                #                            maxshape=(None,))

                # Stocker quelques métadonnées essentielles
                # metadata_group.create_dataset('created_date',
                #                            data=np.array([str(timestamp)], dtype=h5py.string_dtype()))
                # metadata_group.create_dataset('extractor',
                #                            data=np.array(['GoogLeNet (CPU optimized)'], dtype=h5py.string_dtype()))
                # metadata_group.create_dataset('target_fps',
                #                            data=np.array([float(target_fps)], dtype=float))
                # metadata_group.create_dataset('feature_dim',
                #                            data=np.array([1024], dtype=int))

            # Initialiser les groupes si nouveau fichier Aziz correction2 bug
            if mode == "w" or not "features" in hf:
                if "features" in hf:
                    del hf["features"]
                if "metadata" in hf:
                    del hf["metadata"]
                features_group = hf.create_group("features")
                metadata_group = hf.create_group("metadata")
            else:
                features_group = hf["features"]
                metadata_group = hf["metadata"]

                # Créer les datasets avec une donnée initiale fictive qu'on remplacera
                # Utiliser des tableaux numpy au lieu de listes vides
                metadata_group.create_dataset(
                    "video_ids",
                    data=np.array(["temp_id"], dtype=h5py.special_dtype(vlen=str)),
                )
                metadata_group.create_dataset(
                    "video_names",
                    data=np.array(["temp_name"], dtype=h5py.special_dtype(vlen=str)),
                )
                metadata_group.create_dataset(
                    "frame_counts", data=np.array([0], dtype=np.int32)
                )

                # Stocker les métadonnées essentielles
                metadata_group.create_dataset(
                    "created_date",
                    data=np.array([str(timestamp)], dtype=h5py.special_dtype(vlen=str)),
                )
                metadata_group.create_dataset(
                    "extractor",
                    data=np.array(
                        ["GoogLeNet (CPU optimized)"],
                        dtype=h5py.special_dtype(vlen=str),
                    ),
                )
                metadata_group.create_dataset(
                    "target_fps", data=np.array([float(target_fps)], dtype=np.float32)
                )
                metadata_group.create_dataset(
                    "feature_dim", data=np.array([1024], dtype=np.int32)
                )

            # else:
            #     features_group = hf['features']
            #     metadata_group = hf['metadata']

            # Compter les vidéos déjà traitées
            existing_videos = list(features_group.keys())
            logger.info(f"📊 {len(existing_videos)} vidéos déjà dans le HDF5")

            # Traiter chaque vidéo
            successful_videos = []
            failed_videos = []

            for i, video_path in enumerate(video_files, 1):
                video_name = video_path.stem
                video_id = f"video_{i:03d}"

                # Vérifier si déjà traitée
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
                        compression_opts=9,
                        chunks=True,  # Important pour les gros datasets
                    )

                    # Stocker les métadonnées dans des datasets (pas .attrs)
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, np.ndarray)):
                            vid_group.create_dataset(key, data=value)

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
                    logger.error(f"❌ Erreur avec {video_name}: {str(e)}")
                    failed_videos.append(video_name)
                    checkpoint["failed_videos"].append(video_name)
                    self.save_checkpoint(checkpoint)
                    continue

            # Finaliser les statistiques
            total_time = time.time() - total_start_time

            # Calculer les statistiques finales
            all_frame_counts = []
            for vid_id in features_group.keys():
                features = features_group[f"{vid_id}/features"]
                all_frame_counts.append(features.shape[0])

            if all_frame_counts:
                total_frames = sum(all_frame_counts)
                total_size_mb = sum(
                    features_group[f"{vid_id}/features"].size
                    * features_group[f"{vid_id}/features"].dtype.itemsize
                    for vid_id in features_group.keys()
                ) / (1024 * 1024)

                # Ajouter les statistiques finales
                metadata_group.create_dataset(
                    "total_videos", data=[len(features_group)], dtype=int
                )
                metadata_group.create_dataset(
                    "total_frames", data=[total_frames], dtype=int
                )
                metadata_group.create_dataset(
                    "total_size_gb", data=[total_size_mb / 1024], dtype=float
                )
                metadata_group.create_dataset(
                    "avg_frames_per_video",
                    data=[total_frames / len(features_group)],
                    dtype=float,
                )
                metadata_group.create_dataset(
                    "extraction_time_hours", data=[total_time / 3600], dtype=float
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
                dset[-1] = value
            else:
                # Créer le dataset s'il n'existe pas - Aziz
                # metadata_group.create_dataset(
                #    dset_name,
                #    data=[value],
                #    maxshape=(None,),
                #    dtype=h5py.string_dtype() if isinstance(value, str) else int
                # )

                # Créer le dataset s'il n'existe pas - Aziz - bug correction
                if isinstance(value, str):
                    data = np.array([value], dtype=h5py.string_dtype())
                    metadata_group.create_dataset(
                        dset_name, data=data, maxshape=(None,)
                    )
                else:
                    data = np.array([value], dtype=type(value))
                    metadata_group.create_dataset(
                        dset_name, data=data, maxshape=(None,)
                    )

    def estimate_extraction_time(self, video_folder, target_fps=1):
        """Estime le temps d'extraction total"""
        video_files = self.get_video_files(video_folder)

        if not video_files:
            return 0

        # Temps moyen par minute de vidéo (empirique, basé sur des tests)
        avg_time_per_frame_seconds = 0.3  # Optimiste mais réaliste

        total_frames_estimate = 0
        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    )  # Aziz ajout

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

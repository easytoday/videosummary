1. 📦 PRÉPARATION DE L'ENVIRONNEMENT
2. 🎬 ÉTAPE 1 : Organisation des Vidéos
3. 🔍 ÉTAPE 2 : Extraction des Features GoogLeNet
4. ⏸️  Gestion des Points d'Arrêt
5. 🧠 ÉTAPE 3 : Entraînement du Modèle DSN
6. 📊 ÉTAPE 4 : Évaluation et Génération de Résumés
7. 📁 Structure des Fichiers
8. 🚨 Dépannage et FAQ

## preparation de l'environnement
```bash
conda activate projet
python scripts/check_environment.py
```
## Structure des dossiers :
- Créez la structure de base
video_summarization_project/
├── videos/                          # Vos vidéos originales
├── scripts/                         # Tous les scripts
├── datasets/                        # Datasets générés
├── models/                          # Modèles entraînés
├── outputs/                         # Résumés générés
└── logs/                            # Logs d'exécution

# Preparation des videos
** Objectif : Préparer les videos pour l'extraction **

## Préparation des videos
Placer les videos dans le dossier videos:
video_summarization_project/videos/
├── votre_video1.mp4
├── votre_video2.avi
└── ...

## Estimation du temps
``` bash
cd video_summarization_project
python scripts/estimate_time.py --video_folder ./videos --fps 1.0
```
🎬 10 vidéos trouvées
📊 Estimation du temps d'extraction:
   Durée totale vidéo: 85.2 minutes
   Frames à extraire: 5100
   Temps estimé: 3.0 heures


# ETAPE 2 : Extraction des features GooglLeNet
objectif : Extraire les features et les stocker dans HDF5

## Premier lancement
``` bash
cd video_summarization_project
python scripts/extract_features_conda.py \
    --video_folder ./videos \
    --output_dir ./datasets \
    --fps 1.0 \
    --batch_size 8
```

## Suvi de la progression
Pendant l'extraction, suivi de l'extraction
``` bash
cd video_summarization_project
python scripts/extract_features_conda.py \
    --video_folder ./videos \
    --output_dir ./datasets \
    --fps 1.0 \
    --batch_size 8
```
## Suivi de la progression
Pendant l'extraction, suivi de la progression
``` bash
# Dans un autre terminal
tail -f datasets/extraction.log
```

## Structure du HDF5 généré
datasets/
├── googlenet_features_20240115_143022.h5  # Fichier principal
├── extraction.log                         # Logs détaillés
└── checkpoint.json                       # État d'avancement

** Format HDF5: **
/features/video_0001/features    # (n_frames, 1024)
/features/video_0002/features
...
/metadata/video_ids              # Liste des IDs
/metadata/video_names           # Noms originaux

## Vérification après extraction
``` bash
python scripts/verify_hdf5.py --hdf5_file ./datasets/googlenet_features_*.h5
```

# Gestion des points d'arrêts
## Interruption propre
Appuyez sur Ctrl+C. Le script sauvegarde automatiquement :

    - L'état dans checkpoint.json

    - Les features déjà extraites dans le HDF5

## Reprise après interruption
``` bash
python scripts/extract_features_conda.py \
    --video_folder ./videos \
    --output_dir ./datasets \
    --fps 1.0 \
    --batch_size 8 \
    --resume
```
Le script :

    Lit checkpoint.json

    Identifie les vidéos déjà traitées

    Continue avec les vidéos restantes

4.3 État d'Avancement

Pour voir où vous en êtes :
bash

python scripts/check_progress.py --checkpoint_file ./datasets/checkpoint.json

Sortie :
text

📊 État d'avancement :
✅ Traitées : 5/10 vidéos
❌ Échouées : 1 vidéo
⏳ Restantes : 4 vidéos
📅 Dernière mise à jour : 2024-01-15 14:30:22

5. 🧠 ÉTAPE 3 : Entraînement du Modèle DSN
Objectif : Entraîner le Deep Summarization Network
5.1 Préparation des Données d'Entraînement

Divisez votre dataset :
bash

python scripts/split_dataset.py \
    --hdf5_file ./datasets/googlenet_features_*.h5 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15

Résultat :
text

datasets/
├── splits.json                    # Répartition train/val/test
└── googlenet_features_*.h5       # Mêmes features, split dans métadonnées

5.2 Configuration d'Entraînement

Créez un fichier de configuration :
yaml

# configs/training_config.yaml
model:
  feature_dim: 1024
  hidden_dim: 256
  lambda_temporal: 20

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 60
  n_episodes: 5
  
regularization:
  beta1: 0.01      # Poids régularisation pourcentage
  beta2: 0.0001    # Poids régularisation L2
  epsilon: 0.15    # Pourcentage cible de sélection

5.3 Lancement de l'Entraînement
bash

python scripts/train_dsn.py \
    --config configs/training_config.yaml \
    --hdf5_file ./datasets/googlenet_features_*.h5 \
    --output_dir ./models \
    --experiment_name first_training

5.4 Suivi de l'Entraînement

Pendant l'entraînement :
bash

# Suivre les logs
tail -f models/first_training/training.log

# Visualiser les métriques
tensorboard --logdir models/first_training/tensorboard

Fichiers générés :
text

models/first_training/
├── checkpoint_epoch_10.pth       # Checkpoint toutes les 10 époques
├── best_model.pth                # Meilleur modèle
├── training_history.json         # Historique des métriques
├── config.yaml                   # Configuration sauvegardée
└── tensorboard/                  # Logs TensorBoard

5.5 Reprise de l'Entraînement

Pour reprendre un entraînement interrompu :
bash

python scripts/train_dsn.py \
    --config configs/training_config.yaml \
    --hdf5_file ./datasets/googlenet_features_*.h5 \
    --output_dir ./models \
    --experiment_name first_training \
    --resume_from_checkpoint models/first_training/checkpoint_epoch_20.pth

5.6 Early Stopping

L'entraînement s'arrête automatiquement si :

    Pas d'amélioration depuis 10 époques

    Atteint le nombre maximum d'époques (60)

    Vous appuyez sur Ctrl+C

6. 📊 ÉTAPE 4 : Évaluation et Génération de Résumés
6.1 Évaluation sur le Test Set
bash

python scripts/evaluate_model.py \
    --model_path ./models/first_training/best_model.pth \
    --hdf5_file ./datasets/googlenet_features_*.h5 \
    --split test \
    --output_dir ./outputs/evaluation

Métriques calculées :

    F-score (si annotations disponibles)

    R_div (diversité)

    R_rep (représentativité)

6.2 Génération de Résumés

Pour une vidéo spécifique :
bash

python scripts/generate_summary.py \
    --model_path ./models/first_training/best_model.pth \
    --video_path ./videos/votre_video.mp4 \
    --output_dir ./outputs/summaries \
    --summary_percentage 0.15

6.3 Visualisation des Résultats
bash

python scripts/visualize_summary.py \
    --summary_file ./outputs/summaries/votre_video_summary.h5 \
    --video_path ./videos/votre_video.mp4 \
    --output_image ./outputs/visualizations/summary_visualization.png

Fichiers générés :
text

outputs/summaries/
├── votre_video_summary.h5          # Résumé structuré
├── votre_video_scores.npy          # Scores d'importance
└── votre_video_selected_frames.txt # Indices des frames sélectionnées

outputs/visualizations/
└── summary_visualization.png       # Graphique visuel

7. 📁 Structure Complète des Fichiers
text

video_summarization_project/
│
├── videos/                          # VIDÉOS SOURCES
│   ├── raw/                        # Vidéos originales
│   └── processed/                  # Vidéos prétraitées (optionnel)
│
├── scripts/                         # TOUS LES SCRIPTS
│   ├── 01_environment_check.py
│   ├── 02_extract_features.py
│   ├── 03_train_model.py
│   ├── 04_evaluate.py
│   ├── 05_generate_summary.py
│   ├── 06_visualize.py
│   ├── utils/                      # Fonctions utilitaires
│   └── configs/                    # Fichiers de configuration
│
├── datasets/                       # DONNÉES GÉNÉRÉES
│   ├── raw/                        # Features brutes
│   ├── processed/                  # Données préparées
│   ├── splits.json                 # Division train/val/test
│   └── checkpoints/                # Points de reprise extraction
│
├── models/                         # MODÈLES ENTRÂINÉS
│   ├── experiment_1/
│   │   ├── best_model.pth
│   │   ├── training_history.json
│   │   └── checkpoints/
│   └── experiment_2/
│
├── outputs/                        # RÉSULTATS
│   ├── evaluations/                # Métriques d'évaluation
│   ├── summaries/                  # Résumés générés
│   └── visualizations/             # Graphiques et visualisations
│
├── logs/                           # LOGS D'EXÉCUTION
│   ├── extraction_20240115.log
│   ├── training_20240116.log
│   └── evaluation_20240117.log
│
└── README.md                       # Cette documentation


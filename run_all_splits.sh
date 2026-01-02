#!/bin/bash

# Configuration
DATASET="dataset_cpu/alt_dataset_kts.h5"
SPLITS="datasets/alt_splits_kts.json"
BASE_LOG_DIR="log/alt_run_kts_cv" # CV pour Cross-Validation

echo "Démarrage de la Cross-Validation sur 5 splits..."

for i in {0..4}
do
   echo "------------------------------------------------"
   echo "Lancement de l'entraînement pour le SPLIT $i"
   echo "------------------------------------------------"
   
   # On crée un dossier spécifique : log/alt_run_kts_cv/split_0, split_1, etc.
   CURRENT_SAVE_DIR="${BASE_LOG_DIR}/split_${i}"
   mkdir -p $CURRENT_SAVE_DIR
   
   python3 main.py \
     -d $DATASET \
     -s $SPLITS \
     --split-id $i \
     --save-dir $CURRENT_SAVE_DIR \
     --verbose
     
   echo "Split $i terminé. Résultats sauvegardés dans $CURRENT_SAVE_DIR"
done

echo "Toute la Cross-Validation est terminée !"

#!/bin/bash

# Configuration
DATASET="dataset_original/eccv16_dataset_tvsum_google_pool5.h5"
SPLITS="dataset_original/tvsum_splits.json"
BASE_LOG_DIR="log/eccv16_dataset_tvsum_google_pool5_cv" # cross validation

echo "Démarrage de la Cross-Validation sur 5 splits..."

for i in {0..4}
do
   echo "------------------------------------------------"
   echo "Lancement de l'entraînement pour le SPLIT $i"
   echo "------------------------------------------------"
   
   CURRENT_SAVE_DIR="${BASE_LOG_DIR}/split_${i}"
   mkdir -p $CURRENT_SAVE_DIR
   
   # Ajout de -m tvsum pour corriger l'erreur
   python3 main.py \
     -d $DATASET \
     -s $SPLITS \
     --split-id $i \
     -m tvsum \
     --save-dir $CURRENT_SAVE_DIR \
     --verbose \
     --save-results

   # Vérification si la commande a réussi
   if [ $? -eq 0 ]; then
      echo "Split $i terminé avec succès."
   else
      echo "Erreur lors de l'entraînement du Split $i. Arrêt du script."
      exit 1
   fi
done

echo "Toute la Cross-Validation est terminée !"

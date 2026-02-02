#!/bin/bash
# Script Bash pour exécuter les expériences DSNsup et DR-DSNsup sur les jeux de données SumMe et TVSum

# Déterminer la commande Python appropriée à utiliser
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
elif command -v py &>/dev/null; then
    PYTHON_CMD="py"
else
    echo "ERREUR : Commande Python introuvable. Veuillez installer Python et vous assurer qu'il est dans le PATH."
    exit 1
fi

echo "Utilisation de la commande Python : $PYTHON_CMD"

# Définition des paramètres
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)

# Fonction pour exécuter une expérience spécifique
run_experiment() {
    local dataset=$1
    local split=$2
    local model_name=$3     # "DSNsup" ou "DR-DSNsup"
    local sup_only=$4       # "--sup-only" ou ""
    
    local dataset_file="datasets/eccv16_dataset_${dataset}_google_pool5.h5"
    local split_file="datasets/${dataset}_splits.json"
    local save_dir="log/${model_name}-${dataset}-split${split}"
    local model_path="${save_dir}/model_epoch60.pth.tar"
    
    # Créer le répertoire pour cette expérience
    mkdir -p "$save_dir"
    
    if [ ! -d "$save_dir" ]; then
        echo "ERREUR : Impossible de créer le répertoire ${save_dir}. Vérifiez les permissions."
        return 1
    fi
    
    # Étape 1 : Entraîner le modèle
    echo "Entraînement de ${model_name} sur ${dataset} (Split ${split})"
    $PYTHON_CMD main.py -d "$dataset_file" \
                  -s "$split_file" \
                  -m "$dataset" \
                  --split-id "${split}" \
                  --supervised \
                  ${sup_only} \
                  --save-dir "$save_dir" \
                  --gpu 0 \
                  --verbose
                  
    # Vérifier le résultat de l'entraînement
    if [ $? -ne 0 ]; then
        echo "ERREUR : L'entraînement de ${model_name} sur ${dataset} split ${split} a échoué"
        return 1
    fi
    
    # Étape 2 : Évaluer le modèle (si le checkpoint existe)
    if [ -f "$model_path" ]; then
        echo "Évaluation de ${model_name} sur ${dataset} (Split ${split})"
        $PYTHON_CMD main.py -d "$dataset_file" \
                      -s "$split_file" \
                      -m "$dataset" \
                      --split-id "${split}" \
                      --supervised \
                      ${sup_only} \
                      --save-dir "$save_dir" \
                      --gpu 0 \
                      --evaluate \
                      --resume "$model_path" \
                      --verbose \
                      --save-results
        
        if [ $? -eq 0 ]; then
            echo "Expérience ${model_name} sur ${dataset} split ${split} terminée"
        else
            echo "AVERTISSEMENT : L'évaluation de ${model_name} sur ${dataset} split ${split} a échoué"
        fi
    else
        echo "AVERTISSEMENT : Checkpoint de modèle introuvable : $model_path"
        return 1
    fi
    
    return 0
}

# Fonction pour collecter et calculer les scores F1
collect_scores() {
    local model_name=$1
    local dataset=$2
    
    # Collecter les scores F1 pour chaque split
    local scores=()
    for split in "${splits[@]}"; do
        local log_file="log/${model_name}-${dataset}-split${split}/log_test.txt"
        if [ -f "$log_file" ]; then
            local f1=$(grep -oP "Average F-score \K[0-9.]+(?=%)" "$log_file")
            if [ -n "$f1" ]; then
                scores+=("$f1")
                echo "${model_name} sur ${dataset} (Split ${split}) : Score F1 = ${f1}%"
            else
                echo "${model_name} sur ${dataset} (Split ${split}) : Impossible de récupérer le score F1" >&2
            fi
        else
            echo "${model_name} sur ${dataset} (Split ${split}) : Fichier log introuvable" >&2
        fi
    done
    
    # Calculer la moyenne des scores F1 si des résultats existent
    if [ ${#scores[@]} -gt 0 ]; then
        local sum=0
        for score in "${scores[@]}"; do
            sum=$(echo "$sum + $score" | bc)
        done
        local avg=$(echo "scale=2; $sum / ${#scores[@]}" | bc)
        echo "${model_name} sur ${dataset} : Score F1 moyen = ${avg}% (${#scores[@]}/5 splits)"
    else
        echo "${model_name} sur ${dataset} : Aucun résultat valide" >&2
    fi
}

# Créer un fichier de résumé des résultats
summary_file="supervised_experiment_summary.txt"
echo "Résultats des expériences d'apprentissage supervisé - $(date)" > "$summary_file"
echo "=============================================" >> "$summary_file"

# Exécuter DSNsup sur SumMe et TVSum
echo "=== Début de l'entraînement DSNsup ==="
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DSNsup" "--sup-only"
    done
done

# Exécuter DR-DSNsup sur SumMe et TVSum  
echo "=== Début de l'entraînement DR-DSNsup ==="
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        run_experiment "$dataset" "$split" "DR-DSNsup" ""
    done
done

# Collecter et afficher les résultats
echo ""
echo "=== Résumé des résultats ==="
for dataset in "${datasets[@]}"; do
    collect_scores "DSNsup" "$dataset" | tee -a "$summary_file"
    collect_scores "DR-DSNsup" "$dataset" | tee -a "$summary_file"
done

echo ""
echo "Calcul détaillé des scores F1"
$PYTHON_CMD calculate_f1_scores.py

# Ajouter des informations d'environnement au fichier de résumé pour le débogage
echo -e "\n=== Informations sur l'environnement ===" >> "$summary_file"
echo "Commande Python utilisée : $PYTHON_CMD" >> "$summary_file"
echo "Version de Python : $($PYTHON_CMD --version 2>&1)" >> "$summary_file"
echo "Chemin de Python : $(which $PYTHON_CMD 2>&1)" >> "$summary_file"
echo "Système d'exploitation : $(uname -a 2>&1)" >> "$summary_file"
echo "Heure de fin : $(date)" >> "$summary_file"

echo -e "\nRésumé des expériences enregistré dans $summary_file"
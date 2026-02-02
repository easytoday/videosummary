#!/bin/bash

# Script Bash pour exécuter des expériences avec différents modèles (DR-DSN, R-DSN, D-DSN, D-DSN-nolambda)
# sur deux jeux de données (SumMe et TVSum) et calculer les scores F1 moyens

# Définition des paramètres
datasets=("summe" "tvsum")
splits=(0 1 2 3 4)  # 5 splits différents

# Types de récompense et noms de modèles correspondants
declare -A reward_types
reward_types["dr"]="DR-DSN"
reward_types["d"]="D-DSN"
reward_types["r"]="R-DSN"
reward_types["d-nolambda"]="D-DSN-nolambda"

# Fonction pour exécuter une expérience spécifique
run_experiment() {
    local dataset=$1
    local split=$2
    local reward_type=$3
    local model_name=$4
    local train=$5  # true ou false
    local evaluate=$6  # true ou false
    
    local dataset_file="dataset_original/eccv16_dataset_${dataset}_google_pool5.h5"
    local split_file="dataset_original/${dataset}_splits.json"
    local save_dir="log/${model_name}-${dataset}-split${split}"
    local model_path="${save_dir}/model_epoch60.pth.tar"
    
    # Créer le répertoire s'il n'existe pas
    mkdir -p "$save_dir"

    # Entraîner le modèle si demandé
    if [ "$train" = true ]; then
        echo "Entraînement de ${model_name} sur ${dataset} (Split ${split})"
        python main.py -d "$dataset_file" -s "$split_file" -m "$dataset" --save-dir "$save_dir" --gpu 0 --split-id "$split" --verbose --reward-type "$reward_type"
    fi

    # Évaluer le modèle si demandé et si le modèle existe
    if [ "$evaluate" = true ] && [ -f "$model_path" ]; then
        echo "Évaluation de ${model_name} sur ${dataset} (Split ${split})"
        python main.py -d "$dataset_file" -s "$split_file" -m "$dataset" --save-dir "$save_dir" --gpu 0 --split-id "$split" --evaluate --resume "$model_path" --verbose --save-results --reward-type "$reward_type"
    elif [ "$evaluate" = true ]; then
        echo "Évaluation ignorée pour ${model_name} sur ${dataset} (Split ${split}) : Modèle non trouvé"
    fi
}

# Fonction pour extraire le score F1 d'un fichier log
get_f1_score() {
    local log_file=$1
    
    if [ -f "$log_file" ]; then
        f1=$(grep -oP "Average F-score \K[0-9.]+(?=%)" "$log_file")
        if [ -n "$f1" ]; then
            echo "scale=4; $f1/100" | bc
            return 0
        fi
    fi
    return 1
}

# Créer un fichier pour stocker les résultats
summary_file="experiment_summary.txt"
echo "Résultats des Expériences - $(date)" > "$summary_file"

# Exécuter les expériences pour chaque combinaison : type de récompense, jeu de données et split
for reward_type in "${!reward_types[@]}"; do
    model_name=${reward_types["$reward_type"]}
    
    for dataset in "${datasets[@]}"; do
        results_key="${model_name}-${dataset}"
        declare -a scores
        
        for split in "${splits[@]}"; do
            # Exécuter l'expérience
            run_experiment "$dataset" "$split" "$reward_type" "$model_name" true true
            
            # Collecter les résultats
            log_file="log/${model_name}-${dataset}-split${split}/log_test.txt"
            f1_score=$(get_f1_score "$log_file")
            
            if [ -n "$f1_score" ]; then
                scores+=("$f1_score")
                echo "${model_name} sur ${dataset} (Split ${split}) : Score F1 = $f1_score"
            else
                echo "${model_name} sur ${dataset} (Split ${split}) : Impossible de récupérer le score F1" >&2
            fi
        done
        
        # Calculer la moyenne des scores F1 si des résultats existent
        if [ ${#scores[@]} -gt 0 ]; then
            sum=0
            for score in "${scores[@]}"; do
                sum=$(echo "$sum + $score" | bc)
            done
            avg=$(echo "scale=4; $sum / ${#scores[@]}" | bc)
            formatted=$(printf "%.2f%%" $(echo "$avg * 100" | bc))
            echo "${results_key} : $formatted (Splits : ${#scores[@]}/5)"
            
            # Enregistrer le résultat dans le fichier
            echo "${results_key} : $formatted (Scores individuels : ${scores[*]})" >> "$summary_file"
        else
            echo "${results_key} : Aucun résultat valide" >&2
            echo "${results_key} : Aucun résultat valide" >> "$summary_file"
        fi
    done
done

echo -e "\nRésumé des expériences enregistré dans $summary_file"

import os
import h5py
import torch
# On importe vos fonctions de récompense
from rdiv import compute_diversity_reward 
# On importe vos fonctions de représentativité
from rrep import compute_representativeness_reward

# --- 1. CONFIGURATION DES CHEMINS ---
features_path = "dataset_cpu/features.h5"
log_dir = "outputs/logs"
log_path = os.path.join(log_dir, "training_log.csv")
#log_path = "outputs/logs/training_log.csv"

# Création du dossier de log si inexistant
os.makedirs("outputs/logs", exist_ok=True)

# --- 2. INITIALISATION DU FICHIER DE LOG (L'entête) ---
# On le fait UNE SEULE FOIS au début du script
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,video_id,reward_div,reward_rep,reward_total\n")

# --- 3. BOUCLE D'ENTRAÎNEMENT PRINCIPALE ---
def train():
    # Chargement de vos données GoogLeNet
    with h5py.File(features_path, 'r') as hdf:
        video_ids = list(hdf.keys()) # ['v1', 'v2', 'v10'...]
        print(f"Vidéos détectées dans le H5 : {len(video_ids)}")
        
        for epoch in range(1, 11): # Exemple : 10 époques
            for v_id in video_ids:
                # Lecture des features 1024 de la vidéo
                features = torch.tensor(hdf[v_id][:])
                
                # --- SIMULATION DU MODÈLE (En attendant le BiLSTM) ---
                # Ici le modèle prédira quelles images choisir. 
                # Pour le test, on prend des images au hasard :
                indices = torch.randperm(len(features))[:5] 
                selected_features = features[indices]

                # --- 4. CALCUL DES RÉCOMPENSES (L'article de Zhou) ---
                # Calcul de la Diversité (nécessite uniquement les frames choisies)
                r_div = compute_diversity_reward(selected_features)
                       
                # Calcul de la Représentativité 
                # (Nécessite TOUTES les frames ET les frames choisies)
                r_rep = compute_representativeness_reward(features, selected_features)
                
                # Calcul de la récompense totale combinée
                reward_total = r_div + r_rep

                # --- 5. ENREGISTREMENT (C'est ici qu'on utilise le CSV !) ---
                print(f"Vidéo {v_id} | R_div: {r_div:.4f} | R_rep: {r_rep:.4f} | Total: {reward_total:.4f}")
                #print(f"Epoch {epoch} - Vidéo {v_id} : Reward Div = {r_div:.4f}")

                
                with open(log_path, 'a') as f:
                    # 'a' signifie "append" (ajouter à la fin du fichier)
                    f.write(f"{epoch},{v_id},{r_div:.4f},{r_rep:.4f},{reward_total:.4f}\n")

if __name__ == "__main__":
    train()

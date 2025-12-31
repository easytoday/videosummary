import os
import h5py
import torch
import torch.optim as optim
from torch.distributions import Bernoulli

# Importation de vos modules personnels
from model import DSN
from rdiv import compute_diversity_reward
from rrep import compute_representativeness_reward

# 1. CONFIGURATION
features_path = "dataset_cpu/features.h5"
log_path = "outputs/logs/training_log.csv"
save_path = "outputs/models/dsn_model.pth"
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# 2. INITIALISATION
# On crée le modèle défini dans model.py
model = DSN(input_dim=1024, hidden_dim=256)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialisation du log CSV avec l'entête
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write("epoch,video_id,reward_div,reward_rep,reward_total,loss\n")

def train(epochs=50): # Aziz epoch=50
    model.train() # Mode entraînement
    
    with h5py.File(features_path, 'r') as hdf:
        video_ids = list(hdf.keys())
        
        for epoch in range(1, epochs + 1):
            for v_id in video_ids:
                # A. Chargement des données (T, 1024)
                features = torch.tensor(hdf[v_id][:]).unsqueeze(0) # Ajout dimension batch (1, T, 1024)
                
                # B. Forward Pass (Appel à model.py)
                # Le modèle prédit une probabilité pour chaque frame
                probs = model(features) # Sortie: (1, T, 1)
                probs = probs.squeeze()  # Devient (T)
                
                # C. Sélection d'actions (REINFORCE)
                # On utilise une distribution de Bernoulli pour décider de prendre ou non une frame
                dist = Bernoulli(probs)
                actions = dist.sample() # Vecteur de 0 et 1 (T)
                log_probs = dist.log_prob(actions) # Pour le calcul du gradient
                
                # Indices des frames sélectionnées
                selected_indices = torch.where(actions > 0.5)[0]
                
                # Sécurité : Si aucune frame n'est choisie, on passe
                if len(selected_indices) < 2:
                    continue
                
                # D. Calcul des récompenses (Appel à rdiv.py et rrep.py)
                raw_features = features.squeeze(0)
                selected_features = raw_features[selected_indices]
                
                r_div = compute_diversity_reward(selected_features)
                r_rep = compute_representativeness_reward(raw_features, selected_features)
                R = (r_div + r_rep) / 2 # Récompense totale
                
                # E. Calcul de la Loss (Algorithme REINFORCE)
                # L'article de Zhou minimise : -Reward * log_prob
                # On ajoute une pénalité de régularisation pour la longueur (optionnel mais conseillé)
                loss = -R * log_probs.mean()
                
                # F. Backpropagation (Mise à jour des poids du BiLSTM)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # G. Sauvegarde des logs
                with open(log_path, 'a') as f:
                    f.write(f"{epoch},{v_id},{r_div:.4f},{r_rep:.4f},{R:.4f},{loss.item():.4f}\n")
            
            print(f"Époque {epoch} terminée. Modèle sauvegardé.")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()

import torch
import torch.nn.functional as F

def compute_diversity_reward(selected_features):
    """
    Calcule la récompense de diversité selon l'équation (10) de l'article.
    
    Args:
        selected_features (torch.Tensor): Les vecteurs de 1024 features des images 
                                          choisies par le modèle (forme : N x 1024)
    Returns:
        float: La valeur de la récompense de diversité.
    """
    # Si moins de 2 images sont sélectionnées, la diversité est nulle
    n_selected = selected_features.size(0)
    if n_selected < 2:
        return 0.0

    # 1. Normalisation des vecteurs pour faciliter le calcul de la similarité cosinus
    # On divise chaque vecteur par sa norme L2
    norm_features = F.normalize(selected_features, p=2, dim=1)

    # 2. Calcul de la matrice de similarité cosinus (N x N)
    # S_ij = cos_sim(v_i, v_j)
    # Puisque les vecteurs sont normalisés, le produit matriciel donne la similarité
    sim_matrix = torch.mm(norm_features, norm_features.t())

    # 3. Application de l'équation de l'article
    # L'article définit la diversité comme la moyenne des (1 - similarité)
    # On ne prend pas en compte la diagonale (car l'image est identique à elle-même)
    
    # Somme de toutes les similarités
    total_sim = sim_matrix.sum() - torch.trace(sim_matrix)
    
    # Moyenne des similarités
    avg_sim = total_sim / (n_selected * (n_selected - 1))
    
    # La récompense est l'inverse de la similarité moyenne
    reward_div = 1.0 - avg_sim
    
    return reward_div.item()

# --- EXEMPLE D'UTILISATION ---
# Imaginons que le modèle ait sélectionné 3 images (index 10, 50, 100 de votre dataset v1)
# features_v1 est chargé depuis votre fichier .h5
# features_selectionnees = features_v1[[10, 50, 100], :] 

# r_div = compute_diversity_reward(torch.tensor(features_selectionnees))
# print(f"Récompense de Diversité : {r_div}")

import torch

def compute_representativeness_reward(all_features, selected_features):
    """
    Calcule la récompense de représentativité (Eq 9 de l'article).
    
    Args:
        all_features (torch.Tensor): Toutes les frames de la vidéo (T x 1024)
        selected_features (torch.Tensor): Frames sélectionnées (N x 1024)
    """
    if selected_features.size(0) == 0:
        return 0.0

    # 1. Calcul des distances Euclidiennes au carré entre toutes les frames 
    # et les frames sélectionnées. Résultat: matrice (T x N)
    # dist(x, y)^2 = ||x||^2 + ||y||^2 - 2 * x.t * y
    dist_matrix = torch.cdist(all_features, selected_features, p=2)

    # 2. Pour chaque frame de la vidéo originale, on trouve la distance 
    # vers la frame sélectionnée la plus proche
    min_distances, _ = torch.min(dist_matrix, dim=1)

    # 3. La récompense est l'exponentielle négative de la moyenne de ces distances
    # Plus les distances sont petites, plus le score est proche de 1
    avg_min_dist = torch.mean(min_distances)
    reward_rep = torch.exp(-avg_min_dist)

    return reward_rep.item()

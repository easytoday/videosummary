import torch

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Calcul de la récompense combinée (Diversité + Représentativité)
    Basé sur l'implémentation officielle de Kaiyang Zhou.
    """
    _seq = seq.detach()
    _actions = actions.detach()
    
    # Récupérer les indices des frames sélectionnées (où action == 1)
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    
    if pick_idxs.numel() == 0:
        reward = torch.tensor(0.)
        return reward.cuda() if use_gpu else reward

    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    _seq = _seq.squeeze()
    n = _seq.size(0)

    # --- 1. CALCUL DE LA DIVERSITÉ ---
    if num_picks == 1:
        reward_div = torch.tensor(0.)
    else:
        # Normalisation L2 pour la similarité cosinus
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        # Matrice de dissimilarité (1 - Cosine Sim)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        
        if ignore_far_sim:
            # On ignore la similarité des frames très éloignées dans le temps
            pick_mat = pick_idxs.view(num_picks, 1).expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
            
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # --- 2. CALCUL DE LA REPRÉSENTATIVITÉ ---
    # Calcul efficace des distances euclidiennes au carré
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    
    dist_mat = dist_mat[:, pick_idxs]
    # Pour chaque frame de la vidéo, on trouve la distance à la frame sélectionnée la plus proche
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())

    # --- 3. COMBINAISON ---
    reward = (reward_div + reward_rep) * 0.5
    return reward.cuda() if use_gpu else reward
